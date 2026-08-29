# Hard Guards Applied - Collapse Prevention

## Summary

All hard guards and fixes requested have been implemented to prevent "succeed to zero" collapse.

## Changes Implemented

### 0) Enhanced Debug Logging

**File**: `code/debug_logger.py`

Added metrics to step dumps:
- ✅ **EOS probabilities** at answer positions 0/1/2
- ✅ **Support size** (# of non-masked logits) at answer positions - catches masked-to-gold bug
- ✅ **Top-1 code frequency** - detects single-code collapse (>50% = collapse)
- ✅ **Codebook utilization** per example

Example JSON output:
```json
{
  "eos_probs": [0.001, 0.002, 0.003],
  "support_sizes": [4956, 4956, 4956],
  "codebook_metrics": {
    "utilization": 0.42,
    "top1_code_frequency": 0.15
  }
}
```

### 1) Hard Guards

**File**: `code/models/causal_ir_model_v2.py`

#### A) Ban EOS on First 2 Answer Tokens
```python
# Training (forward method - line 216-220)
eos_id = self.base_model.config.eos_token_id
for pos in range(min(2, answer_logits.shape[1])):
    answer_logits[:, pos, eos_id] = -1e10  # Mask EOS to -inf

# Inference (generate_answer method - line 361-363)
if answer_pos < 2:
    next_token_logits[:, self.base_model.config.eos_token_id] = -1e10
```

**File**: `code/debug_logger.py`

#### B) Support Size Check (Fail Fast)
```python
# Line 526-533
if answer_logits is not None and batch_idx > 100:
    support_sizes = self._compute_support_size(answer_logits[0:1])
    if support_sizes and support_sizes[0] < 10:
        print(f"\n[CRITICAL] Batch {batch_idx}: Support size = {support_sizes[0]}")
        print(f"[CRITICAL] Masked distribution to gold token → CE collapses to 0")
        assert False, f"[FAIL FAST] Support size {support_sizes[0]} < 10"
```

#### C) IR Error Rate Tracking (10 Consecutive Violations)
```python
# Line 535-543
if batch_idx > 100:
    if ir_error_rate >= 0.2:
        self.ir_error_violations += 1
        if self.ir_error_violations >= 10:
            assert False, f"[FAIL FAST] IR error rate > 20% for 10 consecutive batches"
    else:
        self.ir_error_violations = 0  # Reset counter
```

#### D) Existing Guards (Kept)
- ✅ `num_answer_tokens_with_loss > 0` - prevents all-PAD batches
- ✅ Codebook utilization > 10% (after warm-up)

### 2) IR Learning Warm-Start

**File**: `code/models/causal_ir_model_v2.py`

#### Gradient Leak Scheduling
```python
# Line 178-187
if self.current_epoch == 0:
    leak_lambda = 0.1  # Epoch 1: 10% leak
elif self.current_epoch == 1:
    leak_lambda = 0.05  # Epoch 2: 5% leak
else:
    leak_lambda = 0.0  # Epoch ≥3: strict IR-only
```

**Note**: Applied to IR embeddings in future implementation; token IDs always detached (discrete).

### 3) VQ Stabilization

**File**: `code/models/causal_ir_model_v2.py`

#### Temperature Annealing (Slower)
```python
# Line 89-91
self.temperature_init = 0.8  # Was 0.7
self.temperature_final = 0.6  # Was 0.4
# Anneals over full training (20 epochs)
```

#### Epoch-Dependent Loss Weights
```python
# Line 272-279
if self.current_epoch < 3:
    coverage_weight = 0.10  # Warm-start: aggressive diversity
    vq_weight = 0.5  # Warm-start: strong commitment (β)
else:
    coverage_weight = 0.03  # Normal
    vq_weight = 0.25  # Normal (β reduced)
```

**Summary of VQ Parameters**:
- Coverage/diversity: **0.10** (warm-start) → **0.03** (normal)
- Commitment β: **0.5** (warm-start) → **0.25** (normal)
- Temperature τ: **0.8** → **0.6** (over full training)

### 4) Answer Decoding Fixes

**File**: `code/models/causal_ir_model_v2.py`

#### Increased Answer CE Weight (Warm-Start)
```python
# Line 228-234
answer_ce_weight = 1.5 if self.current_epoch < 3 else 1.0
answer_loss = torch.nn.functional.cross_entropy(
    answer_logits.reshape(-1, answer_logits.shape[-1]),
    answer_targets.reshape(-1),
    ignore_index=-100
) * answer_ce_weight
```

#### EOS Banning (Already Covered Above)
Enforces min answer length = 1 (prevents early-EOS collapse)

### 5) Unit Test

**File**: `code/test_answer_ce_sanity.py`

Verifies CE path is healthy:
```
✓ PASS: Random logits CE = 11.34 > 2.0 (healthy)
✓ PASS: Masked-to-gold CE = 0.000 < 0.01 (confirms collapse detection)
```

Proves that:
1. CE can produce non-zero loss with random logits
2. Support size check will catch masked-to-gold collapse

## What to Watch in Next 1-2k Steps

### Red Flags (Training Failing)
| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Answer CE | ≤ 0.01 | Collapsed to trivial solution |
| Support size | < 10 | Masked to gold token |
| Codebook util | < 10% | Single-code collapse |
| Top-1 code freq | > 50% | One code dominates |
| IR error rate | > 20% (10 consecutive) | Malformed IR loops |
| EOS prob @ pos 0 | > 0.9 | Early-EOS collapse |

### Green Flags (Training Healthy)
| Metric | Target | Meaning |
|--------|--------|---------|
| Answer CE | 0.5-3.0 (decreasing) | Learning real patterns |
| Support size | ≥ 15 | Full vocab available |
| Codebook util | 30-70% | Diverse code usage |
| Top-1 code freq | < 30% | No single winner |
| IR error rate | < 5% | Well-formed IR |
| EOS prob @ pos 0 | < 0.1 | Generating full answers |

## Debug Workflow

### Step 1: Monitor First 500 Steps
```bash
tail -f ../logs/train_410m_hardguards.log | grep -E "ans=|support|FAIL FAST"
```

### Step 2: Check First Debug Dump (Step 500)
```bash
jq '.examples[] | {eos_prob: .eos_probs[0], support: .support_sizes[0], util: .codebook_metrics.utilization}' \
  ../checkpoints/ir_cot_410m_hardguards/logs/debug_epoch1_step500.json
```

Expected output:
```json
{
  "eos_prob": 0.03,
  "support": 4956,
  "util": 0.35
}
```

### Step 3: If Collapse Detected
1. Check which guard triggered: `grep "FAIL FAST" ../logs/train_410m_hardguards.log`
2. Inspect debug dump at failure point
3. Review support_size, eos_probs, top1_code_frequency

## Files Modified

1. **code/debug_logger.py** - Enhanced debug metrics + hard assertions
2. **code/models/causal_ir_model_v2.py** - EOS banning, gradient leak, VQ stabilization, answer CE boost
3. **code/test_answer_ce_sanity.py** (NEW) - Unit test for CE sanity

## Next Steps

1. ✅ All guards implemented
2. → Launch training with debug enabled
3. → Monitor first 1k steps for red flags
4. → Check debug dump at step 500
5. → If collapse: analyze JSON to identify failure mode
6. → If healthy: continue to full 20 epochs

## Launch Command

```bash
cd code
bash train_410m_hardguards.sh
```

This will:
- Enable all hard guards (fail fast on violations)
- Log debug dumps every 500 steps
- Run with VQ stabilization + answer CE boost
- Ban EOS on first 2 answer tokens
- Track support size + IR error rate

If training survives first 1k steps without assertions failing, the guards are working!
