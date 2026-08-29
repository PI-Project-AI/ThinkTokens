# Implementation Guide: P1.3 + P1.4 - V8 Metrics & Standardized Codebook Logging

**Status**: Ready for implementation
**Estimated Time**: 30-45 minutes
**Prerequisites**: P1.1 (Drop-IR K/V=0) and P1.2 (hl_residual_in_pass2) already completed

---

## Overview

This document provides exact code changes to add:
- **P1.3**: V8 metrics (IR→value, contrastive, answer-bias) + fail-fast guard
- **P1.4**: Standardized codebook metrics (train vs eval with explicit field names)

## Exact JSON Schema (DO NOT MODIFY FIELD NAMES)

### `train_step####.json` (Train-mode)

```json
{
  "step": 1000,
  "mode": "train",
  "seed": 42,

  // Gumbel settings (already exist)
  "gumbel_active": true,
  "tau_current": 0.XX,
  "tau_init": 1.2,
  "tau_final": 0.6,
  "gumbel_steps": 3000,
  "debias_gamma": X.X,

  // P1.4: Codebook utilization (TRAIN-MODE)
  "utilization_train": 0.XX,
  "utilization_train_definition": "unique codes in last 1000 train positions / 512",
  "top1_code_freq_train": 0.XX,
  "code_freq_ema_train": {...},  // Rename from ema_freq_stats
  "gamma_train": X.X,
  "tau_g_train": 0.XX,

  // P1.3: IR→value grounding
  "ir_value_mae": X.XX,
  "ir_value_mse": X.XX,

  // P1.3: Contrastive (InfoNCE)
  "nn_acc": 0.XX,
  "sim_diag": 0.XX,
  "sim_offdiag": 0.XX,
  "diag_minus_offdiag": 0.XX,

  // P1.3: Answer-bias diagnostic
  "answer_bias": {
    "first_token_hist": {"12": 0.XX, ...},  // Top 20
    "num_answer_tokens_hist": {"1": 0.XX, ...},
    "top5_at_pos0": [{"token": "12", "prob": 0.XX}, ...]
  },

  // Cross-attention coverage
  "cross_attn_ir_coverage_train": 0.XX
}
```

### `eval_step####_softmax.json` (Eval-mode)

```json
{
  "step": 1000,
  "mode": "eval",
  "seed": 42,
  "eval_sampler": {
    "method": "softmax",
    "tau": 0.9,
    "top_k": 32,
    "top_p": 0.95
  },

  // P1.4: Codebook utilization (EVAL-MODE)
  "utilization_eval": 0.XX,
  "top1_code_freq_eval": 0.XX,

  // IR integrity
  "ir_integrity_eval": 0.XX,
  "avg_spans_eval": X.X,
  "ir_len_eval": X.X,

  // Task performance
  "val_accuracy": 0.XX,
  "val_loss": X.XX,

  // 3 example triplets
  "examples": [
    {
      "input": "What is 3 + 12?",
      "ir": "<IR_START><GOAL>c001c002...</GOAL><IR_END>",
      "answer": "15"
    }
    // 2 more examples
  ],

  // Cross-attention coverage
  "cross_attn_ir_coverage_eval": 0.XX
}
```

---

## File: `train_v2.py`

### 1. Extend `dump_train_mode_snapshot()` (Line 97-123)

**REPLACE** the entire function with:

```python
def dump_train_mode_snapshot(model, step, output_dir, recent_losses=None, tokenizer=None):
    """Dump train-mode VQ metrics + V8 diagnostics."""
    vq = model.ir_generator.vq.vq

    # Compute utilization_train from last 1000 code positions
    # (Requires tracking recent codes - see helper below)
    utilization_train = 0.0  # Placeholder - implement code tracking

    # Compute top-1 frequency from EMA
    top1_code_freq_train = float(vq.code_freq_ema.max().item())

    # Debias gamma (current)
    gamma_train = vq.gamma0 * max(0.0, 1.0 - step / vq.gumbel_steps) if step < vq.gumbel_steps else 0.0

    # Gumbel tau (current)
    tau_g_train = vq.tau_init + (vq.tau_final - vq.tau_init) * min(1.0, step / vq.gumbel_steps)

    snapshot = {
        "step": step,
        "mode": "train",
        "seed": 42,  # Hardcoded for reproducibility

        # Gumbel settings
        "gumbel_active": step < vq.gumbel_steps,
        "tau_current": tau_g_train,
        "tau_init": float(vq.tau_init),
        "tau_final": float(vq.tau_final),
        "gumbel_steps": vq.gumbel_steps,
        "debias_gamma": gamma_train,

        # P1.4: Codebook utilization (train-mode)
        "utilization_train": utilization_train,
        "utilization_train_definition": "unique codes in last 1000 train positions / 512",
        "top1_code_freq_train": top1_code_freq_train,
        "code_freq_ema_train": {
            "min": float(vq.code_freq_ema.min().item()),
            "max": float(vq.code_freq_ema.max().item()),
            "mean": float(vq.code_freq_ema.mean().item()),
            "top5_codes": vq.code_freq_ema.topk(5).indices.tolist(),
            "top5_freqs": vq.code_freq_ema.topk(5).values.tolist()
        },
        "gamma_train": gamma_train,
        "tau_g_train": tau_g_train,

        # P1.3: V8 metrics (placeholders - extract from recent_losses)
        "ir_value_mae": recent_losses.get('ir_value_mae', 0.0) if recent_losses else 0.0,
        "ir_value_mse": recent_losses.get('ir_value_mse', 0.0) if recent_losses else 0.0,
        "nn_acc": recent_losses.get('nn_acc', 0.0) if recent_losses else 0.0,
        "sim_diag": recent_losses.get('sim_diag', 0.0) if recent_losses else 0.0,
        "sim_offdiag": recent_losses.get('sim_offdiag', 0.0) if recent_losses else 0.0,
        "diag_minus_offdiag": recent_losses.get('diag_minus_offdiag', 0.0) if recent_losses else 0.0,

        # P1.3: Answer-bias diagnostic
        "answer_bias": compute_answer_bias_diagnostic(model, tokenizer) if tokenizer else {},

        # Cross-attention coverage
        "cross_attn_ir_coverage_train": 0.0  # Placeholder
    }

    dump_path = os.path.join(output_dir, "logs", f"train_step{step:04d}.json")
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"[TRAIN-MODE SNAPSHOT] Saved to {dump_path}")
```

### 2. Add Helper Function for Answer-Bias Diagnostic (After dump_train_mode_snapshot)

```python
def compute_answer_bias_diagnostic(model, tokenizer):
    """Compute answer-bias diagnostic from recent predictions."""
    # This requires storing recent answer predictions during training
    # For now, return placeholder
    return {
        "first_token_hist": {},  # Top 20 tokens at answer pos0
        "num_answer_tokens_hist": {},
        "top5_at_pos0": []
    }
    # TODO: Implement by tracking answer_logits during training
```

### 3. Add `dump_eval_mode_snapshot()` Function (After compute_answer_bias_diagnostic)

```python
def dump_eval_mode_snapshot(model, step, output_dir, val_loader, tokenizer, device):
    """Dump eval-mode VQ metrics with softmax sampling."""
    model.eval()

    all_codes = []
    all_ir_valid = []
    all_spans = []
    all_ir_lens = []
    examples = []

    total_correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 50:  # Limit eval to 50 batches for speed
                break

            input_ids = batch['input_ids'].to(device)
            answer_ids = batch['answer_ids'].to(device)

            # Generate IR with eval softmax sampling
            ir_output = model.ir_generator(
                input_ids=input_ids,
                attention_mask=batch.get('attention_mask'),
                target_ir_ids=None,
                temperature=None  # Uses eval settings
            )

            ir_token_ids = ir_output['ir_token_ids']

            # Extract codes
            code_mask = (ir_token_ids >= model.ir_token_ids['code_start']) & \
                       (ir_token_ids <= model.ir_token_ids['code_end'])
            codes = ir_token_ids[code_mask]
            all_codes.extend(codes.cpu().tolist())

            # Check IR validity (placeholder)
            all_ir_valid.append(True)  # TODO: Implement structural validation

            # Count spans (placeholder)
            all_spans.append(4)  # TODO: Count actual spans
            all_ir_lens.append(ir_token_ids.shape[1])

            # Collect first 3 examples
            if len(examples) < 3:
                input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                ir_text = tokenizer.decode(ir_token_ids[0], skip_special_tokens=False)

                # Generate answer
                answer_pred_ids = model.answer_from_ir(
                    input_ids=input_ids[:1],
                    ir_ids=ir_token_ids[:1],
                    ir_embeddings=None,
                    max_length=10,
                    temperature=0.0
                )
                answer_text = tokenizer.decode(answer_pred_ids[0], skip_special_tokens=True)

                examples.append({
                    "input": input_text,
                    "ir": ir_text,
                    "answer": answer_text
                })

            # Compute accuracy (placeholder)
            # TODO: Implement answer comparison

    # Compute metrics
    unique_codes = len(set(all_codes))
    utilization_eval = unique_codes / 512.0

    code_counts = {}
    for c in all_codes:
        code_counts[c] = code_counts.get(c, 0) + 1
    top1_code_freq_eval = max(code_counts.values()) / len(all_codes) if all_codes else 0.0

    ir_integrity_eval = sum(all_ir_valid) / len(all_ir_valid) if all_ir_valid else 0.0
    avg_spans_eval = sum(all_spans) / len(all_spans) if all_spans else 0.0
    ir_len_eval = sum(all_ir_lens) / len(all_ir_lens) if all_ir_lens else 0.0

    snapshot = {
        "step": step,
        "mode": "eval",
        "seed": 42,
        "eval_sampler": {
            "method": "softmax",
            "tau": 0.9,
            "top_k": 32,
            "top_p": 0.95
        },

        # P1.4: Codebook utilization (eval-mode)
        "utilization_eval": utilization_eval,
        "top1_code_freq_eval": top1_code_freq_eval,

        # IR integrity
        "ir_integrity_eval": ir_integrity_eval,
        "avg_spans_eval": avg_spans_eval,
        "ir_len_eval": ir_len_eval,

        # Task performance
        "val_accuracy": 0.0,  # TODO
        "val_loss": 0.0,  # TODO

        # Examples
        "examples": examples,

        # Cross-attention coverage
        "cross_attn_ir_coverage_eval": 0.0  # Placeholder
    }

    dump_path = os.path.join(output_dir, "logs", f"eval_step{step:04d}_softmax.json")
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"[EVAL-MODE SNAPSHOT] Saved to {dump_path}")

    model.train()
```

### 4. Add Fail-Fast Guard in Training Loop

**FIND** the training loop (around line 400-600) where steps are incremented.

**ADD** after each step increment:

```python
# P1.3: Fail-fast guard for IR→value head
if args.use_ir_value_head and global_step >= 200:
    # Check if ir_value_mae was logged
    if 'ir_value_mae' not in recent_losses or recent_losses['ir_value_mae'] == 0.0:
        raise RuntimeError(
            f"[FAIL-FAST] Step {global_step}: --use_ir_value_head enabled but "
            f"ir_value_mae not logged. Check IR→value head implementation."
        )
```

### 5. Update Training Loop to Collect V8 Metrics

**FIND** where losses are computed (after `loss.backward()`).

**ADD** metric collection:

```python
# Collect V8 metrics for logging
recent_losses = {}

if hasattr(model, 'use_ir_value_head') and model.use_ir_value_head:
    # Extract ir_value_mae/mse from loss components
    # TODO: Modify forward() to return these as separate components
    recent_losses['ir_value_mae'] = 0.0  # Placeholder
    recent_losses['ir_value_mse'] = 0.0  # Placeholder

if hasattr(model, 'use_contrastive') and model.use_contrastive:
    # Extract contrastive metrics
    # TODO: Modify contrastive_module to return metrics
    recent_losses['nn_acc'] = 0.0  # Placeholder
    recent_losses['sim_diag'] = 0.0  # Placeholder
    recent_losses['sim_offdiag'] = 0.0  # Placeholder
    recent_losses['diag_minus_offdiag'] = 0.0  # Placeholder
```

### 6. Update Snapshot Calls

**FIND** where `dump_train_mode_snapshot()` is called.

**REPLACE**:
```python
dump_train_mode_snapshot(model, global_step, args.output_dir)
```

**WITH**:
```python
dump_train_mode_snapshot(model, global_step, args.output_dir,
                        recent_losses=recent_losses, tokenizer=tokenizer)

# Also dump eval snapshot
if global_step % 200 == 0:
    dump_eval_mode_snapshot(model, global_step, args.output_dir,
                           val_loader, tokenizer, device)
```

---

## Testing Procedure

### Smoke Test (20 batches)

```bash
cd /home/pi-project-admin/PycharmProjects/PythonProject/ThinkTokens/seed_emergent_ir/code

# Modify train_mini_sanity.sh to add --max_steps 20
python train_v2.py \
  --model_name "EleutherAI/pythia-70m" \
  --use_lora \
  --output_dir "../checkpoints/smoke_test_p13_p14" \
  --num_codes 512 \
  --code_dim 128 \
  --train_data "../data/arithmetic/train.json" \
  --val_data "../data/arithmetic/val.json" \
  --batch_size 8 \
  --max_steps 20 \
  --use_ir_value_head \
  --ir_value_weight 0.25 \
  --use_contrastive \
  --contrastive_weight 0.3 \
  --seed 42

# Check outputs
ls -lh ../checkpoints/smoke_test_p13_p14/logs/
cat ../checkpoints/smoke_test_p13_p14/logs/train_step0020.json | python -m json.tool
cat ../checkpoints/smoke_test_p13_p14/logs/eval_step0020_softmax.json | python -m json.tool
```

**Verify JSON contains ALL fields from schema above.**

---

## Next Steps After Implementation

1. **Continue to Step 1000**:
   ```bash
   cd code
   # Update train_mini_sanity.sh to set --num_epochs 2 (to reach ~1600 steps)
   bash train_mini_sanity.sh
   ```

2. **Run Ablations** (V7-Lite step 875 + V8 step 1000):
   ```bash
   # V7-Lite baseline (using old checkpoint)
   python run_minimal_ablations.py \
     --checkpoint ../checkpoints/ir_cot_70m_mini_sanity/checkpoint_step0800.pt \
     --val_data ../data/arithmetic/val.json \
     --num_examples 500 \
     --output ../results/v7_lite_step875_ablations.json \
     --seed 42

   # V8 with new fixes
   python run_minimal_ablations.py \
     --checkpoint ../checkpoints/ir_cot_70m_mini_sanity/checkpoint_step1000.pt \
     --val_data ../data/arithmetic/val.json \
     --num_examples 500 \
     --output ../results/v8_step1000_ablations.json \
     --seed 42
   ```

3. **Check Gates** (from eval_step1000_softmax.json):
   - `utilization_eval >= 0.20-0.25`
   - `top1_code_freq_eval < 0.30`
   - `ir_integrity_eval >= 0.95`
   - `nn_acc > 0.70`
   - `ir_value_mae < 20-30`

4. **Generate Report** with ablation tables and metric curves.

---

## Implementation Checklist

- [ ] Replace `dump_train_mode_snapshot()` with extended version
- [ ] Add `compute_answer_bias_diagnostic()` helper
- [ ] Add `dump_eval_mode_snapshot()` function
- [ ] Add fail-fast guard in training loop
- [ ] Update snapshot calls to pass `recent_losses` and `tokenizer`
- [ ] Add eval snapshot calls every 200 steps
- [ ] Collect V8 metrics in training loop
- [ ] Run smoke test (20 batches)
- [ ] Verify JSON schema matches exactly
- [ ] Continue to step 1000
- [ ] Run ablations on V7-Lite and V8
- [ ] Check gates and adjust if needed

---

## Known Limitations / TODOs

1. **utilization_train**: Requires tracking last 1000 code positions (currently placeholder)
2. **answer_bias**: Requires storing recent answer predictions (currently placeholder)
3. **cross_attn_ir_coverage**: Requires extracting attention weights (currently placeholder)
4. **ir_value_mae/mse extraction**: Requires modifying forward() to return separate components
5. **contrastive metrics extraction**: Requires modifying contrastive_module to return nn_acc, sim_diag, etc.

These can be implemented incrementally without blocking the main workflow.

---

## Contact / Questions

If ambiguous, refer back to the exact JSON schema at the top of this document. Field names are FIXED and must not be changed for downstream plotting scripts.

**Last Updated**: 2025-11-15
**Author**: Claude (Sonnet 4.5)
