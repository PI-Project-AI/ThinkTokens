# Debug Bundle Implementation Guide

## Overview

Comprehensive debug infrastructure has been implemented to diagnose the subtle training collapse where:
- Answer loss drops to ~0.0000
- Validation accuracy remains at 0%
- Training metrics appear numerically healthy (tok=8, temperature annealing correct)
- VQ loss collapses to ~0.001

## Components Implemented

### A) Single-Batch Step Dumps (Every 500 Steps)

**Location**: `code/debug_logger.py` → `log_step_dump()`

**Output**: `{output_dir}/logs/debug_epoch{E}_step{S}.json`

**Content** (5 random examples per dump):
```json
{
  "epoch": 1,
  "step": 500,
  "examples": [
    {
      "index": 3,
      "hl_input": "What is 234 + 567? [truncated to 128 chars]...",
      "ir_generated": {
        "text": "<IR_START><GOAL>c_45 c_102 c_78</GOAL>...",
        "structure": [
          {"tag": "GOAL", "num_codes": 3, "codes": [45, 102, 78]},
          {"tag": "STEP", "num_codes": 4, "codes": [12, 89, 34, 56]}
        ],
        "validity": {
          "starts_with_ir_start": true,
          "balanced_tags": true,
          "codes_per_step": [3, 4],
          "spans_count": 2
        }
      },
      "answer_generated": "801",
      "answer_topk_positions": {
        "pos_0": {
          "tokens": ["8", "7", "9", "6", "5"],
          "logits": [12.3, 11.2, 10.8, 10.1, 9.7],
          "probs": [0.42, 0.28, 0.19, 0.08, 0.03]
        },
        "pos_1": {...},
        "pos_2": {...}
      },
      "code_topk_positions": {
        "note": "Code topk requires IR generator modification",
        "positions": [0, 1, 2]
      },
      "eos_probs": [0.001, 0.002, 0.003],
      "counts": {
        "num_answer_tokens_with_loss": 8,
        "answer_len": 20,
        "ir_len": 15
      }
    }
  ]
}
```

**What to Look For**:
- **Early-EOS**: `eos_probs` > 0.9 at position 0 → Model predicting EOS immediately
- **Single-code collapse**: All `ir_generated.structure[*].codes` are identical
- **Malformed IR loops**: `validity.balanced_tags = false` or `codes_per_step` outside [3-6]
- **Masked-out answers**: `num_answer_tokens_with_loss = 0` despite non-PAD input

### B) Distribution & Health Metrics (Every Epoch)

**Location**: `code/debug_logger.py` → `log_epoch_metrics()`

**Output**: `{output_dir}/logs/epoch{E}_metrics.json`

**Content**:
```json
{
  "epoch": 1,
  "answer_ce": {
    "mean": 1.234,
    "std": 0.456,
    "min": 0.8,
    "max": 2.1
  },
  "vq_loss": {
    "mean": 0.145,
    "std": 0.023
  },
  "gradient_norms": {
    "mean": 2.34,
    "max": 8.92
  },
  "ir_integrity_pct": 87.5,
  "codebook_utilization": {
    "mean": 0.45,
    "std": 0.12
  },
  "coverage_loss": {
    "mean": 0.012,
    "std": 0.004
  },
  "temperature": {
    "mean": 0.655,
    "final": 0.640
  }
}
```

**What to Look For**:
- **Answer CE collapse**: `mean` < 0.01 by epoch 3 → Trivial solution learned
- **Codebook collapse**: `utilization.mean` < 0.1 → Codes collapsed to single point
- **Gradient vanishing**: `gradient_norms.mean` < 0.01 → No learning signal
- **VQ collapse**: `vq_loss.mean` < 0.01 → Codebook not being used

### C) Hard Assertions (Fail Fast)

**Location**: `code/debug_logger.py` → `assert_training_health()`

**Assertions** (enabled with `--enable_assertions`):
1. **ASSERT**: `num_answer_tokens_with_loss > 0` → Prevents all-PAD batches
2. **ASSERT**: `ir_error_rate < 0.2` (after 100 batches) → Prevents malformed IR
3. **ASSERT**: `codebook_utilization > 0.1` (after 100 batches) → Prevents single-code collapse

**Behavior**: Training **immediately terminates** with clear error message when assertion fails.

### D) Control Experiments

**Location**: `code/debug_logger.py` → `run_control_experiment_random_ir()`

**Usage**: Add `--run_control_random_ir` flag

**Test**: Replace generated IR with random codes, measure accuracy drop
- **Expected**: Accuracy < 10% if IR is genuinely used
- **Failure**: Accuracy > 50% indicates model ignoring IR

## Usage

### Quick Start

```bash
cd code
bash train_410m_debug.sh
```

This script runs 5 epochs with:
- Debug dumps every 500 steps
- Epoch metrics every epoch
- Hard assertions enabled
- Random-IR control at end

### Manual Usage

```bash
python train_v2.py \
  --model_name EleutherAI/pythia-410m \
  --batch_size 8 \
  --num_epochs 5 \
  --enable_debug \
  --debug_step_frequency 500 \
  --enable_assertions \
  --run_control_random_ir \
  --output_dir ../checkpoints/debug_run
```

### Inspect Debug Logs

```bash
# View step dumps
cat ../checkpoints/debug_run/logs/debug_epoch1_step500.json | jq

# View epoch metrics
cat ../checkpoints/debug_run/logs/epoch1_metrics.json | jq

# Check for early-EOS collapse
jq '.examples[].eos_probs[0]' ../checkpoints/debug_run/logs/debug_epoch*.json

# Check codebook utilization trend
jq '.codebook_utilization.mean' ../checkpoints/debug_run/logs/epoch*.json
```

## Diagnostic Workflow

### Step 1: Run Debug Training
```bash
bash code/train_410m_debug.sh
```

### Step 2: Check Epoch Metrics
```bash
for i in {1..5}; do
  echo "Epoch $i:"
  jq '{answer_ce: .answer_ce.mean, codebook_util: .codebook_utilization.mean, vq_loss: .vq_loss.mean}' \
    checkpoints/ir_cot_410m_debug/logs/epoch${i}_metrics.json
done
```

**Red flags**:
- Answer CE < 0.01 by epoch 3
- Codebook util < 0.15
- VQ loss < 0.01

### Step 3: Inspect Step Dumps (If Collapse Detected)
```bash
# Check first dump of collapsed epoch
jq '.examples[] | {eos_prob: .eos_probs[0], answer: .answer_generated, ir_valid: .ir_generated.validity}' \
  checkpoints/ir_cot_410m_debug/logs/debug_epoch3_step*.json | head -20
```

**Diagnosis patterns**:
1. **Early-EOS collapse**: All `eos_prob` > 0.9 at position 0
2. **Single-answer collapse**: All `answer` identical (e.g., all "0")
3. **IR malformed**: `ir_valid.balanced_tags = false` or `codes_per_step` out of range
4. **Code collapse**: All codes in `ir_generated.structure[*].codes` are same value

### Step 4: Check Control Experiment
```bash
tail -20 logs/train_410m_debug.log | grep "CONTROL"
```

**Expected output**:
```
[CONTROL] Random-IR Accuracy: 8.45%
[CONTROL] Expected: <10% if IR is genuinely used for reasoning
```

**If accuracy > 50%**: Model is ignoring IR entirely

## Current Training Status

The current training (train_410m_fixed.sh) is showing collapse at Epoch 5:
- Answer loss: 0.0000-0.0003 (collapsed)
- VQ loss: 0.0004-0.0020 (collapsed)
- Validation accuracy: 0% (confirmed in previous epochs)

**Next step**: Run debug training to identify root cause:
- Is it early-EOS prediction?
- Is it single-code collapse?
- Is it constant-answer prediction?
- Is the IR being ignored entirely?

## Files Modified

1. **code/debug_logger.py** (NEW): Debug logging infrastructure
2. **code/train_v2.py** (MODIFIED): Integrated debug logger with CLI flags
3. **code/train_410m_debug.sh** (NEW): Launch script with debug enabled

## CLI Arguments Added

```
--enable_debug              Enable comprehensive debug logging
--debug_step_frequency N    Log step dumps every N steps (default: 500)
--enable_assertions         Enable hard assertions for fail-fast
--run_control_random_ir     Run Random-IR control experiment after training
```

## Notes

- Debug logging adds ~5-10% overhead (step dumps are expensive)
- Hard assertions may terminate training early if collapse is severe
- Step dumps at 500-step frequency generate ~10-15 JSON files per epoch (7k examples / 8 batch = 875 batches → 2 dumps/epoch)
- Each step dump is ~50-100KB (5 examples with full metadata)
