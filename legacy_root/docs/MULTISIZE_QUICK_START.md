# Multi-Scale Training - Quick Start Guide

## Overview

You now have a complete multi-scale training system. You can:
- Train any model size independently
- Keep results separate and organized
- Compare results across scales
- Re-train any model anytime

---

## Quick Commands

### Training

```bash
# Train 1.4B with quick dataset (2 epochs, 500 samples) - ~2 hours
python train_multisize.py --model 1.4B --dataset quick

# Train 410M with quick dataset - ~24 minutes
python train_multisize.py --model 410M --dataset quick

# Train with medium dataset (3 epochs, 2000 samples)
python train_multisize.py --model 1.4B --dataset medium

# Train with full dataset (3 epochs, 7473 samples)
python train_multisize.py --model 1.4B --dataset full

# Train all sizes at once
python train_multisize.py --model all --dataset quick
```

### Evaluation

```bash
# Evaluate 1.4B model
python eval_multisize.py --model 1.4B

# Evaluate 410M model
python eval_multisize.py --model 410M

# Evaluate all trained models
python eval_multisize.py --model all

# Evaluate with different number of samples
python eval_multisize.py --model 1.4B --samples 200
```

### Comparison

```bash
# Compare 410M vs 1.4B
python compare_scales.py --models 410M,1.4B

# Compare all trained models
python compare_scales.py --models 160M,410M,1.4B,2.8B
```

---

## File Organization

```
results/                    ← Original 410M baseline results
checkpoints/               ← Original 410M checkpoints

results_410M/              ← 410M VQ results (if retrained)
checkpoints_410M/          ← 410M checkpoints (if retrained)

results_1.4B/              ← 1.4B VQ results
checkpoints_1.4B/          ← 1.4B checkpoints

results_160M/              ← 160M results (if trained)
checkpoints_160M/          ← 160M checkpoints

results_2.8B/              ← 2.8B results (if trained)
checkpoints_2.8B/          ← 2.8B checkpoints

comparison_all_scales.txt  ← Side-by-side comparison
```

Each model size has its own:
- **results_XXX/baseline.json** - Baseline metrics
- **results_XXX/vq_results.json** - VQ evaluation results
- **results_XXX/training_history.json** - Training progress
- **results_XXX/training_config.json** - Configuration used

---

## Recommended Workflow

### Step 1: Validate Scaling (Next)
```bash
# Train 1.4B with quick dataset to validate hypothesis
python train_multisize.py --model 1.4B --dataset quick

# Evaluate
python eval_multisize.py --model 1.4B

# Compare to 410M
python compare_scales.py --models 410M,1.4B
```

**Time:** ~2.5 hours
**Output:** Answers "Does scaling help?"

### Step 2: Improve Codebook Usage (If needed)
```bash
# Re-train 410M with more data
python train_multisize.py --model 410M --dataset full

# Re-train 1.4B with more data
python train_multisize.py --model 1.4B --dataset full

# Compare improved results
python compare_scales.py --models 410M,1.4B
```

**Time:** ~6-7 hours total
**Output:** Better compression and higher code usage

### Step 3: Full Scale Suite (If publishing)
```bash
# Test all sizes
python train_multisize.py --model all --dataset full

# Evaluate all
python eval_multisize.py --model all

# Generate comprehensive comparison
python compare_scales.py --models 160M,410M,1.4B,2.8B
```

**Time:** ~20+ hours
**Output:** Complete scaling curve for publication

---

## Configuration Details

### Available Models
- **160M** - Pythia-160M (smallest)
- **410M** - Pythia-410M (baseline)
- **1.4B** - Pythia-1.4B (scaling test)
- **2.8B** - Pythia-2.8B (larger scale)

### Available Datasets
- **quick** - 500 samples, 2 epochs (24min for 410M, 2hr for 1.4B)
- **medium** - 2000 samples, 3 epochs (1.5hr for 410M, 4hr for 1.4B)
- **full** - 7473 samples, 3 epochs (5hr for 410M, 15hr for 1.4B)

### Batch Sizes (Auto-optimized for RTX 4070)
- 160M: batch_size=8
- 410M: batch_size=4
- 1.4B: batch_size=2
- 2.8B: batch_size=1

---

## Expected Results

### 410M (What you have)
```
Accuracy:   3.0%
Tokens:     205 (-31.9% vs baseline)
Codebook:   42.6% (218/512 codes)
Time:       24 min (quick), 1.5 hours (full)
```

### 1.4B (What to expect)
```
Accuracy:   7-15% (should improve with scale)
Tokens:     150-180 (better compression expected)
Codebook:   45-55% (should improve with scale)
Time:       2 hours (quick), 5 hours (full)
```

### Scaling Hypothesis
If 1.4B shows:
- ✅ Higher accuracy: Real reasoning improvement
- ✅ Better codebook usage: Codes more refined at scale
- ✅ Token reduction persists: Compression is real
→ **Hypothesis confirmed: Scale helps**

If 1.4B shows:
- ✗ Lower accuracy: Scaling hurts
- ✗ Lower code usage: Pattern collapse
- ✗ No token reduction: Scaling doesn't help
→ **Alternative hypothesis: Scaling limitation**

---

## Troubleshooting

### Out of Memory for 1.4B
```python
# Reduce batch size further
# Edit train_multisize.py, MODEL_CONFIGS["1.4B"]["batch_size"] = 1
```

### Models Not Found
```bash
# First time downloads models automatically
# This takes time on first run
# Check disk space: df -h
# Check internet connection
```

### Want to Re-run 410M
```bash
# Simply run:
python train_multisize.py --model 410M --dataset quick

# This creates new results_410M/ without affecting original results/
```

### Want to Compare All Results
```bash
# After training, run:
python compare_scales.py --models 410M,1.4B

# This creates comparison_all_scales.txt
```

---

## Next Steps

### Immediate (Now)
1. Run: `python train_multisize.py --model 1.4B --dataset quick`
2. Wait ~2 hours
3. Run: `python eval_multisize.py --model 1.4B`
4. Run: `python compare_scales.py --models 410M,1.4B`
5. Check `comparison_all_scales.txt` for results

### If Scaling Looks Good
6. Run: `python train_multisize.py --model 1.4B --dataset full`
7. Get higher quality results for publication

### If You Want More Validation
8. Run: `python train_multisize.py --model all --dataset quick`
9. Test 160M and 2.8B as well
10. Generate complete scaling curve

---

## Files Created

| File | Purpose |
|------|---------|
| `train_multisize.py` | Main training script |
| `eval_multisize.py` | Evaluation script |
| `compare_scales.py` | Comparison script |
| `MULTISIZE_QUICK_START.md` | This file |

All are self-contained and use argparse for easy configuration changes.

---

## Key Advantages

✅ **Flexibility** - Train any model size anytime
✅ **Organization** - Separate results per model
✅ **Repeatability** - Config saved with each model
✅ **Comparison** - Easy side-by-side metrics
✅ **Scaling** - Add 160M/2.8B later without changes
✅ **Memory** - Auto-optimized batch sizes per model

---

**Start with:** `python train_multisize.py --model 1.4B --dataset quick`

Good luck! 🚀
