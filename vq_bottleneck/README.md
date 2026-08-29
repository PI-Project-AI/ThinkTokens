# VQ Bottleneck Experiment (COMPLETED)

**Status:** Completed October 24, 2025
**Findings:** Hard VQ bottlenecks work technically but lack causality for reasoning

## What This Experiment Did

- Implemented hard VQ bottleneck with forward hooks
- Trained 410M and 1.4B Pythia models on GSM8K
- Measured codebook utilization and task accuracy

## Key Results

| Metric | 410M | 1.4B |
|--------|------|------|
| Codebook Usage | 313/512 (61.1%) | 314/512 (61.3%) |
| Accuracy | 0% | 0% |
| Avg Tokens | 180.4 | 66.9 |

## Key Finding

✅ Codes WERE used (61% utilization proves hard bottleneck works)
❌ But codes were NOT causal (model bypassed them for task solution)

## Critical Insight

This experiment taught us: **Creating a discrete bottleneck is easy. Making it causal is hard.**

The model learned to compress information through codes but routed around them for actual reasoning.

## Documentation

- Full report: `../docs/results/EXPERIMENT_REPORT.md`
- Visualizations: `../docs/results/figures/`
- Checkpoints: `checkpoints/`

## Code

- `code/vq_model_v2.py` - VQ model with forward hooks
- `code/train_multisize.py` - Training script
- `code/eval_multisize.py` - Evaluation script

## Next Step

This experiment identified the problem. The next experiment (**Seed+Emergent IR**) enforces causality architecturally.

See: `../seed_emergent_ir/README.md`
