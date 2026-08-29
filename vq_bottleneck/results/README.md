# VQ Bottleneck Experiment Results

## Structure

```
vq_bottleneck/results/
├── results_410M/           ← 410M model evaluation results
│   ├── vq_results.json     (accuracy, tokens, codebook stats)
│   └── [evaluation data]
├── results_1.4B/           ← 1.4B model evaluation results
│   ├── vq_results.json     (accuracy, tokens, codebook stats)
│   └── [evaluation data]
└── results_old_baseline/   ← Old baseline results (early experiment)
    ├── vq_results.json
    ├── baseline.json
    ├── training_history.json
    ├── training_config.json
    ├── training_history.png
    ├── comparison_plot.png
    └── analysis_report.txt
```

## Key Results

### 410M Model
- **Accuracy:** 0%
- **Codebook Usage:** 313/512 (61.1%)
- **Avg Tokens:** 180.4
- **File:** `results_410M/vq_results.json`

### 1.4B Model
- **Accuracy:** 0%
- **Codebook Usage:** 314/512 (61.3%)
- **Avg Tokens:** 66.9
- **File:** `results_1.4B/vq_results.json`

## Key Finding

Both models show:
- ✅ High codebook utilization (codes genuinely used)
- ❌ Zero accuracy (codes not causal for task)
- ❌ Scale-invariant performance (410M ≈ 1.4B)

See: `../../docs/results/EXPERIMENT_REPORT.md` for full analysis
