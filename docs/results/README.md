# Experimental Results: Hard VQ Bottleneck for Discrete Reasoning

This directory contains comprehensive documentation of our experiments with hard vector quantization bottlenecks for discrete reasoning tokens.

## Contents

### 📊 Main Report
- **[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)** - Complete academic-style report with:
  - Abstract and introduction
  - Detailed methodology
  - Quantitative results and visualizations
  - Analysis and discussion
  - Future directions
  - Reproducibility information

### 📈 Visualizations

All figures are located in `figures/`:

1. **codebook_utilization.png** - Shows how many discrete codes each model uses
2. **generation_length.png** - Compares output token statistics between models
3. **performance_summary.png** - Overall comparison of key metrics
4. **scaling_analysis.png** - How performance changes with model size

## Quick Summary

### What We Did
- Implemented hard VQ bottleneck (Option A) using forward hooks
- Trained two Pythia models: 410M and 1.4B parameters
- Evaluated on GSM8K mathematical reasoning benchmark
- Forced all information through 512 discrete codes at layer 12

### What We Found

| Metric | 410M | 1.4B |
|--------|------|------|
| **Accuracy** | 0% | 0% |
| **Codebook Usage** | 61.1% | 61.3% |
| **Avg Tokens** | 180.4 | 66.9 |

**Key Insights:**
1. ✅ Hard bottleneck successfully enforced (61% code usage proves this)
2. ❌ No reasoning capability learned (0% accuracy on both models)
3. ❌ No scaling benefit (1.4B = 410M performance)
4. 🤔 Bottleneck constraint dominates model capacity

## Reading Guide

### For Quick Overview
- Read Section 1 (Introduction) and Section 3 (Results) of EXPERIMENT_REPORT.md
- Look at the 4 visualization figures

### For Technical Details
- Read Section 2 (Methodology) for architecture and training details
- Read Section 4 (Analysis) for why things didn't work
- Read Section 6 (Future Directions) for next steps

### For Implementation
- See Section 8 (Reproducibility) for commands and code structure
- All code is in the repository root directory

## Key Takeaways

### What This Proves
**Technical Success:**
- Hard VQ bottlenecks CAN be implemented in language models
- Forward hooks successfully enforce discrete information flow
- Training is stable (no collapse, no divergence)

**Task Failure:**
- Current approach does NOT achieve reasoning through bottleneck
- Simple fine-tuning insufficient for discrete reasoning
- Need fundamentally different training methodology

### What This Means
Discrete reasoning tokens (like Opus) are NOT "plug-and-play." They likely require:
- Massive scale (100B+ parameters)
- Specialized training (curriculum learning, from-scratch training)
- Much more data (millions of examples, not 2000)
- Possibly different architecture (not VQ)

## Next Steps

Based on these results, the most promising directions are:

1. **Immediate (Low Effort):**
   - Train much longer (50+ epochs)
   - Use full GSM8K dataset (7.5K examples)
   - Test on simpler tasks (arithmetic only)

2. **Short-term (Medium Effort):**
   - Implement curriculum learning (soft → hard bottleneck)
   - Try different codebook sizes (1024, 2048, 4096)
   - Add auxiliary losses for code diversity

3. **Long-term (High Effort):**
   - Alternative discretization methods (FSQ, Gumbel-Softmax)
   - Multi-bottleneck hierarchy
   - Train from scratch instead of fine-tuning

## Citation

If you reference this work, please cite:

```
ThinkTokens: Hard VQ Bottleneck Experiments (2025)
Experimental investigation of discrete reasoning tokens via vector quantization
https://github.com/[your-repo]/ThinkTokens
```

## Contact & Contributions

This is a research project documenting both successes and failures. Contributions, discussions, and alternative approaches are welcome!

See the main [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for detailed findings.

---

**Last Updated:** October 24, 2025
