# Seed + Emergent IR: Causal Reasoning via Discrete Intermediate Representations

**Status:** V8 Implementation Complete - Investigating IR Semantic Grounding
**Current Phase:** Analyzing causality results and planning next experiments

## Overview

This project implements a hybrid architecture combining structural tags with emergent VQ codes to create interpretable intermediate representations (IR) for multi-step reasoning. The goal is to enforce genuine causal reasoning where the model must use the IR buffer to solve problems.

## Architecture

```
Input → [Pass 1: IR Generation] → IR Buffer → [Pass 2: Answer Generation] → Output

IR Buffer Structure:
<GOAL>[c047][c089]</GOAL>
<STEP>[c124][c256]</STEP>
<CHECK>[c078]</CHECK>

Where:
- Structural tags (<GOAL>, <STEP>, etc.) provide reasoning scaffolding
- Emergent codes (c000-c511) are learned via vector quantization
- Cross-attention in Pass 2 forces IR dependency
```

## Key Components

### V8 Architecture (Current)
- **VQ Codebook:** 512 discrete codes, 128-dim embeddings
- **Gumbel-Softmax Warm-start:** Temperature annealing (1.2 → 0.6) for stable training
- **Contrastive Learning:** InfoNCE loss for HL-IR alignment
- **IR→Value Head:** Auxiliary regression head for explicit IR supervision
- **Grammar Enforcement:** Structural constraints on IR generation
- **Anti-Collapse Measures:** Diversity loss, frequency debiasing, coverage loss

### Training Configuration
- **Model:** Pythia-70M (105.9M params) with LoRA
- **Dataset:** Arithmetic reasoning (8000 train, 1000 val)
- **Loss Components:**
  - Answer cross-entropy (primary task)
  - VQ commitment loss (codebook alignment)
  - Contrastive loss (HL-IR grounding)
  - IR→Value loss (semantic supervision)
  - Coverage loss (attention distribution)

## Current Results (V8)

### Training Metrics (2 epochs, ~1600 steps)
| Metric | Value | Status |
|--------|-------|--------|
| Val Accuracy | 6.27% | ✓ Above 5% threshold |
| IR Integrity | 100% | ✓ Perfect structural validity |
| Codebook Util (eval) | 26.95% | ✓ Above 20% threshold |
| Codebook Util (train) | 12.34% | - |
| Val Loss | 8.70 | - |

### Causality Ablation Results
| Condition | Accuracy | Relative Drop |
|-----------|----------|---------------|
| **intact** | 6.00% | — |
| **random-IR** | 6.00% | 0.0% |
| **shuffle-IR** | 6.00% | 0.0% |
| **drop-IR** | 0.00% | 100.0% |

**Interpretation:** IR is **structurally necessary but semantically weak**
- Removing IR completely causes 100% accuracy drop (model requires IR to generate)
- Random/shuffled IR performs identically to intact IR (model ignores IR content)
- Conclusion: Model has learned a shortcut that bypasses IR semantics

## Project Structure

```
seed_emergent_ir/
├── code/
│   ├── models/
│   │   ├── causal_ir_model_v2.py    # Main V8 model (two-pass architecture)
│   │   └── ir_value_head.py         # IR→Value regression head
│   ├── ir_generator_v2.py           # IR generation with contrastive loss
│   ├── vq.py                        # Vector quantization with diversity enforcement
│   ├── vq_tied_generation.py        # VQ-tied code generation
│   ├── ir_grammar.py                # Structural grammar enforcement
│   ├── train_v2.py                  # Training script with V8 metrics
│   ├── run_minimal_ablations.py     # Causality testing harness
│   └── tools/
│       └── run_single_batch_debug.py # Single-batch debugging
├── data/arithmetic/                 # Training and validation data
├── checkpoints/                     # Model checkpoints and logs
├── reports/                         # Training reports
├── V8_RESULTS_P1.1-P1.4.md         # Complete V8 results and analysis
├── IMPLEMENTATION_P1.3_P1.4.md     # P1.3-P1.4 technical details
└── README.md                        # This file
```

## Quick Start

### Training V8 Model
```bash
cd code
bash train_mini_sanity.sh  # 2 epochs on Pythia-70M
```

### Running Causality Ablations
```bash
cd code
python run_minimal_ablations.py \
  --checkpoint ../checkpoints/ir_cot_70m_mini_sanity/best_model.pt \
  --val_data ../data/arithmetic/val.json \
  --num_examples 150 \
  --output results.json
```

### Key Training Parameters
```bash
--use_ir_value_head          # Enable IR→Value auxiliary head
--ir_value_weight 0.25       # Weight for IR→Value loss
--use_contrastive            # Enable contrastive HL-IR alignment
--contrastive_weight 0.3     # Weight for contrastive loss
--use_gumbel_warmstart       # Enable Gumbel temperature annealing
--gumbel_tau 0.6             # Final Gumbel temperature
--eval_code_sampling softmax # Eval-mode sampling (softmax/argmax/gumbel)
```

## Next Steps

Based on V8 ablation results, three main directions:

### Option A: Strengthen IR Semantics
- Increase `ir_value_weight` (0.25 → 0.5+)
- Increase `contrastive_weight` (0.3 → 0.5+)
- Add explicit IR→answer alignment loss
- Experiment with different contrastive temperatures

### Option B: Architectural Constraints
- Mask input embeddings during Pass-2 (force IR dependency)
- Multi-head IR attention (separate reasoning streams)
- Hierarchical IR (global + local codes)
- Explicit IR→scratchpad→answer pipeline

### Option C: Training Improvements
- Longer training (10+ epochs)
- Curriculum learning (simple → complex problems)
- Data augmentation (paraphrase questions)
- Larger model (Pythia-410M or 1B)

## Key Findings

### What Works
- ✅ VQ codebook trains stably (26.95% utilization in eval)
- ✅ IR generation maintains 100% structural integrity
- ✅ Grammar enforcement prevents malformed IR
- ✅ Anti-collapse measures keep codes diverse
- ✅ Model learns basic arithmetic (6.27% validation accuracy)

### What Doesn't Work Yet
- ❌ IR codes don't encode semantic information (0% drop with random IR)
- ❌ Model ignores IR content during answer generation
- ❌ Low overall accuracy (6.27% vs target 70%+)
- ❌ IR→Value head metrics not captured in snapshots

### Critical Insight
The two-pass architecture creates a structural dependency (model needs *some* IR to generate answers) but hasn't learned semantic dependency (model doesn't care *what* the IR contains). This suggests the model is solving problems directly in the input→answer mapping, using IR merely as a positional delimiter.

## Documentation

- **V8_RESULTS_P1.1-P1.4.md:** Complete V8 implementation, training results, and ablation analysis
- **IMPLEMENTATION_P1.3_P1.4.md:** Technical details of P1.3 (V8 metrics) and P1.4 (standardized codebook metrics)
- **V8_ANTICOLLAPSE_SUMMARY.md:** Anti-collapse measures and codebook diversity techniques
- **PROGRESS_REPORT.md:** Overall project progress and milestones
- **reports/V8_Training_Report_20251114.md:** Historical training report

## Related Work

**Previous Experiment:** VQ Bottleneck (`../vq_bottleneck/`)
- ✅ Achieved 61% codebook utilization
- ❌ But codes were not causally used for reasoning

**This Experiment:** Seed + Emergent IR
- ✅ Forces IR usage through two-pass architecture
- 🔄 Working to achieve genuine semantic causality

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)

## Citation

If you use this code or findings in your research, please cite:

See `CITATION.cff` at the repository root.

## License

Apache-2.0 (see `LICENSE` at the repository root).

## Contact

Paul Provost · PI Project · https://pi-project.ai · pi.project@outlook.com