# Experiment Separation: VQ Bottleneck vs. Seed + Emergent IR

**Purpose:** Clarify the difference between the two experiments to avoid confusion and allow clean execution.

---

## Experiment 1: VQ Bottleneck (COMPLETED)

**Location:** `ThinkTokens/` (root directory)

**What It Did:**
- Took pretrained Pythia models (410M, 1.4B)
- Added hard VQ bottleneck at middle layer (layer 12/24)
- Forced ALL information through 512 discrete codes
- Trained on GSM8K (2000 samples, 3 epochs)

**Key Code Files:**
- `vq_model_v2.py` - VQ model with forward hooks
- `train_multisize.py` - Training script
- `eval_multisize.py` - Evaluation script

**Results:**
- ✅ **Technical Success:** 61% codebook utilization (codes genuinely used)
- ❌ **Task Failure:** 0% accuracy on GSM8K
- ❌ **Scaling Insight:** 410M = 1.4B performance (bottleneck dominates)

**Why It Failed:**
Model was compressing information through codes but routing around them for task solution. The codes were passive (used but not causal).

**Output:**
- `docs/results/EXPERIMENT_REPORT.md` - Full analysis
- `docs/results/figures/` - 4 visualization graphs
- Checkpoints in `checkpoints_410M/`, `checkpoints_1.4B/`

---

## Experiment 2: Seed + Emergent IR (NEW - Starting Now)

**Location:** `ThinkTokens/seed_emergent_ir/` (NEW DIRECTORY)

**What It Will Do:**
- Train SMALL model from scratch (not fine-tuning)
- Use structural tags (GOAL, STEP, etc.) + emergent codes (VQ)
- **FORCE** decoder to use IR buffer (architectural constraint)
- Verify causality with diagnostic tests
- Start simple (arithmetic), scale to GSM8K

**Key Code Files (To Be Built):**
- `code/ir_generator.py` - IRBufferGenerator + VQ codebook
- `code/decoder.py` - ConstrainedDecoder (forced cross-attention)
- `code/models/causal_ir.py` - CausalIRModel (integration)
- `training/train.py` - Training loop
- `evaluation/causal_tests.py` - Diagnostic suite

**Expected Improvements Over VQ Experiment:**
1. **Causality enforced** - Architectural constraint (not hoped-for)
2. **Testable** - Run diagnostic tests to prove IR is used
3. **Simpler first** - Start with arithmetic before GSM8K
4. **Hybrid approach** - Seeds (tags) + emergence (codes)

**Expected Output:**
- Trained model on arithmetic + GSM8K
- Causal test results (must show IR is essential)
- Analysis of learned codes
- Publication-ready findings

---

## Key Differences

| Aspect | VQ Bottleneck | Seed + Emergent IR |
|--------|---------------|-------------------|
| **Model** | Pretrained (Pythia) | From scratch (nanoGPT base) |
| **Bottleneck** | Hard VQ, no structure | Structured tags + VQ codes |
| **Causality** | Not enforced | FORCED (architecture) |
| **Testing** | No causality checks | Full diagnostic suite |
| **Dataset** | GSM8K immediately | Arithmetic → GSM8K |
| **Scale** | Large (410M, 1.4B) | Small (10M-50M initially) |
| **Status** | ✅ Complete | 🚀 Ready to start |

---

## Why Experiment 2 Solves Experiment 1's Problem

**Experiment 1 Problem:** Codes used but not causal
```
Input → VQ Code (ignored for task) → Output
           ↓
        (Passive compression)
```

**Experiment 2 Solution:** Codes are causal
```
Input → Encoder → IR Buffer → Decoder (MUST use IR)
                   ↓
              <GOAL>[code]</GOAL>
              <STEP>[code]</STEP>
                   ↓
            Forced Cross-Attention
           (No bypass allowed)
                   ↓
                 Output
```

**Architectural difference:**
- VQ Bottleneck: Codes computed, but decoder can ignore them
- Seed+Emergent: Decoder has ZERO access to input without IR buffer

---

## File & Directory Management

### Experiment 1 Files (DO NOT TOUCH)
```
ThinkTokens/
├── vq_model_v2.py              ← VQ experiment
├── train_multisize.py          ← VQ experiment
├── eval_multisize.py           ← VQ experiment
├── checkpoints_410M/           ← VQ results
├── checkpoints_1.4B/           ← VQ results
├── results_410M/               ← VQ results
├── results_1.4B/               ← VQ results
└── docs/results/               ← VQ report
```

### Experiment 2 Files (NEW - Everything Here)
```
ThinkTokens/seed_emergent_ir/   ← NEW DIRECTORY
├── code/
│   ├── models/
│   │   └── causal_ir.py        ← NEW
│   ├── ir_generator.py         ← NEW
│   ├── decoder.py              ← NEW
│   ├── training/
│   │   └── train.py            ← NEW
│   └── evaluation/
│       └── causal_tests.py     ← NEW
├── data/
│   ├── arithmetic_generator.py ← NEW
│   └── arithmetic_*.json       ← Generated
├── models/
│   └── causal_ir_arithmetic_final.pt  ← Checkpoint
├── results/
│   ├── phase4_results.json     ← Results
│   ├── phase5_causal_tests.json ← Diagnostic results
│   └── figures/                ← Analysis figures
└── logs/
    └── training.log            ← Training log
```

---

## Execution Workflow

### How to Keep Them Separate

**When running Experiment 1 code:**
```bash
cd ThinkTokens/
python train_multisize.py --model 410M
python eval_multisize.py --model 410M
```

**When running Experiment 2 code:**
```bash
cd ThinkTokens/seed_emergent_ir/
python training/train.py
python evaluation/causal_tests.py
```

### Git Workflow

**Experiment 1 is DONE - don't commit again:**
```bash
# (Already committed in previous session)
git log | grep "VQ\|bottleneck"
```

**Experiment 2 is NEW - clean commits by phase:**
```bash
cd seed_emergent_ir
git init  # New repo or add to main
git add code/ir_generator.py code/decoder.py code/models/causal_ir.py
git commit -m "Phase 1: Implement seed+emergent IR architecture"

# Then after each phase:
git add data/arithmetic_*.json
git commit -m "Phase 2: Generate arithmetic dataset"

# etc.
```

---

## Prevention of Mixing

**To ensure no accidental mixing:**

1. **Use separate directories** (✅ seed_emergent_ir/)
2. **Use separate git branches** (optional)
3. **Use separate conda envs** (optional, but good practice)
4. **Different requirements.txt** (if using different dependencies)

**Set reminder:**
- When working on Seed+Emergent IR, ONLY touch `seed_emergent_ir/`
- If you need VQ code, read it from `docs/results/EXPERIMENT_REPORT.md`, don't copy
- Never import from vq_model_v2.py into seed_emergent_ir code

---

## Documentation

### For Experiment 1 (Reference Only)
- `docs/results/EXPERIMENT_REPORT.md` - Full details
- `docs/results/README.md` - Overview
- `docs/results/figures/` - Results visualization

### For Experiment 2 (Active Work)
- `docs/SEED_EMERGENT_IR_GUIDE.md` - Detailed implementation (6 phases)
- `docs/QUICK_START_SEED_EMERGENT.md` - Quick checklist (this file)
- `docs/EXPERIMENT_SEPARATION.md` - This document
- `docs/SEED_EMERGENT_IR_ARCHITECTURE.md` - Architecture details (to be created)

---

## If You Get Confused

**Quick Check:**
- Working on codes and VQ? → Experiment 1 (DONE, read-only)
- Working on tags and IR buffer? → Experiment 2 (active, in progress)
- Not sure? → Check which directory you're in

**To verify you're in Experiment 2:**
```bash
pwd
# Should output: .../ThinkTokens/seed_emergent_ir
# NOT: .../ThinkTokens
```

---

## Summary

- **Experiment 1 (VQ):** Showed what doesn't work (codes passive) ✅ COMPLETE
- **Experiment 2 (Seed+Emergent):** Tests what should work (codes active) 🚀 READY TO START

**Clear separation = clean execution = better research.**

When starting Experiment 2, use `docs/QUICK_START_SEED_EMERGENT.md` as your checklist.

Do NOT mix files, do NOT reuse VQ code, do NOT modify Experiment 1 results.

---

**Next Action:** Begin Phase 0 of Experiment 2
```bash
cd ThinkTokens/
mkdir -p seed_emergent_ir/{data,code/models,code/training,code/evaluation,logs,results}
echo "✓ Project structure ready for Seed + Emergent IR experiment"
```