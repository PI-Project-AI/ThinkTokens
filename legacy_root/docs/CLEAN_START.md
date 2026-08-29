# Clean Project Start Guide

**Date:** October 25, 2025
**Status:** Project reorganized and ready for Experiment 2

---

## What Changed

The project has been reorganized for **clarity and consistency**:

### Before (Messy)
```
ThinkTokens/
├── vq_model_v2.py          ← Scattered files
├── train_multisize.py
├── eval_multisize.py
├── checkpoints_410M/        ← Results scattered
├── results_410M/
├── checkpoints_1.4B/
├── results_1.4B/
└── [30+ other files]
```

### After (Organized)
```
ThinkTokens/
├── vq_bottleneck/          ← Experiment 1 (complete)
│   ├── code/
│   ├── checkpoints/
│   ├── results/
│   ├── logs/
│   └── README.md
│
├── seed_emergent_ir/       ← Experiment 2 (ready)
│   ├── code/
│   ├── data/
│   ├── checkpoints/
│   ├── results/
│   ├── logs/
│   └── README.md
│
├── docs/                   ← Documentation
│   ├── results/            (VQ experiment report)
│   ├── SEED_EMERGENT_IR_GUIDE.md
│   ├── QUICK_START_SEED_EMERGENT.md
│   └── [other guides]
│
└── PROJECT_STRUCTURE.md    ← Navigation map
```

---

## Files to Ignore (Old Root Level)

These are old files from initial experiments. They can stay but are not used:

- `analyze_results.py` (old script)
- `compare_scales.py` (old script)
- `eval_baseline.py` (old script)
- `train_simple.py` (old script)
- `train_vq.py` (old script)
- `run_pipeline.py` (old script)
- `vq_model.py` (old, replaced by `vq_bottleneck/code/vq_model_v2.py`)
- `checkpoints/` (old)
- `results/` (old)
- Various `*.md` files (old)

**These don't hurt anything; they're just not in the new structure.**

---

## What You Need to Know

### For Experiment 1 (Reference Only)
```
Location: vq_bottleneck/
Status: ✅ Complete
Don't modify these files - they're archived results.
Read the report: docs/results/EXPERIMENT_REPORT.md
```

### For Experiment 2 (Active)
```
Location: seed_emergent_ir/
Status: 🚀 Ready to begin Phase 1
All code templates in: docs/SEED_EMERGENT_IR_GUIDE.md
Daily checklist: docs/QUICK_START_SEED_EMERGENT.md
```

---

## How to Start Working on Experiment 2

### Step 1: Understand the Setup
```bash
# Read the master map
cat PROJECT_STRUCTURE.md

# Read the quick start
cat docs/QUICK_START_SEED_EMERGENT.md
```

### Step 2: Navigate to Experiment 2
```bash
cd seed_emergent_ir
pwd
# Should output: .../ThinkTokens/seed_emergent_ir
```

### Step 3: Begin Phase 1
```bash
# Read the implementation guide
cat ../docs/SEED_EMERGENT_IR_GUIDE.md | head -200

# Start implementing:
# - code/ir_generator.py
# - code/decoder.py
# - code/models/causal_ir.py
```

---

## Key Principle: Clean Separation

**Never mix experiments:**
- Work on Experiment 2? Stay in `seed_emergent_ir/`
- Need to reference Experiment 1? Read from `docs/results/`
- Never copy code from `vq_bottleneck/` into `seed_emergent_ir/`

**Why?** Clarity. Each experiment is self-contained and tells a complete story.

---

## Project Navigation Map

**Start here when confused:**
```
1. Where am I?
   → pwd  (should show seed_emergent_ir or vq_bottleneck)

2. What am I building?
   → cat [experiment]/README.md

3. How do I build it?
   → cat ../../docs/SEED_EMERGENT_IR_GUIDE.md (for Experiment 2)

4. What's the overall structure?
   → cat PROJECT_STRUCTURE.md

5. I have a specific question about...
   → Architecture? → docs/SEED_EMERGENT_IR_ARCHITECTURE.md
   → Timeline? → docs/QUICK_START_SEED_EMERGENT.md
   → Separation? → docs/EXPERIMENT_SEPARATION.md
   → Project map? → PROJECT_STRUCTURE.md
```

---

## Git Workflow for Experiment 2

**Each phase gets its own commit:**

```bash
# Phase 1: Architecture
git add seed_emergent_ir/code/ir_generator.py seed_emergent_ir/code/decoder.py seed_emergent_ir/code/models/
git commit -m "Phase 1: Implement Seed+Emergent IR architecture"

# Phase 2: Data
git add seed_emergent_ir/data/
git commit -m "Phase 2: Generate arithmetic dataset"

# Phase 4: Training
git add seed_emergent_ir/models/ seed_emergent_ir/results/phase4_results.json
git commit -m "Phase 4: Train on arithmetic, verify causality passes"

# Etc.
```

**This means:** If you hit a problem and need to restart, you can always `git checkout` the last working phase.

---

## Success Indicators

### Right Now ✅
- [x] VQ experiment complete and archived
- [x] New experiment directory created
- [x] All documentation written
- [x] Project structure organized
- [x] Ready to code

### Soon (After Phase 1)
- [ ] Architecture implemented (3 Python files)
- [ ] All modules tested independently

### Later (After Phase 4)
- [ ] Training complete on arithmetic
- [ ] Causality tests all pass
- [ ] Ready to scale to GSM8K

### Publication ✅ (After Phase 6)
- [ ] Full results on GSM8K
- [ ] Analysis complete
- [ ] Ready for paper/blog

---

## If You Get Confused

**"I don't remember what we're building"**
→ Read: `seed_emergent_ir/README.md`

**"I don't know what to do next"**
→ Read: `docs/QUICK_START_SEED_EMERGENT.md`

**"I want to understand the architecture"**
→ Read: `docs/SEED_EMERGENT_IR_GUIDE.md` Section 1-2

**"I'm about to make a mess mixing experiments"**
→ Read: `docs/EXPERIMENT_SEPARATION.md`

**"I need the master map"**
→ Read: `PROJECT_STRUCTURE.md`

---

## Final Checklist

Before starting Phase 1, verify:

- [ ] You understand the VQ experiment (read the report)
- [ ] You understand why it failed (seeds + emergent IR solves it)
- [ ] You've read the quick start guide
- [ ] You're in the `seed_emergent_ir/` directory
- [ ] You have the implementation guide (`SEED_EMERGENT_IR_GUIDE.md`)
- [ ] You understand the 3 core components (IRGen, Decoder, CausalIR)
- [ ] You're ready to implement Phase 1

---

## You're Ready

The project is clean, organized, and documented.

**Everything you need is here. Let's build Experiment 2.**

```bash
cd seed_emergent_ir
cat ../docs/QUICK_START_SEED_EMERGENT.md

# Begin Phase 1: Implement Architecture
```

**Questions? Re-read the documentation map above. All answers are there.**