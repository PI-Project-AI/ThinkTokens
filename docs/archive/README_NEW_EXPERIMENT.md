# New Experiment: Seed + Emergent IR (Starting October 25, 2025)

**Status:** Planning complete, ready for Phase 0
**Context:** Previous VQ experiment identified the problem; this experiment tests the solution

---

## The Problem (From VQ Experiment)

The previous VQ bottleneck experiment showed:
- ✅ Codes WERE used (61% codebook utilization)
- ❌ But they were NOT causal (model bypassed them for task solution)

**Conclusion:** Hard VQ alone doesn't work. We need to enforce causality architecturally.

---

## The Solution (This Experiment)

**Hybrid Seed + Emergent IR:**

```
[Structural Tags]  +  [Emergent Codes]  =  [Causal IR]

<GOAL>[code_47]</GOAL>
<STEP>[code_89]</STEP>
      ↓
  FORCED Cross-Attention
  (No bypass to input)
      ↓
    Answer
```

**Key Innovation:** 
- Minimal structure (5-6 tags)
- Maximum emergence (512 learned codes)
- Mandatory causality (architectural constraint)

---

## Three Core Documents

1. **SEED_EMERGENT_IR_GUIDE.md** (20KB)
   - Comprehensive 6-phase implementation guide
   - All architecture code (IRBufferGenerator, ConstrainedDecoder, CausalIRModel)
   - Dataset generation, training, evaluation
   - **Read this when building**

2. **QUICK_START_SEED_EMERGENT.md** (5KB)
   - Phase-by-phase checklist
   - Key testing commands
   - Success criteria
   - Decision points
   - **Read this as your daily checklist**

3. **EXPERIMENT_SEPARATION.md** (4KB)
   - How VQ experiment differs from this one
   - File organization to avoid mixing
   - Git workflow for clean separation
   - **Read this to stay organized**

---

## Quick Timeline

| Phase | Days | Task | Status |
|-------|------|------|--------|
| 0 | 1 | Project setup | Ready |
| 1 | 2-5 | Implement architecture (IR, Decoder, Model) | Ready |
| 2 | 5-6 | Generate arithmetic dataset | Ready |
| 3 | 7-8 | Implement causal diagnostic tests | Ready |
| 4 | 9-12 | Train on arithmetic + test causality | Ready |
| 5 | 13-16 | Scale to GSM8K | Ready |
| 6 | 17-20 | Analysis + documentation | Ready |

**Total:** ~3-4 weeks to publication-ready results

---

## Success Criteria

**Must Pass (Critical):**
- ✅ Baseline accuracy ≥70% on arithmetic
- ✅ Random IR test: -30% accuracy drop
- ✅ Shuffle IR test: -30% accuracy drop
- ✅ Drop IR test: -40% accuracy drop
- ✅ Codebook usage 50-70%

**Nice to Have:**
- 🎯 +2-5% accuracy on GSM8K
- 🎯 20% token efficiency gain vs CoT
- 🎯 Learned codes cluster by problem type

---

## Start Now

### Step 1: Read & Understand
```bash
# Read the three core documents
cat docs/SEED_EMERGENT_IR_GUIDE.md
cat docs/QUICK_START_SEED_EMERGENT.md
cat docs/EXPERIMENT_SEPARATION.md
```

### Step 2: Create Project Structure
```bash
cd ThinkTokens/
mkdir -p seed_emergent_ir/{data,code/{models,training,evaluation},logs,results}
git add docs/SEED_EMERGENT_IR_GUIDE.md docs/QUICK_START_SEED_EMERGENT.md docs/EXPERIMENT_SEPARATION.md
git commit -m "Add Seed+Emergent IR experiment documentation"
```

### Step 3: Begin Phase 0
```bash
cd seed_emergent_ir
# Follow QUICK_START_SEED_EMERGENT.md Phase 0 section
```

---

## Key Insight

This experiment answers the question: **"Can we make discrete reasoning tokens that are genuinely causal?"**

By enforcing causality architecturally (not hoping for it in the loss), we can test whether emergent codes are actually useful for reasoning.

---

## Contact Points

- **Architecture questions?** → Read SEED_EMERGENT_IR_GUIDE.md Section 2
- **Step-by-step confused?** → Read QUICK_START_SEED_EMERGENT.md
- **File organization?** → Read EXPERIMENT_SEPARATION.md
- **Running into blocker?** → Check QUICK_START_SEED_EMERGENT.md "Common Issues"

---

**Ready? Begin Phase 0 now.**

This is well-defined, achievable, and addresses the exact limitation we discovered in the VQ experiment.

Let's build something that works.
