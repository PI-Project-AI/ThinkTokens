# Intermediate Reasoning Language for LLMs - Research Documentation

**Project Lead:** Paul PROVOST
**Documentation:** Claude Code
**Date:** October 21, 2025
**Status:** Pre-POC Planning Phase

---

## 📚 Documentation Structure

This research project is documented across multiple files for different purposes. **Start here** to navigate:

### 🎯 **For Quick Overview (5 min read)**
**→ [Executive Summary](#executive-summary)** (below)

### 🔬 **For Understanding the Research (15 min read)**
**→ [Memo - Proof of Concept for an Intermediate Reasoning Language in LLMs.md](Memo - Proof of Concept for an Intermediate Reasoning Language in LLMs.md)**
- Original research proposal
- Problem statement & hypothesis
- High-level architecture options

### ⚠️ **For Critical Concerns (20 min read)**
**→ [Scaling Concerns - Intermediate Reasoning Language POC.md](Scaling Concerns - Intermediate Reasoning Language POC.md)**
- Scale dependency analysis
- Risk mitigation strategies
- Multi-scale experimental protocol

### 🛠️ **For Implementation (2-3 hours read, ready to code)**
**→ [POC Implementation Guide - Intermediate Reasoning Language for LLMs.md](POC Implementation Guide - Intermediate Reasoning Language for LLMs.md)**
- Complete technology stack
- Production-ready code
- Training procedures
- Evaluation framework

---

## Executive Summary

### The Big Question

**Can LLMs reason more efficiently in a learned discrete intermediate representation (IR) rather than natural language chain-of-thought?**

### The Hypothesis

Current LLMs reason in natural language, which is:
- ❌ **Inefficient** - wastes 10-15x tokens on linguistic scaffolding
- ❌ **Ambiguous** - introduces redundancy and imprecision
- ❌ **Human-centric** - optimized for communication, not computation

**What if models could:**
- ✅ Learn their own symbolic reasoning vocabulary
- ✅ Reason in compact abstract tokens (not English)
- ✅ Reduce token cost by 30-50% without sacrificing accuracy

### The Approach

Insert a **VQ-VAE bottleneck** into a transformer:

```
Input Question
    ↓
Transformer Layers [1-N/2] ← Encode
    ↓
[8 Discrete Reasoning Codes] ← VQ Bottleneck (512-code vocabulary)
    ↓
Transformer Layers [N/2-N] ← Decode
    ↓
Output Answer
```

**Key innovation:** The reasoning vocabulary is **not hand-designed**. The model discovers its own codes through training.

### Critical Success Factors

#### ✅ **For this to be meaningful:**
1. Codebook utilization >50% (avoid collapse)
2. Token reduction >20% vs baseline
3. Accuracy within 5% of baseline
4. **IR benefits increase with model scale** (410M → 1.4B → 7B)

#### ⚠️ **Major risks:**
1. **Codebook collapse** - only 20-50 codes used (wasted capacity)
2. **Scale dependency** - works at 7B but not at 1B (can't validate cheaply)
3. **Shallow compression** - compresses surface patterns, not reasoning
4. **Interpretability loss** - can't debug what we can't understand

### Resource Requirements

#### **Minimum Viable POC** (validates basic concept)
- **Models:** Pythia-160M baseline + VQ
- **Dataset:** GSM8K (7.5K math problems)
- **Compute:** 1x GPU (RTX 3090 or V100), 4 hours
- **Cost:** ~$2-10 on cloud
- **Timeline:** 1 week
- **Risk:** Might not generalize to larger scales

#### **Scale-Robust POC** (validates scaling trend) ⭐ **RECOMMENDED**
- **Models:** Pythia-410M + 1.4B (baseline + VQ each)
- **Dataset:** GSM8K + SVAMP (transfer test)
- **Compute:** 1x GPU (A100), ~50 GPU-hours total
- **Cost:** ~$80-120 on cloud
- **Timeline:** 2-3 weeks
- **Benefit:** Know if scaling helps or hurts

#### **Full Research Paper** (publishable results)
- **Models:** Pythia 70M/410M/1.4B/2.8B (4 scales)
- **Datasets:** GSM8K, SVAMP, ARC
- **Ablations:** 3 codebook sizes × 3 token counts = 9+ runs
- **Compute:** ~150-200 GPU-hours
- **Cost:** ~$200-300
- **Timeline:** 1-2 months
- **Deliverable:** Conference submission (NeurIPS, ICLR, etc.)

### Expected Outcomes

#### **Optimistic Scenario** (30% probability)
- 40-60% token reduction at 7B scale
- Clear interpretable code clusters
- Cross-task generalization
- **Impact:** Changes how we think about LLM reasoning

#### **Realistic Scenario** (50% probability)
- 20-30% token reduction
- Partial interpretability
- Some accuracy trade-offs
- **Impact:** Interesting research contribution, limited practical use

#### **Pessimistic Scenario** (20% probability)
- Codebook collapse or no efficiency gain
- Doesn't scale beyond 1B models
- **Impact:** Negative result (still publishable - shows reasoning isn't easily compressible)

### Key Decisions to Make

#### 🤔 **Decision 1: Which experimental scale?**

| Option | Cost | Risk | Recommendation |
|--------|------|------|----------------|
| Minimum (160M only) | $10 | High (scaling unknown) | ❌ Too risky |
| **Scale-Robust (410M + 1.4B)** | **$100** | **Medium** | **✅ Sweet spot** |
| Full Suite (4 scales) | $300 | Low | ⏰ Do after validation |

**Paul's decision:** _______________

#### 🤔 **Decision 2: What's your success threshold?**

What would you consider "worth continuing"?

- [ ] Any measurable improvement (>5% token reduction)
- [ ] Modest improvement (>20% token reduction, <5% accuracy drop)
- [ ] Strong improvement (>30% token reduction, no accuracy drop)
- [ ] Only if interpretable AND efficient

**Paul's answer:** _______________

#### 🤔 **Decision 3: Timeline vs. Thoroughness?**

- [ ] **Fast iteration** (1 week, minimal validation, higher risk)
- [ ] **Balanced** (2-3 weeks, scale-robust, medium risk) ← Recommended
- [ ] **Comprehensive** (2 months, full research paper, low risk)

**Paul's choice:** _______________

---

## Next Steps

### ✅ **Completed**
- [x] Literature review and hypothesis formation
- [x] Architecture design (VQ bottleneck)
- [x] Implementation guide with code
- [x] Scaling risk analysis
- [x] Resource estimates

### 🔄 **To Do Next**

1. **Make key decisions** (see above)
2. **Set up environment** (see Implementation Guide, Section 6)
3. **Run baseline benchmarks** (establish what "normal" looks like)
4. **Implement VQ bottleneck** (code ready in Implementation Guide)
5. **Train first model** (start with recommended scale)
6. **Analyze results** (use provided analysis scripts)
7. **Decide: continue, pivot, or publish?**

### 📅 **Suggested Timeline (Scale-Robust Path)**

**Week 1:**
- Day 1-2: Environment setup, baseline training
- Day 3-4: VQ implementation and debugging
- Day 5-7: Training experiments (410M + 1.4B)

**Week 2:**
- Day 1-2: Evaluation on GSM8K + SVAMP
- Day 3-4: Analysis (codebook, probes, visualization)
- Day 5-7: Interpretation and decision point

**Week 3 (if positive):**
- Day 1-3: Additional ablations
- Day 4-7: Documentation and writeup

---

## Quick Reference: File Guide

### When to Read Each Document

**Before starting:**
1. This README (5 min)
2. Original Memo (15 min)
3. Scaling Concerns (20 min)

**When implementing:**
4. Implementation Guide - Sections 1-5 (architecture overview)
5. Implementation Guide - Sections 6-8 (code and training)

**When debugging:**
6. Implementation Guide - Section 10 (troubleshooting)

**When analyzing results:**
7. Implementation Guide - Section 9 (analysis tools)
8. Scaling Concerns - Section 3 (scale diagnostics)

**When deciding next steps:**
9. Scaling Concerns - Section 5 (decision tree)
10. Implementation Guide - Section 12 (future extensions)

---

## Critical Insights (Save You Hours of Reading)

### 💡 **Insight 1: Scale is THE key question**
Don't trust small model results alone. The trend matters more than absolute performance.

### 💡 **Insight 2: Codebook collapse is your #1 enemy**
Monitor codebook utilization from epoch 1. If <20%, something's wrong.

### 💡 **Insight 3: Even "failure" is valuable**
If this doesn't work, it tells us reasoning ISN'T easily compressible - that's a scientific contribution.

### 💡 **Insight 4: Interpretability vs Efficiency tradeoff**
You're trading human-readable CoT for efficient codes. Make sure you can still validate correctness.

### 💡 **Insight 5: Transfer testing is crucial**
Train on GSM8K, test on SVAMP. If it doesn't transfer, it's just memorization.

---

## Contact & Collaboration

**Primary researcher:** Paul PROVOST
**Documentation:** Claude Code (Anthropic)
**Date:** October 21, 2025

**Status:** Pre-implementation phase, seeking feedback and collaborators

---

## Licensing & Usage

This research documentation is provided for:
- Academic research
- Reproducibility studies
- Educational purposes

If you use this work, please cite appropriately and share your findings!

---

**Ready to start? → Go to [POC Implementation Guide](POC Implementation Guide - Intermediate Reasoning Language for LLMs.md)**

**Need to understand risks first? → Go to [Scaling Concerns](Scaling Concerns - Intermediate Reasoning Language POC.md)**

**Want the original vision? → Go to [Original Memo](Memo - Proof of Concept for an Intermediate Reasoning Language in LLMs.md)**
