> **Archive note:** historical AI-assisted working document, kept verbatim and not maintained.

# Scaling Concerns: Will This Work Only At Large Scale?

**Author:** Claude Code (with Paul PROVOST)
**Date:** October 21, 2025
**Context:** Critical analysis of scale dependencies for IR reasoning POC

---

## The Core Concern

**Your observation:**
> "A 1B model can say total nonsense but scaled to 7B it's okay"

This is absolutely correct and represents one of the **biggest risks** in this research direction.

### What We Know About Scaling in LLMs

**Emergent capabilities appear suddenly at specific scales:**

| Capability | Emergence Point | Example |
|------------|----------------|---------|
| Multi-step reasoning | ~1-3B params | GPT-2 (1.5B) starts, GPT-3 (6.7B) reliable |
| In-context learning | ~1B params | Few-shot learning "clicks" |
| Chain-of-thought | ~10-100B params | Really works at GPT-3 scale |
| Complex abstraction | ~7B+ params | Llama-7B vs Llama-1B huge gap |

**The pattern:** Small models (<1B) often memorize surface patterns rather than learn true reasoning.

### Why This Matters for Our POC

**Risk scenario:**
1. We train Pythia-160M with VQ bottleneck
2. It achieves 30% token reduction on GSM8K
3. BUT it's just learned pattern matching: "if question contains 'total cost' → emit code #47"
4. This "pseudo-reasoning" doesn't transfer or scale
5. At 7B scale, the pattern breaks because real reasoning emerges

**Result:** Our PoC could show "success" that's actually meaningless.

---

## Evidence-Based Analysis: Will This Scale?

### Case Study 1: VQ-VAE Scaling in Vision

**What we know from DALL-E and similar:**

- **Small codebooks (256-512 codes) work at all scales** (from tiny autoencoders to DALL-E 2)
- **Emergent structure improves with scale** - larger models use codes more meaningfully
- **Core mechanism is scale-invariant** - quantization bottleneck works at 10M and 10B params

**Implication:** The VQ mechanism itself should scale. Question is whether *reasoning compression* scales.

### Case Study 2: Distillation & Compression

**Student-teacher distillation research shows:**

- **Small models CAN learn compressed representations** of larger models' knowledge
- **BUT they struggle with multi-hop reasoning** (Hinton et al., various papers)
- **Compression works better for recognition than reasoning** (classification > generation)

**Implication:** Our small models might learn "reasoning shortcuts" that don't generalize.

### Case Study 3: Sparse Attention & Bottlenecks

**Longformer, Reformer, etc. demonstrate:**

- **Attention bottlenecks work across scales** (128-token bottleneck in 100M-1B models)
- **Small models learn simpler patterns** in bottleneck, but structure persists when scaled
- **Key insight:** The bottleneck forces structure; scale determines quality

**Implication:** Even if small model codes are "dumb," scaling might make them "smart."

---

## The Scaling Hypothesis for IR Reasoning

### Optimistic View: "Bottleneck Enforces Structure"

**Argument:**
1. Small models already do *some* reasoning (just poorly)
2. VQ bottleneck forces this reasoning to be explicit/discrete
3. At larger scale, the *same mechanism* captures richer reasoning
4. Codes discovered at 160M might be "proto-reasoning" that matures at 7B

**Evidence supporting this:**
- Mechanistic interpretability shows reasoning circuits exist even in small models (Neel Nanda's work)
- Superposition hypothesis suggests small models compress many features - bottleneck might help
- Code semantics could improve continuously with scale (no sharp phase transition needed)

### Pessimistic View: "Reasoning is Emergent, Compression Isn't"

**Argument:**
1. True reasoning emerges only at 1B+ scale
2. Below that threshold, models just pattern-match
3. Compressing non-existent reasoning is meaningless
4. We'll see "success" at small scale that's actually fake

**Evidence supporting this:**
- GSM8K requires genuine multi-step reasoning (Wei et al. show CoT doesn't work <1B)
- Quantization might just compress the *wrong thing* (surface patterns, not reasoning)
- Small models might use codes as "lookup table" (question type → answer template)

### Most Likely Reality: "It Depends on What We Measure"

**Nuanced prediction:**

At **small scale (160M-410M)**:
- Codes will compress *something* (token efficiency likely)
- That "something" will be partial reasoning + pattern matching
- We'll see modest improvements but not deep abstraction

At **medium scale (1B-3B)**:
- Real reasoning begins to emerge
- Codes start capturing genuine multi-step logic
- Transfer across tasks improves
- This is the **critical validation zone**

At **large scale (7B+)**:
- Codes should represent rich reasoning primitives
- Efficiency gains compound
- Cross-domain generalization
- **This is where the hypothesis truly pays off**

---

## Scale-Robust Experimental Design

### Strategy 1: Multi-Scale Training from Day 1

**Instead of just testing 160M, test 3-4 points on the scaling curve:**

```yaml
Models to test:
  - Pythia-70M      # Below reasoning threshold (control)
  - Pythia-410M     # At reasoning threshold
  - Pythia-1.4B     # Above threshold (critical test)
  - Pythia-2.8B     # Larger (if budget allows)
```

**Why this matters:**
- If 70M shows same gains as 1.4B → probably just compression, not reasoning
- If gains *increase* with scale → genuine reasoning being captured
- If gains *only* appear at 1.4B+ → need that scale for it to work

**Compute cost:**
- 70M + 410M + 1.4B: ~$50-80 total on cloud (6-10 GPU-hours each)
- Totally feasible for PoC budget

### Strategy 2: Scale-Diagnostic Metrics

**Don't just measure accuracy - measure *reasoning quality*:**

#### Metric 1: Cross-Task Transfer

```python
# Train on GSM8K, test on SVAMP (different phrasing)
transfer_score = accuracy_svamp / accuracy_gsm8k

# If transfer_score high (>0.8) → genuine reasoning
# If low (<0.5) → task-specific pattern matching
```

**Prediction:**
- Small models (160M): transfer ~0.4-0.6 (weak reasoning)
- Large models (1.4B+): transfer ~0.7-0.9 (real reasoning)

#### Metric 2: Compositional Generalization

```python
# Test on problems requiring N steps
# Compare baseline vs VQ model

for num_steps in [1, 2, 3, 4, 5]:
    vq_accuracy[num_steps] / baseline_accuracy[num_steps]

# If VQ helps more on complex problems → real reasoning
# If VQ helps equally (or hurts) on complex → shallow compression
```

#### Metric 3: Code Reuse Analysis

```python
# How many codes are reused across different problem types?

code_overlap = len(codes_addition ∩ codes_multiplication) / num_codes

# High overlap (>0.5) → general reasoning primitives
# Low overlap (<0.2) → task-specific lookup table
```

**This metric is scale-diagnostic:**
- Small models: expect low overlap (memorization)
- Large models: expect high overlap (abstraction)

### Strategy 3: Probe Classifier Scaling Test

**Train probes to predict reasoning properties from codes:**

```python
# At each model scale, train probe to predict:
# - Number of reasoning steps needed
# - Arithmetic operations required
# - Problem difficulty

probe_accuracy_by_scale = {
    "70M": 0.35,   # Barely above random (0.25)
    "410M": 0.52,  # Modest signal
    "1.4B": 0.71,  # Clear structure
    "2.8B": 0.79,  # Strong structure
}
```

**Interpretation:**
- If probe accuracy scales smoothly → codes capture increasing reasoning structure
- If probe accuracy flat → codes are noise/random
- If probe accuracy jumps at specific scale → that's the emergence threshold

### Strategy 4: Qualitative Code Analysis at Each Scale

**Manually inspect what codes represent:**

**At 160M scale:**
```
Code #47: Activates for "total cost" in question (surface pattern)
Code #82: Activates for problems with 2 numbers (pattern matching)
Code #13: Activates for... unclear (possibly noise)
```

**At 1.4B scale (hypothetically):**
```
Code #47: Activates for "aggregation operations" (addition, sum, total)
Code #82: Activates for "sequential reasoning" (first X, then Y)
Code #13: Activates for "unit conversion" (dollars to cents, etc.)
```

**How to do this:**
1. Cluster problems by which codes activate
2. Read 10-20 problems per cluster
3. Identify common semantic features
4. Score "interpretability" (1-5 scale)

**Scale prediction:**
- Small models: interpretability ~2/5 (vague patterns)
- Large models: interpretability ~4/5 (clear concepts)

---

## Updated Resource Requirements for Scale-Robust PoC

### Minimum Scale-Aware Experiment

**Models:** Pythia-70M, 410M, 1.4B (3 models)
**Configurations:** Baseline + 2 VQ variants per model (9 runs)
**Compute:** ~50 GPU-hours total
**Cost:** ~$80-120 on cloud (Lambda Labs @ $1.50/hr for A100)
**Timeline:** 3-4 days of compute, 2 weeks elapsed

### What This Buys You

✅ **Scale trend data** - does it get better or worse?
✅ **Emergence threshold** - at what size does real reasoning appear?
✅ **Risk mitigation** - if 1.4B works but 70M doesn't, you know it needs scale
✅ **Publishable** - "we tested across 3 orders of magnitude"

### Budget-Constrained Alternative

**If you can only test 2 models:**

**Option A (Conservative):** 410M + 1.4B
- Skip 70M (probably too small anyway)
- Focus on the critical transition zone
- Still get scaling trend

**Option B (Ambitious):** 1.4B + 6.7B
- Skip small models entirely
- Assume you need scale, test how much
- Higher compute (~150 GPU-hours) but clearer signal

**My recommendation:** Option A (410M + 1.4B) as minimum viable scale-robust PoC.

---

## Experimental Protocol: Scale-Aware Version

### Phase 1: Baseline Establishment (All Scales)

```bash
# Run baselines to establish scaling behavior WITHOUT IR
python train_baseline.py --model pythia-70m
python train_baseline.py --model pythia-410m
python train_baseline.py --model pythia-1.4b

# Expected results:
# 70M:  ~5-10% GSM8K accuracy (mostly guessing)
# 410M: ~15-25% accuracy (some reasoning)
# 1.4B: ~30-40% accuracy (decent reasoning)
```

**This establishes:** What does reasoning scaling look like without IR?

### Phase 2: VQ Training (All Scales)

```bash
# Train VQ models at each scale
for model in 70m 410m 1.4b; do
    python train_vq.py --model pythia-$model --codes 512
done
```

**Key question:** Does VQ gap (VQ_acc - Baseline_acc) increase or decrease with scale?

**Scenario A: Gap increases with scale**
```
70M:  Baseline 8%  → VQ 9%   (+1%, noise)
410M: Baseline 20% → VQ 23%  (+3%, modest)
1.4B: Baseline 35% → VQ 42%  (+7%, strong!)
```
**Interpretation:** ✅ IR reasoning benefits from scale - GOOD SIGN

**Scenario B: Gap constant across scale**
```
70M:  Baseline 8%  → VQ 11%  (+3%)
410M: Baseline 20% → VQ 23%  (+3%)
1.4B: Baseline 35% → VQ 38%  (+3%)
```
**Interpretation:** ⚠️ IR provides scale-invariant compression (might just be surface-level)

**Scenario C: Gap decreases with scale**
```
70M:  Baseline 8%  → VQ 12%  (+4%)
410M: Baseline 20% → VQ 22%  (+2%)
1.4B: Baseline 35% → VQ 36%  (+1%)
```
**Interpretation:** ❌ IR doesn't capture what makes large models good - BAD SIGN

### Phase 3: Scale-Diagnostic Metrics

For each model scale, measure:

1. **Token Efficiency Scaling**
   ```python
   token_reduction_70m = 15%   # Modest compression
   token_reduction_410m = 28%  # Better
   token_reduction_1.4b = 42%  # Best

   # Q: Does compression improve with scale?
   # If yes → richer reasoning = more compressible
   ```

2. **Transfer Scaling**
   ```python
   # Train on GSM8K, test on SVAMP
   transfer_70m = 0.42   # Poor
   transfer_410m = 0.68  # Decent
   transfer_1.4b = 0.84  # Good

   # Q: Does IR improve transfer at larger scales?
   ```

3. **Code Interpretability Scaling**
   ```python
   # Probe classifier accuracy
   probe_70m = 0.38
   probe_410m = 0.56
   probe_1.4b = 0.73

   # Q: Do codes become more meaningful with scale?
   ```

### Phase 4: Analysis & Decision Point

**After collecting data, ask:**

1. ✅ Do VQ benefits increase with scale? → CONTINUE, test 6.7B next
2. ⚠️ Are benefits constant across scale? → INVESTIGATE, might be shallow compression
3. ❌ Do benefits decrease with scale? → STOP, hypothesis likely wrong

**Decision tree:**

```
IF (vq_gap_1.4b > vq_gap_410m > vq_gap_70m):
    CONCLUSION: "IR reasoning scales positively - pursue larger models"
    NEXT: Train Pythia-6.7B or Llama-7B

ELIF (token_reduction increases but accuracy doesn't):
    CONCLUSION: "IR compresses non-reasoning content"
    NEXT: Investigate what's being compressed (error analysis)

ELIF (all metrics flat):
    CONCLUSION: "Scale-invariant pattern matching"
    NEXT: Try different architecture or give up

ELSE:
    CONCLUSION: "Hypothesis doesn't hold"
    NEXT: Publish negative results, try alternative approaches
```

---

## Mitigating Scale Risk: Hybrid Approach

### If Small Models Fail But You Suspect Scale Helps

**Strategy: Knowledge Distillation from Large Model**

```python
# Step 1: Get reasoning traces from large model (GPT-4, Claude, Llama-70B)
large_model_codes = extract_reasoning_codes_from_large_model(gsm8k)

# Step 2: Train small model to predict these codes
small_model.train(
    input=questions,
    target_codes=large_model_codes,  # Supervision from large model
    loss=code_matching_loss
)

# Step 3: Test if small model learned to use codes meaningfully
```

**Why this helps:**
- Provides "ground truth" for what good reasoning codes should look like
- Small model learns structure from large model's reasoning
- Tests whether scale was needed for discovery vs. usage

**If this works:**
→ Small models CAN use IR, they just can't discover it alone
→ Larger models needed for initial codebook learning

**If this fails:**
→ Small models fundamentally can't represent rich reasoning
→ Need larger models end-to-end

---

## Concrete Recommendation for Your POC

### What I Would Do (Balancing Risk & Resources)

**Phase 1: Quick Validation (1 week, ~$30)**

Train 3 models:
1. Pythia-410M baseline
2. Pythia-410M VQ (512 codes)
3. Pythia-1.4B VQ (512 codes)

**Success criteria:**
- 1.4B VQ > 410M VQ (shows scale helps)
- 1.4B VQ token efficiency >20% better than 1.4B baseline
- Probe accuracy on 1.4B > 0.6 (codes capture structure)

**If all three met:**
→ Strong signal, proceed to Phase 2

**Phase 2: Full Scale Test (2 weeks, ~$100)**

Train full suite:
1. Pythia-70M, 410M, 1.4B, 2.8B (baselines)
2. Same models with VQ (2-3 configurations each)
3. Full evaluation (GSM8K, SVAMP, ARC)
4. Cross-task transfer tests
5. Probe classifiers at each scale

**Deliverable:**
- 8-page paper with scaling analysis
- Clear answer to "does IR reasoning scale?"
- Publishable either way (positive or negative)

**Phase 3: If Positive (1-2 months, ~$500)**

Train 6.7B or Llama-7B with VQ:
- Validate that scaling trend continues
- Test if codes transfer across model families
- Explore modular pipeline (Option B)
- Investigate RL fine-tuning

---

## Scale-Aware Success Metrics (Updated)

### Tier 1: Minimum Viable Success

- [ ] VQ benefits increase from 410M → 1.4B (scaling trend positive)
- [ ] Token reduction >15% at 1.4B
- [ ] Transfer accuracy (GSM8K→SVAMP) within 10% of baseline

**Interpretation:** Basic hypothesis holds, worth continuing

### Tier 2: Strong Success

- [ ] VQ benefits increase across all scales (70M → 2.8B)
- [ ] Token reduction >30% at 2.8B
- [ ] Probe accuracy >0.7 at 2.8B (interpretable codes)
- [ ] Transfer accuracy matches or exceeds baseline

**Interpretation:** Clear evidence of reasoning compression, publish & scale further

### Tier 3: Publication-Grade Success

- [ ] All Tier 2 criteria
- [ ] Codes transfer across tasks (train on GSM8K, work on ARC)
- [ ] Qualitative analysis shows interpretable reasoning primitives
- [ ] Scaling law established (log-linear relationship between model size and IR efficiency)

**Interpretation:** Major contribution, submit to top conference

---

## Final Thoughts on Scale

### Your Concern is Valid and Important

You're right to worry that "1B says nonsense but 7B is okay" - this is a **real risk** in AI research.

Many papers show impressive results on small models that don't scale (or vice versa).

### But Here's Why I'm Cautiously Optimistic

1. **VQ-VAE mechanism is scale-invariant** (proven in vision)
2. **We can test scaling explicitly** (410M vs 1.4B vs 2.8B)
3. **Even "failure" is informative** (shows reasoning isn't compressible)
4. **Cost is manageable** (~$100-200 for full scale test)

### The Key Insight

**You don't need to solve the full problem at small scale.**

What you need is a **positive scaling trend**:
- If 70M gets +1% from IR...
- And 410M gets +3%...
- And 1.4B gets +7%...

→ Then extrapolating to 7B might get +15-20%, which is huge!

**Conversely, if the trend is flat or negative** → abandon early, save resources.

### Actionable Next Step

**Start with 410M + 1.4B comparison** (~$50, 1 week):

```bash
# Baseline
python train.py --model pythia-410m --no-bottleneck
python train.py --model pythia-1.4b --no-bottleneck

# VQ
python train.py --model pythia-410m --vq --codes 512
python train.py --model pythia-1.4b --vq --codes 512

# Compare
python analyze_scaling.py --models 410m,1.4b
```

**If 1.4B shows stronger VQ benefits than 410M:**
→ Green light, full steam ahead

**If 1.4B shows same or weaker benefits:**
→ Red flag, rethink approach before investing more

---

**Does this address your scaling concern? Should I add this analysis to the main implementation guide?**
