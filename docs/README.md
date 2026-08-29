# IR-CoT: Air-Gapped Internal Reasoning Tokens

> **Goal:** Show that a model that *must think in a machine-native discrete language* (IR tokens, “internal CoT”) can match or beat traditional CoT that reasons in human language.

We explore an **air-gap architecture** where the model is forced to go:

> **English → IR (internal reasoning tokens) → English**

IR tokens are **discrete**, **causally necessary**, and (ideally) **interpretable**.
We compare this to standard transformers / CoT that reason purely in natural language.

---

## Part 1 — Intuitions Behind the Project

This section explains the motivating ideas and why they are at least *plausible*, not just sci-fi.

### 1.1. Machines can invent better internal languages than humans

**Intuition**

* Human language is optimized for **humans**: social communication, ambiguity, redundancy.
* A transformer is not a person. It might benefit from a very different internal “language”:

  * More compact,
  * More regular,
  * More aligned with the statistics of its data and its architecture.

So instead of forcing the model to reason in English (`"let's think step by step"`), we let it invent its own **“Language of Thought”** (LoT) in the middle.

**Why this is plausible**

* LLMs already learn **internal latent codes** in their hidden layers. We just don’t see them.
* Neural compression models (e.g. VQ-VAE, discrete bottlenecks) show that networks can discover **discrete internal codes** that correlate with meaningful factors (phonemes, object parts, etc.).
* Transformers handle **huge vocabularies** well: nothing stops them from using 1k–4k “ideogram” tokens internally.

So the bet is: if we **make that internal code explicit and necessary**, we get:

* A more **efficient reasoning medium**.
* A more **inspectable** representation.
* Potentially **better generalization** than human-language CoT.

---

### 1.2. CoT in human language is powerful but suboptimal

**Intuition**

* Chain-of-Thought (CoT) works, but it forces the model to reason in **human language**:

  * Long, verbose sequences,
  * Syntactic noise (“the”, “a”, “slowly”),
  * Ambiguity and redundancy.
* For a machine, it might be strictly better to think in a **denser code**:

  * Fewer tokens,
  * Each token carries more semantics,
  * No need to be grammatical or readable.

**Why this is plausible**

* We already know LLMs can solve problems without printing human-readable CoT: internal activations carry the signal.
* CoT sometimes helps performance just by **slowing down** and **expanding context**, not because natural language is inherently ideal for reasoning.
* In math and algorithmic tasks, the “true” underlying state is short (numbers, positions, logical states). English text is a noisy wrapper.

The project takes this seriously and tries to **separate the thinking medium (IR)** from the **communication medium (English)**.

---

### 1.3. Air-gap: forcing IR to be causally necessary

**Intuition**

* In a standard transformer, reasoning is spread everywhere: activations, residual paths, and output directly depend on the input.
* To prove the existence of a **machine-native CoT**, we want something stronger:

  * A **hard bottleneck** in the middle,
  * No residual skip around it,
  * So that *all* information must pass through the IR.

Architecture:

> Encoder (English) → VQ bottleneck (IR tokens) → Decoder (English / answers)

**Causal necessity**

* If we **scramble** IR tokens and performance collapses, IR is not just a side effect; it’s **causal**.
* If we can **interpret** patterns in IR (e.g., particular codes correspond to certain states), we get a handle on the internal “language of thought”.

**Why this is plausible**

* V10–V12 and V16 already showed:

  * On synthetic tasks, scrambling IR breaks performance.
  * IR tokens form a meaningful internal channel.
* This validates the **mechanism**: discrete, causal internal communication via an air-gap is possible.

---

### 1.4. Vector Quantization (VQ) as the mechanism for discrete IR

**Intuition**

* We want IR tokens to be:

  * **Discrete** (like words),
  * **Learned** (not hand-coded),
  * **Reused** across examples (a code means something stable).
* Vector Quantization (VQ) provides exactly that:

  * A codebook of embeddings,
  * Each continuous encoder output is snapped to its nearest code,
  * Gradients are handled via straight-through or EMA updates.

**Is VQ a good idea here?**

For this project, **yes, VQ is a reasonable and aligned choice**, because:

* It gives you an **explicit codebook** (inspectable “ideograms”).
* It forces **discreteness** at a specific layer (the air-gap).
* It’s a **standard technique** with known behavior (VQ-VAE-style).

Tradeoffs / caveats:

* VQ introduces **non-smoothness**: harder optimization, dead codes, sparse gradient issues.
* If the codebook is too small → over-compression, lost semantics.
  If too big → harder to learn, some codes unused.
* It does not automatically guarantee “nice” semantics; it only enforces discreteness + reuse.

Given your aim — **explicit, discrete IR tokens as a machine-native language** — VQ is actually a pretty natural tool. Alternatives (e.g. pure Gumbel-softmax without codebook) would be less interpretable.

---

### 1.5. System 1 vs System 2 (pretraining vs reasoning)

**Intuition**

* System 1: fast pattern recognition (language understanding, syntax, basic associations) → lives in the **weights**.
* System 2: slow, explicit reasoning, stepwise thinking → lives in **tokens** (IR stream).

We should not expect a tiny model to learn both:

1. Language understanding (System 1),
2. Internal reasoning language (IR),
3. Reasoning over that IR (System 2),

from scratch, all at once.

**Why this matters**

* The experiments suggest:

  * For **math**, where input tokens ≈ state, IR can emerge even in small models.
  * For **natural language**, the entropy is too high; we need:

    * More capacity,
    * A curriculum / two-phase training.

This motivates a **two-phase training method** (see Part 2).

---

### 1.6. Information density / entropy as the real wall

**Intuition**

* Synthetic data (small vocab, rigid templates) has **low entropy**:

  * A story like “The king fought the dragon” can be compressed into a couple of integers: (king, fought, dragon).
* Real language (TinyStories) has **much higher entropy**:

  * Larger vocab (~3k),
  * Free-form grammar,
  * Richer semantics (“sad”, “lost”, “toy” all distinct).

At ~26.5M parameters, with a discrete IR in the middle, trying to:

1. Parse real English,
2. Compress it into IR,
3. Then reason on top of that,

proved too much.

This led to the **“Real Language Wall”** observed in V17: success on synthetic data, collapse on TinyStories.

---

## Part 2 — Training Method: Hypotheses & Choices

This section states the key **hypotheses** and the **training recipes** we use to test them.

### 2.1. Core hypothesis: IR as a learned internal language of thought

> **Hypothesis H1**
> A transformer with an air-gapped, VQ-based discrete bottleneck can learn an internal “language of thought” (IR) that:
>
> * is **causally necessary** for performance,
> * is **more compact** than surface text,
> * can support **non-trivial reasoning** on real language tasks.

To test H1, we:

* Use an **encoder → VQ IR → decoder** architecture.
* Enforce **no bypass** around IR.
* Scramble IR to check **causal necessity**.
* Inspect code usage and structure to assess emergent “language”.

---

### 2.2. Two-phase training: grounding then reasoning

> **Hypothesis H2**
> Learning a useful IR requires first **grounding** it in the statistics of the input (System 1), then **training it for reasoning** (System 2). Trying to do both from scratch in one step is too hard.

#### Phase 1 — Grounding / Semantic Pretraining

**Goal:**
Make IR carry enough information about the input to support prediction, without forcing it to memorize exact surface form.

We consider multiple variants:

* **Early versions (V15, V17)**:

  * Phase 1: **Reconstruction**
    `Input (English) → IR → Reconstruct Input`
  * Works well on synthetic data but forces IR to encode **syntax**, not just semantics.

* **Improved variant (V17_ter & V18)**:

  * Phase 1: **Next-segment prediction** for narratives:
    `Story Segment 1 → IR → Predict Story Segment 2`
  * Math side: still some **auto-encoding** or task-specific prediction, to ground math tokens.

**Rationale:**

* Prediction encourages IR to encode the **causal state** (“what happens next?”), not just exact wording.
* For math, input tokens are already close to state, so auto-encoding is less harmful.

#### Phase 2 — Reasoning Fine-tune

**Goal:**
Teach the model to **solve tasks** via IR, not just predict text.

Examples:

* `Story → IR → Answer (QA / classification / next event)`
* `Math Problem → IR → Final numeric answer or reasoning step`

**Rationale:**

* Phase 1 builds a **shared protocol** between encoder/decoder through IR.
* Phase 2 shapes that protocol toward **task-relevant reasoning**.

In practice, Phase 2 can be:

* Pure fine-tune (Phase 1 → Phase 2 separated), or
* Joint training with reduced Phase 1 loss as auxiliary.

---

### 2.3. Why we avoid pure reconstruction for real language

> **Hypothesis H3**
> Exact reconstruction as the main objective is **misaligned** with semantic IR for high-entropy language.

Reason:

* Reconstruction requires IR to store:

  * Function words,
  * Adverbs, exact word order,
  * Any syntactic trivia that doesn’t matter for reasoning.
* A good “language of thought” should:

  * Keep **entities, relations, goals, states**,
  * Discard or compress superficial syntax.

Therefore, in V18 we:

* Replace exact reconstruction for narratives with **next-segment prediction**.
* Keep math auto-encoding only in a limited, controlled way (and plan to move toward more semantic math objectives later).

---

### 2.4. Capacity and IR configuration (V18 direction)

> **Hypothesis H4**
> Real-language air-gap training at ~26.5M parameters is fragile and likely still
> capacity-limited, but the observed failure mode depends materially on the
> exact data, vocabulary, and task setup.

Evidence:

* V16: 22M model + IR reported 100% on synthetic Math/Story/Chat (no shuffle controls recorded for that run, so it does not meet the causal-necessity bar; treated as unverified).
* Original December 2025 freeze configs for V17 and V17_ter collapsed on
  TinyStories + Math.
* Later reruns after vocab-alignment fixes recovered strong signal in V17
  (`math_acc=1.00`, `story_f1=0.633`, `story_shuffle=0.109`) and weaker but
  non-zero signal in V17_ter (`math_acc=1.00`, `story_pred_f1=0.262`,
  `story_shuffle=0.188`).
* This shifts the interpretation: ~26.5M is not a uniform hard failure on real
  language, but it remains brittle, objective-sensitive, and plausibly
  bandwidth/capacity-limited.

V18 therefore scales:

* Model size: ~26.5M → ~190M params.
* IR:

  * Codebook: 1024 → 4096 codes,
  * Length: 64 tokens (unchanged).

Goal: avoid over-compression and give the model enough bandwidth and capacity to learn a stable IR on real language.

---

### 2.5. Dataset strategy

We use:

* **TinyStories**

  * Real English, moderate complexity, ~3k vocab.
  * Stress-test: can IR handle real language entropy?

* **Synthetic Math/Logic generators**

  * Precise, controllable reasoning tasks.
  * Stress-test: can IR support **exact reasoning** while also handling language?

This hybrid setup allows us to ask:

> “Can a single IR-CoT model compress real stories and still perform precise reasoning through the same bottleneck?”

---

## Part 3 — Short Recap of V1x / V17 Variants & Results

This is not an exhaustive log; just the key milestones relevant to the story so far.

### V10–V12 (early air-gap on synthetic tasks)

* **Setup:** Small models, air-gapped architecture, synthetic math/logic.
* **Result:**

  * IR tokens became **causally necessary** (scrambling broke performance).
  * Demonstrated that discrete IR communication works on simple domains.
* **Takeaway:**

  * **Mechanism proof**: air-gap + IR is viable in principle.

---

### V16 — “Synthetic Trinity”: Math + Story + Chat (22M, IR=32)

* **Data:** Synthetic, templated Math + Story + Chat, low-entropy vocab (~90 words).
* **Training:** Two-phase with reconstruction.
* **Result:**

  * 100% accuracy reported on all three tasks (caveat: no shuffle controls were recorded for this run; unverified against the causal-necessity bar).
  * IR with 32 tokens was sufficient.
* **Takeaway:**

  * Architecture + training pipeline **work reliably** in low-entropy regimes.
  * Internal “language” emerges and is used.

---

### V17 — TinyStories + Math, Reconstruction (~26.5M params, IR=32/64)

* **Data:** TinyStories (real English) + synthetic Math.
* **Training:** Two-phase with **exact reconstruction** for narratives.
* **Result:**

  * Original December 2025 freeze:
    * Phase 1 reconstructed reasonably.
    * Phase 2 collapsed on both 32-token and 64-token IR settings.
  * Later rerun after vocab-alignment fixes:
    * Math accuracy reached `1.00`.
    * Story F1 reached `0.633` at the last epoch (`0.637` best), versus
      shuffle baseline `0.109`.
* **Takeaway:**

  * The original setup was unstable and fed the "real-language wall" diagnosis.
  * After the later vocab/eval fixes, V17 became a valid small-scale baseline
    rather than a pure failure case.
  * Remaining limitations are robustness and scale, not total absence of IR
    signal.
  * See `docs/air_gap/v17s_full_run_analysis_2026-01-04.md` for the
    post-fix run record.

---

### V17_ter — TinyStories + Math, Predictive Phase 1 (~26.5M params, IR=64)

* **Data:** TinyStories + synthetic Math.
* **Training:**

  * Phase 1: **predictive** objective
    `Segment 1 → IR → Predict Segment 2` (stories), plus math.
  * Phase 2: reasoning fine-tune.
* **Results:**

  * Original December 2025 freeze:
    * Phase 1 achieved low loss, perplexity around `376` on a ~3k vocabulary.
    * Phase 2 failed as a reasoning run.
  * Later rerun after vocab-alignment fixes:
    * Math accuracy reached `1.00`.
    * Story-prediction F1 reached `0.262` at the last epoch (`0.275` best),
      versus shuffle baseline `0.188`.
    * Diagnostics still showed high teacher-forced loss on the `story_pred`
      channel.
* **Takeaway:**

  * Better objective design alone did not make the story channel strong at ~26.5M params.
  * V17_ter now reads more as an objective/task-definition weakness than as
    proof of total architectural failure.
  * The line remains useful diagnostically, but it is weaker than V17 as a
    story baseline.
  * See `docs/air_gap/v17s_full_run_analysis_2026-01-04.md` for the
    post-fix run record and ablations.

---

### V18 (H100 run attempted) — Real-World Scale-Up (~190M params)

* **Goal (unchanged):**

  * Cross the “generalization threshold” on real language.
  * Answer: “Can a larger IR-CoT model handle TinyStories + Math through an air-gapped IR?”

* **Key changes:**

  1. Model: ~190M params (GPT-2 scale).
  2. IR: 4096-code VQ, 64-token IR sequence.
  3. Objective:

     * Narrative: **next-segment prediction** in Phase 1.
     * Math: (initially) auto-encoding / predictive mix.
  4. Dataset: TinyStories + synthetic Math/Logic.

* **Current status (December 2025 H100 run):**

  * Phase 1 checkpoint exists (`phase1_ae.pt`).
  * Phase 2 collapsed (`<unk>`-dominated outputs, ~0% task accuracy).
  * Run export was incomplete and eval logging was insufficient for scientific acceptance.
  * Therefore `air_gap/v18/h100_snapshot/` is an archive snapshot, not an exploitable result.

* **Success criteria (for next valid rerun):**

  * Non-collapse on TinyStories + Math.
  * Clear **task performance** above trivial baselines.
  * IR is **causal** (scramble ablations).
  * IR is **meaningful** enough to probe (code usage not degenerate).

* **Relation to main aim (IR-CoT > CoT):**

  * V18 is the **first serious test** of IR-CoT on real language at a realistic scale.
  * Once V18 is successful, the next step is to **add a standard same-scale CoT baseline** and run head-to-head comparisons.
