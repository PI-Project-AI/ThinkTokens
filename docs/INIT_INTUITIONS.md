# Initial/concepts of the project ThinkTokens

## Part 1 — Core Intuitions Behind the Project (Three Pillars)

This project is built on **three distinct but complementary ideas**.
Each can be wrong independently; the experiments are designed to test them separately whenever possible.

---

### **Idea 1 — A Machine-Native Language of Thought Is Likely More Efficient Than Human Language**

**Claim**

A transformer should not be forced to reason in human language.
Instead, it should be allowed (and forced) to invent its own **machine-native internal language** that is:

* discrete,
* compact,
* reused across contexts,
* optimized for reasoning rather than communication.

We call this language **IR (Internal Reasoning tokens)**.

**Why this makes sense**

Human language is optimized for:

* social interaction,
* ambiguity tolerance,
* redundancy,
* learnability by humans.

It is *not* optimized for:

* symbolic manipulation,
* compact state encoding,
* systematic reasoning under noise.

By contrast:

* Transformers already learn **latent abstractions** in hidden states.
* Discrete bottlenecks (VQ, latent codebooks) have repeatedly shown that neural networks can discover **stable symbolic representations** when forced to compress.
* There is no architectural reason a transformer cannot handle:

  * thousands of discrete internal symbols,
  * with denser semantics than words.

**Implication**

If reasoning truly lives in a compact state space, then:

* fewer IR tokens should express what would require many English tokens,
* reasoning through IR may be **more stable, more efficient, or more generalizable** than English CoT.

This idea motivates **creating IR explicitly** instead of relying on opaque hidden activations.

---

### **Idea 2 — Causal Separation Is Required to Prove “Thinking” (Air-Gap Architecture)**

**Claim**

If a model is allowed to bypass its “thinking representation,” we cannot claim it is *actually* reasoning in that representation.

Therefore, IR must be:

* **causally necessary**,
* not just a byproduct or interpretability artifact.

**Why this makes sense**

Standard CoT models suffer from ambiguity:

* The model may output reasoning text, but it may not *use* it.
* CoT can be epiphenomenal (helpful for training, not for actual inference).

The air-gap architecture enforces causality:

```
English → Encoder → IR (discrete bottleneck) → Decoder → English
```

* No residual path around IR.
* No skip connections.
* All task-relevant information must pass through IR.

**Empirical backing**

* In early synthetic experiments (V10–V12, V16):

  * Scrambling IR tokens destroys performance.
  * Therefore IR is not decoration; it is functionally required.
* This is strictly stronger evidence than showing that IR is “interpretable.”

**Implication**

This architecture allows strong claims:

* “The model **uses** IR”
* “Reasoning failure corresponds to IR corruption”
* “IR is a legitimate internal reasoning medium”

Without causal separation, those claims are not defensible.

---

### **Idea 3 — Reasoning Is a Two-Stage Process (Language Grounding → Reasoning), Not One**

**Claim**

A model cannot efficiently learn:

1. language understanding,
2. an internal reasoning language,
3. reasoning *over* that language,

all at once through a tight discrete bottleneck—especially at small scale.

Instead, reasoning should be learned in **two conceptual phases**:

1. **Grounding the IR** (System 1–like)
2. **Using the IR for reasoning** (System 2–like)

**Why this makes sense**

Humans do this:

* Children learn to understand language long before abstract reasoning.
* Reasoning builds on a *stable representation of the world*.

Transformers seem to behave similarly:

* Math works early because tokens ≈ state.
* Natural language has high entropy and must first be **parsed and compressed**.

Your experiments show this clearly:

* Synthetic tasks → IR emerges easily.
* Real language → model collapses when compression + reasoning are demanded simultaneously.

**Implication**

This motivates:

* **Two-phase training**
* Careful choice of Phase 1 objective
* Avoiding objectives that force IR to encode surface syntax

This idea is orthogonal to IR itself:
even a perfect IR mechanism will fail if learned under the wrong curriculum.

---

## How the Three Ideas Fit Together

| Idea               | What it answers                            |
| ------------------ | ------------------------------------------ |
| Machine-native IR  | *What the model should think in*           |
| Air-gap causality  | *How we prove it really thinks that way*   |
| Two-phase learning | *How the model can realistically learn it* |

Air-gap separation and VQ/discrete bottlenecks are enforcement mechanisms to make IR causally necessary; they are not ends in themselves. The aim of the current experiment cycle is to show **IR-CoT > English-CoT**.

Together, they form this thesis:

> **If a model is forced to communicate through a discrete internal language, trained with a curriculum that encourages semantic abstraction, it may develop a more efficient reasoning process than models constrained to reason in human language.**

---

## Relation to the Experiments

* **V10–V16** validate:

  * Idea 2 (air-gap causality)
  * Partial support for Idea 1 (IR emerges in low-entropy regimes)
*   **V17 / V17_ter** reveal:

  * Idea 3 is critical (e.g., initially, severe data truncation due to insufficient context window, which was later corrected, but these versions were not re-run as V18 took precedence).
* **V18** tests:

  * Whether Idea 1 + 2 + 3 still hold when real language entropy is present at realistic model scale.

Only *after* V18 can the project seriously test:

> **IR-CoT > CoT**

— because until then, IR-CoT hasn’t fully crossed the realism threshold.
