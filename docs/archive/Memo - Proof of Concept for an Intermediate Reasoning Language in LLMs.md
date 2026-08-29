
**Date:** [21/10/2025]  
**Prepared by:** [Paul PROVOST]

---

## 1. Context and Problem Statement

Current Large Language Models (LLMs) rely on **chain-of-thought (CoT) in natural language** for reasoning. While interpretable for humans, this approach introduces drawbacks:

- **Inefficiency:** Natural language reasoning expands token usage, increasing cost and latency.
- **Ambiguity:** English phrasing introduces redundancy and imprecision.
- **Human-centric bias:** The model is forced to reason in a format designed for communication, not optimized for internal computation.

The central hypothesis: _Models may reason more effectively in a compact, abstract intermediate representation (IR) that is neither raw hidden states nor verbose human language._

---

## 2. Concept

We introduce a **middle reasoning language** between input and output:

- **Input:** User intent in natural language.
- **Reasoning:** Encoded in discrete, non-human tokens (IR).
- **Output:** Translated back to natural language for the user.

### Key Principle

The **vocabulary of reasoning tokens is not hand-designed**. Instead:

- The model should **discover its own symbolic units** and **choose the appropriate level of abstraction** through training.
- Human engineers provide only the mechanism (a bottleneck / discrete code layer), not the semantics.
- Over time, tokens may cluster into meaningful categories (e.g., assumptions, logical negation, tool calls), but that structure is emergent.

This ensures the reasoning “language” is **optimized for the model’s efficiency**, not constrained by human intuition.

---

## 3. Proof of Concept Plan

### 3.1 Baselines

- **Models:** EleutherAI Pythia 70M/160M/410M, plus TinyLlama-1.1B as a reference point.
- **Tasks:** GSM8K (math word problems), SVAMP (robust arithmetic), ARC-Challenge (science).
- **Evaluation framework:** lm-evaluation-harness (accuracy, token usage, latency).

### 3.2 Architecture Options

- **Option A: Single-Model with Discrete Bottleneck (Recommended for PoC)**
    - Insert a VQ-VAE–style codebook inside a Transformer.
    - Lets the model self-organize a reasoning vocabulary.
    - Minimal engineering overhead, trainable on student-level compute.
    - Ideal for validating whether IR tokens emerge and whether they improve efficiency.
- **Option B: Modular Pipeline (Future Extension)**
    
    - Separate stages:
        1. Input parser (maps NL → IR).
        2. Reasoning core (operates purely on IR).
        3. Translator/monitor (maps IR → output and explanations).
    - Advantages: strong separation of concerns, easier to audit, swappable modules.
    - Drawbacks: higher implementation complexity, requires stable IR definition and more compute.

**Recommendation:** Begin with **Option A** to obtain first empirical results; explore **Option B** only if early results are promising or resources allow.

### 3.3 Training & Evaluation

- Train with standard LM loss on reasoning datasets.
- Add auxiliary objectives:
    - Codebook usage diversity (avoid collapse).
    - Optional cycle-consistency (IR → output → recover IR).
- Evaluate vs baselines on GSM8K, SVAMP, ARC.
- Analyze:
    - Accuracy gains/losses.
    - Token efficiency (vs CoT).
    - Emergent structure in IR codes.

---

## 4. Expected Outcomes

- **Efficiency:** IR reasoning should reduce token overhead compared to English CoT.
- **Emergent abstractions:** The model should discover a non-human symbolic vocabulary that clusters into useful reasoning primitives.
- **Comparative insight:** Even if accuracy is similar to baselines, efficiency and interpretability of code usage will be informative.

Risk: the IR may degenerate into trivial or uninterpretable codes, requiring careful regularization and monitoring.

---

## 5. Next Steps

1. Establish baseline benchmarks with Pythia models.
2. Implement Transformer + VQ bottleneck (PyTorch).
3. Train on GSM8K subset, scale to SVAMP/ARC.
4. Evaluate against baselines, with ablations on codebook size and bottleneck placement.
5. Document findings; prepare report/blogpost; consider workshop submission.

---

**Summary:**  
The proposed PoC will test whether models can **self-learn an abstract reasoning vocabulary** more efficient than natural-language CoT. The most effective starting point is a **single-model VQ bottleneck architecture**, as it allows emergent discovery of reasoning tokens while staying tractable for student-scale experiments.

---

Would you like me to also sketch a **diagram** (input → discrete reasoning tokens → output) to include in the memo so the architecture choice is immediately clear at a glance?