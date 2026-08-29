# V15 Plan: Grounded General Reasoning (The "Developmental" Approach)

**Objective:** Prove the "Thinking Tokens" thesis by mimicking cognitive development: First, learn to represent the world (Phase 1); Second, learn to reason with those representations (Phase 2). This solves the "Cold Start" problem where models fail to invent a language and a logic simultaneously.

---

## 1. The Core Philosophy
*   **Original Intent:** Verify if models can reason via an emergent, discrete, inspectable, and causally necessary "Language of Thought" (IR).
*   **The V15 Shift:** Instead of asking the model to invent the language *while* solving the test, we first help it establish a **Vocabulary of Thought** (Grounding) via compression, then ask it to **Sequence those Thoughts** (Reasoning). This separates "Perception" from "Logic".

## 2. Dataset: The "General Reasoning Mix"
To ensure the IR is a *general* thinking language, not a domain-specific hack, we blend 3 distinct "worlds":

*   **Source A: Algorithmic Traces (40%)**
    *   `x=5; for i in range(3): x+=1`
    *   *Concept:* State updates, loops, variables.
*   **Source B: Natural Language Instructions (40%)**
    *   `Mary is in the kitchen. She moves to the garden.`
    *   *Concept:* Spatial state, possession, actors.
*   **Source C: Symbolic Logic (20%)**
    *   `A=T, B=F, C=A&B`
    *   *Concept:* Abstract boolean causality.

## 3. Architecture (Air-Gap VQ)
*   **Scale:** 22M Parameters (Standard Nano-scale).
*   **Bridge:** 1024 Codes.
*   **IR Length:** **32 Tokens** (The "Compression Constraint").
    *   *Rationale:* Thoughts must be denser than speech. Forcing 2x-3x compression prevents the model from just copying the input ("Parrot Mode") and forces it to extract the semantic core ("Abstraction Mode").

## 4. The Two-Phase Pipeline

### Phase 1: "Naming the World" (Auto-Encoder)
*   **Task:** `Input -> [IR] -> Reconstruction`
*   **Goal:** Force the model to map diverse, noisy inputs (English, Code) into the discrete Codebook.
*   **Result:** The codes acquire meaning. Code 42 $\approx$ "Loop", Code 99 $\approx$ "Location".
*   **Metric:** Reconstruction Acc > 90%.

### Phase 2: "Learning to Think" (Reasoning Fine-Tune)
*   **Task:** `Input -> [IR] -> Answer`
*   **Mechanism:** Load Phase 1 weights. The model now "knows" the words. It must now learn to *select and sequence* the relevant Thinking Tokens to derive the answer.
*   **Goal:** Prove causal necessity and generalization.

## 5. Success Criteria
1.  **Phase 1:** High Reconstruction (The model *can* speak the thought language).
2.  **Phase 2:** Non-trivial Accuracy (>15%) on the mixed task (The model *uses* the language to think).
3.  **Universality:** Codebook analysis should show code reuse across domains (e.g., same tokens for "State Change" in Code and English).

## 6. Implementation Steps
1.  `air_gap/v15/data.py`: The Mixed Corpus generator.
2.  `air_gap/v15/model.py`: Air-Gap model with weight loading.
3.  `air_gap/v15/train_phase1.py`: The Compressor.
4.  `air_gap/v15/train_phase2.py`: The Thinker.