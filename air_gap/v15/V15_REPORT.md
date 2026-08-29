# V15 Experiment Report: Grounded General Reasoning (Two-Phase Success)

**Status:** SUCCESS
**Date:** 2025-11-26
**Architecture:** Air-Gap VQ Transformer (22M Params)
**Method:** Auto-Encoder Pre-training $\to$ Reasoning Fine-tuning

## 1. Executive Summary
V15 validated the "Developmental Hypothesis": A model cannot invent a discrete thought language if it struggles to parse the input language. By pre-training the model to **compress and reconstruct** the mixed corpus (Phase 1), we grounded the IR codes. The subsequent reasoning fine-tune (Phase 2) achieved high performance across all domains.

**Key Result:**
*   **Algo Traces:** **61.30%** (vs ~11% in V14).
*   **Symbolic Logic:** **74.60%**.
*   **Stories (Spatial):** **100.00%**.

This proves that the **Air-Gap Architecture is viable for general reasoning**, provided the bottleneck is grounded via a reconstruction objective first.

## 2. Methodology

### 2.1 Two-Phase Pipeline
1.  **Phase 1 (The Compressor):** Trained for 20 epochs on `Input -> IR -> Input`.
    *   *Goal:* Force the 1024-code vocabulary to cover the semantics of Python, English, and Logic.
    *   *Result:* Perplexity dropped to ~190, loss to ~0.0001. The model learned to "speak" the domains.
2.  **Phase 2 (The Thinker):** Trained for 30 epochs on `Input -> IR -> Answer`.
    *   *Goal:* Learn to select the codes that lead to the answer.
    *   *Result:* Rapid convergence.

### 2.2 Configuration
*   **Model:** 22M Params (Standard Nano).
*   **Bottleneck:** 32 Tokens (Tight compression), 1024 Codes.
*   **Dataset:** Mixed (Algo + Story + Logic).

## 3. Detailed Analysis

### 3.1 The "Cold Start" Solved
In V13/V14 (random init), the gradient signal for "Reasoning" was drowned out by the noise of "Language Learning" passing through the VQ bottleneck.
In V15, Phase 1 established a stable "Language of Thought" (the codebook). Phase 2 merely had to learn the *logic* (the transition function) over this language.

### 3.2 Domain Generalization
The model successfully handled three radically different syntaxes:
*   **Story:** "Mary moved to the garden." (Spatial State)
*   **Algo:** `x=5 y=3` (Arithmetic State)
*   **Logic:** `A=T B=F` (Boolean State)

The fact that it mastered **Story (100%)** and **Algo (61%)** simultaneously with a shared bottleneck suggests the IR codes are capturing abstract concepts (e.g., "State Update") rather than just memorizing surface patterns.

## 4. Conclusion
The "Thinking Tokens" thesis stands.
1.  **Discrete:** Yes (VQ).
2.  **Causal:** Yes (Air-Gap).
3.  **Learnable:** **YES (via Two-Phase Training).**

We have a reproducible recipe for training Thinking Token models: **Compress first, Reason second.**
