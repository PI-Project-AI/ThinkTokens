# V10 Experiment Report: The "Air-Gap" Proof of Concept

**Status:** SUCCESS
**Date:** 2025-11-21
**Architecture:** Air-Gapped VQ Transformer (Nano Scale)

## 1. Executive Summary
V10 successfully demonstrated **Semantic Necessity** of a discrete Intermediate Representation (IR) for reasoning tasks. Unlike previous single-decoder designs (V8/V9) where the model bypassed the IR via residual streams, V10's physically separated "Reasoner-Speaker" architecture forces the model to encode all necessary task information into the discrete IR.

**Key Result:**
- **Intact IR Accuracy:** 59.3% (on 2-digit arithmetic)
- **Randomized IR Accuracy:** 0.6% (Chance level)
- **Shuffled IR Accuracy:** 1.7%

This >98% relative drop proves the model is not using the IR as a mere "compute buffer" or timing signal, but as the sole carrier of semantic information.

## 2. Methodology

### 2.1 Architecture: "The Air-Gap"
The model consists of two distinct NanoGPT modules connected *only* by a discrete VQ bottleneck. No residual connections link the modules.

*   **Module A (Reasoner/Encoder):**
    *   Input: Natural language problem (e.g., "12 + 34 =")
    *   Config: 4 layers, 4 heads, 128 dim (NanoGPT)
    *   Output: 16 discrete code indices (the IR)
*   **The Bridge (VQ Layer):**
    *   Codebook: 512 codes, 128 dim
    *   Mechanism: Vector Quantization with Straight-Through Estimator (training) / Argmax (inference)
*   **Module B (Speaker/Decoder):**
    *   Input: Sequence of 16 VQ code embeddings
    *   Config: 4 layers, 4 heads, 128 dim (NanoGPT)
    *   Output: Natural language answer (e.g., "46")
    *   **Constraint:** Does NOT see the original input text.

### 2.2 Training Setup
*   **Dataset:** 2-digit integer arithmetic (Addition/Subtraction).
*   **Size:** 50k training examples, 1k test examples.
*   **Loss Function:** `CrossEntropy(Answer) + VQ_Commitment`.
*   **Optimization:** AdamW, lr=3e-4, 50 epochs.
*   **Schedule:** Gumbel-Softmax warm-up (first 10 epochs, temp 2.0 -> 0.5), then hard VQ.

## 3. Results & Analysis

### 3.1 Quantitative Metrics (Test Set)
| Metric | Value |
| :--- | :--- |
| **Final Accuracy** | 59.30% |
| **Random-IR Acc** | 0.60% |
| **Shuffle-IR Acc** | 1.70% |
| **Perplexity** | ~3.8 |
| **Active Codes** | ~14 (effective) |

### 3.2 Qualitative Examples
The model learned a compact protocol using a small subset of codes to transmit operands and operations.

**Example 1 (Success):**
*   **Input:** `3 + 94 =`
*   **Target:** `97`
*   **IR Sequence:** `[261, 261, 261, 261, 343, 261, ... 346, 60, 261]`
*   **Output:** `97`

**Example 2 (Failure Case):**
*   **Input:** `94 + 13 =`
*   **Target:** `107`
*   **Output:** `106` (Off-by-one error suggests lossy compression in the bottleneck)

### 3.3 Why V10 Succeeded
V8/V9 failed because standard Transformers have a **Continuous Residual Highway**. Even with masks and stop-gradients on specific tokens, information leaks through the continuous state vectors that bypass the discrete bottleneck.

V10 succeeds by **severing the highway**. The Reasoner's output is *quantized* before reaching the Speaker. The Speaker has no other path to the input. This forces the "Language of Thought" to be the *only* bridge, guaranteeing its causal necessity.

## 4. Artifacts
*   **Code:** `air_gap/`
*   **Model:** `air_gap/results/model_final.pt`
*   **Logs:** `air_gap/results/history.json`
