# V10 Design Doc: Nano-Scale Air-Gapped VQ Transformer

**Date:** 2025-11-21
**Objective:** Prove **semantic necessity** of an Intermediate Representation (IR) by architecturally enforcing an information bottleneck.

---

## 1. Core Hypothesis
Standard Decoder-only Transformers bypass discrete bottlenecks via the residual stream. To force an LLM to rely on a discrete internal language, we must physically separate the "Reasoning" phase from the "Answering" phase, connecting them *only* via a discrete channel.

## 2. Architecture: "The Air-Gap"

The model consists of two distinct Transformer modules connected by a VQ bottleneck.

### Module A: The Reasoner (Encoder)
*   **Input:** Natural language problem (e.g., `12 + 15 =`).
*   **Architecture:** Small NanoGPT (e.g., 6 layers, 384 dim, 6 heads).
*   **Output:** A fixed-length sequence of $K$ discrete codes (the IR).
*   **Mechanism:**
    *   Processes input text.
    *   A `VQHead` projects the final hidden states (or a specific learned query sequence) to the codebook dimension.
    *   Outputs `ir_indices` (indices into the codebook).

### The Bridge: Discrete VQ Layer
*   **Codebook:** $N=512$ codes, $D=128$ dimension.
*   **Operation:**
    *   Forward: `z_q = codebook(ir_indices)` (stop-gradient on indices effectively, though we use straight-through estimator for learning).
    *   Backward: Gradients flow from Module B -> z_q -> Module A.
*   **Constraint:** **NO** residual connection from Module A to Module B. The only information passed is the sequence of code embeddings.

### Module B: The Speaker (Decoder)
*   **Input:** The sequence of VQ code embeddings `z_q` (from the Bridge).
*   **Architecture:** Small NanoGPT (e.g., 6 layers, 384 dim, 6 heads).
*   **Context:** It attends *only* to the IR embeddings (as a prefix or via cross-attention). It does **not** see the original input tokens.
*   **Output:** Natural language answer (e.g., `27`).

## 3. Training & Data

### Dataset
*   **Task:** Integer Arithmetic (Addition/Subtraction/Multiplication).
*   **Format:**
    *   Input: `12 + 15`
    *   Target: `27`
*   **Complexity:** 3-5 digit operands to ensure non-trivial reasoning.

### Loss Function
$$L = L_{answer} + \beta \cdot L_{vq}$$
*   $L_{answer}$: Cross-Entropy on the target answer tokens.
*   $L_{vq}$: Vector Quantization commitment loss via EMA or standard MSE.
*   **No auxiliary losses** (no concept heads, no contrastive loss). The architecture *is* the constraint.

## 4. Evaluation & Instrumentation

### Primary Metric: Semantic Necessity
We measure accuracy under 4 conditions on a held-out test set:
1.  **Intact:** Standard inference.
2.  **Random-IR:** Replace generated codes with random codes from the codebook.
3.  **Shuffle-IR:** Permute the generated code positions.
4.  **Drop-IR:** (Not strictly applicable since Module B *needs* input, but we can replace with a fixed "blank" code).

**Success Criterion:**
*   Intact Accuracy: High (>90%)
*   Random/Shuffle Accuracy: Near zero (chance).

### Monitoring
*   **Codebook Usage:** Active codes, perplexity of code distribution.
*   **Snapshots:** Periodically dump `(Input, IR_Codes, Answer)` triplets to `json` to inspect the emergent protocol.

## 5. Implementation Plan
1.  `air_gap/data.py`: Arithmetic dataset.
2.  `air_gap/model.py`: `AirGapVQTransformer`.
3.  `air_gap/train.py`: Training loop with instrumented evaluation.
4.  `air_gap/run_v10.sh`: Execution script.
