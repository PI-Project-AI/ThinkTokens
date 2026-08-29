# V12 Proposal: General Reasoning Corpus (Logic & Planning)

**Objective:** Verify that the "Air-Gap" VQ architecture can learn **generalizable state tracking** beyond arithmetic. Test if the model partitions or re-uses the codebook across disparate tasks.

---

## 1. Architecture (V12)
Same as V11 (22M parameters) but with extended IR capacity and task awareness.

*   **Reasoner (Module A):**
    *   6 Layers, 8 Heads, 384 Dim.
    *   Context: 128 tokens (increased for verbose logic traces).
*   **Bridge:**
    *   Codebook: 1024 codes, 384 Dim.
    *   **IR Length: 64 tokens** (Doubled from V11 to handle step-by-step deduction).
*   **Speaker (Module B):**
    *   6 Layers, 8 Heads, 384 Dim.
    *   Context: 64 (IR) + 64 (Answer) tokens.

## 2. The "Proto-Reasoning" Corpus
A mixed dataset of 3 distinct tasks (equal probability):

### Task A: Arithmetic (Control)
*   3-digit mixed arithmetic (`+`, `-`, `*`).
*   *Purpose:* Anchor the model on known capabilities.

### Task B: Symbolic Logic (Deduction)
*   **Input:** `A=True, B=False, C=(A or B), D=(not C). Value of D?`
*   **Target:** `True` (with intermediate steps implicit in IR).
*   **Complexity:** 3-5 variables, nested boolean ops.
*   *Purpose:* Test boolean state tracking and causal chains.

### Task C: Grid Navigation (Spatial/Planning)
*   **Input:** `Grid 5x5. Start (2,2). U, U, R, D, L. End?`
*   **Target:** `(2, 3)`
*   **Complexity:** 5-10 steps, boundary checks (stay in grid).
*   *Purpose:* Test coordinate state tracking $(x, y)$ and path integration.

## 3. Evaluation Strategy
*   **Per-Task Accuracy:** Does it master all 3?
*   **Ablations:** Does `Random-IR` kill performance on *all* tasks?
*   **Codebook Overlap:**
    *   We will compute the **Jaccard Similarity** of code usage between tasks.
    *   $J(A, B) = \frac{|Codes_A \cap Codes_B|}{|Codes_A \cup Codes_B|}$
    *   **Hypothesis:** High overlap suggests "Universal Reasoning Codes" (e.g., "increment" used for math and grid steps). Low overlap suggests partitioning.

## 4. Implementation Plan
1.  **`general_reasoning_data.py`**: Implement the `MixedReasoningDataset`.
2.  **`model_v12.py`**: Minor config update for sequence length.
3.  **`train_v12.py`**: Add per-task metrics logging.
