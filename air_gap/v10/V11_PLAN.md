# V11 Proposal & Scaling Strategy

**Objective:** Scale the validated "Air-Gap" architecture to handle higher complexity (3-digit arithmetic + multi-op) while maintaining semantic necessity.

---

## 1. V11 Configuration (Moderate Scale-Up)

We aim for a model that is ~4x larger/deeper than V10 but still trains on a single GPU in <12 hours.

### Architecture
*   **Reasoner (Module A):**
    *   6 Layers, 8 Heads, 384 Hidden Dim (vs 4L/4H/128D in V10).
    *   Context: 64 tokens.
*   **Bridge:**
    *   Codebook: 1024 codes, 384 dim.
    *   IR Length: 32 tokens (doubled capacity for harder problems).
*   **Speaker (Module B):**
    *   6 Layers, 8 Heads, 384 Hidden Dim.
    *   Context: 32 (IR) + 32 (Answer) tokens.

### Training
*   **Dataset:** Mixed Arithmetic
    *   3-digit Addition/Subtraction (primary).
    *   2-digit Multiplication (harder task).
    *   Size: 200k examples.
*   **Optimization:**
    *   Batch Size: 128.
    *   Steps: ~50k steps.
    *   Loss: CE + VQ + *Entropy Reg* (keep V10's diversity loss).

---

## 2. Future Scaling Strategy (H100 Target)

If V11 succeeds, the path to a "Coup d'Éclat" demonstration involves:

### 2.1 Dataset Strategy: "General Reasoning Corpus"
Arithmetic is too narrow. We should synthesize a **"Logic & Planning"** corpus:
1.  **Symbolic Logic:** "If A then B, A is true..." (requires step-by-step deduction).
2.  **Pathfinding:** Grid navigation descriptions (requires spatial state tracking).
3.  **Algorithmic Trace:** Execution traces of simple Python programs (loops, variables).

**Recommendation:** Create a mixed dataset of **Arithmetic + Symbolic Logic + Program Trace**. This forces the IR to become a *general* reasoning language, not just a math notation.

### 2.2 H100 "Coup d'Éclat" Model
*   **Scale:** ~150M - 300M Parameters total.
    *   Reasoner: 12 Layers, 768 Dim (GPT-2 Small equivalent).
    *   Speaker: 12 Layers, 768 Dim.
*   **Budget:** ~10B tokens (Standard Chinchilla-ish for this size).
*   **Goal:** Show that the model invents *human-interpretable* concepts (e.g., "carrying" markers, "variable" slots) in the IR without supervision.

---

## 3. Immediate Next Steps (V11 Execution)
1.  Refactor `data.py` to generate **3-digit arithmetic** and **mixed operations**.
2.  Create `run_v11.sh` with the larger model config.
3.  Launch training and monitor codebook usage.
