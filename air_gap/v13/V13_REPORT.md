# V13 Experiment Report: The "Air-Gap" on bAbI

**Status:** Completed (Mixed Result)
**Date:** 2025-11-26
**Architecture:** Air-Gap VQ Transformer (22M Params)
**Dataset:** bAbI Tasks 1-20 (Joint Training)

## 1. Executive Summary
V13 attempted to apply the Air-Gap architecture (proven on arithmetic/logic in V11/V12) to **Natural Language Reasoning** using the bAbI benchmark.

**Key Findings:**
1.  **Causal Necessity Confirmed:**
    *   **Intact IR:** 3.08% Overall Accuracy (Peaks of 43% on Task 7, 14% on Task 16).
    *   **Random IR:** 0.03% (Complete Collapse).
    *   **Shuffle IR:** 0.26% (Complete Collapse).
    *   *Conclusion:* The model *is* using the IR to solve the tasks it can solve. The bottleneck is functional.

2.  **Performance Ceiling:**
    *   The model struggled significantly to converge on the bAbI suite compared to arithmetic.
    *   It only learned **Task 7 (Counting)** and **Task 16 (Basic Induction)** to any meaningful degree.
    *   It failed completely on Story Memory tasks (Tasks 1-5), suggesting the 64-token IR bottleneck might be too lossy for compressing full natural language stories into discrete state *from scratch* with only 22M parameters.

## 2. Configuration
*   **Model:** 22M Params (6L/384D Reasoner + Speaker).
*   **VQ:** 1024 Codes, 384 Dim.
*   **IR Length:** 64 Tokens.
*   **Training:** 50 Epochs (~156k steps), Batch 64.

## 3. Detailed Results (Test Set)

| Task ID | Task Name | Intact Acc | Random Acc |
| :--- | :--- | :--- | :--- |
| 1 | Single Supporting Fact | 0.0% | 0.0% |
| ... | ... | ... | ... |
| **7** | **Counting** | **43.7%** | **0.0%** |
| **16** | **Basic Induction** | **14.5%** | **0.0%** |
| 20 | Agents Motivations | 0.9% | 0.0% |
| **Overall** | | **3.08%** | **0.03%** |

## 4. Analysis & Next Steps
The low overall performance compared to V11/V12 suggests that **Natural Language Compression is Harder than Logic Compression**.

*   **Hypothesis:** Arithmetic/Grid-Nav state is low-entropy (just variables/coords). A story is high-entropy. Compressing a multi-sentence story into 64 discrete codes requires a level of linguistic abstraction that a tiny 22M model struggles to invent from scratch.
*   **Recommendation for V14:**
    *   **Option A (Scale):** Needs significantly larger model/codebook to learn "Language" + "Compression" simultaneously.
    *   **Option B (Curriculum):** Pre-train the Reasoner/Speaker on Language Modeling first, *then* insert the Air-Gap.
    *   **Option C (Simplify):** Return to Synthetic Logic (V12) but scale complexity there, where we know it works.

**Verdict:** The Air-Gap mechanism is valid (causality holds), but the 22M/64-token config is underpowered for *jointly* learning to parse English and reason about it.
