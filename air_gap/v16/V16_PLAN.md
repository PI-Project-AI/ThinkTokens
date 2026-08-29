# V16 Plan: The "Conversational Reasoner"

**Objective:** Test the "Modal Collapse" hypothesis. Can a single Air-Gap VQ model handle both Low-Entropy Logic (Math) and High-Entropy Communication (Chat/Story) without degrading either?

---

## 1. The "Trinity" Corpus (Synthetic Mix)
To ensure clean control, we synthesize the 3 modes (200k samples total):

1.  **Reasoning (Math Word Problems):**
    *   *Format:* "Tom has 3 apples. He buys 2 more. How many?" $\to$ "5"
    *   *Metric:* Exact Match Accuracy.
2.  **Narrative (TinyStories-style):**
    *   *Format:* "Once upon a time there was a [N]. It liked to [V]..." (Completion)
    *   *Metric:* Perplexity / BLEU (Structural coherence).
3.  **Chit-Chat (Persona-style):**
    *   *Format:* "Hi, how are you?" $\to$ "I am good, thanks." (Response)
    *   *Metric:* Perplexity / Response Relevance.

## 2. Architecture (V16)
*   **Scale:** 22M Params (6L/384D).
*   **Bridge:** 1024 Codes, **32 Tokens** (The Constraint).
    *   *Crucial Test:* Is 32 tokens enough to encode the "vibe" of a chat message *and* the "state" of a math problem?

## 3. The Two-Phase Pipeline
*   **Phase 1 (The Universal Compressor):**
    *   `Input -> IR -> Reconstruction`
    *   Goal: Ground the codebook in English/Math semantics.
*   **Phase 2 (The Multi-Task Thinker):**
    *   Task A (Math): `Problem -> IR -> Answer`
    *   Task B (Chat): `Message -> IR -> Response`
    *   Task C (Story): `Start -> IR -> Continuation`

## 4. Instrumentation
We need specific metrics per mode to detect collapse:
*   **Math:** Accuracy.
*   **Chat/Story:** Perplexity (PPL).
*   **Codebook Analysis:** Do "Chat Codes" overlap with "Math Codes"? (Jaccard).

## 5. Implementation
1.  `data.py`: Trinity generator.
2.  `train_phase1.py`: Auto-Encoder.
3.  `train_phase2.py`: Multi-task fine-tune with per-task logging.
