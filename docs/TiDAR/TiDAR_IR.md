# **TiDAR-mini: Think in IR Diffusion, Talk in Autoregression**

I picked one *specific*, implementable design and walk it end-to-end. You can swap pieces later, but this give a working target.

---

## 0. High-level design

We’ll build three main components:

1. **IR encoder** `E_θ`

   * Takes an English prompt `x`
   * Produces a short sequence of **discrete IR tokens** `z` (our internal language).

2. **IR diffusion model** `D_φ`

   * Runs a **discrete diffusion / denoising process** over `z`.
   * This is the “thinking” step: iterative refinement of IR (`z₀ → z_T → z*`).

3. **AR decoder (speaker)** `S_ψ`

   * Takes the refined IR `z*` (and optionally the prompt)
   * Generates the English answer `y` autoregressively.

Air-gap is treated as a **training tool**, not a strict law: we’ll have modes where `S_ψ` only sees IR, and modes where it also sees the original text.

The **toy task** will be simple arithmetic word problems so you can get signal quickly.

---

## 1. Setup: environment & task

### 1.1 Tech stack

* **Language**: Python
* **Framework**: PyTorch
* **Libraries**:

  * `transformers` (for basic Transformer blocks + tokenizers),
  * `sentencepiece` or `tokenizers` for BPE,
  * or roll your own small tokenizer if you want minimal dependencies.

Hardware: a single RTX 3090/4090 or A5000-class GPU is plenty for this POC.

### 1.2 Toy task (dataset)

Make synthetic arithmetic word problems:

* Expressions:

  * Depth 1–3: `a + b`, `(a + b) * c`, `a * (b - c)`, etc.
  * Integers between 0–99.
* Templates (examples):

  * `"What is (12 + 7) * 3?" → "57"`
  * `"Alice has 3 apples and Bob has 5. How many apples in total?" → "8"`
  * `"If you divide 20 by 4, what do you get?" → "5"`

Generate:

* ~100k training examples
* ~10k validation examples

Store each as:

```json
{
  "prompt": "Alice has 3 apples and Bob has 5. How many apples in total?",
  "answer": "8",
  "expr": "(3 + 5)"    // optional, for debugging and IR probing
}
```

---

## 2. Tokenization & IR space

### 2.1 Text tokenizer

Use a small BPE over your prompts + answers:

* Vocab size: 2k–4k.
* Special tokens: `[PAD]`, `[BOS]`, `[EOS]`.

`prompt_ids` = tokenized prompt, `answer_ids` = tokenized answer.

### 2.2 IR vocabulary

We want **emergent** IR, not manually symbolic, so:

* **IR vocab size** `|V_ir|`: 256–1024 codes.
* **IR length** `L_ir`: 16 slots (you can bump to 32 later).

So IR is:

```text
z ∈ {0..|V_ir|-1}^L_ir    # a small discrete sequence
```

Discreteness is enforced via a **VQ / Gumbel bottleneck**, but diffusion works over **embeddings** of these codes.

---

## 3. IR encoder – “from text to internal language”

We want a module `E_θ: x → z`:

1. **Prompt encoder** (Transformer)

   * Input: `prompt_ids`
   * Output: hidden states `H ∈ ℝ^{L_x × d_model}`

2. **IR slot extractor**

   * Use cross-attention or pooling to map `H` to `L_ir` latent vectors:
   * E.g. initialize `L_ir` learnable queries `Q_ir ∈ ℝ^{L_ir × d_model}`, run cross-attention to `H`:

     ```python
     # pseudo
     ir_queries = Q_ir  # (L_ir, d_model)
     ir_hidden = cross_attention(ir_queries, H)  # (L_ir, d_model)
     ```

3. **Discrete bottleneck (VQ)**

   * Codebook `C ∈ ℝ^{|V_ir| × d_ir}`
   * Map each `ir_hidden[i]` to nearest code:

     * `z[i] = argmin_k || ir_hidden[i] - C[k] ||²`
     * Embedding `e[i] = C[z[i]]`
   * Straight-through estimator for backprop.

4. **Optional projection**

   * If `d_ir ≠ d_model`, add linear layer between `ir_hidden` and VQ.

**Loss (Phase 1)**: you need IR to actually contain useful information:

* Reconstruction or predictive losses:

  * Predict the final answer from IR directly (via a tiny MLP or small decoder).
  * Possibly reconstruct the arithmetic expression or key operands from IR.

Simple version:

```python
# MLP over pooled IR embeddings predicts numeric answer tokens
ir_pooled = ir_hidden.mean(dim=0)       # (d_model,)
answer_logits = MLP(ir_pooled)          # map to vocabulary or number class
L_pred = CE(answer_logits, true_answer_tokens_or_class)
L_vq = vq_commitment_and_codebook_loss

L_phase1 = L_pred + β * L_vq
```

Train `E_θ` + VQ so IR becomes a compact summary of “what you need to answer”.

---

## 4. IR diffusion model – “think in IR”

Now we build `D_φ` to do **denoising diffusion over the IR slots**.

### 4.1 Noise process (discrete, MaskGIT-like)

We’ll use a simple discrete mask-corruption scheme:

* Noise steps `t = 1..T` (e.g. `T=4`).
* At each step, a fraction `p_t` of positions is replaced with a special `[MASK_IR]` code.
* Forward process `q(z_t | z_0)`:

  * Sample each position independently:

    * With prob `p_t`: `z_t[i] = MASK_IR`
    * Else: `z_t[i] = z_0[i]`

### 4.2 Denoiser model

`D_φ` is a Transformer that sees:

* Noisy IR embeddings `E_ir(z_t)` (with `[MASK_IR]` embedding),
* Prompt context (either full `H` from encoder or a pooled summary),
* Time embedding `τ_t`.

And predicts **original IR tokens** `z_0`.

Architecture:

```text
Inputs:
  - IR tokens (noisy) z_t ∈ {0..|V_ir|}^L_ir
  - Prompt summary h_prompt ∈ ℝ^{d_model} (e.g. mean pool of H)
  - Time step t

Steps:
  - Embed IR tokens: E_ir(z_t) → (L_ir, d_model)
  - Add time embedding to IR tokens
  - Concatenate h_prompt as an extra token OR condition via FiLM / biases
  - Run several Transformer layers
  - Project to |V_ir| logits per slot
```

Loss:

```python
logits = D_phi(z_t, prompt_cond, t_embed)
L_diffusion = CE(logits, target=z_0)   # per-slot cross entropy
```

Train by:

* Sampling `t ∼ Uniform({1..T})`,
* Creating `z_t` with mask corruption at level `p_t`,
* Minimizing `L_diffusion`.

You can train this **after** you have a stable `E_θ` (Phase 2) or jointly with some freezing.

---

## 5. AR decoder – “talk in autoregression”

Speaker `S_ψ` generates the answer tokens.

Inputs at inference:

* Refined IR `z*` (from diffusion),
* Prompt encoding `H` (optional, if you want soft air-gap),
* Previously generated answer tokens.

Design it as an **encoder-decoder**:

1. **IR encoder for speaker**

   * Take IR token IDs `z*`,
   * Use the same codebook embeddings `C[z*]`,
   * Run a few Transformer layers to produce `H_ir ∈ ℝ^{L_ir × d_model}`.

2. **Answer decoder**

   * Standard autoregressive Transformer:

     * Self-attention over past answer tokens,
     * Cross-attention over `H_ir` (and optionally `H` from prompt).
   * Outputs logits over text vocab.

Training loss:

```python
logits = S_psi(z_star, prompt_ids, answer_inp)
L_speaker = CE(logits, target=answer_out)
```

Where:

* `answer_inp` is `[BOS] y_1 ... y_{n-1}`
* `answer_out` is `y_1 ... y_n [EOS]`

---

## 6. Training phases (end-to-end recipe)

### Phase 1 – Learn IR encoder

**Goal:** IR tokens summarize the prompt in a way that’s predictive of the answer.

1. Freeze diffusion and speaker (don’t build them yet).

2. Train `E_θ` + VQ with:

   * `L_pred` (predict answer from IR),
   * `L_vq` (commitment + codebook),
   * Optional aux losses (predict expr structure, operands).

3. Monitor:

   * Answer accuracy from IR-only predictor,
   * Code usage (are many codes used? or collapsing?).

Stop when:

* IR-only accuracy is **well above random** (e.g. >80% on this toy),
* Code usage is healthy (entropy not too low).

---

### Phase 2 – Train IR diffusion

**Goal:** diffusion can reconstruct/refine IR, making it a real “thinking” process.

1. Freeze `E_θ` / VQ.

2. For each training batch:

   * Encode prompt: `z_0 = E_θ(prompt)`.
   * Sample `t ∼ {1..T}`, generate `z_t` via masking.
   * Feed into `D_φ` with prompt conditioning.
   * Minimize `L_diffusion`.

3. Monitor:

   * Token accuracy of predicted `z_0` from `z_t`,
   * Degradation vs noise level `t`.

You should see good reconstruction for small `t` and decent even for large `t`.

---

### Phase 3 – Train AR speaker (with IR, no gap yet)

**Goal:** speaker can answer tasks from IR (optionally supported by prompt).

1. For each training batch:

   * Encode prompt: `z_0 = E_θ(prompt)`.
   * Option A (simpler): **no diffusion yet** → use `z_0` directly.
   * Option B: apply diffusion denoising to get `z*`.

2. Train `S_ψ` with teacher forcing:

   ```python
   logits = S_psi(z_star_or_z0, prompt_ids, answer_inp)
   L_speaker = CE(logits, answer_out)
   ```

3. Monitor:

   * Answer accuracy on validation.
   * Attention patterns: does AR cross-attend to IR tokens?

At this point you already have:

* Encode → IR → Speak → answer

(with optional diffusion in the middle).

---

### Phase 4 – Integrate diffusion + introduce “gap curriculum”

Now make it truly “Think in Diffusion, Talk in AR” and **encourage IR usage**.

Training loop with *three modes*:

---

#### Mode A – Full-context (text + IR) [e.g. 50% of batches]

* AR sees both prompt and IR; easiest regime.

Steps:

1. `z_0 = E_θ(prompt)`
2. Run diffusion (one pass or multi-step) to get refined IR `z*`.
3. Speaker input: `z*` + prompt encoding `H`.
4. Loss: `L_speaker`.

---

#### Mode B – IR-only (hard air-gap episodes) [e.g. 40% of batches]

Here AR **cannot see the raw prompt**, only IR. This forces IR to be sufficient.

Steps:

1. Same `z_0`, `z*` as above.
2. Speaker input: `z*` only (no `H` / prompt text).
3. Loss: `L_speaker`.
4. You can occasionally add **noise to `z*`** to test robustness.

---

#### Mode C – Text-only (baseline LM) [e.g. 10% of batches]

Let AR see only the prompt (no IR) to avoid over-reliance on a potentially shaky IR early on.

Steps:

1. Skip IR / diffusion.
2. Speaker input: prompt encoding `H` only.
3. Loss: `L_speaker`.

---

#### Joint loss and training

For each batch:

```python
if mode == "A":
    L = L_speaker(prompt+IR)
elif mode == "B":
    L = L_speaker(IR_only)
else:  # mode C
    L = L_speaker(prompt_only)

# optionally add small diffusion loss on the side:
L_total = L + λ * L_diffusion_in_this_batch
L_total.backward()
step_optim()
```

Over time, you can:

* Increase IR-only proportion if it’s working.
* Decrease text-only.

By the end, you should have:

* A model that **works well in full-context mode**,
* And still works (maybe slightly worse) in pure **IR-only mode** – evidence that IR carries real causal information.

---

## 7. Inference pipeline

For a new prompt `x`:

1. **Encode to IR**

   ```python
   z0 = E_θ(x)  # (L_ir,)
   ```

2. **Diffusion “thinking”**

   Simple multi-step schedule (for T small, like 4):

   ```python
   z = mask_all_or_partial(z0)  # z_T
   for t in reversed(range(1, T+1)):
       logits = D_φ(z, prompt_cond, t_embed(t))
       z = sample_or_argmax(logits)  # approximate z_{t-1}
   z_star = z
   ```

   For a tiny POC you can even do **one-step** denoising from heavily masked `z_T`.

3. **AR “talking”**

   ```python
   answer_ids = sample_autoregressive(
       S_ψ,
       z_star,            # and optionally prompt_ids
       max_len=16
   )
   answer_text = decode_tokens(answer_ids)
   ```

4. **Check gap behavior (for science)**

   * Run inference in **full-context** vs **IR-only** modes.
   * Compare accuracy / style.
   * Try scrambling `z_star` → output should degrade a lot in IR-only mode.

---

## 8. What to log / how to know it’s working

Key things to log:

1. **Phase 1: IR quality**

   * IR-only answer accuracy.
   * Codebook usage histogram.
   * IR length vs performance (sweep L_ir).

2. **Phase 2: Diffusion**

   * Reconstruction accuracy of z₀ from z_t for different t.
   * Visualize how accuracy improves as t decreases.

3. **Phase 3–4: Speaker**

   * Answer accuracy across modes A/B/C.
   * Do IR-only answers stay good?
   * Sensitivity of answers to changes in IR vs changes in prompt text.

4. **Qualitative**:

   * For a fixed prompt, print:

     * `z0` IDs,
     * `z*` IDs after diffusion,
     * Interpret changes (e.g. do certain codes correlate with operations or magnitudes?).

---

## 9. Minimal hyperparams (good starting point)

* `d_model`: 256
* `n_layers` (encoder / diffusion / decoder): 4–6 each
* `n_heads`: 4
* `IR vocab`: 512
* `IR length`: 16
* `Diffusion steps T`: 4
* Batch size: 128
* Optimizer: AdamW, lr=1e-4, warmup 2k steps, cosine decay.

---

Next option example:
* Sketch actual **PyTorch class skeletons** (`IRVQEncoder`, `IRDiffusionTransformer`, `IRSpeakerDecoder`),
* Or write a **single training loop** in pseudo-code that ties all three phases together.
