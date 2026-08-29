# Phase 1 Implementation Complete: Seed + Emergent IR-CoT

**Date:** 2025-11-01
**Status:** ✅ All core components implemented and tested
**Next:** Ready for Phase 2 (Training on arithmetic dataset)

---

## What Was Built

Phase 1 implemented the complete **architectural CoT (IR-CoT)** system with:

### 1. Core Architecture Components

**Two-Pass Decoder-Only Design:**
- **Pass 1:** Input → IR Buffer (autoregressive, structured tags + VQ codes)
- **Stop-Gradient:** Hard boundary prevents backprop through IR
- **Pass 2:** IR Buffer Only → Answer (0% HL bypass by construction)

**Key Innovation:** No cross-attention modules needed. We simply exclude HL tokens from Pass 2 context, ensuring 0% bypass architecturally.

### 2. Implemented Modules

#### `tokenizer_utils.py` (seed_emergent_ir/code/)
- Extends Pythia tokenizer with 524 new special tokens:
  - 12 structural tags: `<IR_START>`, `<IR_END>`, `<GOAL>`, `</GOAL>`, etc.
  - 512 VQ code tokens: `c000`, `c001`, ..., `c511`
- Provides token ID mapping and utility functions

**Location:** `seed_emergent_ir/code/tokenizer_utils.py`

#### `vq.py` (seed_emergent_ir/code/)
- `VectorQuantizer`: L2-nearest neighbor quantization with straight-through estimator
- `ProjectionVQ`: Automatic projection layer (hidden_dim → code_dim)
- Codebook utilization tracking
- Default: 512 codes, 128-dim embeddings

**Location:** `seed_emergent_ir/code/vq.py`

#### `ir_generator.py` (seed_emergent_ir/code/)
- `IRBufferGenerator`: Autoregressive IR emission
- Generates structured IR: `<IR_START> <GOAL> c047 c089 </GOAL> <STEP> c201 ... <IR_END>`
- Teacher forcing for training
- Enforces span budgets (4-12 spans, 3-6 codes per span)
- VQ integration for code selection

**Location:** `seed_emergent_ir/code/ir_generator.py`

#### `local_cycle.py` (seed_emergent_ir/code/)
- `LocalCycleHead`: Reconstructs 5-10 token HL snippets from IR spans
- Light supervision to anchor IR codes to task content
- Small transformer decoder (2 layers)
- Samples 10% of spans to keep loss light

**Location:** `seed_emergent_ir/code/local_cycle.py`

#### `causal_ir_model.py` (seed_emergent_ir/code/models/)
- `CausalIRModel`: Main orchestrator integrating all components
- Two-pass forward with stop-gradient
- Combined losses:
  - **Answer CE loss** (primary)
  - **IR LM loss** (0.5x weight)
  - **VQ commitment loss** (0.1x weight)
  - **Local cycle loss** (0.05x weight)
  - **Coverage/diversity losses** (0.02x weight)
- Inference: `generate_answer()` method

**Location:** `seed_emergent_ir/code/models/causal_ir_model.py`

#### `causal_tests.py` (seed_emergent_ir/code/evaluation/)
- `CausalityTester`: Three diagnostic tests
  1. **Random-IR:** Replace codes with random → expect ≥70% accuracy drop
  2. **Shuffle-IR:** Swap IR between examples → expect ≥70% drop
  3. **Drop-IR:** Remove IR entirely → expect ≥70% drop (near random)
- Automated pass/fail checking
- Formatted result reporting

**Location:** `seed_emergent_ir/code/evaluation/causal_tests.py`

#### `arithmetic_generator.py` (seed_emergent_ir/data/)
- Generates 1-2 step arithmetic problems
- Curriculum learning: easy → medium → hard
- 5 difficulty levels:
  1. Single-step addition (1-20)
  2. Single-step subtraction (1-20)
  3. Two-step: add then add
  4. Two-step: add then subtract
  5. Two-step: multiply then add
- Includes snippets for local cycle loss

**Location:** `seed_emergent_ir/data/arithmetic_generator.py`

#### `train.py` (seed_emergent_ir/code/)
- Complete training loop with all losses
- Checkpoint saving (best validation loss)
- Periodic causality tests (every 5 epochs)
- Learning rate scheduling with warmup
- Gradient clipping
- Comprehensive logging

**Location:** `seed_emergent_ir/code/train.py`

---

## Dataset Generated

**Total:** 10,000 examples
**Split:** 70% train (7,000) / 15% val (1,500) / 15% test (1,500)

**Difficulty Distribution (Train):**
- Easy: 3,472 examples (49.6%)
- Medium: 2,842 examples (40.6%)
- Hard: 686 examples (9.8%)

**Operation Distribution (Train):**
- Addition: 1,728 (24.7%)
- Subtraction: 1,744 (24.9%)
- Addition → Addition: 1,403 (20.0%)
- Addition → Subtraction: 1,439 (20.6%)
- Multiplication → Addition: 686 (9.8%)

**Sample Problems:**
```
What is 7 + 3?                    → 10  (easy, 1-step)
What is 15 - 8?                   → 7   (easy, 1-step)
What is 5 + 12 + 3?               → 20  (medium, 2-step)
What is 18 + 6 - 11?              → 13  (medium, 2-step)
What is (4 * 6) + 9?              → 33  (hard, 2-step)
```

**Location:** `seed_emergent_ir/data/arithmetic/`

---

## Architecture Decisions Locked In

Based on user specifications:

### Structural Tags (Fixed)
```
<IR_START>, <IR_END>
<GOAL>, </GOAL>
<ASSUME>, </ASSUME>
<STEP>, </STEP>
<CHECK>, </CHECK>
<BRANCH>, </BRANCH>
```

### VQ Codebook
- **Default:** 512 codes (ablate 256/1024 later)
- **Dimension:** 128
- **Per-span:** 3-6 codes
- **Spans per example:** 4-12

### IR Buffer Format
```
<IR_START> <GOAL> c047 c089 </GOAL> <STEP> c201 c033 c177 </STEP> <CHECK> c055 </CHECK> <IR_END>
```

### No-Bypass Mechanism
- **Pass 2 context:** IR tokens ONLY (no input tokens)
- **HL→Answer attention:** 0% by construction (no architectural bypass possible)
- **Stop-grad:** Applied at IR→Answer boundary

### Loss Weights
- Answer CE: 1.0
- IR LM: 0.5
- VQ commitment: 0.1 (β=0.25)
- Local cycle: 0.05
- Coverage/diversity: 0.02

### Target Metrics
- **Codebook utilization:** 50-70%
- **Accuracy gain:** +2-5% vs baseline
- **Token efficiency:** 20% reduction vs NL CoT
- **Causality tests:** All three must show ≥70% relative drop

---

## File Structure

```
seed_emergent_ir/
├── code/
│   ├── tokenizer_utils.py          # Tokenizer extension (524 tokens)
│   ├── vq.py                        # VectorQuantizer + ProjectionVQ
│   ├── ir_generator.py              # IRBufferGenerator (Pass 1)
│   ├── local_cycle.py               # LocalCycleHead (IR→HL snippet)
│   ├── train.py                     # Main training script
│   ├── models/
│   │   ├── __init__.py
│   │   └── causal_ir_model.py       # CausalIRModel (orchestrator)
│   └── evaluation/
│       ├── __init__.py
│       └── causal_tests.py          # Random/Shuffle/Drop IR tests
├── data/
│   ├── arithmetic_generator.py      # Dataset generation script
│   └── arithmetic/
│       ├── train.json               # 7,000 examples
│       ├── val.json                 # 1,500 examples
│       ├── test.json                # 1,500 examples
│       └── dataset_stats.json       # Statistics
├── checkpoints/                     # (empty, for training outputs)
├── results/                         # (empty, for evaluation results)
├── logs/                            # (empty, for training logs)
└── PHASE1_COMPLETE.md              # This document
```

---

## How to Use

### 1. Generate Dataset (Already Done)
```bash
cd seed_emergent_ir/data
python arithmetic_generator.py
```

### 2. Start Training
```bash
cd seed_emergent_ir/code
python train.py \
    --model_name EleutherAI/pythia-70m \
    --train_data ../data/arithmetic/train.json \
    --val_data ../data/arithmetic/val.json \
    --batch_size 16 \
    --num_epochs 20 \
    --lr 5e-5 \
    --output_dir ../checkpoints/ir_cot_v1 \
    --test_frequency 5
```

**Key Training Args:**
- `--model_name`: Base Pythia model (default: pythia-70m)
- `--num_codes`: Codebook size (default: 512)
- `--code_dim`: Code dimension (default: 128)
- `--use_local_cycle`: Enable local cycle loss (default: True)
- `--test_frequency`: Run causality tests every N epochs (default: 5)

### 3. Monitor Training
Training script outputs:
- **Per-epoch:** Total loss + loss components breakdown
- **Validation:** Loss + accuracy (simplified, first-token matching)
- **Checkpoints:** Saved to `output_dir/best_model.pt` when val loss improves
- **Causality tests:** JSON results saved to `output_dir/causality_tests_epoch{N}.json`

### 4. Load and Evaluate
```python
from tokenizer_utils import extend_tokenizer_for_ir
from models.causal_ir_model import CausalIRModel
from evaluation.causal_tests import CausalityTester
import torch
import json

# Setup
tokenizer, ir_token_ids = extend_tokenizer_for_ir()
model = CausalIRModel("EleutherAI/pythia-70m", ir_token_ids)

# Load checkpoint
checkpoint = torch.load("../checkpoints/ir_cot_v1/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run causality tests
with open("../data/arithmetic/test.json", 'r') as f:
    test_data = json.load(f)

tester = CausalityTester(model, tokenizer, ir_token_ids)
results = tester.run_all_tests(test_data, batch_size=8)

print(f"Baseline accuracy: {results['baseline_accuracy']:.2%}")
print(f"Random-IR drop: {results['random_ir_drop']:.2%}")
print(f"Shuffle-IR drop: {results['shuffle_ir_drop']:.2%}")
print(f"Drop-IR drop: {results['drop_ir_drop']:.2%}")
print(f"All tests passed: {results['all_tests_passed']}")
```

---

## Success Criteria for Phase 2

Phase 2 (Training) is successful if:

1. **Training converges:**
   - ✅ Loss decreases steadily
   - ✅ Validation accuracy reaches ≥70% on arithmetic
   - ✅ No loss divergence or NaN

2. **Causality tests pass:**
   - ✅ Random-IR: ≥70% relative accuracy drop
   - ✅ Shuffle-IR: ≥70% relative accuracy drop
   - ✅ Drop-IR: ≥70% relative accuracy drop (or near random)

3. **Codebook health:**
   - ✅ Utilization: 50-70%
   - ✅ No mode collapse (all codes in <5% range)
   - ✅ Diverse code usage across span types

4. **Efficiency:**
   - ✅ IR buffer length: 20-50 tokens (vs 100+ for NL CoT)
   - ✅ Token reduction: ≥20%

**If tests don't pass initially:**
- Increase VQ loss weight (0.1 → 0.2)
- Increase local cycle weight (0.05 → 0.1)
- Add explicit no-empty-span penalty
- Try stronger HL attenuation in Pass 2 (currently 0%)

---

## Known Limitations & Future Work

### Current Implementation Gaps:
1. **IR generation is simplified:**
   - State machine for tag selection is basic
   - May need more sophisticated span planning
   - Consider using a separate small controller

2. **Answer evaluation is simplified:**
   - Currently just compares first token
   - Should extract full numeric answer and compare

3. **No-empty-span penalty not implemented:**
   - Coverage loss placeholder (returns 0)
   - Need to count empty spans and penalize

4. **Diversity loss simplified:**
   - Should track per-tag code distributions
   - Add entropy regularization per span type

### Phase 3-6 TODOs:
- **Phase 3:** Implement full diagnostic infrastructure
- **Phase 4:** Train on arithmetic, iterate until causality tests pass
- **Phase 5:** Scale to GSM8K subset (100-500 examples)
- **Phase 6:** Analysis & visualization (code clustering, per-tag semantics)

---

## Technical Details

### Stop-Gradient Implementation
```python
# In CausalIRModel.forward()
ir_token_ids = ir_output['ir_token_ids']       # (batch, ir_len)
ir_token_ids_detached = ir_token_ids.detach()   # HARD STOP

# Pass 2: Only IR tokens in context
full_sequence = torch.cat([ir_token_ids_detached, answer_ids[:, :-1]], dim=1)
outputs = self.base_model(input_ids=full_sequence)
```

### VQ Integration
```python
# In IRBufferGenerator.forward()
last_hidden = outputs.hidden_states[-1][:, -1, :]  # (batch, hidden_dim)

# At code positions:
if self.use_vq and is_code_position(next_token):
    code_hidden = last_hidden[code_mask].unsqueeze(1)
    _, vq_loss, vq_indices = self.vq(code_hidden)

    # Convert VQ indices to token IDs
    code_tokens = vq_indices + self.ir_token_ids['code_start']
    next_tokens[code_mask] = code_tokens
```

### Causality Test Logic
```python
# Random-IR test
def _randomize_codes(ir_token_ids):
    code_mask = (ir_token_ids >= code_start) & (ir_token_ids <= code_end)
    random_codes = torch.randint(code_start, code_end + 1, size=(code_mask.sum(),))
    ir_token_ids[code_mask] = random_codes
    return ir_token_ids

# Expected: accuracy crashes if IR is genuinely causal
```

---

## Next Steps

### Immediate (Phase 2 - Start Training):
```bash
# 1. Activate environment
source /path/to/venv/bin/activate

# 2. Install dependencies (if needed)
pip install torch transformers tqdm

# 3. Start training
cd seed_emergent_ir/code
python train.py --num_epochs 20 --batch_size 16

# 4. Monitor training
# Watch for:
# - Loss convergence
# - Val accuracy ≥70%
# - Causality test results every 5 epochs
```

### Debug Checklist if Training Fails:
- [ ] Check tokenizer extension worked (vocab size increased by 524)
- [ ] Verify VQ loss is non-zero (codes being used)
- [ ] Check IR buffer format (should have tags + codes)
- [ ] Verify stop-grad is applied (grad_fn should be None)
- [ ] Monitor codebook utilization (50-70% target)
- [ ] Check answer loss is decreasing
- [ ] Verify no NaN in any loss component

### Phase 3 (Causality Verification):
Once training reaches ~70% accuracy:
```python
# Run comprehensive causality tests
python -c "
from evaluation.causal_tests import CausalityTester
# ... load model and test data ...
results = tester.run_all_tests(test_data)
"
```

**If tests fail:**
1. Increase loss weights (VQ, cycle)
2. Add no-empty-span penalty
3. Try harder stop-grad (also stop during IR gen)
4. Reduce HL→Answer attention further (already 0%)

### Phase 4-6 (Scale & Analyze):
- Scale to GSM8K
- Analyze learned code semantics
- Visualize code clusters
- Write final report

---

## Summary

**Phase 1 Status:** ✅ COMPLETE

**Implemented:**
- ✅ Full two-pass architecture
- ✅ VQ with 512 codes
- ✅ IR buffer generator (autoregressive)
- ✅ Local cycle loss
- ✅ Causality tests (Random/Shuffle/Drop)
- ✅ Training script with all losses
- ✅ Arithmetic dataset (10K examples)

**Ready for:**
- Phase 2: Training on arithmetic (target ≥70% accuracy)
- Phase 3: Causality verification (all tests must pass)

**Key Insight:**
The architecture enforces 0% bypass by construction (IR-only context in Pass 2). Success depends on:
1. IR quality (must encode task-relevant information)
2. Code semantics (VQ must learn meaningful discretization)
3. Causality (tests must show IR is genuinely used)

If causality tests pass after training, we have architectural CoT working.

---

**Date:** 2025-11-01
**Author:** Claude Code
**Next Action:** Start training with `python train.py`
