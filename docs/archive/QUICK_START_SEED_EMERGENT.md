# Quick Start: Seed + Emergent IR Experiment

**Status:** Ready to begin Phase 0 (Setup)
**Previous Work:** Separate directory (do NOT mix with VQ experiment)
**Time Estimate:** 3-4 weeks for full completion

---

## TL;DR - What We're Building

```
Input → Encoder → IR Buffer  → Decoder
                  ↓
             <GOAL>[code]</GOAL>
             <STEP>[code]</STEP>
                  ↓
            FORCED Cross-Attn
            (No bypass allowed)
                  ↓
               Answer
```

**Key Innovation:** Structural tags (GOAL, STEP, etc.) + emergent codes
**Key Constraint:** Decoder must use IR (not input directly)

---

## Execution Steps

### Phase 0: Setup (Day 1)
```bash
# Create separate project
mkdir -p ThinkTokens/seed_emergent_ir/{data,code,logs,results}

# Copy reference docs
cp docs/SEED_EMERGENT_IR_GUIDE.md seed_emergent_ir/
cp docs/SEED_EMERGENT_IR_ARCHITECTURE.md seed_emergent_ir/

# Initialize git
cd seed_emergent_ir
git init
git add .
git commit -m "Phase 0: Project initialization for Seed+Emergent IR experiment"
```

**Deliverable:** Clean project structure, no VQ code mixed in

---

### Phase 1: Architecture (Days 2-5)

Implement 3 files:
1. **seed_emergent_ir/code/ir_generator.py**
   - VectorQuantizer (256 codes, 64-dim)
   - IRBufferGenerator (tags + VQ)

2. **seed_emergent_ir/code/decoder.py**
   - ConstrainedDecoder (forced cross-attention)
   - DecoderLayer (with cross-attn)

3. **seed_emergent_ir/code/models/causal_ir.py**
   - CausalIRModel (full integration)

**Test Each Component:**
```bash
# Test IRBufferGenerator
python -c "
import torch
from code.ir_generator import IRBufferGenerator
gen = IRBufferGenerator()
hidden = torch.randn(4, 10, 256)
ir_tokens, vq_loss, diag = gen(hidden)
print(f'✓ IRBufferGenerator works. VQ Loss: {vq_loss:.4f}')
"

# Test ConstrainedDecoder
python -c "
import torch
from code.decoder import ConstrainedDecoder
decoder = ConstrainedDecoder()
ir_buffer = torch.randn(4, 5, 256)
input_ids = torch.randint(0, 50257, (4, 10))
logits = decoder(ir_buffer, input_ids)
print(f'✓ ConstrainedDecoder works. Logits shape: {logits.shape}')
"

# Test full model
python -c "
import torch
from code.models.causal_ir import CausalIRModel
model = CausalIRModel()
input_ids = torch.randint(0, 50257, (2, 20))
labels = torch.randint(0, 50257, (2, 20))
logits, loss, ir_buffer, diag = model(input_ids, labels)
print(f'✓ CausalIRModel works. Loss: {loss:.4f}')
"
```

**Deliverable:** All 3 components tested, working independently

---

### Phase 2: Dataset (Days 5-6)

```bash
cd seed_emergent_ir/data
python arithmetic_generator.py

# Verify outputs
ls -lh arithmetic_*.json
head -5 arithmetic_train_1k.json
```

**Deliverable:** arithmetic_train_1k.json (1000 examples) + arithmetic_test_100.json

---

### Phase 3: Causal Tests (Days 7-8)

Implement: `seed_emergent_ir/evaluation/causal_tests.py`

**Do NOT run yet** (wait until after training)

**Deliverable:** Causal test suite ready

---

### Phase 4: Training (Days 9-12)

```bash
cd seed_emergent_ir
python training/train.py

# Monitor:
# - Loss should decrease smoothly
# - No NaNs or crashes
# - Codebook usage should increase (start low, reach 50-70%)

# Save checkpoint
cp models/causal_ir_arithmetic.pt models/causal_ir_arithmetic_final.pt
```

**Deliverable:** Trained model on arithmetic

---

### Phase 5: Run Causal Tests (Days 12-13)

```bash
python -c "
from evaluation.causal_tests import CausalDiagnosticTests
from torch.utils.data import DataLoader
from data.arithmetic_dataset import ArithmeticDataset

# Load model
model = CausalIRModel()
model.load_state_dict(torch.load('models/causal_ir_arithmetic.pt'))

# Load test data
test_dataset = ArithmeticDataset('data/arithmetic_test_100.json')
test_loader = DataLoader(test_dataset, batch_size=16)

# Run tests
tester = CausalDiagnosticTests(model, test_loader, device='cuda')
results = tester.run_all_tests()
"
```

**Critical Checks:**
- Baseline accuracy ≥ 70%
- Random IR: accuracy drop ≥ 30%
- Shuffle IR: accuracy drop ≥ 30%
- Drop IR: accuracy drop ≥ 40%

**If tests fail:** Debug before scaling to GSM8K

**Deliverable:** Causal test results (must pass)

---

### Phase 6: Scale to GSM8K (Days 14-17)

```bash
# Download GSM8K (100-500 subset first)
# Convert to same format as arithmetic

# Retrain or fine-tune
python training/train.py --dataset gsm8k_subset_100.json

# Run causal tests on GSM8K
python -c "
tester = CausalDiagnosticTests(model, gsm8k_loader, device='cuda')
results = tester.run_all_tests()
"
```

**Deliverable:** Results on GSM8K + causality verified

---

### Phase 7: Analysis (Days 18-20)

```bash
# Analyze learned codes
python analysis/analyze_codes.py

# Generate figures
python analysis/visualize_embeddings.py  # t-SNE
python analysis/compare_baselines.py     # Accuracy vs no IR
```

**Deliverable:** Analysis report + figures

---

## Expected Results (Success Criteria)

| Metric | Phase 4 (Arithmetic) | Phase 6 (GSM8K) |
|--------|----------------------|-----------------|
| **Accuracy** | ≥70% | ≥5-10% (from ~0%) |
| **Causality Tests** | ✓ All pass | ✓ All pass |
| **Token Efficiency** | ≤50 tokens | 20% reduction vs CoT |
| **Codebook Usage** | 50-70% | 50-70% |

---

## Decision Points

**If Phase 4 fails (causality tests don't pass):**
- Debug the architecture (may need stronger attenuation)
- Consider stronger constraint: freeze input tokens entirely?
- Add diagnostic to trace information flow

**If Phase 6 shows no accuracy gain:**
- Problem may be too hard for 3-4 layer decoder
- Add more decoder layers (8 instead of 4)
- Try curriculum: warm-up on simple task first

**If codebook collapse occurs:**
- Increase entropy loss weight
- Try codebook reset (reset unused codes)
- Lower code commitment loss

---

## Files to Track

**Core Implementation:**
```
seed_emergent_ir/code/
├── ir_generator.py       (VQ + IRBufferGenerator)
├── decoder.py            (ConstrainedDecoder)
└── models/causal_ir.py   (CausalIRModel)
```

**Training & Evaluation:**
```
seed_emergent_ir/
├── training/train.py
├── evaluation/causal_tests.py
└── data/arithmetic_generator.py
```

**Results (accumulate over phases):**
```
seed_emergent_ir/results/
├── phase4_arithmetic_results.json
├── phase5_causal_tests.json
├── phase6_gsm8k_results.json
└── figures/
```

---

## Git Workflow

After each phase, commit:

```bash
# Phase 1
git add code/
git commit -m "Phase 1: Implement IR architecture (IRGen, Decoder, CausalIR)"

# Phase 2
git add data/arithmetic_*.json
git commit -m "Phase 2: Generate arithmetic dataset (1k train, 100 test)"

# Phase 4
git add models/causal_ir_arithmetic_final.pt
git commit -m "Phase 4: Train on arithmetic, baseline 70% accuracy"

# Phase 5
git add results/phase5_causal_tests.json
git commit -m "Phase 5: Verify causality - all diagnostic tests pass"

# Phase 6
git add results/phase6_gsm8k_results.json
git commit -m "Phase 6: Scale to GSM8K, achieve +5-10% vs baseline"

# Phase 7
git add results/analysis/, results/figures/
git commit -m "Phase 7: Analysis - learned codes cluster by problem type"
```

---

## Common Issues & Solutions

**Issue:** VQ loss diverges
**Fix:** Lower VQ loss weight (0.1 → 0.01)

**Issue:** Model ignores IR (causality tests fail)
**Fix:** Increase input attenuation (0.1 → 0.01)

**Issue:** Training is slow
**Fix:** Reduce IR buffer length, smaller model (128→64 dim)

**Issue:** Codebook collapse (only using 10/256 codes)
**Fix:** Increase entropy loss weight, add code reset

---

## Success = "Next Paper"

When you complete all 7 phases and causality tests pass:

**You have:**
- ✅ Working hybrid IR architecture
- ✅ Proof that emergent codes are causal
- ✅ Comparison to baseline (VQ experiment showed this doesn't work)
- ✅ Clean codebase + reproducible results

**This is publishable as:**
- "Seed + Emergent Intermediate Reasoning: Forcing Models to Use Discrete Thinking"
- "Causal Reasoning Tokens: Architecture for Verifiable Discrete Thought"
- Workshop paper at minimum

---

## Start Here

**Begin Phase 0 now:**

```bash
cd ThinkTokens/
mkdir -p seed_emergent_ir/{data,code/models,code/training,code/evaluation,logs,results}
echo "Phase 0: Project structure initialized" > seed_emergent_ir/README.md
```

**Then follow the SEED_EMERGENT_IR_GUIDE.md step-by-step.**

When you hit any blocker, report it with:
- What phase
- What failed
- What you've tried

Good luck!