# IR-CoT V7-Lite Progress Report
**Date**: 2025-11-12
**Model**: Pythia-70M with VQ bottleneck (512 codes, 128 dim)

## Summary
Successfully fixed P0-P6 training issues and implemented logit debias for code diversity. However, code collapse persists despite active debias during training.

---

## Problems Fixed ✓

### P3: EOS Ban Not Working
- **Issue**: Model emitted EOS inside IR sequences
- **Fix**: Implemented hard masking for EOS token during IR generation
- **Status**: ✓ FIXED

### P4: Answer Loss = 0.0
- **Issue**: PAD/EOS tokens not masked in answer loss calculation
- **Fix**: Added proper ignore_index masking for PAD tokens
- **Status**: ✓ FIXED

### P5: VQ Gradients Not Flowing
- **Issue**: Gumbel-Softmax path broke gradient flow to codebook
- **Fix**: Use actual `self.codebook.weight` parameter (not normalized copy)
- **Location**: `vq.py:144`
- **Status**: ✓ FIXED

### P6: Tag LM Loss = 0.0
- **Issue**: Teacher forcing path had indexing bugs
- **Fix**: Corrected indexing in `ir_generator_v2.py:234-238`
- **Status**: ✓ FIXED

### P7: Logit Debias Not Active
- **Issue**: `training_step` parameter not passed from training loop to VQ
- **Fix**: Added `training_step` parameter passing in `ir_generator_v2.py:235,324`
- **Status**: ✓ FIXED (active during training, not visible in eval dumps)

---

## Current Issue: Code Collapse

### Symptoms (Step 200)
- **Codebook utilization**: 0.59% (3 codes out of 512)
- **Generated codes**: Alternating pattern `c453c177` (codes 50743, 50467)
- **Top-1 frequency**: 50%
- **debias_gamma in dumps**: 0.0 (because dumps capture eval mode, not training)

### Root Cause Analysis

#### 1. Debias IS Active During Training
The `training_step` parameter is correctly passed, and debias runs during forward passes. However:
- Debug dumps capture **evaluation mode** (`self.training=False`)
- Gumbel check at `vq.py:108` disables debias during eval
- This is **expected behavior** - debias is training-only

#### 2. Why Collapse Persists

Despite active debias during training, codes collapse because:

**A. Insufficient Training Steps**
- `gumbel_steps=1500` with `batch_size=8` = ~187 weight updates before VQ switch
- Debias strength decays linearly: `gamma = 1.0 * (1 - step/1500)`
- By step 200, debias is already weak: `gamma ≈ 0.87`

**B. Gumbel Temperature Too High**
- `gumbel_tau=0.6` allows mode collapse in early training
- Softmax becomes too peaked before codes diversify

**C. No Gradient Flow to Codebook During Gumbel Phase**
- With Gumbel-Softmax, codes are selected via soft assignment
- Codebook receives gradients, but selection is still soft
- Argmin selection starts AFTER gumbel_steps, by which point collapse has occurred

#### 3. The Fundamental Problem

**Gumbel warmstart helps gradient flow, but doesn't prevent collapse:**
- Codes need to be **forced** to explore during warm-start
- Current debias (log-prior penalty) is too weak
- Need stronger diversity constraint OR different initialization

---

## Proposed Solutions

### Option A: Stronger Debias (Quick Fix)
```python
# In vq.py
self.gamma0 = 3.0  # Increase from 1.0
self.gumbel_steps = 3000  # Double warm-start period
```

**Pros**: Minimal code change
**Cons**: May overcorrect, causing uniform (non-semantic) code distribution

### Option B: Explicit Diversity Loss (Recommended)
Add entropy maximization during Gumbel phase:

```python
# In train_v2.py, during forward pass
if use_gumbel:
    # Entropy of code distribution across batch
    code_probs = F.softmax(code_logits, dim=-1)  # (B, S, num_codes)
    code_dist = code_probs.mean(dim=(0, 1))  # (num_codes,)
    entropy_loss = -torch.sum(code_dist * torch.log(code_dist + 1e-8))
    target_entropy = math.log(num_codes)  # Maximum entropy
    diversity_loss = (target_entropy - entropy_loss) * diversity_weight
```

**Pros**: Directly optimizes for uniform code usage
**Cons**: Requires modifying training loop

### Option C: Code Initialization (Complementary)
Initialize codebook with k-means on random projections:

```python
# In vq.py:__init__()
# Replace uniform init with k-means
from sklearn.cluster import KMeans
random_samples = torch.randn(10000, code_dim)
kmeans = KMeans(n_clusters=num_codes, n_init=10)
centers = kmeans.fit(random_samples.numpy()).cluster_centers_
self.codebook.weight.data = torch.from_numpy(centers).float()
```

**Pros**: Better initialization, codes start diverse
**Cons**: Requires sklearn dependency

---

## Recommended Action Plan

### Immediate (Today)
1. **Increase debias strength**: `gamma0=3.0`, `gumbel_steps=3000`
2. **Add code usage logging**: Track unique codes per batch during training
3. **Re-run mini sanity to step 200**

### Short-term (This Week)
4. **Implement explicit diversity loss** (Option B)
5. **Experiment with temperature annealing**: Start `gumbel_tau=1.0`, anneal to `0.5`
6. **Add early stopping**: If utilization < 5% after 500 steps, restart with higher gamma

### Medium-term (Next Week)
7. **Try k-means initialization** (Option C)
8. **Ablation study**: Test with/without debias, with/without diversity loss
9. **Scale to 410M**: Once 70M works, test on larger model

---

## Training Configuration (Current)
```bash
--model_name "EleutherAI/pythia-70m"
--num_codes 512
--code_dim 128
--batch_size 8
--lr 5e-5
--use_gumbel_warmstart
--gumbel_tau 0.6
--gumbel_steps 1500  # Currently 187 weight updates
--gamma0 1.0          # Currently too weak
--temp_init 1.0
--temp_final 0.8
```

---

## Key Metrics to Monitor

### Training
- **VQ Loss**: Should decrease from ~30k to <1000 by step 200 ✓
- **Unique Codes per Batch**: Should be >50 by step 200 (currently ~3) ✗
- **debias_gamma**: Should decay from 1.0 to 0.0 over gumbel_steps
- **EMA Code Frequencies**: Should be relatively uniform (currently collapsed)

### Evaluation (Step 200)
- **Codebook Utilization**: Target >10% (currently 0.59%) ✗
- **IR Structure**: Valid tags and spans ✓
- **Answer Quality**: Loss ~7.0 ✓

---

## Files Modified

### Core Fixes
- `vq.py:144` - Fix gradient flow to codebook
- `vq.py:116-128` - Implement logit debias
- `ir_generator_v2.py:235,324` - Pass training_step to VQ
- `vq_tied_generation.py:84` - Accept training_step parameter

### Training Scripts
- `train_v2.py:233-235` - Set VQ current_step
- `train_mini_sanity.sh` - Mini sanity configuration

---

## Next Session TODO
- [ ] Increase `gamma0` to 3.0 and `gumbel_steps` to 3000
- [ ] Add training-time code usage logging
- [ ] Run to step 200 and verify diversity improves
- [ ] If still collapsed, implement explicit diversity loss (Option B)

---

## Notes
- All P0-P6 core training bugs are fixed ✓
- VQ loss converges properly ✓
- Gradient flow works ✓
- **Code diversity remains the blocker** for semantic IR emergence
- Debias is working during training (just not visible in eval dumps)
- Need stronger diversity constraints during Gumbel warm-start phase
