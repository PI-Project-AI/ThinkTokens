#!/bin/bash
# PHASE 0 BOOTSTRAP V6: Two-phase VQ codebook priming + emergent IR
# Fixes batch-149 collapse by seeding codebook exposure before emergent learning

set -e

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "=========================================="
echo "IR-CoT PHASE 0 BOOTSTRAP V6: Pythia-70M"
echo "Phase A: Arithmetic (7k train, 20 epochs)"
echo "=========================================="
echo ""
echo "V6 STRATEGY (Two-Phase VQ Bootstrap):"
echo "  ===== PHASE 0 (Steps 0-1500): VQ Bootstrap ====="
echo "  ✓ Balanced code sampler: 3-6 codes/span, rotating reservoir"
echo "  ✓ Code CE loss: weight 0.1 (teach projection head all codes)"
echo "  ✓ VQ temperature: τ = 2.0 (exploration vs standard 1.0)"
echo "  ✓ Soft guards: WARN for util < 5% or top-1 > 50%"
echo "  ✓ Catastrophic guards: FAIL only if util < 1% (200 steps) or top-1 > 70% (50 steps)"
echo "  ✓ Early exit: if util ≥ 20% by step 600, ramp code CE → 0 by step 800"
echo ""
echo "  ===== PHASE 1 (Steps 1500+): Emergent IR ====="
echo "  ✓ V3 baseline: Entropy-based diversity only"
echo "  ✓ Code CE weight: 0.0 (codebook already primed)"
echo "  ✓ VQ temperature: 1.0 → 0.8 (standard annealing)"
echo "  ✓ Hard guards: FAIL if util < 10% or top-1 > 50%"
echo "  ✓ Coverage weight: 0.5 (epochs 1-2) → 0.25 (epochs 3+)"
echo ""
echo "TARGET METRICS:"
echo "  Step ~300:  Util ≥10%, top-1 <50%, integrity ≥60%"
echo "  Step ~800:  Util 20-30%, top-1 <40%, integrity ≥70%"
echo "  Step ~1500: Util 30-50%, top-1 <30%, integrity ≥90%"
echo ""
echo "DEBUG DUMPS:"
echo "  Step 300, 800, 1500 (util/top-1/integrity/code_ce_weight)"
echo "=========================================="
echo ""

# Run unit test first
echo "Running CE sanity test..."
python test_answer_ce_sanity.py
echo ""

python train_v2.py \
  --model_name EleutherAI/pythia-70m \
  --num_codes 512 \
  --code_dim 128 \
  --temp_init 1.0 \
  --temp_final 0.8 \
  --phase0_seeded_steps 1500 \
  --phase0_code_ce_weight 0.1 \
  --phase0_vq_tau 2.0 \
  --train_data ../data/arithmetic/train.json \
  --val_data ../data/arithmetic/val.json \
  --batch_size 8 \
  --num_epochs 20 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --use_lora \
  --gradient_checkpointing \
  --use_8bit_adam \
  --output_dir ../checkpoints/ir_cot_70m_phase0_v6 \
  --test_frequency 5 \
  --no_ir_teacher_forcing \
  --enable_debug \
  --debug_step_frequency 300 \
  --enable_assertions \
  --run_control_random_ir \
  2>&1 | tee ../logs/train_70m_phase0_v6.log

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoint: ../checkpoints/ir_cot_70m_phase0_v6/best_model.pt"
echo "Log: ../logs/train_70m_phase0_v6.log"
echo "Debug logs: ../checkpoints/ir_cot_70m_phase0_v6/logs/"
echo "=========================================="
