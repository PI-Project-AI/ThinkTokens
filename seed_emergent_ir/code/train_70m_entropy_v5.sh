#!/bin/bash
# ENTROPY V5: V3 baseline stability + EMA infrastructure for future logit debias
# Conservative rollback from failed V4 global diversity push

set -e

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "=========================================="
echo "IR-CoT ENTROPY V5: Pythia-70M"
echo "Phase A: Arithmetic (7k train, 20 epochs)"
echo "=========================================="
echo ""
echo "V5 STRATEGY (Conservative Rollback):"
echo "  ✓ V3 baseline: Entropy-based diversity ONLY"
echo "  ✓ Coverage weight: 0.5 (epochs 1-2) → 0.25 (epochs 3+)"
echo "  ✓ NO global KL loss (removed from V4)"
echo "  ✓ EMA frequency tracking (ready for logit debias)"
echo "  ✓ VQ temperature: Hold at 1.0 until step 2000, then → 0.8"
echo ""
echo "V3 PROVEN COMPONENTS (RETAINED):"
echo "  ✓ Local entropy loss: log(C) - H(avg_batch_dist)"
echo "  ✓ Gradient leak λ=0 (strict IR-only)"
echo "  ✓ Grammar: ≥3 spans, no consecutive codes, ≥2 distinct/span"
echo "  ✓ Early guards: step ≥100, 50-step window"
echo ""
echo "TARGET METRICS:"
echo "  Step ~300:  Util ≥5%, top-1 <60%, integrity ≥50%"
echo "  Step ~800:  Util ≥10%, top-1 <50%, integrity ≥70%"
echo "  Step ~1500: Util 15-25%, top-1 <40%, integrity ≥85%"
echo ""
echo "DEBUG DUMPS:"
echo "  Step 300, 800, 1500 (util/top-1/integrity/coverage_loss)"
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
  --train_data ../data/arithmetic/train.json \
  --val_data ../data/arithmetic/val.json \
  --batch_size 8 \
  --num_epochs 20 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --use_lora \
  --gradient_checkpointing \
  --use_8bit_adam \
  --output_dir ../checkpoints/ir_cot_70m_entropy_v5 \
  --test_frequency 5 \
  --no_ir_teacher_forcing \
  --enable_debug \
  --debug_step_frequency 300 \
  --enable_assertions \
  --run_control_random_ir \
  2>&1 | tee ../logs/train_70m_entropy_v5.log

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoint: ../checkpoints/ir_cot_70m_entropy_v5/best_model.pt"
echo "Log: ../logs/train_70m_entropy_v5.log"
echo "Debug logs: ../checkpoints/ir_cot_70m_entropy_v5/logs/"
echo "=========================================="
