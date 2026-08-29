#!/bin/bash
# GLOBAL DIVERSITY V4: Push utilization from ~1% to 20-30%
# Entropy-based + EMA global diversity + top-1 blacklisting

set -e

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "=========================================="
echo "IR-CoT GLOBAL DIVERSITY V4: Pythia-70M"
echo "Phase A: Arithmetic (7k train, 20 epochs)"
echo "=========================================="
echo ""
echo "GLOBAL DIVERSITY ENHANCEMENTS:"
echo "  ✓ Coverage weight: 1.0 (epochs 1-2) → 0.3 (epochs 3+)"
echo "  ✓ EMA-based global diversity (KL from uniform, weight 0.5)"
echo "  ✓ Top-1 code blacklisting (updated every 200 steps)"
echo "  ✓ VQ temperature: Hold at 1.0 until step 2000, then → 0.8"
echo ""
echo "RETAINED FROM V3 (ENTROPY-BASED):"
echo "  ✓ Local entropy loss: log(C) - H(avg_batch_dist)"
echo "  ✓ Gradient leak λ=0 (strict IR-only)"
echo "  ✓ Grammar: ≥3 spans, no consecutive codes, ≥2 distinct/span"
echo "  ✓ Early guards: step ≥100, 50-step window"
echo ""
echo "TARGET METRICS:"
echo "  Step ~800:  Util ≥10%, top-1 <40%, integrity ≥70%"
echo "  Step ~1500: Util 20-30%, top-1 <30%, integrity ≥85-95%"
echo ""
echo "DEBUG DUMPS:"
echo "  Step 500, 800, 1000, 1500 (util/top-1/integrity)"
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
  --output_dir ../checkpoints/ir_cot_70m_global_div_v4 \
  --test_frequency 5 \
  --no_ir_teacher_forcing \
  --enable_debug \
  --debug_step_frequency 500 \
  --enable_assertions \
  --run_control_random_ir \
  2>&1 | tee ../logs/train_70m_global_div_v4.log

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoint: ../checkpoints/ir_cot_70m_global_div_v4/best_model.pt"
echo "Log: ../logs/train_70m_global_div_v4.log"
echo "Debug logs: ../checkpoints/ir_cot_70m_global_div_v4/logs/"
echo "=========================================="
