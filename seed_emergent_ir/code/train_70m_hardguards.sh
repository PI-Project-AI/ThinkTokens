#!/bin/bash
# HARD GUARDS TRAINING: Pythia-70M with all collapse prevention measures
# Phase A: Arithmetic dataset, 20 epochs

set -e

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "=========================================="
echo "IR-CoT V2: Pythia-70M HARD GUARDS Training"
echo "Phase A: Arithmetic (7k train, 20 epochs)"
echo "=========================================="
echo ""
echo "HARD GUARDS APPLIED:"
echo "  ✓ EOS banned on first 2 answer tokens"
echo "  ✓ Support size check (fail fast if < 10)"
echo "  ✓ IR error rate tracking (abort if 10 consecutive violations)"
echo "  ✓ Codebook utilization check (fail fast if < 10%)"
echo ""
echo "VQ STABILIZATION:"
echo "  ✓ Temperature: 1.0 → 0.7 (slower annealing for exploration)"
echo "  ✓ Coverage weight: 0.10 (epochs 1-2) → 0.05 (epochs 3+)"
echo "  ✓ Commitment β: 0.5 (warm-start) → 0.25 (normal)"
echo "  ✓ Early diversity guards: step≥500, util<5% or top-1>50%"
echo ""
echo "IR WARM-START:"
echo "  ✓ Gradient leak: λ = 0.1 → 0.05 → 0 (epochs 1-2-3+)"
echo "  ✓ Teacher forcing: tags only (no CE on codes)"
echo ""
echo "ANSWER DECODING:"
echo "  ✓ Answer CE weight: 1.5x (warm-start) → 1.0x (normal)"
echo "  ✓ Min answer length: 1 (EOS banned on pos 0-1)"
echo ""
echo "IR INTEGRITY:"
echo "  ✓ EOS banned inside IR until <IR_END> emitted"
echo "  ✓ Grammar masks enforce valid tag structure"
echo ""
echo "DEBUG:"
echo "  ✓ Step dumps every 500 steps with full diagnostics"
echo "  ✓ EOS prob, support size, top-1 code frequency"
echo "  ✓ Validation accuracy logged every epoch"
echo "  ✓ Hard assertions enabled (fail fast)"
echo "=========================================="
echo ""

# Run unit test first to verify CE path
echo "Running CE sanity test..."
python test_answer_ce_sanity.py
echo ""

python train_v2.py \
  --model_name EleutherAI/pythia-70m \
  --num_codes 512 \
  --code_dim 128 \
  --temp_init 1.0 \
  --temp_final 0.7 \
  --train_data ../data/arithmetic/train.json \
  --val_data ../data/arithmetic/val.json \
  --batch_size 8 \
  --num_epochs 20 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --use_lora \
  --gradient_checkpointing \
  --use_8bit_adam \
  --output_dir ../checkpoints/ir_cot_70m_hardguards \
  --test_frequency 5 \
  --no_ir_teacher_forcing \
  --enable_debug \
  --debug_step_frequency 500 \
  --enable_assertions \
  --run_control_random_ir \
  2>&1 | tee ../logs/train_70m_hardguards.log

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoint: ../checkpoints/ir_cot_70m_hardguards/best_model.pt"
echo "Log: ../logs/train_70m_hardguards.log"
echo "Debug logs: ../checkpoints/ir_cot_70m_hardguards/logs/"
echo "=========================================="
