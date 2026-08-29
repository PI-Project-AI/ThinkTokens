#!/bin/bash
# Debug training run: Pythia-410M with comprehensive diagnostics
# Phase A: Arithmetic dataset, 5 epochs for quick diagnosis

set -e

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "=========================================="
echo "IR-CoT V2: Pythia-410M DEBUG Training"
echo "Phase A: Arithmetic (7k train, 5 epochs)"
echo "=========================================="
echo ""
echo "Debug Features:"
echo "  ✓ Step dumps every 500 steps"
echo "  ✓ Epoch-level metrics (JSON)"
echo "  ✓ Hard assertions (fail fast)"
echo "  ✓ Random-IR control experiment"
echo "=========================================="
echo ""

python train_v2.py \
  --model_name EleutherAI/pythia-410m \
  --num_codes 512 \
  --code_dim 128 \
  --temp_init 0.7 \
  --temp_final 0.4 \
  --train_data ../data/arithmetic/train.json \
  --val_data ../data/arithmetic/val.json \
  --batch_size 8 \
  --num_epochs 5 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --use_lora \
  --gradient_checkpointing \
  --use_8bit_adam \
  --output_dir ../checkpoints/ir_cot_410m_debug \
  --test_frequency 5 \
  --no_ir_teacher_forcing \
  --enable_debug \
  --debug_step_frequency 500 \
  --enable_assertions \
  --run_control_random_ir \
  2>&1 | tee ../logs/train_410m_debug.log

echo ""
echo "=========================================="
echo "Debug training complete!"
echo "Checkpoint: ../checkpoints/ir_cot_410m_debug/best_model.pt"
echo "Log: ../logs/train_410m_debug.log"
echo "Debug logs: ../checkpoints/ir_cot_410m_debug/logs/"
echo "=========================================="
