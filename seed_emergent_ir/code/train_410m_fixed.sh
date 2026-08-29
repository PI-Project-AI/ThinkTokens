#!/bin/bash
# Fixed training run: Pythia-410M with collapse fixes
# Phase A: Arithmetic dataset, 20 epochs

set -e

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "=========================================="
echo "IR-CoT V2: Pythia-410M FIXED Training"
echo "Phase A: Arithmetic (7k train, 20 epochs)"
echo "=========================================="
echo ""
echo "Fixes Applied:"
echo "  ✓ Teacher forcing DISABLED (VQ-guided learning)"
echo "  ✓ Gradient leak: λ=0.1→0 over 3 epochs"
echo "  ✓ Batch safety checks + hard asserts"
echo "  ✓ Increased signals: VQ=0.5, coverage=0.05, cycle=0.1"
echo "  ✓ Temperature annealing: 0.7→0.4"
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
  --num_epochs 20 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --use_lora \
  --gradient_checkpointing \
  --use_8bit_adam \
  --output_dir ../checkpoints/ir_cot_410m_fixed \
  --test_frequency 5 \
  --no_ir_teacher_forcing \
  2>&1 | tee ../logs/train_410m_fixed.log

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoint: ../checkpoints/ir_cot_410m_fixed/best_model.pt"
echo "Log: ../logs/train_410m_fixed.log"
echo "=========================================="
