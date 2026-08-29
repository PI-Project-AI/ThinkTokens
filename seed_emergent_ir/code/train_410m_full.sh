#!/bin/bash
# Full training run: Pythia-410M with all fixes
# Phase A: Arithmetic dataset, 20 epochs

set -e

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "=========================================="
echo "IR-CoT V2: Pythia-410M Full Training"
echo "Phase A: Arithmetic (10k examples, 20 epochs)"
echo "=========================================="

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
  --output_dir ../checkpoints/ir_cot_410m_full \
  --test_frequency 5 \
  --ir_teacher_forcing \
  2>&1 | tee ../logs/train_410m_full.log

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoint: ../checkpoints/ir_cot_410m_full/best_model.pt"
echo "Log: ../logs/train_410m_full.log"
echo "=========================================="
