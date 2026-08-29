#!/bin/bash
# Mini Sanity Training (2 epochs) with debug dumps at 20/100/200/600/800/1000
set -e
cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "=========================================="
echo "MINI SANITY: Pythia-70M (2 epochs, V8 with P1.1-P1.4)"
echo "Debug dumps at steps: 20, 100, 200, 600, 800, 1000"
echo "=========================================="

python train_v2.py \
  --model_name "EleutherAI/pythia-70m" \
  --use_lora \
  --output_dir "../checkpoints/ir_cot_70m_mini_sanity" \
  --num_codes 512 \
  --code_dim 128 \
  --temp_init 1.0 \
  --temp_final 0.8 \
  --train_data "../data/arithmetic/train.json" \
  --val_data "../data/arithmetic/val.json" \
  --batch_size 8 \
  --num_epochs 2 \
  --lr 5e-5 \
  --gradient_checkpointing \
  --use_contrastive \
  --contrastive_weight 0.3 \
  --contrastive_T 0.07 \
  --use_gumbel_warmstart \
  --gumbel_tau 0.6 \
  --gumbel_steps 3000 \
  --diversity_weight 0.5 \
  --phase0_seeded_steps 0 \
  --seed 42 \
  --fp16 \
  --max_grad_norm 1.0 \
  --enable_debug \
  --debug_step_frequency 200 \
  --log_file "../logs/mini_sanity_70m.log" \
  --eval_code_sampling softmax \
  --eval_tau 0.9 \
  --eval_topk 32 \
  --eval_topp 0.95 \
  --use_ir_value_head \
  --ir_value_weight 0.25 \
  --answer_ce_boost_steps 2000

echo ""
echo "=========================================="
echo "Mini sanity complete! Check:"
echo "  - Checkpoint: ../checkpoints/ir_cot_70m_mini_sanity/"
echo "  - Logs: ../logs/mini_sanity_70m.log"
echo "  - Debug dumps: ../checkpoints/ir_cot_70m_mini_sanity/logs/"
echo "=========================================="
