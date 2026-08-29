#!/usr/bin/env bash

# V9: Concept Bottleneck IR Training Script (CORRECTED)
# Adds IR→concept head for semantic grounding
# FIX: answer_ce_boost_steps=0, concept_weight=0.1, num_epochs=3

source ../../.venv/bin/activate

python train_v2.py \
  --model_name "EleutherAI/pythia-70m" \
  --use_lora \
  --output_dir "../checkpoints/ir_cot_70m_v9_fixed" \
  --num_codes 512 \
  --code_dim 128 \
  --temp_init 1.0 \
  --temp_final 0.8 \
  --train_data "../data/arithmetic_v9/train_v9.json" \
  --val_data "../data/arithmetic_v9/val_v9.json" \
  --batch_size 8 \
  --num_epochs 3 \
  --lr 5e-5 \
  --gradient_checkpointing \
  --use_contrastive \
  --contrastive_weight 0.3 \
  --contrastive_T 0.07 \
  --use_gumbel_warmstart \
  --gumbel_tau 0.6 \
  --gumbel_steps 1500 \
  --phase0_seeded_steps 0 \
  --use_ir_value_head \
  --ir_value_weight 0.25 \
  --answer_ce_boost_steps 0 \
  --use_concept_head \
  --concept_weight 0.1 \
  --seed 42 \
  --fp16 \
  --max_grad_norm 1.0 \
  --enable_debug \
  --debug_step_frequency 200 \
  --log_file "../logs/v9_training_fixed.log" \
  2>&1 | tee ../logs/v9_training_fixed.log
