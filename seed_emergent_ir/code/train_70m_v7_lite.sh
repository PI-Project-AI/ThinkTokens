#!/bin/bash
# V7-LITE: Contrastive + Gumbel warm-start (single 512 codebook)
# Fixes IR information collapse by forcing HL-IR alignment and differentiable learning
set -e
cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "=========================================="
echo "IR-CoT V7-LITE: Pythia-70M"
echo "Phase A: Arithmetic (7k train, 20 epochs)"
echo "=========================================="
echo ""
echo "V7-Lite Changes:"
echo "  1. InfoNCE Contrastive Loss (weight=0.3, T=0.07)"
echo "  2. Gumbel-Softmax warm-start (steps 0-1500, tau=0.6)"
echo "  3. Single 512 codebook (no product codes)"
echo "  4. Entropy-based diversity (coverage 0.5 → 0.25)"
echo "  5. Success gates: Step 600 (15% util, 60% NN-acc), Step 1500 (30-50% util, 80% NN-acc)"
echo ""
echo "Key Differences from V6:"
echo "  - Contrastive loss forces IR to capture input information"
echo "  - Gumbel provides differentiable path through VQ selection"
echo "  - No seeded codes (Gumbel handles exploration)"
echo ""

python train_v2.py \
  --model_name "EleutherAI/pythia-70m" \
  --use_lora \
  --output_dir "../checkpoints/ir_cot_70m_v7_lite" \
  --num_codes 512 \
  --code_dim 128 \
  --temp_init 1.0 \
  --temp_final 0.8 \
  --train_data "../data/arithmetic/train.json" \
  --val_data "../data/arithmetic/val.json" \
  --batch_size 8 \
  --num_epochs 20 \
  --lr 5e-5 \
  --gradient_checkpointing \
  --use_contrastive \
  --contrastive_weight 0.3 \
  --contrastive_T 0.07 \
  --use_gumbel_warmstart \
  --gumbel_tau 0.6 \
  --gumbel_steps 1500 \
  --phase0_seeded_steps 0 \
  --seed 42 \
  --fp16 \
  --max_grad_norm 1.0 \
  --log_file "../logs/train_70m_v7_lite.log" 2>&1 | tee -a "../logs/train_70m_v7_lite.log"

echo ""
echo "=========================================="
echo "Training complete! Check results:"
echo "  - Checkpoint: ../checkpoints/ir_cot_70m_v7_lite/"
echo "  - Logs: ../logs/train_70m_v7_lite.log"
echo "  - Debug dumps (step 600, 1200, 1500): ../checkpoints/ir_cot_70m_v7_lite/logs/"
echo ""
echo "Expected outcomes by step 1500:"
echo "  ✓ Utilization: 30-50% (150-256 codes)"
echo "  ✓ Top-1 frequency: <30%"
echo "  ✓ IR integrity: ≥90%"
echo "  ✓ NN-accuracy: ≥80%"
echo "  ✓ Val accuracy: >0%"
echo ""
echo "If util <15% or NN-acc <60% at step 1500, pivot to Option C (supervised pretrain)"
echo "=========================================="
