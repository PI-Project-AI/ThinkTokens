#!/bin/bash
# ENTROPY-BASED DIVERSITY V3: Pythia-70M with differentiable entropy loss
# Phase A: Arithmetic dataset, 20 epochs

set -e

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "=========================================="
echo "IR-CoT ENTROPY V3: Pythia-70M Training"
echo "Phase A: Arithmetic (7k train, 20 epochs)"
echo "=========================================="
echo ""
echo "CRITICAL FIX: ENTROPY-BASED DIVERSITY LOSS"
echo "  ✓ FIX #5: Diversity loss = log(C) - H(avg_code_dist)"
echo "      - Uses VQ logits → soft probs → entropy"
echo "      - FULLY DIFFERENTIABLE (proper gradient flow!)"
echo "      - Coverage weight: 0.50 (epochs 1-2) → 0.25 (epochs 3+)"
echo ""
echo "PREVIOUS FIXES (ALL RETAINED):"
echo "  ✓ FIX #1: Diversity loss returns proper tensor (not Python float)"
echo "  ✓ FIX #2: Gradient leak λ=0 (no warm-start attractor)"
echo "  ✓ FIX #3: Grammar constraints:"
echo "      - Min 3 spans (was 4)"
echo "      - Ban consecutive identical codes"
echo "      - Require ≥2 distinct codes per span"
echo "  ✓ FIX #4: Early diversity guards:"
echo "      - Activate at step ≥100 (was 500)"
echo "      - 50-step window (was 200)"
echo ""
echo "ROOT CAUSE IDENTIFIED:"
echo "  Previous diversity loss had NO GRADIENTS:"
echo "    - utilization = len(unique_codes) / num_codes"
echo "    - torch.unique() breaks gradient path"
echo "    - len() returns Python int (no grad_fn)"
echo "  New entropy loss HAS GRADIENTS:"
echo "    - Soft probs from VQ logits (differentiable)"
echo "    - Entropy computed via log + sum (differentiable)"
echo "    - Direct gradient path to VQ codebook!"
echo ""
echo "EXPECTED HEALTH TARGETS:"
echo "  Step ~50:   Entropy loss > 0 (diversity penalty active)"
echo "  Step ~200:  Codebook util ≥10-20%, top-1 <50%"
echo "  Step ~500:  Codebook util ≥30-60%, top-1 <30%"
echo "  Ongoing:    IR integrity >80%, answer CE >0.2"
echo ""
echo "VQ STABILIZATION:"
echo "  ✓ Temperature: 1.0 → 0.7 (slower annealing)"
echo "  ✓ Coverage weight: 0.50 (epochs 1-2) → 0.25 (epochs 3+)"
echo "  ✓ Commitment β: 0.5 (warm-start) → 0.25 (normal)"
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
  --output_dir ../checkpoints/ir_cot_70m_entropy_v3 \
  --test_frequency 5 \
  --no_ir_teacher_forcing \
  --enable_debug \
  --debug_step_frequency 500 \
  --enable_assertions \
  --run_control_random_ir \
  2>&1 | tee ../logs/train_70m_entropy_v3.log

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoint: ../checkpoints/ir_cot_70m_entropy_v3/best_model.pt"
echo "Log: ../logs/train_70m_entropy_v3.log"
echo "Debug logs: ../checkpoints/ir_cot_70m_entropy_v3/logs/"
echo "=========================================="
