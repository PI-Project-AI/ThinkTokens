#!/bin/bash
# Launch IR-CoT V2 Training with VQ-Tied Codes
# Optimized for Pythia-410M on 16GB GPU

set -e

echo "=========================================="
echo "IR-CoT V2 Training Launch"
echo "=========================================="

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Training will use CPU (very slow)."
else
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Navigate to code directory
cd "$(dirname "$0")/code"

# Default configuration for 410M on 16GB GPU
MODEL="EleutherAI/pythia-410m"
BATCH_SIZE=16
NUM_EPOCHS=20
LR="5e-5"
OUTPUT_DIR="../checkpoints/ir_cot_v2_410m"

# Memory optimizations
USE_LORA="--use_lora"
USE_8BIT="--use_8bit_adam"
GRAD_CKPT="--gradient_checkpointing"

# Temperature annealing
TEMP_INIT=0.7
TEMP_FINAL=0.4

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LR"
echo "  Temperature: $TEMP_INIT → $TEMP_FINAL"
echo "  Memory opts: LoRA + 8-bit Adam + Gradient Checkpointing"
echo ""

# Check dependencies
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

if python -c "import peft" 2>/dev/null; then
    python -c "import peft; print(f'PEFT: {peft.__version__}')"
else
    echo "Warning: PEFT not installed. Install with: pip install peft"
    USE_LORA=""
fi

if python -c "import bitsandbytes" 2>/dev/null; then
    python -c "import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')"
else
    echo "Warning: bitsandbytes not installed. Install with: pip install bitsandbytes"
    USE_8BIT=""
fi

echo ""
echo "Starting training in 3 seconds..."
sleep 3

# Launch training
python train_v2.py \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --temp_init $TEMP_INIT \
    --temp_final $TEMP_FINAL \
    --output_dir $OUTPUT_DIR \
    --test_frequency 5 \
    $USE_LORA \
    $USE_8BIT \
    $GRAD_CKPT

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Check causality test results in $OUTPUT_DIR/causality_tests_*.json"
echo "  2. Review training curves"
echo "  3. If causality tests pass, proceed to GSM8K (Phase 5)"
