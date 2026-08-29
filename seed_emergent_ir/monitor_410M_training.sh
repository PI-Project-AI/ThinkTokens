#!/bin/bash
# Monitor 410M training and extract epoch 5 metrics

LOG_FILE="logs/train_410m_fixed.log"

echo "=========================================="
echo "410M Training Monitor"
echo "=========================================="
echo ""

# Check if training is still running
if pgrep -f "train_v2.py.*pythia-410m" > /dev/null; then
    echo "✓ Training process is ACTIVE"
else
    echo "✗ Training process is NOT running"
fi

echo ""
echo "--- Current Progress ---"
tail -3 "$LOG_FILE" | grep "Epoch"

echo ""
echo "--- Epochs Completed ---"
grep -E "^Epoch [0-9]+ completed" "$LOG_FILE" | tail -5

echo ""
echo "--- Validation Results ---"
grep -E "Val Accuracy|Val Loss" "$LOG_FILE" | tail -10

echo ""
echo "--- IR Integrity & Utilization ---"
grep -E "IR integrity|Codebook utilization" "$LOG_FILE" | tail -10

echo ""
echo "--- Causality Tests ---"
grep -E "Random-IR|Shuffle-IR|Drop-IR" "$LOG_FILE" | tail -20

echo ""
echo "=========================================="
echo "To see full log: tail -f $LOG_FILE"
echo "=========================================="
