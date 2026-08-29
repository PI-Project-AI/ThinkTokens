#!/bin/bash
# Monitor 410M training progress with corrected architecture

LOG_FILE="train_410M_corrected_medium.log"
CHECK_INTERVAL=30  # Check every 30 seconds

echo "=========================================="
echo "Monitoring 410M Training (Hard Bottleneck)"
echo "=========================================="
echo "Dataset: Medium (2000 samples, 3 epochs)"
echo "Log file: $LOG_FILE"
echo ""

while true; do
    if [ ! -f "$LOG_FILE" ]; then
        echo "Waiting for training to start..."
        sleep $CHECK_INTERVAL
        continue
    fi

    # Check if training complete
    if grep -q "✓ Training complete for 410M" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "=========================================="
        echo "🎉 TRAINING COMPLETE!"
        echo "=========================================="
        echo ""
        echo "Results: results_410M/"
        echo "Checkpoints: checkpoints_410M/"
        echo ""
        echo "Next: python eval_multisize.py --model 410M"

        # Desktop notification
        if command -v notify-send &> /dev/null; then
            notify-send "410M Training Complete" "Hard bottleneck model finished!" -u critical
        fi

        break
    fi

    # Check for errors
    if grep -q "Traceback\|Error\|CUDA out of memory" "$LOG_FILE" 2>/dev/null; then
        if ! grep -q "✓ Training complete" "$LOG_FILE" 2>/dev/null; then
            echo "❌ ERROR DETECTED - Check $LOG_FILE"
            break
        fi
    fi

    # Get current progress
    CURRENT_EPOCH=$(grep -oP "Epoch \K\d+/\d+" "$LOG_FILE" 2>/dev/null | tail -1)
    CURRENT_BATCH=$(grep -oP "Epoch \d+/\d+:\s+\K\d+%" "$LOG_FILE" 2>/dev/null | tail -1)
    CURRENT_LOSS=$(grep -oP "loss=\K[\d\.]+" "$LOG_FILE" 2>/dev/null | tail -1)
    CODE_USAGE=$(grep -oP "codes=\K\d+/512" "$LOG_FILE" 2>/dev/null | tail -1)

    # Show summary
    clear
    echo "=========================================="
    echo "410M Training Progress (Hard Bottleneck)"
    echo "=========================================="
    echo ""
    echo "Epoch:       $CURRENT_EPOCH"
    echo "Progress:    $CURRENT_BATCH"
    echo "Loss:        $CURRENT_LOSS"
    echo "Codes Used:  $CODE_USAGE"
    echo ""
    echo "Refresh: Every ${CHECK_INTERVAL}s"
    echo "Log: tail -f $LOG_FILE"
    echo ""
    echo "Expected completion: ~2-2.5 hours from start"
    echo ""

    sleep $CHECK_INTERVAL
done

echo ""
echo "Monitoring stopped."
