#!/bin/bash
# Monitor 1.4B training and notify when complete

LOG_FILE="train_1.4B_fixed.log"
CHECK_INTERVAL=60  # Check every 60 seconds

echo "Monitoring training progress..."
echo "Log file: $LOG_FILE"
echo "Checking every $CHECK_INTERVAL seconds"
echo ""

while true; do
    # Check if training is complete
    if grep -q "✓ Training complete for 1.4B" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "=========================================="
        echo "🎉 TRAINING COMPLETE! 🎉"
        echo "=========================================="
        echo ""
        echo "Model: Pythia-1.4B"
        echo "Checkpoint: checkpoints_1.4B/final_model.pt"
        echo "Results: results_1.4B/"
        echo ""
        echo "Next steps:"
        echo "  1. python eval_multisize.py --model 1.4B"
        echo "  2. python compare_scales.py --models 410M,1.4B"
        echo ""

        # Try to send desktop notification (if available)
        if command -v notify-send &> /dev/null; then
            notify-send "Training Complete" "1.4B model training finished!" -u critical
        fi

        # Play a beep sound (if available)
        if command -v paplay &> /dev/null && [ -f /usr/share/sounds/freedesktop/stereo/complete.oga ]; then
            paplay /usr/share/sounds/freedesktop/stereo/complete.oga
        elif command -v beep &> /dev/null; then
            beep -f 1000 -l 500
        fi

        break
    fi

    # Check if training failed
    if grep -q "Traceback\|Error\|CUDA out of memory" "$LOG_FILE" 2>/dev/null; then
        if ! grep -q "✓ Training complete" "$LOG_FILE" 2>/dev/null; then
            echo ""
            echo "=========================================="
            echo "❌ TRAINING ERROR DETECTED"
            echo "=========================================="
            echo ""
            echo "Check $LOG_FILE for details"
            echo ""

            if command -v notify-send &> /dev/null; then
                notify-send "Training Error" "1.4B model training encountered an error" -u critical
            fi

            break
        fi
    fi

    # Show periodic progress update
    if [ -f "$LOG_FILE" ]; then
        PROGRESS=$(grep -oP "Epoch \d+/\d+:\s+\K\d+%" "$LOG_FILE" 2>/dev/null | tail -1)
        EPOCH=$(grep -oP "Epoch \K\d+/\d+" "$LOG_FILE" 2>/dev/null | tail -1)
        LOSS=$(grep -oP "loss=\K[\d\.]+" "$LOG_FILE" 2>/dev/null | tail -1)

        if [ -n "$PROGRESS" ]; then
            echo "[$(date '+%H:%M:%S')] Epoch $EPOCH - Progress: $PROGRESS - Loss: $LOSS"
        fi
    fi

    sleep $CHECK_INTERVAL
done

echo "Monitoring stopped."