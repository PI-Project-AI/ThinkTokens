#!/bin/bash
# Monitor training progress

echo "========================================================================================"
echo "IR REASONING LANGUAGE - TRAINING MONITOR"
echo "========================================================================================"
echo "Time: $(date)"
echo ""

# Check baseline
echo "BASELINE RESULTS:"
if [ -f "results/baseline.json" ]; then
    echo "✓ Baseline completed"
    python -c "import json; b = json.load(open('results/baseline.json')); print(f\"  Accuracy: {b['accuracy']:.1%}\"); print(f\"  Avg Tokens: {b['avg_tokens']:.0f}\")"
else
    echo "⏳ Baseline running or not started"
fi

echo ""
echo "TRAINING STATUS:"
if [ -f "results/training_history.json" ]; then
    echo "✓ Training started"
    python -c "import json; h = json.load(open('results/training_history.json')); e = len(h['epochs']); print(f\"  Epochs completed: {e}\"); print(f\"  Latest loss: {h['losses'][-1]['total_loss']:.4f}\"); print(f\"  Code usage: {h['losses'][-1]['code_usage']:.1f}%\")"
    echo "  Latest checkpoint:"
    ls -lh checkpoints/ 2>/dev/null | tail -1 || echo "    (no checkpoints yet)"
else
    echo "⏳ Training not started or initializing"
fi

echo ""
echo "GPU STATUS:"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,unit=C

echo ""
echo "DISK USAGE:"
du -sh results/ checkpoints/ 2>/dev/null | awk '{print "  " $0}'

echo ""
echo "========================================================================================"
