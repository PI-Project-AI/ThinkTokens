# Execution Guide: Complete IR Reasoning Experiment

## Current Status

✅ **Implementation Complete**
✅ **Dependencies Installed**
🔄 **Baseline Evaluation Running** (Started at ~14:10 UTC)

---

## What Has Been Done

### Code Files Created
1. **vq_model.py** - Vector Quantized transformer model
2. **train_vq.py** - Training script with full pipeline
3. **eval_baseline.py** - Baseline model evaluation
4. **eval_vq.py** - VQ model evaluation
5. **analyze_results.py** - Analysis and visualization
6. **run_pipeline.py** - Master orchestration script
7. **IMPLEMENTATION.md** - Technical documentation
8. **PROJECT_STATUS.md** - Detailed status report
9. **requirements.txt** - All dependencies

### Current Execution
```bash
# Baseline evaluation running in background
# Task: Download Pythia-410M model, evaluate on 100 GSM8K test samples
# Status: Running
# ETA: 30-60 minutes
```

---

## How to Proceed

### Option 1: Let Baseline Complete Automatically (Recommended)

The baseline is already running. Just monitor progress:

```bash
# Check on baseline progress (in another terminal)
cd /home/pi-project-admin/PycharmProjects/PythonProject/ThinkTokens
source .venv/bin/activate

# Check if baseline.json has been created
ls -lh results/baseline.json 2>/dev/null && echo "✓ Baseline complete" || echo "⏳ Still running..."

# Check GPU usage to monitor progress
nvidia-smi

# Monitor memory
free -h
```

Once baseline.json appears, you can proceed to Step 2.

### Option 2: Run Full Pipeline After Baseline Completes

Once baseline completes, run the full pipeline:

```bash
source .venv/bin/activate
python run_pipeline.py
```

This will:
1. ✓ Complete baseline evaluation (if not done)
2. ▶ Train VQ model (6-8 hours)
3. ▶ Evaluate trained model (30-60 min)
4. ▶ Generate analysis and visualizations (5-10 min)

### Option 3: Run Individual Components Sequentially

If you want manual control:

```bash
source .venv/bin/activate

# Step 1: Wait for baseline, or run it
python eval_baseline.py

# Step 2: Train VQ model
python train_vq.py

# Step 3: Evaluate VQ model
python eval_vq.py

# Step 4: Analyze results
python analyze_results.py
```

---

## Monitoring Progress

### Check Baseline Status
```bash
# Method 1: Check results file
ls -la results/baseline.json

# Method 2: Check for active Python processes
ps aux | grep python

# Method 3: Monitor GPU
watch -n 2 nvidia-smi

# Method 4: Check results file size (grows as evaluation progresses)
watch -n 10 'du -sh results/baseline.json 2>/dev/null || echo "Not started"'
```

### Check Training Progress
During training, monitor:

```bash
# Watch training output live
tail -f results/training_history.json

# Monitor GPU/Memory
nvidia-smi -l 1  # Refresh every 1 second

# Check checkpoint creation
ls -lt checkpoints/ | head -10
```

### Check Final Results
```bash
# Once evaluation complete
ls -la results/

# View summary report
cat results/analysis_report.txt

# View comparison plots
# On Linux with display
eog results/comparison_plot.png

# On Windows/Mac, open with image viewer
open results/comparison_plot.png  # macOS
```

---

## Timeline Estimates

### Phase 1: Baseline (Currently Running)
- **Task:** Download Pythia-410M, evaluate on 100 samples
- **GPU Usage:** ~20-24GB
- **Time:** 30-60 minutes
- **Output:** `results/baseline.json`

### Phase 2: Training
- **Task:** Train VQ model for 3 epochs on full GSM8K (~7,500 samples)
- **GPU Usage:** ~20-24GB
- **Time:** 6-8 hours
- **Checkpoints:** Created every 500 steps + per epoch
- **Outputs:**
  - `checkpoints/epoch_1.pt`, `epoch_2.pt`, `epoch_3.pt`
  - `checkpoints/final_model.pt`
  - `results/training_history.json`

### Phase 3: Evaluation
- **Task:** Evaluate trained model on 100 test samples
- **GPU Usage:** ~20-24GB
- **Time:** 30-60 minutes
- **Output:** `results/vq_results.json`

### Phase 4: Analysis
- **Task:** Compare results, generate visualizations
- **CPU Only:** No GPU needed
- **Time:** 5-10 minutes
- **Outputs:**
  - `results/analysis_report.txt`
  - `results/comparison_plot.png`
  - `results/training_history.png`

**Total Pipeline Time:** ~8-10 hours

---

## Expected Outputs

### After Baseline
```
results/
└── baseline.json
    ├── model_name: "EleutherAI/pythia-410m"
    ├── accuracy: 0.18-0.25  (18-25%)
    ├── avg_tokens: 300-500
    └── samples: [...]
```

### After Training
```
results/
├── training_history.json
│   ├── epochs: [1, 2, 3]
│   └── losses: [...]
│
results/training_config.json
```

```
checkpoints/
├── epoch_1.pt
├── epoch_2.pt
├── epoch_3.pt
└── final_model.pt
```

### After Evaluation
```
results/
└── vq_results.json
    ├── accuracy: 0.15-0.23
    ├── avg_tokens: 200-400
    ├── codebook_stats: {...}
    └── samples: [...]
```

### After Analysis
```
results/
├── analysis_report.txt        # Full comparison report
├── comparison_plot.png        # Accuracy/token/codebook charts
└── training_history.png       # Loss and usage curves

results/pipeline.json          # Execution log with timestamps
```

---

## Interpreting Results

### Key Metrics to Look For

#### 1. Accuracy
```
Baseline:  18-25%
VQ Model:  15-23%
Target:    Within 5% of baseline (change = ±5pp)
```

#### 2. Token Efficiency
```
Baseline:  300-500 avg tokens
VQ Model:  200-400 avg tokens
Target:    >20% reduction (60-400 avg tokens)
Success:   Token reduction % = (Baseline - VQ) / Baseline * 100
```

#### 3. Codebook Utilization
```
Target:    >50% codes used
Example:   256 codes used / 512 total = 50%
Warning:   <20% indicates collapse (failure mode)
Good:      50-80% indicates healthy usage
Excellent: >80% indicates strong learning
```

#### 4. Success Criteria Met
Look in `analysis_report.txt`:
```
Codebook utilization >50%: ✓ PASS / ✗ FAIL
Token reduction >20%: ✓ PASS / ✗ FAIL
Accuracy within 5%: ✓ PASS / ✗ FAIL
No catastrophic loss: ✓ PASS / ✗ FAIL

Criteria met: 4/4
```

---

## Troubleshooting During Execution

### Baseline Still Running After 2 Hours?
```bash
# Check if process is alive
ps aux | grep eval_baseline

# Check disk space
df -h

# Check if model is still downloading
ls -lh ~/.cache/huggingface/hub/

# If stuck, kill and restart
pkill -f eval_baseline.py
python eval_baseline.py
```

### Training Crashes with CUDA Out of Memory?
Edit `train_vq.py`:
```python
CONFIG['batch_size'] = 4  # was 8
CONFIG['max_length'] = 256  # was 512
```

Then restart:
```bash
python train_vq.py
```

### Training Loss Not Decreasing?
```python
CONFIG['learning_rate'] = 1e-5  # Lower learning rate
CONFIG['epochs'] = 5  # Train longer
```

### Codebook Collapse (all loss, no codes used)?
```python
# In vq_model.py VectorQuantizer init:
self.commitment_cost = 0.5  # Increase from 0.25
```

---

## Advanced Monitoring

### Real-Time Training Monitor
```bash
# In one terminal, watch losses
watch -n 30 'tail -20 results/training_history.json | python -m json.tool'

# In another, watch GPU
watch -n 2 nvidia-smi

# In another, watch checkpoints
watch -n 60 'ls -lrt checkpoints/ | tail -5'
```

### Save Logs for Later Review
```bash
# Start with tee to capture output
python run_pipeline.py | tee execution_log.txt

# Later, review the log
cat execution_log.txt | grep -i "error\|warning\|loss"
```

---

## After Completion

### Review Results
```bash
# 1. Check main report
cat results/analysis_report.txt

# 2. View plots (if display available)
eog results/comparison_plot.png
eog results/training_history.png

# 3. Examine JSON data
python -m json.tool < results/baseline.json | head -50
python -m json.tool < results/vq_results.json | head -50
```

### Decide Next Steps

Based on results, decide:

1. **Success (all criteria met)?**
   - ✓ Proceed to scale-robust validation (410M + 1.4B)
   - ✓ Explore cross-task transfer (SVAMP)
   - ✓ Analyze emergent code structure

2. **Partial success (2-3 criteria met)?**
   - ⏰ Investigate what went right/wrong
   - ⏰ Try different hyperparameters
   - ⏰ Longer training or larger codebook

3. **Failure (0-1 criteria met)?**
   - ✗ May indicate hypothesis not viable at this scale
   - ✗ Document findings as negative result
   - ✗ Consider architectural changes

---

## Quick Commands Reference

```bash
# Navigate to project
cd /home/pi-project-admin/PycharmProjects/PythonProject/ThinkTokens

# Activate environment
source .venv/bin/activate

# Run full pipeline
python run_pipeline.py

# Run individual steps
python eval_baseline.py        # Step 1
python train_vq.py             # Step 2
python eval_vq.py              # Step 3
python analyze_results.py      # Step 4

# Monitor execution
nvidia-smi -l 2               # GPU every 2 sec
tail -f results/training_history.json  # Live losses
ps aux | grep python          # Active processes

# View results
cat results/analysis_report.txt
ls -lh results/
ls -lrt checkpoints/
```

---

## Expected Behavior

### Baseline Evaluation Output
```
Device: cuda
Loading model: EleutherAI/pythia-410m
Loading GSM8K dataset...

Evaluating EleutherAI/pythia-410m on 100 samples:
100%|██████████| 100/100 [45:30<00:00, 27.30s/it]

============================================================
Baseline Results: EleutherAI/pythia-410m
============================================================
Accuracy: 21.00% (21/100)
Avg tokens: 387.5
Total tokens: 38750
Min tokens: 156
Max tokens: 512

Results saved to results/baseline.json
```

### Training Output
```
VQ Reasoning Model initialized:
  Base model: EleutherAI/pythia-410m
  Hidden size: 1024
  Number of codes: 512
  Bottleneck layer: 12/24

Loading tokenizer and dataset...
Processing dataset...
Dataset size: 7473
Num batches per epoch: 934

====================================================================
Epoch 1/3
====================================================================
Epoch 1: 100%|██| 934/934 [2:15:30<00:00,  8.71s/it]

Epoch 1 Summary:
  Avg Loss: 3.8432
  Avg LM Loss: 3.7821
  Avg VQ Loss: 0.0611
  Avg Code Usage: 68.2%
Checkpoint saved: checkpoints/epoch_1.pt
```

### Analysis Report Output
```
================================================================================
INTERMEDIATE REASONING LANGUAGE - RESULTS ANALYSIS
================================================================================

1. BASELINE MODEL
Accuracy: 21.00%
Avg Tokens: 387.5

2. VQ MODEL
Accuracy: 19.50%
Avg Tokens: 310.0

3. COMPARATIVE ANALYSIS
Accuracy Change: -1.50 percentage points (-7.1%)
Token Reduction: +77.5 tokens (+20.0%)

4. SUCCESS CRITERIA
✓ PASS: Codebook utilization >50%
✓ PASS: Token reduction >20%
⚠ WARNING: Accuracy within 5% of baseline
✓ PASS: No catastrophic accuracy loss

Criteria met: 3/4
```

---

## Next Steps After Completion

1. **Review Results**
   - Read analysis_report.txt
   - View comparison_plot.png
   - Check if success criteria met

2. **If Successful:**
   - Scale to Pythia-1.4B
   - Test on SVAMP dataset
   - Analyze codebook structure

3. **If Partial Success:**
   - Modify hyperparameters
   - Try larger codebook (1024)
   - Extend training (5 epochs)

4. **Document Findings:**
   - Save results to shared location
   - Write summary report
   - Compare to baseline research

---

**Status:** Ready to Execute ✅

Start with monitoring baseline, then run full pipeline when baseline completes.
