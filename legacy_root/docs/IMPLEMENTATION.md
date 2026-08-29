# Intermediate Reasoning Language Implementation

## Project Overview

This is an end-to-end implementation of the Intermediate Reasoning Language (IR) research project. The system tests whether LLMs can reason more efficiently using discrete learned codes rather than verbose natural language chain-of-thought.

## Architecture

### Core Components

1. **VQ Bottleneck** (`vq_model.py`)
   - Vector Quantizer module implementing straight-through estimator
   - Inserts discrete codes into transformer midpoint
   - 512-code vocabulary by default
   - Monitors codebook utilization to detect collapse

2. **Training Pipeline** (`train_vq.py`)
   - Trains VQReasoningModel on GSM8K dataset
   - Configurable architecture (bottleneck position, codebook size)
   - Tracks multiple loss components (LM loss, VQ loss, commitment loss)
   - Monitors codebook usage per epoch

3. **Evaluation Framework** (`eval_vq.py`, `eval_baseline.py`)
   - Baseline evaluation on Pythia-410M
   - VQ model evaluation with same protocol
   - Answer extraction and accuracy calculation
   - Codebook statistics analysis

4. **Analysis Tools** (`analyze_results.py`)
   - Comparative metrics (accuracy, token efficiency, codebook usage)
   - Success criteria validation
   - Training history visualization
   - Comprehensive report generation

## Quick Start

### 1. Setup
```bash
cd /home/pi-project-admin/PycharmProjects/PythonProject/ThinkTokens
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python run_pipeline.py
```

This executes all steps:
- Baseline evaluation
- VQ model training
- VQ model evaluation
- Results analysis & visualization

### 3. Individual Steps

**Baseline only:**
```bash
python eval_baseline.py
```

**Training only:**
```bash
python train_vq.py
```

**Evaluation only (requires trained model):**
```bash
python eval_vq.py
```

**Analysis only (requires baseline & VQ results):**
```bash
python analyze_results.py
```

## Configuration

Edit configuration in the scripts:

### eval_baseline.py
- `num_samples`: Number of test samples (default: 100)

### train_vq.py
```python
CONFIG = {
    'model_name': "EleutherAI/pythia-410m",
    'num_codes': 512,
    'batch_size': 8,
    'epochs': 3,
    'learning_rate': 5e-5,
    'max_length': 512,
}
```

### vq_model.py
```python
VQReasoningModel(
    base_model_name,
    num_codes=512,           # Codebook size
    bottleneck_position="middle",  # Middle of transformer
    commitment_cost=0.25     # VQ loss weight
)
```

## Expected Output

### Results Directory
```
results/
├── baseline.json              # Baseline metrics & samples
├── vq_results.json           # VQ model metrics & samples
├── training_history.json     # Per-epoch loss tracking
├── training_config.json      # Training hyperparameters
├── analysis_report.txt       # Comprehensive analysis
├── comparison_plot.png       # Accuracy/token/codebook plots
├── training_history.png      # Loss & usage curves
└── pipeline.json             # Pipeline execution log
```

### Checkpoints Directory
```
checkpoints/
├── epoch_1.pt               # Checkpoint after epoch 1
├── epoch_2.pt               # Checkpoint after epoch 2
├── epoch_3.pt               # Checkpoint after epoch 3
└── final_model.pt           # Final trained model
```

## Success Criteria

The experiment is considered successful if:

1. **Codebook Utilization > 50%**
   - Indicates codes are being learned and used
   - Avoids "codebook collapse" failure mode

2. **Token Reduction > 20%**
   - VQ model produces fewer tokens than baseline
   - Demonstrates compression benefit

3. **Accuracy Within 5% of Baseline**
   - Model maintains reasoning quality
   - No catastrophic accuracy loss

4. **Stable Training**
   - Loss converges smoothly
   - No NaN or divergence

## Key Metrics to Monitor

### During Training
- **Total Loss**: Should decrease smoothly
- **LM Loss**: Language modeling objective
- **VQ Loss**: Quantization quality
- **Code Usage %**: Should remain >50%

### During Evaluation
- **Accuracy**: Percentage of correct answers
- **Avg Tokens**: Average token count per sample
- **Token Reduction**: Efficiency gain vs baseline
- **Codebook Usage**: Percentage of codes utilized

## Troubleshooting

### Out of Memory
```python
CONFIG['batch_size'] = 4  # Reduce batch size
CONFIG['max_length'] = 256  # Reduce sequence length
```

### Training Loss Not Decreasing
```python
CONFIG['learning_rate'] = 1e-5  # Lower learning rate
CONFIG['epochs'] = 5  # Train longer
```

### Codebook Collapse
```python
# In vq_model.py VectorQuantizer.__init__:
commitment_cost = 0.5  # Increase commitment loss weight
```

### Model Out of Memory During Generation
Edit `eval_vq.py`:
```python
max_new_tokens=128  # Reduce generation length
```

## Scaling to Larger Models

To test on multiple scales:

```bash
# Edit train_vq.py to use different models:
CONFIG['model_name'] = "EleutherAI/pythia-1.4b"

python train_vq.py
python eval_vq.py
```

Expected resource usage:
- **Pythia-410M**: ~8 hours on A100, 8GB GPU
- **Pythia-1.4B**: ~20 hours on A100, 16GB GPU
- **Pythia-2.8B**: ~40 hours on A100, 24GB GPU

## Research Next Steps

If results are positive:

1. **Scale Validation**
   - Train on Pythia-1.4B + 2.8B
   - Check if gains improve with scale

2. **Transfer Testing**
   - Evaluate GSM8K-trained model on SVAMP
   - Test cross-domain generalization

3. **Ablation Studies**
   - Try different bottleneck positions
   - Vary codebook sizes (256, 512, 1024)
   - Test different commitment weights

4. **Interpretability Analysis**
   - Cluster codes by semantic meaning
   - Use probe classifiers to detect reasoning patterns
   - Visualize codebook structure

5. **Modular Architecture**
   - Implement separate encoding/reasoning/decoding stages
   - Enable code reuse across tasks

## Hardware Requirements

### Minimum (Single Epoch POC)
- 8GB GPU (RTX 3060 or better)
- ~4 hours training
- Pythia-410M only

### Recommended (Full POC)
- 24GB GPU (A100 or RTX 3090)
- ~8-12 hours total
- Pythia-410M + evaluation

### Comprehensive Research
- 40GB+ GPU (A100)
- Multi-scale experiments
- 200+ GPU-hours for full ablations

## References

- Original research memo: `docs/Memo - Proof of Concept...md`
- Implementation guide: `docs/POC Implementation Guide...md`
- Scaling analysis: `docs/Scaling Concerns...md`
- Quick start: `docs/Quick Start Guide...md`

## Citation

If you use this implementation, please cite:

```bibtex
@research{think_tokens_2025,
  title={Intermediate Reasoning Language Implementation},
  author={PROVOST, Paul and Claude Code},
  year={2025},
  note={Anthropic Research}
}
```

## Support

For issues or questions:
- Check troubleshooting section above
- Review detailed docs in `docs/` directory
- Check pipeline logs in `results/pipeline.json`
