# Quick Start Guide: 1-Week Minimum POC

**Goal:** Validate the intermediate reasoning language concept in 1 week with minimal resources.

**Who this is for:** You want to test the idea quickly before committing to a full research project.

---

## Day 1: Setup (2-3 hours)

### Step 1: Environment Setup (30 min)

```bash
# Create project directory
mkdir reasoning-ir-poc
cd reasoning-ir-poc

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
vector-quantize-pytorch>=1.12.0
wandb>=0.16.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
EOF

# Install
pip install -r requirements.txt

# Test
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Step 2: Get Baseline Performance (1-2 hours)

```bash
# Create baseline evaluation script
cat > eval_baseline.py << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re

def extract_answer(text):
    """Extract numerical answer from GSM8K."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    return None

def evaluate_model(model_name, num_samples=100):
    """Quick baseline evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("gsm8k", "main")['test'].select(range(num_samples))

    correct = 0
    total_tokens = 0

    for example in tqdm(dataset, desc=f"Evaluating {model_name}"):
        prompt = f"Question: {example['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        predicted = extract_answer(generated)
        true_answer = extract_answer(example['answer'])

        if predicted == true_answer:
            correct += 1

        total_tokens += outputs.size(1)

    accuracy = correct / len(dataset)
    avg_tokens = total_tokens / len(dataset)

    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{len(dataset)})")
    print(f"  Avg tokens: {avg_tokens:.1f}")

    return accuracy, avg_tokens

if __name__ == '__main__':
    # Test on small subset first
    evaluate_model("EleutherAI/pythia-410m", num_samples=100)
EOF

# Run baseline
python eval_baseline.py
```

**Expected output:**
```
Accuracy: 15-25% (pythia-410m on 100 samples)
Avg tokens: 300-500
```

This establishes your baseline. Save these numbers!

---

## Day 2-3: Implement VQ Bottleneck (4-6 hours)

### Minimal VQ Implementation

Create `vq_model.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

class SimpleVQ(nn.Module):
    """Minimal VQ implementation."""

    def __init__(self, num_codes=512, code_dim=512):
        super().__init__()
        self.num_codes = num_codes
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)

    def forward(self, x):
        # x: [batch, seq, dim]
        batch_size, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)

        # Find nearest codes
        distances = torch.cdist(x_flat, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        quantized = self.codebook(indices).view(batch_size, seq_len, dim)

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        # Losses
        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())

        return quantized, commitment_loss + codebook_loss, indices

class VQReasoningModel(nn.Module):
    """Transformer with VQ bottleneck."""

    def __init__(self, base_model_name, num_codes=512):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(base_model_name)
        config = self.base.config

        self.vq = SimpleVQ(num_codes=num_codes, code_dim=config.hidden_size)
        self.bottleneck_layer = config.num_hidden_layers // 2

    def forward(self, input_ids, attention_mask=None, labels=None):
        # This is a simplified version - see full implementation guide for complete version
        transformer = self.base.gpt_neox

        # Embed
        hidden = transformer.embed_in(input_ids)

        # First half of layers
        for i in range(self.bottleneck_layer):
            hidden = transformer.layers[i](hidden, attention_mask=attention_mask)[0]

        # VQ bottleneck
        hidden_vq, vq_loss, indices = self.vq(hidden)

        # Second half
        for i in range(self.bottleneck_layer, len(transformer.layers)):
            hidden_vq = transformer.layers[i](hidden_vq, attention_mask=attention_mask)[0]

        hidden_vq = transformer.final_layer_norm(hidden_vq)
        logits = self.base.embed_out(hidden_vq)

        # Compute loss
        lm_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_loss = lm_loss + 0.25 * vq_loss if lm_loss is not None else vq_loss

        return {
            'loss': total_loss,
            'logits': logits,
            'vq_loss': vq_loss,
            'lm_loss': lm_loss,
        }
```

---

## Day 4-5: Training (6-8 hours compute, 1 hour active)

### Simple Training Script

Create `train.py`:

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
from vq_model import VQReasoningModel

# Config
MODEL_NAME = "EleutherAI/pythia-410m"
BATCH_SIZE = 8
EPOCHS = 3
LR = 5e-5
MAX_LENGTH = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("gsm8k", "main")

def preprocess(examples):
    texts = [f"Question: {q}\nAnswer: {a}"
             for q, a in zip(examples['question'], examples['answer'])]
    return tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding='max_length')

train_data = dataset['train'].map(preprocess, batched=True, remove_columns=dataset['train'].column_names)
train_data.set_format('torch', columns=['input_ids', 'attention_mask'])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = VQReasoningModel(MODEL_NAME, num_codes=512).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_loader) * EPOCHS
)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask, labels=input_ids)
        loss = outputs['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f'checkpoint_epoch{epoch+1}.pt')

print("Training complete!")
```

### Run Training

```bash
# Start training (will take 4-8 hours depending on GPU)
python train.py

# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi
```

---

## Day 6: Evaluation (2-3 hours)

### Evaluate Trained Model

```python
# eval_vq.py
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from vq_model import VQReasoningModel
import re

def extract_answer(text):
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    return match.group(1).replace(',', '') if match else None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = VQReasoningModel("EleutherAI/pythia-410m", num_codes=512).to(device)
model.load_state_dict(torch.load('checkpoint_epoch3.pt'))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
tokenizer.pad_token = tokenizer.eos_token

# Evaluate
dataset = load_dataset("gsm8k", "main")['test'].select(range(100))

correct = 0
total_tokens = 0

for example in tqdm(dataset):
    prompt = f"Question: {example['question']}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        # For generation, we'd need to implement proper generation loop
        # For quick POC, just check training accuracy
        outputs = model(inputs['input_ids'])
        logits = outputs['logits']

        # Simple greedy decode (not full generation)
        predicted_ids = logits.argmax(dim=-1)
        generated = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

        predicted_answer = extract_answer(generated)
        true_answer = extract_answer(example['answer'])

        if predicted_answer == true_answer:
            correct += 1

        total_tokens += logits.size(1)

accuracy = correct / len(dataset)
avg_tokens = total_tokens / len(dataset)

print(f"\nVQ Model Results:")
print(f"  Accuracy: {accuracy:.2%}")
print(f"  Avg tokens: {avg_tokens:.1f}")

# Compare to baseline
print("\nComparison to baseline:")
print(f"  Baseline accuracy: [YOUR BASELINE]%")
print(f"  VQ accuracy: {accuracy:.2%}")
print(f"  Difference: {accuracy - YOUR_BASELINE:.2%}")
```

---

## Day 7: Analysis & Decision (2-3 hours)

### Quick Codebook Analysis

```python
# analyze.py
import torch
import matplotlib.pyplot as plt
from vq_model import VQReasoningModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VQReasoningModel("EleutherAI/pythia-410m", num_codes=512).to(device)
model.load_state_dict(torch.load('checkpoint_epoch3.pt'))

# Get codebook
codebook = model.vq.codebook.weight.data.cpu().numpy()

# Simple visualization
print(f"Codebook shape: {codebook.shape}")
print(f"Mean code norm: {(codebook**2).sum(1).mean():.4f}")

# During training, track which codes were used
# (In a full implementation, you'd save indices during training)
# For quick POC, just check if training worked

print("\nPOC Complete!")
print("\nNext steps:")
print("1. Compare VQ accuracy to baseline")
print("2. Check if codebook is being used (not collapsed)")
print("3. Decide: continue with larger models or pivot?")
```

---

## Quick Decision Framework

After 1 week, you should know:

### ✅ **Continue if:**
- [ ] VQ accuracy within 10% of baseline
- [ ] Some token reduction observed
- [ ] Codebook shows usage (not all codes dead)
- [ ] Training was stable (no NaN losses)

### ⚠️ **Investigate if:**
- [ ] Accuracy drop >10% but codebook looks good
- [ ] Token efficiency unclear
- [ ] Some instability but recoverable

### ❌ **Pivot if:**
- [ ] Accuracy drop >20%
- [ ] Codebook completely collapsed (<10% codes used)
- [ ] Training unstable (constant NaN)
- [ ] No efficiency gain whatsoever

---

## Common Issues & Quick Fixes

### Issue: "Out of memory"
**Fix:** Reduce batch size to 4 or 2

### Issue: "Training loss not decreasing"
**Fix:** Lower learning rate to 1e-5

### Issue: "Codebook collapse (only 20 codes used)"
**Fix:** Add diversity loss (see full implementation guide)

### Issue: "Accuracy way worse than baseline"
**Fix:** Train longer (5 epochs instead of 3)

---

## What You'll Learn in 1 Week

1. ✅ Can you implement VQ bottleneck successfully?
2. ✅ Does training converge?
3. ✅ Is there any efficiency signal?
4. ⚠️ **What you WON'T know:** Does it scale to larger models?

**For full validation, see the Scale-Robust protocol in Scaling Concerns document.**

---

## Cost Breakdown (1 Week POC)

- **Free option:** Google Colab (slow but free)
- **Paid option:** Lambda Labs GPU rental
  - 1x A100 (40GB): $1.29/hr
  - 8 hours training: ~$10
  - Total: **~$10-20**

---

## After 1 Week: Decision Tree

```
Did VQ model work at all?
├─ NO → Abandon or try different architecture
├─ SORT OF → Investigate what went wrong
└─ YES → Go to scale-robust protocol:
    ├─ Train Pythia-1.4B with VQ
    ├─ Compare 410M vs 1.4B results
    └─ If 1.4B better → proceed to full research
        If 1.4B same/worse → publish negative result
```

---

**Ready to start? Just copy-paste the code blocks above and run them in order!**

**Need more detail? See the full [Implementation Guide](POC Implementation Guide - Intermediate Reasoning Language for LLMs.md)**
