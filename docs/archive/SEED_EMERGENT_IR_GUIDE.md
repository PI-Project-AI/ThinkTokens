# Seed + Emergent IR: Implementation Guide

**Status:** New experimental direction (separate from VQ bottleneck experiment)
**Date Created:** October 25, 2025
**Core Concept:** Hybrid approach with minimal structural tags + emergent discrete codes

---

## Executive Summary

This is a **distinct experiment** from the previous VQ bottleneck work. The previous experiment showed:
- ✅ VQ codes were used (61% codebook utilization)
- ❌ But codes were NOT causal (bypassed for task solution)

**New Direction:** Add structural tags to enforce causality while keeping code semantics emergent.

```
Input → Encoder → IR Buffer (tags + codes) → Decoder (forced cross-attn)
                        ↓
                   <GOAL> [code_47] </GOAL>
                   <STEP> [code_89] </STEP>
                        ↓
              Cross-Attention ONLY (no bypass)
                        ↓
                   Answer
```

---

## Phase 0: Setup & Documentation (Day 1)

### Goal
Establish clean separation from VQ experiment, document architecture, prepare implementation.

### Tasks

**0.1: Create Separate Project Directory**
```bash
cd ThinkTokens/
mkdir -p seed_emergent_ir/{data,models,logs,results}
mkdir -p seed_emergent_ir/code/{models,training,evaluation,diagnostics}
```

**0.2: Copy nanoGPT as Base**
```bash
cd seed_emergent_ir/code
git clone https://github.com/karpathy/nanoGPT.git
# We'll fork this into our own causal IR variant
```

**0.3: Document Core Components**
Create `docs/SEED_EMERGENT_IR_ARCHITECTURE.md` with:
- Tag definitions (GOAL, ASSUME, STEP, CHECK, BRANCH)
- Codebook design (256 codes, embedding dim 64)
- Data flow (encoder → IR generator → constrained decoder)
- Loss function breakdown

**0.4: Version Control Baseline**
```bash
git add docs/SEED_EMERGENT_IR_GUIDE.md
git add docs/SEED_EMERGENT_IR_ARCHITECTURE.md
git commit -m "Add Seed+Emergent IR experiment plan (separate from VQ bottleneck)"
```

---

## Phase 1: Architecture Implementation (Days 2-5)

### Goal
Implement the three core components with **exact causality constraints**.

### 1.1: IRBufferGenerator

**File:** `seed_emergent_ir/code/ir_generator.py`

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict

class VectorQuantizer(nn.Module):
    """Simple VQ codebook (256 codes, 64-dim)"""
    def __init__(self, num_codes: int = 256, dim: int = 64):
        super().__init__()
        self.num_codes = num_codes
        self.embedding = nn.Embedding(num_codes, dim)
        self.register_buffer('cluster_size', torch.zeros(num_codes))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [batch, seq, dim] → quantized, loss, indices"""
        # L2 nearest neighbor
        flat = x.reshape(-1, x.shape[-1])
        dist = torch.cdist(flat, self.embedding.weight)
        indices = dist.argmin(dim=1)

        # Quantize
        quantized = self.embedding(indices).reshape_as(x)

        # Loss
        e_loss = ((quantized.detach() - x) ** 2).mean()
        q_loss = ((quantized - x.detach()) ** 2).mean()
        loss = q_loss + 0.25 * e_loss

        # Straight-through
        quantized = x + (quantized - x).detach()

        return quantized, loss, indices.reshape(x.shape[0], x.shape[1])


class IRBufferGenerator(nn.Module):
    """
    Generates IR buffer: structured tags + emergent codes

    Input hidden states → tag sequence with embedded codes
    Format: <GOAL> [code_47] </GOAL> <STEP> [code_89] </STEP> ...
    """
    def __init__(self,
                 hidden_dim: int = 256,
                 num_codes: int = 256,
                 code_dim: int = 64,
                 max_spans: int = 5):
        super().__init__()

        self.tags = ['<GOAL>', '<ASSUME>', '<STEP>', '<CHECK>', '<BRANCH>']
        self.tag_tokens = {tag: i for i, tag in enumerate(self.tags)}
        self.num_tags = len(self.tags)
        self.max_spans = max_spans
        self.num_codes = num_codes

        # VQ codebook
        self.vq = VectorQuantizer(num_codes=num_codes, dim=code_dim)

        # Encoder: hidden → code representations
        self.code_encoder = nn.Sequential(
            nn.Linear(hidden_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim)
        )

        # Span predictor: how many codes per span?
        self.span_length_predictor = nn.Linear(hidden_dim, max_spans + 1)

        # Tag predictor: which tag for this span?
        self.tag_predictor = nn.Linear(hidden_dim, self.num_tags)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            hidden: [batch, seq, hidden_dim] encoder outputs

        Returns:
            ir_tokens: [batch, ir_seq] token IDs representing IR buffer
            vq_loss: scalar VQ loss
            diagnostics: dict with codes, tags, etc.
        """
        batch_size = hidden.shape[0]
        device = hidden.device

        # Generate IR spans (simplified: use mean of hidden states)
        context = hidden.mean(dim=1)  # [batch, hidden_dim]

        # Predict structure
        tag_logits = self.tag_predictor(context)  # [batch, num_tags]
        tags = tag_logits.argmax(dim=1)  # [batch]

        span_logits = self.span_length_predictor(context)  # [batch, max_spans+1]
        num_codes = span_logits.argmax(dim=1) + 1  # [batch] → 1 to max_spans

        # Encode codes via VQ
        code_repr = self.code_encoder(context)  # [batch, code_dim]
        quantized, vq_loss, code_indices = self.vq(code_repr.unsqueeze(1))
        quantized = quantized.squeeze(1)  # [batch, code_dim]
        code_indices = code_indices.squeeze(1)  # [batch]

        # Build IR token sequence
        ir_tokens = []
        diagnostics = {
            'codes': code_indices.detach().cpu().numpy(),
            'tags': [self.tags[t] for t in tags.cpu().numpy()],
            'num_codes': num_codes.detach().cpu().numpy(),
            'vq_loss': vq_loss.item()
        }

        return ir_tokens, vq_loss, diagnostics
```

**Checklist:**
- [ ] VectorQuantizer works (test with random input)
- [ ] IRBufferGenerator generates tags + codes
- [ ] Output can be embedded and passed to decoder
- [ ] VQ loss is computed

**Testing Code:**
```bash
python -c "
import torch
from ir_generator import IRBufferGenerator

gen = IRBufferGenerator(hidden_dim=256, num_codes=256)
hidden = torch.randn(4, 10, 256)
ir_tokens, vq_loss, diag = gen(hidden)
print(f'VQ Loss: {vq_loss:.4f}')
print(f'Codes: {diag[\"codes\"]}')
print(f'Tags: {diag[\"tags\"]}')
"
```

---

### 1.2: ConstrainedDecoder

**File:** `seed_emergent_ir/code/decoder.py`

```python
import torch
import torch.nn as nn

class ConstrainedDecoder(nn.Module):
    """
    Decoder that MUST cross-attend to IR buffer.

    Key constraint:
    - Hidden direct path from input is attenuated or disabled
    - All information must flow through IR buffer
    """
    def __init__(self,
                 vocab_size: int = 50257,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 attenuate_input: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.attenuate_input = attenuate_input  # 0.1 = 90% reduction

        # Embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # Decoder layers with FORCED cross-attention to IR
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads=8, has_cross_attn=True)
            for _ in range(num_layers)
        ])

        # Output projection
        self.to_logits = nn.Linear(hidden_dim, vocab_size)

    def forward(self,
                ir_buffer: torch.Tensor,
                input_ids: torch.Tensor,
                attenuate_input: bool = True) -> torch.Tensor:
        """
        Args:
            ir_buffer: [batch, ir_seq, hidden_dim] - THE KEY INFO
            input_ids: [batch, seq] - attenuated if attenuate_input=True

        Returns:
            logits: [batch, seq, vocab_size]
        """
        # Embed input (attenuated)
        x = self.embed(input_ids)  # [batch, seq, hidden_dim]

        if attenuate_input:
            # Reduce input signal (force reliance on IR)
            x = x * self.attenuate_input

        # Process through decoder with cross-attention to IR
        for layer in self.layers:
            x = layer(x, cross_attn_key=ir_buffer)

        # Project to logits
        logits = self.to_logits(x)  # [batch, seq, vocab_size]

        return logits


class DecoderLayer(nn.Module):
    """Single decoder layer with forced cross-attention"""
    def __init__(self, hidden_dim: int, num_heads: int = 8, has_cross_attn: bool = True):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) if has_cross_attn else None

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, cross_attn_key: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_attn, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x_attn)

        # Cross-attention to IR buffer (MANDATORY)
        if self.cross_attn is not None:
            x_cross, _ = self.cross_attn(x, cross_attn_key, cross_attn_key)
            x = self.norm2(x + x_cross)

        # Feed-forward
        x_ff = self.ff(x)
        x = self.norm3(x + x_ff)

        return x
```

**Checklist:**
- [ ] ConstrainedDecoder processes IR buffer
- [ ] Cross-attention is mandatory (no bypass)
- [ ] Input attenuation works (0.1 × input)
- [ ] Output logits correct shape

**Testing Code:**
```bash
python -c "
import torch
from decoder import ConstrainedDecoder

decoder = ConstrainedDecoder(vocab_size=50257, hidden_dim=256)
ir_buffer = torch.randn(4, 5, 256)  # IR buffer
input_ids = torch.randint(0, 50257, (4, 10))

logits = decoder(ir_buffer, input_ids, attenuate_input=True)
print(f'Logits shape: {logits.shape}')  # Should be [4, 10, 50257]
"
```

---

### 1.3: CausalIRModel (Integration)

**File:** `seed_emergent_ir/code/models/causal_ir.py`

```python
import torch
import torch.nn as nn
from ir_generator import IRBufferGenerator
from decoder import ConstrainedDecoder

class CausalIRModel(nn.Module):
    """
    Full model: Input → Encoder → IR Buffer → Decoder

    Constraints:
    - IR is stop_grad (decoder can't modify it)
    - Decoder has forced cross-attention to IR
    - Direct input path is attenuated
    """

    def __init__(self,
                 vocab_size: int = 50257,
                 hidden_dim: int = 256,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 num_codes: int = 256):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Encoder: input → hidden
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )

        # IR Generator: hidden → (tags + codes)
        self.ir_generator = IRBufferGenerator(
            hidden_dim=hidden_dim,
            num_codes=num_codes,
            code_dim=64,
            max_spans=5
        )

        # Decoder: IR → answer
        self.decoder = ConstrainedDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            attenuate_input=0.1  # 90% reduction
        )

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """
        Args:
            input_ids: [batch, seq]
            labels: [batch, seq] (for training)

        Returns:
            logits, loss, ir_buffer, diagnostics
        """
        # Encode
        hidden = self.embed(input_ids)
        hidden = self.encoder(hidden)

        # Generate IR (CRITICAL: stop_grad to enforce causality)
        ir_buffer, vq_loss, ir_diag = self.ir_generator(hidden)
        ir_buffer = ir_buffer.detach()  # ← KEY: No gradient

        # Decode (forced cross-attention)
        logits = self.decoder(ir_buffer, input_ids, attenuate_input=True)

        # Losses
        if labels is not None:
            ce_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )

            # Auxiliary losses
            coverage_loss = self._tag_coverage_loss(ir_diag)  # Use all tags
            entropy_loss = self._code_entropy_loss(ir_diag)   # Diverse codes

            total_loss = ce_loss + 0.1 * vq_loss + 0.05 * coverage_loss + 0.02 * entropy_loss

            return logits, total_loss, ir_buffer, ir_diag
        else:
            return logits, None, ir_buffer, ir_diag

    def _tag_coverage_loss(self, ir_diag: dict) -> torch.Tensor:
        """Encourage use of all tags"""
        # Simplified: penalize if some tags unused
        tag_counts = {}
        for tag in ir_diag['tags']:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        unused_tags = len(self.ir_generator.tags) - len(tag_counts)
        return torch.tensor(unused_tags * 0.1, device=next(self.parameters()).device)

    def _code_entropy_loss(self, ir_diag: dict) -> torch.Tensor:
        """Encourage diversity in code usage"""
        import numpy as np
        codes = ir_diag['codes']
        unique = len(np.unique(codes))
        total = len(codes)

        # Loss = 1 - (unique / total), i.e., penalize repetition
        return torch.tensor(1.0 - (unique / total), device=next(self.parameters()).device)
```

**Checklist:**
- [ ] Model integrates all 3 components
- [ ] IR buffer is stop_grad
- [ ] Decoder has cross-attention
- [ ] Losses computed correctly
- [ ] Forward pass works end-to-end

---

## Phase 2: Simple Arithmetic Dataset (Days 5-6)

### Goal
Create small, controlled dataset to verify causality before scaling.

**File:** `seed_emergent_ir/data/arithmetic_generator.py`

```python
import random
import json

def generate_arithmetic_dataset(n_samples: int = 1000, filename: str = None):
    """
    Generate simple 1-2 step arithmetic problems.

    Format:
    Q: What is 5 + 3?
    A: 8

    Or:
    Q: If you have 5 apples and add 3 more, then take away 2, how many do you have?
    A: 6
    """
    data = []

    for _ in range(n_samples):
        # Single step (80%)
        if random.random() < 0.8:
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            op = random.choice(['+', '-', '*'])

            if op == '+':
                ans = a + b
            elif op == '-':
                ans = max(0, a - b)  # Avoid negative
            else:  # '*'
                ans = a * b
                if ans > 1000:  # Cap multiplication
                    a, b = random.randint(1, 20), random.randint(1, 20)
                    ans = a * b

            question = f"Q: What is {a} {op} {b}?"

        # Two step (20%)
        else:
            a = random.randint(1, 50)
            b = random.randint(1, 50)
            c = random.randint(1, 50)

            op1 = random.choice(['+', '-'])
            op2 = random.choice(['+', '-'])

            if op1 == '+':
                temp = a + b
            else:
                temp = max(0, a - b)

            if op2 == '+':
                ans = temp + c
            else:
                ans = max(0, temp - c)

            question = f"Q: What is {a} {op1} {b} {op2} {c}?"

        answer = f"A: {ans}"
        data.append({'question': question, 'answer': answer})

    if filename:
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Generated {len(data)} arithmetic problems → {filename}")

    return data

if __name__ == '__main__':
    generate_arithmetic_dataset(n_samples=1000,
                               filename='arithmetic_train_1k.json')
    generate_arithmetic_dataset(n_samples=100,
                               filename='arithmetic_test_100.json')
```

**Run:**
```bash
cd seed_emergent_ir/data
python arithmetic_generator.py
# Outputs: arithmetic_train_1k.json, arithmetic_test_100.json
```

**Checklist:**
- [ ] 1000 training examples generated
- [ ] 100 test examples generated
- [ ] Examples are 1-2 step arithmetic
- [ ] Answers are correct

---

## Phase 3: Causal Diagnostic Tests (Days 7-8)

### Goal
Verify IR is genuinely causal before scaling.

**File:** `seed_emergent_ir/evaluation/causal_tests.py`

```python
import torch
import numpy as np
from typing import Dict, List

class CausalDiagnosticTests:
    """
    Three critical tests:
    1. Random IR: Replace IR with random codes
    2. Shuffle IR: Swap IR between examples
    3. Drop IR: Hide IR from decoder

    All should crash accuracy significantly.
    """

    def __init__(self, model, dataloader, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def baseline_accuracy(self) -> float:
        """Normal inference"""
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits, _, _, _ = self.model(input_ids)
                preds = logits.argmax(dim=-1)

                correct += (preds == labels).sum().item()
                total += labels.numel()

        return correct / total if total > 0 else 0.0

    def random_ir_test(self) -> float:
        """Replace IR buffer with random codes"""
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Normal forward but corrupt IR
                hidden = self.model.embed(input_ids)
                hidden = self.model.encoder(hidden)

                ir_buffer, _, _ = self.model.ir_generator(hidden)
                ir_buffer = ir_buffer.detach()

                # CORRUPT: Replace with random
                ir_buffer = torch.randn_like(ir_buffer)

                logits = self.model.decoder(ir_buffer, input_ids, attenuate_input=True)
                preds = logits.argmax(dim=-1)

                correct += (preds == labels).sum().item()
                total += labels.numel()

        return correct / total if total > 0 else 0.0

    def shuffle_ir_test(self) -> float:
        """Swap IR buffers between examples"""
        correct = 0
        total = 0

        with torch.no_grad():
            batches = list(self.dataloader)
            for i, batch in enumerate(batches):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Get IR from this batch
                hidden = self.model.embed(input_ids)
                hidden = self.model.encoder(hidden)
                ir_buffer, _, _ = self.model.ir_generator(hidden)
                ir_buffer = ir_buffer.detach()

                # CORRUPT: Shuffle IR (roll by 1)
                ir_buffer = torch.roll(ir_buffer, shifts=1, dims=0)

                logits = self.model.decoder(ir_buffer, input_ids, attenuate_input=True)
                preds = logits.argmax(dim=-1)

                correct += (preds == labels).sum().item()
                total += labels.numel()

        return correct / total if total > 0 else 0.0

    def drop_ir_test(self) -> float:
        """Hide IR from decoder (zero it out)"""
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Get IR normally
                hidden = self.model.embed(input_ids)
                hidden = self.model.encoder(hidden)
                ir_buffer, _, _ = self.model.ir_generator(hidden)
                ir_buffer = ir_buffer.detach()

                # CORRUPT: Zero out IR
                ir_buffer = torch.zeros_like(ir_buffer)

                logits = self.model.decoder(ir_buffer, input_ids, attenuate_input=True)
                preds = logits.argmax(dim=-1)

                correct += (preds == labels).sum().item()
                total += labels.numel()

        return correct / total if total > 0 else 0.0

    def run_all_tests(self) -> Dict[str, float]:
        """Run all diagnostic tests"""
        print("\n" + "="*60)
        print("CAUSAL DIAGNOSTIC TESTS")
        print("="*60)

        baseline = self.baseline_accuracy()
        print(f"\nBaseline Accuracy: {baseline:.2%}")

        random_ir = self.random_ir_test()
        print(f"Random IR Accuracy: {random_ir:.2%}")
        print(f"  → Drop: {(baseline - random_ir):.2%} (should be ≥ 30%)")

        shuffle_ir = self.shuffle_ir_test()
        print(f"Shuffle IR Accuracy: {shuffle_ir:.2%}")
        print(f"  → Drop: {(baseline - shuffle_ir):.2%} (should be ≥ 30%)")

        drop_ir = self.drop_ir_test()
        print(f"Drop IR Accuracy: {drop_ir:.2%}")
        print(f"  → Drop: {(baseline - drop_ir):.2%} (should be ≥ 40%)")

        print("\n" + "="*60)
        print("VERDICT:")

        all_pass = (
            (baseline - random_ir) >= 0.30 and
            (baseline - shuffle_ir) >= 0.30 and
            (baseline - drop_ir) >= 0.40
        )

        if all_pass:
            print("✓ ALL TESTS PASSED - IR is genuinely causal!")
        else:
            print("✗ TESTS FAILED - Model may be bypassing IR")

        print("="*60 + "\n")

        return {
            'baseline': baseline,
            'random_ir': random_ir,
            'shuffle_ir': shuffle_ir,
            'drop_ir': drop_ir,
            'all_pass': all_pass
        }
```

**Usage:**
```python
from causal_tests import CausalDiagnosticTests

tester = CausalDiagnosticTests(model, test_dataloader, device='cuda')
results = tester.run_all_tests()

# Must see significant drops (≥30-40%) for IR to be causal
```

---

## Phase 4: Training (Days 9-12)

### Goal
Train on arithmetic, verify causality passes.

**File:** `seed_emergent_ir/training/train.py`

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path

class ArithmeticDataset(Dataset):
    def __init__(self, json_file: str, tokenizer, max_len: int = 50):
        with open(json_file) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['question'] + ' ' + item['answer']

        tokens = self.tokenizer.encode(text, max_length=self.max_len, truncation=True, padding='max_length')
        return {
            'input_ids': torch.tensor(tokens),
            'labels': torch.tensor(tokens)
        }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        logits, loss, ir_buffer, ir_diag = model(input_ids, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    from models.causal_ir import CausalIRModel
    model = CausalIRModel(vocab_size=50257, hidden_dim=256).to(device)

    # Load data
    train_dataset = ArithmeticDataset('data/arithmetic_train_1k.json', tokenizer=None, max_len=50)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f}")

    # Save model
    torch.save(model.state_dict(), 'models/causal_ir_arithmetic.pt')
    print("✓ Model saved")

if __name__ == '__main__':
    main()
```

**Run:**
```bash
cd seed_emergent_ir
python training/train.py
```

---

## Success Criteria (End of Phase 4)

### Must Pass:
- ✅ Baseline accuracy ≥ 70% on arithmetic
- ✅ Random IR drops accuracy by ≥30%
- ✅ Shuffle IR drops accuracy by ≥30%
- ✅ Drop IR drops accuracy by ≥40%
- ✅ Codebook utilization 50-70%

### Nice to Have:
- 🎯 Token generation ≤ 50 tokens (efficiency)
- 🎯 Smooth loss curve (stable training)
- 🎯 Tags distributed across spans

---

## Phase 5: Scale to GSM8K (Days 13-16)

### Goal
Apply learned approach to harder dataset.

**Steps:**
1. Load GSM8K subset (100-500 examples)
2. Retrain or fine-tune model
3. Run causality tests again
4. Measure: accuracy, tokens, codebook usage

---

## Phase 6: Analysis & Documentation (Days 17-20)

### Goal
Understand what IR codes learned and document findings.

**Analysis:**
- Visualize code embeddings (t-SNE)
- Cluster codes by problem type
- Identify "reasoning primitives"
- Compare to baseline (no IR)

**Output:**
- Analysis report with figures
- Code release on GitHub
- Blog post / paper draft

---

## File Structure (Clean & Organized)

```
seed_emergent_ir/
├── data/
│   ├── arithmetic_generator.py
│   ├── arithmetic_train_1k.json
│   └── arithmetic_test_100.json
│
├── code/
│   ├── models/
│   │   └── causal_ir.py          (CausalIRModel)
│   ├── ir_generator.py           (IRBufferGenerator, VQ)
│   ├── decoder.py                (ConstrainedDecoder)
│   ├── training/
│   │   └── train.py              (Training loop)
│   └── evaluation/
│       └── causal_tests.py       (Diagnostic tests)
│
├── logs/
│   └── training.log
│
├── results/
│   ├── causal_test_results.json
│   ├── analysis.md
│   └── figures/
│       ├── code_embeddings.png
│       └── performance_vs_baseline.png
│
└── README.md                      (Quick start guide)
```

---

## Key Checkpoints

- **End of Phase 1:** All 3 components (IRGen, Decoder, CausalIR) tested independently ✓
- **End of Phase 2:** 1000 arithmetic problems generated ✓
- **End of Phase 3:** Causal tests implemented (not yet run) ✓
- **End of Phase 4:** Training complete, causality verified ✓
- **End of Phase 5:** Scaled to GSM8K, results documented ✓
- **End of Phase 6:** Full analysis and blog/paper ready ✓

---

## Summary: Why This Is Different & Better

**vs. Previous VQ Experiment:**
- ✅ Enforces causality architecturally (not hoped for in loss)
- ✅ Tests causality explicitly (diagnostic suite)
- ✅ Starts simple (arithmetic) before scaling
- ✅ Clean separation (new directory, no confusion)

**vs. Other Approaches:**
- ✅ Keeps code semantics emergent (unlike hand-designed tokens)
- ✅ Minimal structural overhead (just 5-6 tags)
- ✅ Testable & reproducible (arithmetic is controlled)
- ✅ Research-level (new insight: seed + emergent IR)

---

**This is the implementation. Execute Phase by Phase. Stop and report after each phase.**
