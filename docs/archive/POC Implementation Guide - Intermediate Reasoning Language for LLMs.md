# Proof of Concept Implementation Guide: Intermediate Reasoning Language for LLMs

**Author:** Claude Code (with Paul PROVOST)
**Date:** October 21, 2025
**Status:** Implementation-Ready Guide

---

## Table of Contents

1. [Executive Summary & Critical Analysis](#1-executive-summary--critical-analysis)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Complete Technology Stack](#3-complete-technology-stack)
4. [Detailed Architecture Design](#4-detailed-architecture-design)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Code Implementation Guide](#6-code-implementation-guide)
7. [Training Procedures](#7-training-procedures)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Analysis & Visualization Tools](#9-analysis--visualization-tools)
10. [Risk Mitigation & Debugging](#10-risk-mitigation--debugging)
11. [Resource Requirements](#11-resource-requirements)
12. [Future Extensions](#12-future-extensions)

---

## 1. Executive Summary & Critical Analysis

### 1.1 The Core Hypothesis

**Claim:** LLMs can develop more efficient reasoning by operating in a learned discrete intermediate representation (IR) rather than natural language chain-of-thought (CoT).

**What Makes This Interesting:**
- Current CoT is fundamentally a **communication protocol**, not a computation substrate
- Hidden states already encode non-linguistic structure - why force verbalization?
- Discrete codes could enable compositionality without linguistic overhead

### 1.2 Critical Analysis: Strengths & Concerns

#### Strengths

1. **Efficiency Hypothesis is Well-Motivated**
   - Natural language CoT demonstrably wastes tokens on linguistic scaffolding
   - Math reasoning in particular doesn't need articles, prepositions, etc.
   - Compression could yield 3-10x token reduction

2. **Emergent Discovery is Elegant**
   - Avoiding hand-crafted reasoning languages is wise
   - VQ-VAE framework has proven track record (DALL-E, Muse)
   - Learned abstractions may discover non-obvious reasoning patterns

3. **Tractable Starting Point**
   - Small models (70M-1.1B) are experimentally feasible
   - GSM8K/SVAMP are well-studied benchmarks
   - Can start with supervised learning before RL complexity

#### Concerns & Open Questions

1. **Codebook Collapse Risk**
   - VQ-VAE codebooks notoriously underutilize capacity
   - In vision, only ~20-30% of codes get used despite hundreds available
   - **Mitigation needed:** Commitment loss, exponential moving average updates, restart mechanisms

2. **Interpretability Paradox**
   - Goal is efficiency, but need interpretability to validate reasoning quality
   - How do we know IR codes represent "reasoning" vs. task-specific compression?
   - **Challenge:** Distinguishing genuine abstraction from memorization shortcuts

3. **Generalization Uncertainty**
   - Codes learned on GSM8K may not transfer to ARC or other domains
   - Natural language's generality comes from shared linguistic priors
   - **Question:** Will we need different codebooks per domain?

4. **Training Stability**
   - Discrete bottlenecks create non-differentiable barriers
   - Straight-through estimators can be unstable
   - **Risk:** Model might route around the bottleneck if not carefully regularized

5. **The "Alignment Tax" Problem**
   - If humans can't understand IR, how do we:
     - Debug reasoning errors?
     - Provide oversight for safety?
     - Correct biased reasoning patterns?
   - This echoes broader AI interpretability challenges

### 1.3 My Assessment

**This is worth pursuing as a research experiment**, but with measured expectations:

✅ **Strong PoC Potential:**
- Small-scale implementation is feasible (~1-2 weeks for proficient researcher)
- Token efficiency gains are likely measurable
- Even negative results would be informative

⚠️ **Realistic Outcome Predictions:**
- **Best case:** 40-60% token reduction with maintained accuracy, clear emergent structure
- **Expected case:** 20-30% reduction, some accuracy trade-off, partially interpretable codes
- **Worst case:** Codebook collapse, no efficiency gain, or accuracy degradation

**The real scientific value** lies not in beating state-of-the-art, but in understanding:
1. What level of abstraction emerges naturally?
2. How domain-specific vs. general are the learned codes?
3. What's the Pareto frontier between efficiency and interpretability?

### 1.4 Updated Recommendation

**Start with the VQ bottleneck approach**, but add three critical components from day one:

1. **Codebook usage monitoring** - Track which codes activate, kill/restart dead codes
2. **Probe classifiers** - Train lightweight classifiers to predict reasoning properties from IR
3. **Ablation suite** - Test bottleneck placement, codebook size, commitment strength

If initial results show promise (>10% token reduction, >50% codebook usage, maintained accuracy), then:
- Scale to larger models (1B-7B range)
- Explore cross-task transfer
- Investigate the modular pipeline architecture

---

## 2. Theoretical Foundation

### 2.1 Why Natural Language CoT is Suboptimal

**Token Analysis of a Simple Reasoning Step:**

```
Natural Language CoT:
"To solve this problem, I need to first understand what's being asked.
The question wants me to find the total cost. Let me break this down step by step.
First, I'll calculate the cost per item..."

Tokens: ~45 tokens for setup
Information content: "calculate total cost" (~3 concepts)
Efficiency: ~15:1 overhead ratio
```

**Idealized IR Representation:**
```
[TASK_IDENTIFY] [DECOMPOSE] [CALC_SUM] [MULTIPLY]
Tokens: 4
Information content: Same 3-4 concepts
Efficiency: ~1:1 ratio
```

### 2.2 Hidden State Analysis (Why IR Makes Sense)

Recent research (Anthropic's "Scaling Monosemanticity", Neel Nanda's mechanistic interpretability) shows:

1. **Residual stream contains rich abstractions** beyond what's verbalized
2. **Attention heads specialize** in ways that don't align with word boundaries
3. **MLP layers compute non-linguistic features** (e.g., "entity-has-property" relations)

**Implication:** The model already "thinks" in abstractions. Natural language CoT is a projection into human-readable space, not the native computation.

### 2.3 The VQ-VAE Framework Applied to Reasoning

**Standard VQ-VAE (for images):**
```
Image → Encoder → Continuous latent → Quantize to codebook → Decoder → Reconstructed image
```

**Our Adaptation (for reasoning):**
```
Question → Transformer layers → Continuous reasoning state → Quantize to reasoning codes → Transformer layers → Answer
```

**Key Differences:**
- **No reconstruction loss** on the question (we don't need to recreate it)
- **Supervision on final answer** (standard language modeling loss)
- **Codebook learns reasoning primitives**, not visual patches

### 2.4 Mathematical Formulation

**Model Architecture:**

Let:
- `x` = input question tokens
- `h_pre` = hidden states before bottleneck
- `C = {c_1, c_2, ..., c_K}` = codebook of K reasoning vectors (e.g., K=512, dim=512)
- `z_q` = quantized representation
- `y` = output answer tokens

**Forward Pass:**

1. **Encode:** `h_pre = Transformer_encode(x)`
2. **Quantize:** `z_q = argmin_{c_k ∈ C} ||h_pre - c_k||²`
3. **Decode:** `y = Transformer_decode(z_q)`

**Loss Function:**

```
L_total = L_task + β·L_commit + γ·L_codebook + δ·L_diversity

Where:
L_task = CrossEntropy(y_pred, y_true)  # Standard LM loss
L_commit = ||sg[z_q] - h_pre||²         # Encoder commits to codes
L_codebook = ||z_q - sg[h_pre]||²       # Codebook updates toward encoder
L_diversity = -Σ p_k log(p_k)           # Entropy over code usage

sg[·] = stop_gradient (no backprop through this path)
β, γ, δ = hyperparameters (typically β=0.25, γ=1.0, δ=0.01)
```

**Gradient Flow:**

Straight-through estimator: `∇h_pre ≈ ∇z_q` (copy gradients as if quantization didn't exist)

---

## 3. Complete Technology Stack

### 3.1 Core Framework

**Recommended Stack:**

```yaml
Language: Python 3.10+
Deep Learning: PyTorch 2.0+ (for better compilation)
Transformers: Hugging Face Transformers 4.35+
Acceleration: Flash Attention 2 (optional but recommended)
Experiment Tracking: Weights & Biases (wandb)
Evaluation: lm-evaluation-harness (EleutherAI)
```

**Alternative (if resource-constrained):**
- JAX/Flax instead of PyTorch (better TPU support)
- TensorBoard instead of W&B (free, local)

### 3.2 Model Choices

**Baseline Models (Hugging Face):**

1. **EleutherAI/pythia-70m** (70M params)
   - Smallest, fastest iteration
   - Good for debugging architecture

2. **EleutherAI/pythia-160m** (160M params)
   - Sweet spot for initial experiments

3. **EleutherAI/pythia-410m** (410M params)
   - More capacity, still trainable on single GPU

4. **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (1.1B params)
   - Reference point for "what small models can do"
   - Already has some instruction-following ability

**Why Pythia?**
- Fully open training data and checkpoints
- Multiple sizes for scaling experiments
- Designed for research reproducibility
- No licensing ambiguity

### 3.3 Datasets

**Primary Task: Mathematical Reasoning**

1. **GSM8K** (Grade School Math)
   ```python
   from datasets import load_dataset
   gsm8k = load_dataset("gsm8k", "main")
   # 7,473 training + 1,319 test questions
   ```

2. **SVAMP** (Simple Variations on Arithmetic Math Problems)
   ```python
   svamp = load_dataset("ChilleD/SVAMP")
   # 1,000 problems, tests robustness to question phrasing
   ```

3. **MATH** (Competition Math, subset for difficulty levels 1-2)
   ```python
   math_dataset = load_dataset("hendrycks/competition_math")
   # Use Level 1-2 for small models
   ```

**Secondary Task: Logical Reasoning**

4. **ARC-Challenge** (Science Q&A)
   ```python
   arc = load_dataset("ai2_arc", "ARC-Challenge")
   # 2,590 questions requiring reasoning
   ```

**Dataset Sizes (Training Efficiency):**
- GSM8K: ~7K examples → 1 epoch ≈ 30-60 min on V100
- Combined: ~10K examples → Full training in <6 hours

### 3.4 Compute Requirements

**Minimum Setup:**
- 1x GPU with 16GB VRAM (RTX 4090, V100, T4)
- 32GB system RAM
- ~100GB disk space

**Recommended Setup:**
- 1x GPU with 24GB+ VRAM (A5000, 3090, A100)
- 64GB system RAM
- 500GB SSD

**Training Time Estimates (Pythia-160M on GSM8K):**
- Baseline: 2-3 hours
- With VQ bottleneck: 3-4 hours (10-20% slower due to quantization)
- Full experiment suite (5 runs, 3 codebook sizes): ~15-20 hours

**Free Options:**
- Google Colab Pro (~$10/month, A100 access)
- Kaggle Notebooks (30hr/week free GPU)
- Lightning.ai (limited free tier)

### 3.5 Software Dependencies

**requirements.txt:**
```txt
# Core
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0

# VQ-VAE specific
vector-quantize-pytorch>=1.12.0  # Lucidrains' excellent implementation

# Evaluation
lm-eval>=0.4.0
scikit-learn>=1.3.0

# Experiment tracking
wandb>=0.16.0

# Utilities
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
tqdm>=4.65.0

# Optional but recommended
flash-attn>=2.0.0  # Requires CUDA compilation
einops>=0.7.0
```

---

## 4. Detailed Architecture Design

### 4.1 Baseline Transformer (No Bottleneck)

**Standard autoregressive LM:**

```
Input tokens → Embedding
            ↓
     [Transformer Layer 1]
     [Transformer Layer 2]
          ...
     [Transformer Layer N]
            ↓
      LM Head → Output logits
```

**This is our baseline** - we'll measure token efficiency and accuracy here.

### 4.2 VQ-Bottleneck Architecture (Option A)

**Insertion point:** Between layers 2/3 and 4/5 (depending on model depth)

```
Input tokens → Embedding
            ↓
     [Transformer Layer 1]   ← Encode input
     [Transformer Layer 2]
            ↓
     [VQ Bottleneck]         ← Discrete reasoning codes
            ↓
     [Transformer Layer 3]   ← Decode to answer
     [Transformer Layer 4]
            ↓
      LM Head → Output logits
```

**Why mid-layer insertion?**
- Early layers = token-level features (syntax, entity recognition)
- Mid layers = abstract reasoning (what we want to compress)
- Late layers = answer generation (should stay flexible)

### 4.3 VQ Bottleneck Details

**Components:**

1. **Projection Layer** (optional, if hidden_dim ≠ codebook_dim)
   ```python
   h_pre = linear_project(transformer_hidden)  # [batch, seq, hidden] → [batch, seq, code_dim]
   ```

2. **Codebook**
   ```python
   codebook = nn.Embedding(num_codes, code_dim)  # e.g., [512, 512]
   # Each code is a learned vector representing a reasoning primitive
   ```

3. **Quantization**
   ```python
   # Find nearest code for each position in sequence
   distances = torch.cdist(h_pre, codebook.weight)  # [batch, seq, num_codes]
   code_indices = distances.argmin(dim=-1)          # [batch, seq]
   z_q = codebook(code_indices)                     # [batch, seq, code_dim]
   ```

4. **Straight-Through Estimator**
   ```python
   z_q = h_pre + (z_q - h_pre).detach()  # Forward: use z_q, Backward: gradient flows to h_pre
   ```

### 4.4 Hyperparameter Design Space

**Critical Choices:**

| Parameter | Options | Recommendation | Rationale |
|-----------|---------|----------------|-----------|
| `num_codes` | 128, 256, 512, 1024 | **512** | Balance capacity vs. sparsity |
| `code_dim` | Same as hidden_dim, or 256/512 | **Match hidden_dim** | Avoid projection bottleneck |
| `bottleneck_layer` | Layer 2, 3, or mid-point | **Mid-point** (layer N/2) | Maximize abstraction |
| `num_reasoning_tokens` | 4, 8, 16 | **8** | Enough for multi-step reasoning |
| `commitment_weight` β | 0.1, 0.25, 0.5 | **0.25** | Standard VQ-VAE value |
| `codebook_weight` γ | 0.5, 1.0 | **1.0** | Standard |
| `diversity_weight` δ | 0.0, 0.01, 0.1 | **0.01** | Mild regularization |

**Ablation Matrix (for experiments):**

Test 3 values each for:
- `num_codes`: {256, 512, 1024}
- `num_reasoning_tokens`: {4, 8, 16}
- `bottleneck_layer`: {N/3, N/2, 2N/3}

Total: 27 configurations (prioritize with budget)

### 4.5 Alternative: Gumbel-Softmax (Continuous Relaxation)

**Instead of hard quantization:**

```python
# Temperature-controlled soft assignment
logits = -distances / temperature  # [batch, seq, num_codes]
soft_weights = F.gumbel_softmax(logits, tau=temperature, hard=False)
z_soft = soft_weights @ codebook.weight  # Weighted sum of codes
```

**Pros:**
- Fully differentiable (no straight-through estimator)
- Easier to train initially

**Cons:**
- Not truly discrete (can't count distinct codes used)
- Temperature scheduling adds complexity

**Recommendation:** Start with VQ-VAE (cleaner discrete structure), fall back to Gumbel if training fails.

---

## 5. Implementation Roadmap

### Phase 1: Setup & Baselines (Week 1, Days 1-2)

**Tasks:**
1. Environment setup
2. Download models and datasets
3. Run baseline evaluations
4. Establish metrics tracking

**Deliverables:**
- Baseline accuracy on GSM8K for Pythia 70M/160M/410M
- Token usage statistics
- Inference latency benchmarks

### Phase 2: VQ Architecture (Week 1, Days 3-4)

**Tasks:**
1. Implement VQ bottleneck module
2. Insert into Pythia models
3. Unit tests for quantization
4. Verify gradient flow

**Deliverables:**
- Working VQ-augmented model
- Gradient norm comparisons (with/without bottleneck)
- Codebook usage visualization

### Phase 3: Training (Week 1, Days 5-7)

**Tasks:**
1. Train VQ model on GSM8K (3 configurations)
2. Monitor codebook usage
3. Implement dead code restart
4. Hyperparameter sweep (if time permits)

**Deliverables:**
- Trained checkpoints
- Training curves (loss, codebook usage, accuracy)
- Preliminary results table

### Phase 4: Evaluation (Week 2, Days 1-2)

**Tasks:**
1. Accuracy evaluation on GSM8K test
2. Token efficiency analysis
3. Cross-task transfer (SVAMP, ARC)
4. Ablation studies

**Deliverables:**
- Complete results table
- Token reduction percentages
- Failure case analysis

### Phase 5: Analysis (Week 2, Days 3-5)

**Tasks:**
1. Codebook clustering analysis
2. Probe classifiers for code interpretability
3. Qualitative inspection of reasoning traces
4. Visualization dashboard

**Deliverables:**
- Code interpretation report
- t-SNE/UMAP visualizations
- Case studies (3-5 examples)

### Phase 6: Documentation (Week 2, Days 6-7)

**Tasks:**
1. Technical report
2. Blog post (optional)
3. Code cleanup and README
4. Reproducibility checklist

**Deliverables:**
- 6-10 page report
- Public GitHub repo (if desired)
- Presentation slides

---

## 6. Code Implementation Guide

### 6.1 Project Structure

```
reasoning-ir-poc/
├── configs/
│   ├── baseline.yaml
│   ├── vq_small.yaml
│   └── vq_medium.yaml
├── src/
│   ├── models/
│   │   ├── vq_bottleneck.py
│   │   ├── reasoning_transformer.py
│   │   └── utils.py
│   ├── data/
│   │   ├── gsm8k_loader.py
│   │   └── preprocessing.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── losses.py
│   └── evaluation/
│       ├── metrics.py
│       └── analysis.py
├── scripts/
│   ├── train_baseline.py
│   ├── train_vq.py
│   ├── evaluate.py
│   └── analyze_codes.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── requirements.txt
└── README.md
```

### 6.2 Core Implementation: VQ Bottleneck Module

**File: `src/models/vq_bottleneck.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class VectorQuantizer(nn.Module):
    """
    Vector Quantization bottleneck for reasoning compression.

    Based on VQ-VAE (van den Oord et al., 2017) with improvements:
    - Exponential moving average updates for codebook
    - Dead code restart mechanism
    - Usage tracking for analysis
    """

    def __init__(
        self,
        num_codes: int = 512,
        code_dim: int = 512,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        restart_threshold: int = 10,  # Restart codes unused for N batches
    ):
        super().__init__()

        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.epsilon = epsilon
        self.restart_threshold = restart_threshold

        # Codebook: learnable embeddings
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)

        # EMA tracking (for codebook updates)
        self.register_buffer('ema_cluster_size', torch.zeros(num_codes))
        self.register_buffer('ema_weight', torch.zeros(num_codes, code_dim))
        self.register_buffer('usage_count', torch.zeros(num_codes))

    def forward(
        self,
        z: torch.Tensor,  # [batch, seq_len, code_dim]
        update_codebook: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z: Continuous representations to quantize
            update_codebook: Whether to update codebook (False during eval)

        Returns:
            z_q: Quantized representations (same shape as z)
            loss: VQ loss (commitment + codebook)
            metrics: Dict with usage stats
        """
        batch_size, seq_len, _ = z.shape

        # Flatten for easier processing
        z_flat = z.reshape(-1, self.code_dim)  # [batch*seq, code_dim]

        # Compute distances to all codes
        distances = torch.cdist(
            z_flat,
            self.codebook.weight,
            p=2.0
        )  # [batch*seq, num_codes]

        # Find nearest codes
        code_indices = distances.argmin(dim=-1)  # [batch*seq]

        # Quantize
        z_q_flat = self.codebook(code_indices)  # [batch*seq, code_dim]
        z_q = z_q_flat.view(batch_size, seq_len, self.code_dim)

        # Commitment loss: encourage encoder to commit to codes
        commitment_loss = F.mse_loss(z_q.detach(), z)

        # Codebook loss: move codes toward encoder outputs
        codebook_loss = F.mse_loss(z_q, z.detach())

        # Total VQ loss
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator for gradients
        z_q = z + (z_q - z).detach()

        # Track usage
        if update_codebook:
            self._update_usage(code_indices)
            if self.training:
                self._update_ema(z_flat, code_indices)
                self._restart_dead_codes(z_flat)

        # Metrics
        unique_codes = len(torch.unique(code_indices))
        utilization = unique_codes / self.num_codes

        metrics = {
            'vq_loss': vq_loss.item(),
            'commitment_loss': commitment_loss.item(),
            'codebook_loss': codebook_loss.item(),
            'unique_codes': unique_codes,
            'codebook_utilization': utilization,
            'code_indices': code_indices.view(batch_size, seq_len),
        }

        return z_q, vq_loss, metrics

    def _update_usage(self, code_indices: torch.Tensor):
        """Track which codes are being used."""
        usage = torch.bincount(
            code_indices,
            minlength=self.num_codes
        ).float()
        self.usage_count += usage

    def _update_ema(self, z_flat: torch.Tensor, code_indices: torch.Tensor):
        """Exponential moving average update for codebook."""
        # Count how many vectors assigned to each code
        encodings = F.one_hot(code_indices, self.num_codes).float()  # [N, num_codes]

        # Update cluster sizes
        updated_size = encodings.sum(0)  # [num_codes]
        self.ema_cluster_size.mul_(self.decay).add_(
            updated_size, alpha=1 - self.decay
        )

        # Update code weights
        dw = encodings.t() @ z_flat  # [num_codes, code_dim]
        self.ema_weight.mul_(self.decay).add_(dw, alpha=1 - self.decay)

        # Normalize
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_codes * self.epsilon) * n
        )

        normalized_weight = self.ema_weight / cluster_size.unsqueeze(1)
        self.codebook.weight.data.copy_(normalized_weight)

    def _restart_dead_codes(self, z_flat: torch.Tensor):
        """Replace codes that haven't been used recently."""
        # Find dead codes (not used in last N updates)
        dead_mask = self.usage_count < self.restart_threshold
        num_dead = dead_mask.sum().item()

        if num_dead > 0:
            # Reinitialize dead codes with random encoder outputs
            random_indices = torch.randint(0, z_flat.size(0), (num_dead,))
            self.codebook.weight.data[dead_mask] = z_flat[random_indices]
            self.usage_count[dead_mask] = self.restart_threshold

            print(f"Restarted {num_dead} dead codes")

    def get_codebook_usage_stats(self) -> dict:
        """Return analysis of codebook usage."""
        total_usage = self.usage_count.sum()
        if total_usage == 0:
            return {'error': 'No usage data yet'}

        usage_dist = self.usage_count / total_usage

        return {
            'total_codes': self.num_codes,
            'codes_ever_used': (self.usage_count > 0).sum().item(),
            'usage_entropy': -(usage_dist * torch.log(usage_dist + 1e-10)).sum().item(),
            'top_10_codes': self.usage_count.topk(10).indices.tolist(),
            'top_10_usage': self.usage_count.topk(10).values.tolist(),
        }


class ReasoningBottleneck(nn.Module):
    """
    Wrapper that handles projection and reasoning token generation.

    This module:
    1. Projects hidden states to reasoning space (if needed)
    2. Generates fixed number of reasoning tokens
    3. Quantizes them
    4. Projects back to hidden space
    """

    def __init__(
        self,
        hidden_dim: int,
        num_reasoning_tokens: int = 8,
        num_codes: int = 512,
        code_dim: Optional[int] = None,
        **vq_kwargs
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_reasoning_tokens = num_reasoning_tokens
        code_dim = code_dim or hidden_dim

        # Projection layers (if hidden_dim != code_dim)
        if hidden_dim != code_dim:
            self.pre_projection = nn.Linear(hidden_dim, code_dim)
            self.post_projection = nn.Linear(code_dim, hidden_dim)
        else:
            self.pre_projection = nn.Identity()
            self.post_projection = nn.Identity()

        # Reasoning token generation (compress sequence to fixed size)
        self.reasoning_compress = nn.Linear(hidden_dim, num_reasoning_tokens * code_dim)

        # Vector quantizer
        self.vq = VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
            **vq_kwargs
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_dim]
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            hidden_states: Input hidden states from transformer
            attention_mask: Attention mask (1 for real tokens, 0 for padding)

        Returns:
            reasoning_codes: Quantized reasoning representations
            vq_loss: Vector quantization loss
            metrics: Usage statistics
        """
        batch_size = hidden_states.size(0)

        # Pool sequence into single vector (mean pooling over non-padding tokens)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask_expanded).sum(1) / mask_expanded.sum(1)
        else:
            pooled = hidden_states.mean(1)  # [batch, hidden_dim]

        # Generate reasoning tokens
        reasoning_flat = self.reasoning_compress(pooled)  # [batch, num_tokens * code_dim]
        reasoning_tokens = reasoning_flat.view(
            batch_size,
            self.num_reasoning_tokens,
            -1
        )  # [batch, num_tokens, code_dim]

        # Project if needed
        reasoning_tokens = self.pre_projection(reasoning_tokens)

        # Quantize
        reasoning_quantized, vq_loss, metrics = self.vq(
            reasoning_tokens,
            update_codebook=self.training
        )

        # Project back
        reasoning_quantized = self.post_projection(reasoning_quantized)

        return reasoning_quantized, vq_loss, metrics
```

### 6.3 Modified Transformer with Bottleneck

**File: `src/models/reasoning_transformer.py`**

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from .vq_bottleneck import ReasoningBottleneck
from typing import Optional, Tuple


class ReasoningTransformer(nn.Module):
    """
    Transformer with VQ reasoning bottleneck inserted at mid-layer.

    Architecture:
        Input → Transformer[:split] → VQ Bottleneck → Transformer[split:] → Output
    """

    def __init__(
        self,
        base_model_name: str = "EleutherAI/pythia-160m",
        bottleneck_layer: Optional[int] = None,  # None = auto (mid-point)
        num_reasoning_tokens: int = 8,
        num_codes: int = 512,
        code_dim: Optional[int] = None,
        enable_bottleneck: bool = True,  # For ablation (disable = baseline)
        **vq_kwargs
    ):
        super().__init__()

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        config = self.base_model.config

        # Determine bottleneck insertion point
        num_layers = config.num_hidden_layers
        self.bottleneck_layer = bottleneck_layer or (num_layers // 2)
        self.enable_bottleneck = enable_bottleneck

        print(f"Inserting bottleneck at layer {self.bottleneck_layer}/{num_layers}")

        # Create bottleneck
        if enable_bottleneck:
            self.reasoning_bottleneck = ReasoningBottleneck(
                hidden_dim=config.hidden_size,
                num_reasoning_tokens=num_reasoning_tokens,
                num_codes=num_codes,
                code_dim=code_dim,
                **vq_kwargs
            )
        else:
            self.reasoning_bottleneck = None

        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_reasoning_codes: bool = False,
    ):
        """
        Forward pass with optional bottleneck.

        Returns:
            If return_dict:
                {
                    'logits': output logits,
                    'loss': total loss (LM + VQ),
                    'lm_loss': language modeling loss,
                    'vq_loss': vector quantization loss,
                    'vq_metrics': usage statistics,
                    'reasoning_codes': (optional) discrete code indices
                }
        """

        # If bottleneck disabled, just run base model
        if not self.enable_bottleneck:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

            result = {
                'logits': outputs.logits,
                'loss': outputs.loss,
                'lm_loss': outputs.loss,
                'vq_loss': torch.tensor(0.0, device=input_ids.device),
                'vq_metrics': {},
            }
            return result if return_dict else (result['loss'], result['logits'])

        # Get transformer blocks
        transformer = self.base_model.gpt_neox  # For Pythia; adjust for other models

        # Embed inputs
        hidden_states = transformer.embed_in(input_ids)

        # Apply position embeddings if present
        if hasattr(transformer, 'emb_dropout'):
            hidden_states = transformer.emb_dropout(hidden_states)

        # Run first N layers
        for i in range(self.bottleneck_layer):
            layer_outputs = transformer.layers[i](
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

        # Apply reasoning bottleneck
        reasoning_codes, vq_loss, vq_metrics = self.reasoning_bottleneck(
            hidden_states,
            attention_mask=attention_mask
        )

        # CRITICAL: Replace hidden states with reasoning codes
        # Option 1: Replace entirely (forces all reasoning through bottleneck)
        # hidden_states = reasoning_codes

        # Option 2: Append reasoning codes (preserves input info)
        # This is more forgiving and often works better initially
        hidden_states = torch.cat([hidden_states, reasoning_codes], dim=1)

        # Update attention mask to account for new tokens
        if attention_mask is not None:
            reasoning_mask = torch.ones(
                attention_mask.size(0),
                reasoning_codes.size(1),
                device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, reasoning_mask], dim=1)

        # Run remaining layers
        for i in range(self.bottleneck_layer, len(transformer.layers)):
            layer_outputs = transformer.layers[i](
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

        # Final layer norm
        hidden_states = transformer.final_layer_norm(hidden_states)

        # Project to vocabulary
        logits = self.base_model.embed_out(hidden_states)

        # Compute language modeling loss
        lm_loss = None
        if labels is not None:
            # Shift for autoregressive prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Total loss
        total_loss = lm_loss + vq_loss if lm_loss is not None else vq_loss

        result = {
            'logits': logits,
            'loss': total_loss,
            'lm_loss': lm_loss if lm_loss is not None else torch.tensor(0.0),
            'vq_loss': vq_loss,
            'vq_metrics': vq_metrics,
        }

        if output_reasoning_codes:
            result['reasoning_codes'] = vq_metrics['code_indices']

        return result if return_dict else (total_loss, logits)

    def generate(self, *args, **kwargs):
        """
        Generation wrapper (for inference).

        Note: This is simplified - full implementation needs to handle
        bottleneck during autoregressive generation properly.
        """
        if not self.enable_bottleneck:
            return self.base_model.generate(*args, **kwargs)

        # For PoC: just use base model generation
        # (full implementation would need custom generation loop)
        print("Warning: Using base model generation (bottleneck bypassed)")
        return self.base_model.generate(*args, **kwargs)
```

### 6.4 Training Script

**File: `scripts/train_vq.py`**

```python
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm
import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.reasoning_transformer import ReasoningTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--run_name', type=str, default=None)
    return parser.parse_args()


def load_gsm8k(tokenizer, max_length=512):
    """Load and preprocess GSM8K dataset."""
    dataset = load_dataset("gsm8k", "main")

    def preprocess(examples):
        # Format: Question + " Answer:" + Solution
        texts = [
            f"Question: {q}\nAnswer: {a}"
            for q, a in zip(examples['question'], examples['answer'])
        ]

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Labels = input_ids (for LM loss)
        tokenized['labels'] = tokenized['input_ids'].clone()

        return tokenized

    train_dataset = dataset['train'].map(
        preprocess,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    eval_dataset = dataset['test'].map(
        preprocess,
        batched=True,
        remove_columns=dataset['test'].column_names
    )

    return train_dataset, eval_dataset


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """Single training epoch."""
    model.train()
    total_loss = 0
    total_lm_loss = 0
    total_vq_loss = 0
    total_utilization = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

        optimizer.step()
        scheduler.step()

        # Logging
        total_loss += loss.item()
        total_lm_loss += outputs['lm_loss'].item()
        total_vq_loss += outputs['vq_loss'].item()

        if outputs['vq_metrics']:
            total_utilization += outputs['vq_metrics'].get('codebook_utilization', 0)

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'lm': outputs['lm_loss'].item(),
            'vq': outputs['vq_loss'].item(),
            'lr': scheduler.get_last_lr()[0]
        })

        # Log to wandb
        if config.get('use_wandb') and batch_idx % config.get('log_interval', 10) == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/lm_loss': outputs['lm_loss'].item(),
                'train/vq_loss': outputs['vq_loss'].item(),
                'train/learning_rate': scheduler.get_last_lr()[0],
                'train/codebook_utilization': outputs['vq_metrics'].get('codebook_utilization', 0),
            })

    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'lm_loss': total_lm_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
        'codebook_utilization': total_utilization / num_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluation loop."""
    model.eval()
    total_loss = 0
    total_lm_loss = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        total_loss += outputs['loss'].item()
        total_lm_loss += outputs['lm_loss'].item()

    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'lm_loss': total_lm_loss / num_batches,
    }


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    if config.get('use_wandb', True):
        wandb.init(
            project=config.get('wandb_project', 'reasoning-ir-poc'),
            name=args.run_name or config.get('run_name', 'vq-experiment'),
            config=config
        )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    train_dataset, eval_dataset = load_gsm8k(tokenizer, config['data']['max_length'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Initialize model
    model = ReasoningTransformer(
        base_model_name=config['model']['base_model_name'],
        bottleneck_layer=config['model'].get('bottleneck_layer'),
        num_reasoning_tokens=config['model']['num_reasoning_tokens'],
        num_codes=config['model']['num_codes'],
        code_dim=config['model'].get('code_dim'),
        enable_bottleneck=config['model'].get('enable_bottleneck', True),
        commitment_weight=config['model'].get('commitment_weight', 0.25),
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    num_warmup_steps = int(num_training_steps * config['training']['warmup_ratio'])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    best_eval_loss = float('inf')

    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, config
        )

        print(f"Train loss: {train_metrics['loss']:.4f}")
        print(f"Codebook utilization: {train_metrics['codebook_utilization']:.2%}")

        # Evaluate
        eval_metrics = evaluate(model, eval_loader, device)
        print(f"Eval loss: {eval_metrics['loss']:.4f}")

        # Log to wandb
        if config.get('use_wandb'):
            wandb.log({
                'epoch': epoch,
                'eval/loss': eval_metrics['loss'],
                'eval/lm_loss': eval_metrics['lm_loss'],
            })

        # Save checkpoint
        if eval_metrics['loss'] < best_eval_loss:
            best_eval_loss = eval_metrics['loss']

            output_path = Path(args.output_dir) / f"checkpoint-best"
            output_path.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'eval_loss': eval_metrics['loss'],
                'config': config,
            }, output_path / 'model.pt')

            print(f"Saved best checkpoint (eval_loss: {eval_metrics['loss']:.4f})")

    # Final codebook analysis
    if model.enable_bottleneck:
        usage_stats = model.reasoning_bottleneck.vq.get_codebook_usage_stats()
        print("\nFinal Codebook Usage:")
        print(f"Codes used: {usage_stats['codes_ever_used']}/{usage_stats['total_codes']}")
        print(f"Usage entropy: {usage_stats['usage_entropy']:.3f}")

        if config.get('use_wandb'):
            wandb.log({'final_codebook_stats': usage_stats})

    if config.get('use_wandb'):
        wandb.finish()


if __name__ == '__main__':
    main()
```

### 6.5 Configuration Files

**File: `configs/vq_small.yaml`**

```yaml
# Experiment configuration

# Model settings
model:
  base_model_name: "EleutherAI/pythia-160m"
  bottleneck_layer: null  # null = auto (mid-point)
  num_reasoning_tokens: 8
  num_codes: 512
  code_dim: null  # null = match hidden_dim
  enable_bottleneck: true
  commitment_weight: 0.25

# Data settings
data:
  max_length: 512
  dataset: "gsm8k"

# Training settings
training:
  batch_size: 16
  num_epochs: 5
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0

# Logging
use_wandb: true
wandb_project: "reasoning-ir-poc"
run_name: "vq-pythia160m-codes512"
log_interval: 10

# Reproducibility
seed: 42
```

**File: `configs/baseline.yaml`**

```yaml
# Baseline (no bottleneck) configuration

model:
  base_model_name: "EleutherAI/pythia-160m"
  enable_bottleneck: false  # Disable for baseline

data:
  max_length: 512
  dataset: "gsm8k"

training:
  batch_size: 16
  num_epochs: 5
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0

use_wandb: true
wandb_project: "reasoning-ir-poc"
run_name: "baseline-pythia160m"
log_interval: 10

seed: 42
```

---

## 7. Training Procedures

### 7.1 Setup Commands

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Login to wandb (optional)
wandb login

# Test installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 7.2 Running Experiments

**Step 1: Baseline (No Bottleneck)**

```bash
python scripts/train_vq.py \
    --config configs/baseline.yaml \
    --output_dir outputs/baseline \
    --run_name baseline-pythia160m
```

**Step 2: VQ Bottleneck (Default)**

```bash
python scripts/train_vq.py \
    --config configs/vq_small.yaml \
    --output_dir outputs/vq_512codes \
    --run_name vq-512codes-8tokens
```

**Step 3: Ablations**

```bash
# Different codebook sizes
for num_codes in 256 512 1024; do
    python scripts/train_vq.py \
        --config configs/vq_small.yaml \
        --output_dir outputs/vq_${num_codes}codes \
        --run_name vq-${num_codes}codes
done

# Different reasoning token counts
for num_tokens in 4 8 16; do
    # Edit config to set num_reasoning_tokens = $num_tokens
    python scripts/train_vq.py \
        --config configs/vq_${num_tokens}tokens.yaml \
        --output_dir outputs/vq_${num_tokens}tokens \
        --run_name vq-${num_tokens}tokens
done
```

### 7.3 Training Monitoring

**Key Metrics to Watch:**

1. **Loss Trends**
   - `lm_loss` should decrease steadily
   - `vq_loss` should stabilize (not collapse to 0)
   - Total loss = lm_loss + vq_loss

2. **Codebook Utilization**
   - Target: >50% codes used
   - Warning if <20% (collapse risk)
   - Monitor `unique_codes` and `codebook_utilization`

3. **Gradient Norms**
   - Should be stable (not exploding/vanishing)
   - Clipping should trigger <10% of the time

**What Good Training Looks Like:**

```
Epoch 1/5
Train loss: 3.42 → 2.87
Codebook utilization: 23% → 48%
Eval loss: 2.91

Epoch 2/5
Train loss: 2.74 → 2.51
Codebook utilization: 51% → 56%
Eval loss: 2.58

...converges steadily
```

**Red Flags:**

- Codebook utilization <10% (collapse)
- VQ loss → 0 (bottleneck being ignored)
- LM loss increases (something broken)

### 7.4 Checkpointing Strategy

Save checkpoints:
1. **Best eval loss** (always)
2. **Every epoch** (if disk space allows)
3. **Final model** (after last epoch)

Include in checkpoint:
- Model state dict
- Optimizer state
- Scheduler state
- Config
- Codebook usage stats

---

## 8. Evaluation Framework

### 8.1 Evaluation Script

**File: `scripts/evaluate.py`**

```python
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import argparse
import json
import re
from pathlib import Path

from models.reasoning_transformer import ReasoningTransformer


def extract_answer(text):
    """Extract numerical answer from GSM8K solution."""
    # GSM8K answers end with "#### NUMBER"
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    return None


def evaluate_gsm8k(model, tokenizer, device, max_new_tokens=256, num_samples=None):
    """Evaluate on GSM8K test set."""
    dataset = load_dataset("gsm8k", "main")['test']

    if num_samples:
        dataset = dataset.select(range(num_samples))

    correct = 0
    total = 0
    total_input_tokens = 0
    total_generated_tokens = 0

    results = []

    for example in tqdm(dataset, desc="Evaluating GSM8K"):
        question = example['question']
        true_answer = extract_answer(example['answer'])

        # Format prompt
        prompt = f"Question: {question}\nAnswer:"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_length = inputs['input_ids'].size(1)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        predicted_answer = extract_answer(generated_text)

        # Check correctness
        is_correct = (predicted_answer == true_answer)

        if is_correct:
            correct += 1
        total += 1

        # Token counts
        total_input_tokens += input_length
        total_generated_tokens += outputs.size(1) - input_length

        # Store result
        results.append({
            'question': question,
            'true_answer': true_answer,
            'predicted_answer': predicted_answer,
            'generated_text': generated_text,
            'correct': is_correct,
            'input_tokens': input_length,
            'generated_tokens': outputs.size(1) - input_length,
        })

    accuracy = correct / total
    avg_input_tokens = total_input_tokens / total
    avg_generated_tokens = total_generated_tokens / total

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_input_tokens': avg_input_tokens,
        'avg_generated_tokens': avg_generated_tokens,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='eval_results.json')
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    # Initialize model
    model = ReasoningTransformer(**config['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    # Evaluate
    print("Evaluating on GSM8K...")
    results = evaluate_gsm8k(model, tokenizer, device, num_samples=args.num_samples)

    # Print results
    print(f"\nResults:")
    print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    print(f"Avg input tokens: {results['avg_input_tokens']:.1f}")
    print(f"Avg generated tokens: {results['avg_generated_tokens']:.1f}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to {args.output}")


if __name__ == '__main__':
    main()
```

### 8.2 Metrics

**Primary Metrics:**

1. **Accuracy** (correctness on task)
2. **Token Efficiency** (input + generated tokens)
3. **Codebook Utilization** (% of codes used)
4. **Inference Latency** (optional)

**Comparison Table Template:**

| Model | Accuracy | Avg Tokens | Codebook Usage | Train Time |
|-------|----------|------------|----------------|------------|
| Baseline (no bottleneck) | X% | Y | N/A | Z hrs |
| VQ-256 codes | X% | Y | P% | Z hrs |
| VQ-512 codes | X% | Y | P% | Z hrs |
| VQ-1024 codes | X% | Y | P% | Z hrs |

### 8.3 Token Efficiency Analysis

**Calculate compression ratio:**

```python
baseline_avg_tokens = 487.3  # From baseline eval
vq_avg_tokens = 312.1        # From VQ model eval

compression_ratio = baseline_avg_tokens / vq_avg_tokens  # e.g., 1.56x
token_reduction = (1 - vq_avg_tokens/baseline_avg_tokens) * 100  # e.g., 36%

print(f"Compression: {compression_ratio:.2f}x")
print(f"Token reduction: {token_reduction:.1f}%")
```

---

## 9. Analysis & Visualization Tools

### 9.1 Codebook Analysis Script

**File: `scripts/analyze_codes.py`**

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import Counter
import argparse

from models.reasoning_transformer import ReasoningTransformer


def analyze_codebook(checkpoint_path, output_dir='analysis'):
    """Comprehensive codebook analysis."""

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']

    model = ReasoningTransformer(**config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract codebook
    codebook = model.reasoning_bottleneck.vq.codebook.weight.data.cpu().numpy()
    usage_count = model.reasoning_bottleneck.vq.usage_count.cpu().numpy()

    num_codes, code_dim = codebook.shape

    print(f"Codebook shape: {codebook.shape}")
    print(f"Codes used: {(usage_count > 0).sum()}/{num_codes}")

    # 1. Usage distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(usage_count, bins=50)
    axes[0].set_xlabel('Usage Count')
    axes[0].set_ylabel('Number of Codes')
    axes[0].set_title('Code Usage Distribution')
    axes[0].set_yscale('log')

    # Top-K codes
    top_k = 20
    top_indices = np.argsort(usage_count)[-top_k:][::-1]
    axes[1].barh(range(top_k), usage_count[top_indices])
    axes[1].set_xlabel('Usage Count')
    axes[1].set_ylabel('Code Index')
    axes[1].set_title(f'Top {top_k} Most Used Codes')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/code_usage.png', dpi=150)
    print(f"Saved: {output_dir}/code_usage.png")

    # 2. t-SNE visualization
    print("Computing t-SNE...")

    # Only visualize used codes
    used_mask = usage_count > 0
    used_codes = codebook[used_mask]
    used_usage = usage_count[used_mask]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(used_codes)-1))
    codes_2d = tsne.fit_transform(used_codes)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        codes_2d[:, 0],
        codes_2d[:, 1],
        c=np.log1p(used_usage),  # Log scale for color
        s=100,
        alpha=0.6,
        cmap='viridis'
    )
    plt.colorbar(scatter, label='log(Usage Count + 1)')
    plt.title('t-SNE Visualization of Reasoning Codes')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(f'{output_dir}/code_tsne.png', dpi=150)
    print(f"Saved: {output_dir}/code_tsne.png")

    # 3. Clustering analysis
    print("Performing clustering...")

    n_clusters = min(10, len(used_codes) // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(used_codes)

    # Cluster statistics
    print(f"\nCluster Analysis ({n_clusters} clusters):")
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_size = cluster_mask.sum()
        cluster_total_usage = used_usage[cluster_mask].sum()
        print(f"  Cluster {i}: {cluster_size} codes, {cluster_total_usage:.0f} total usage")

    # Visualize clusters on t-SNE
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        codes_2d[:, 0],
        codes_2d[:, 1],
        c=cluster_labels,
        s=100,
        alpha=0.6,
        cmap='tab10'
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('Clustered Reasoning Codes')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(f'{output_dir}/code_clusters.png', dpi=150)
    print(f"Saved: {output_dir}/code_clusters.png")

    # 4. Code similarity heatmap (for top codes)
    top_k_codes = codebook[top_indices[:15]]
    similarity = np.corrcoef(top_k_codes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True
    )
    plt.title('Similarity Between Top 15 Codes')
    plt.xlabel('Code Index')
    plt.ylabel('Code Index')
    plt.savefig(f'{output_dir}/code_similarity.png', dpi=150)
    print(f"Saved: {output_dir}/code_similarity.png")

    # 5. Summary statistics
    stats = {
        'total_codes': num_codes,
        'codes_used': (usage_count > 0).sum(),
        'utilization': (usage_count > 0).sum() / num_codes,
        'usage_entropy': -np.sum((usage_count / usage_count.sum()) * np.log(usage_count / usage_count.sum() + 1e-10)),
        'max_usage': usage_count.max(),
        'mean_usage': usage_count[usage_count > 0].mean(),
        'top_10_codes': top_indices[:10].tolist(),
        'top_10_usage': usage_count[top_indices[:10]].tolist(),
    }

    print("\nSummary Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='analysis')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    analyze_codebook(args.checkpoint, args.output_dir)


if __name__ == '__main__':
    main()
```

### 9.2 Visualization Examples

**What to look for in t-SNE plots:**

1. **Cluster formation** - Do codes naturally group?
2. **Usage patterns** - Are high-usage codes spread out or clustered?
3. **Dead zones** - Are there regions with no codes (wasted capacity)?

**Interpreting clusters:**

- **Tight clusters** might represent specialized reasoning operations
- **Scattered points** might be general-purpose codes
- **Isolated high-usage codes** could be "hub" concepts

### 9.3 Probe Classifiers (Interpretability)

**Goal:** Train classifiers to predict problem properties from IR codes

```python
def train_probe_classifier(model, dataset, tokenizer, device):
    """
    Train a probe to predict problem type from reasoning codes.

    Example labels:
    - Arithmetic operation type (addition, multiplication, etc.)
    - Number of reasoning steps required
    - Problem difficulty
    """

    # Collect reasoning codes and labels
    code_sequences = []
    labels = []

    for example in dataset:
        # Get reasoning codes for this example
        inputs = tokenizer(example['question'], return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_reasoning_codes=True)
            codes = outputs['reasoning_codes']  # [1, num_tokens]

        code_sequences.append(codes.cpu())

        # Label: count arithmetic operations (simple heuristic)
        num_operations = example['answer'].count('+') + example['answer'].count('*')
        labels.append(min(num_operations, 4))  # Clip to 0-4

    # Stack
    X = torch.cat(code_sequences, dim=0).numpy()  # [N, num_tokens]
    y = np.array(labels)

    # Train simple classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"Probe accuracy (train): {train_acc:.2%}")
    print(f"Probe accuracy (test): {test_acc:.2%}")

    # Interpretation: if test_acc >> random chance, codes encode problem structure
    random_baseline = 1 / len(np.unique(y))
    print(f"Random baseline: {random_baseline:.2%}")

    return clf, test_acc
```

**High probe accuracy** → codes meaningfully represent problem structure
**Low probe accuracy** → codes are opaque or task-specific

---

## 10. Risk Mitigation & Debugging

### 10.1 Common Issues

**Issue 1: Codebook Collapse (Utilization <10%)**

**Symptoms:**
- Only 20-50 codes ever used
- VQ loss → 0 quickly
- Model performance drops

**Solutions:**
1. Increase diversity loss weight (`δ = 0.1`)
2. Lower commitment weight (`β = 0.1`)
3. Restart dead codes more aggressively (`restart_threshold = 5`)
4. Try smaller codebook (256 instead of 512)

**Issue 2: Gradient Explosion**

**Symptoms:**
- Loss spikes to NaN
- Gradient norms >100

**Solutions:**
1. Lower learning rate (try 1e-5)
2. Reduce max_grad_norm (try 0.5)
3. Check bottleneck placement (try later layer)

**Issue 3: No Efficiency Gain**

**Symptoms:**
- Generated tokens same as baseline
- Reasoning codes seem unused

**Solutions:**
1. Check that bottleneck is actually being used (print activations)
2. Try forcing reasoning through bottleneck (replace vs append strategy)
3. Increase number of reasoning tokens (16 instead of 8)

**Issue 4: Accuracy Drop >5%**

**Symptoms:**
- Model performs worse than baseline

**Solutions:**
1. Train longer (more epochs)
2. Lower VQ loss weight (0.1x instead of 1.0x)
3. Try softer quantization (Gumbel-Softmax)
4. Verify gradients flow properly (check backward pass)

### 10.2 Debugging Checklist

Before debugging, verify:

- [ ] Baseline model works (accuracy matches published results)
- [ ] Data loading correct (inspect tokenized examples)
- [ ] Shapes match expectations (print tensor shapes)
- [ ] Gradients flow (check `param.grad` is not None)
- [ ] Loss components reasonable (LM loss ~2-4, VQ loss ~0.1-1.0)

**Debug prints to add:**

```python
# In forward pass
print(f"Input shape: {input_ids.shape}")
print(f"Hidden before bottleneck: {hidden_states.shape}")
print(f"Reasoning codes: {reasoning_codes.shape}")
print(f"VQ loss: {vq_loss.item():.4f}")
print(f"Unique codes this batch: {len(torch.unique(code_indices))}")
```

### 10.3 Ablation Studies

**Test these systematically:**

1. **Bottleneck placement:** Early (layer 2) vs Mid (layer N/2) vs Late (layer 2N/3)
2. **Codebook size:** 128, 256, 512, 1024
3. **Reasoning tokens:** 4, 8, 16, 32
4. **Commitment weight:** 0.1, 0.25, 0.5
5. **Replace vs append:** Full replacement vs concatenation

**Prioritization:** Start with #2 and #3 (biggest impact)

---

## 11. Resource Requirements

### 11.1 Compute Budget

**Minimum viable experiment:**

- **Model:** Pythia-160M
- **Data:** GSM8K (7.5K examples)
- **Training:** 5 epochs, batch size 16
- **GPU:** 1x RTX 3090 (24GB) or equivalent
- **Time:** ~4 hours
- **Cost:** ~$2 on cloud (Lambda Labs, RunPod)

**Full experiment suite:**

- **Models:** Pythia-70M/160M/410M + TinyLlama-1.1B
- **Ablations:** 3 codebook sizes × 3 token counts = 9 runs
- **Training:** ~40 GPU-hours total
- **Cost:** ~$20-40 on cloud

**Scaling to larger models (future):**

- **Model:** Pythia-2.8B or Llama-7B
- **GPU:** 1x A100 (40GB) or 2x 3090
- **Time:** ~12 hours per run
- **Cost:** ~$50-100 for full suite

### 11.2 Storage Requirements

**Per experiment:**
- Model checkpoint: ~1-2 GB (for 160M model)
- Codebook analysis: ~100 MB
- Eval results: ~50 MB

**Total:** ~500 GB for full suite (with all checkpoints)

**Optimization:** Delete intermediate checkpoints, keep only best.

### 11.3 Development Timeline

**Week-by-week breakdown:**

| Week | Tasks | Hours |
|------|-------|-------|
| 1 | Setup, baselines, implementation | 20-25 |
| 2 | Training, evaluation, analysis | 15-20 |
| 3 | Documentation, visualization, writeup | 10-15 |

**Total:** ~50-60 hours for proficient ML researcher

---

## 12. Future Extensions

### 12.1 If Initial Results Are Positive

**Near-term (1-2 months):**

1. **Scale to larger models**
   - Pythia-2.8B, Llama-7B
   - Test if emergent structure persists

2. **Cross-task transfer**
   - Train on GSM8K, test on SVAMP/ARC
   - Measure code reuse across domains

3. **Reasoning trace analysis**
   - Manually inspect what codes activate for different problem types
   - Build taxonomy of discovered primitives

**Medium-term (3-6 months):**

4. **Modular pipeline (Option B)**
   - Separate encoder/reasoner/decoder
   - Enable swappable reasoning cores

5. **Reinforcement learning**
   - Fine-tune with RLHF to improve reasoning quality
   - Reward sparse code usage (efficiency bonus)

6. **Hierarchical codes**
   - Multi-level codebook (coarse + fine reasoning)
   - Test if abstraction hierarchy emerges

### 12.2 If Results Are Negative

**Still valuable to publish:**

- Why did it fail? (codebook collapse, no efficiency, etc.)
- What does this tell us about reasoning in LLMs?
- Negative results prevent others from repeating mistakes

**Pivot directions:**

1. **Soft bottleneck** (continuous, no quantization)
2. **Hybrid approach** (combine IR + natural language)
3. **Supervised reasoning codes** (hand-label some codes, then learn rest)

### 12.3 Theoretical Questions to Explore

1. **What is the information-theoretic lower bound** for reasoning compression?
2. **Do codes compose** (can learned primitives combine for novel problems)?
3. **How does code structure relate to model size** (do larger models need more codes)?
4. **Can we transfer codes between models** (like knowledge distillation)?

---

## 13. Conclusion & Recommendations

### 13.1 Summary

This PoC will test whether:

1. **LLMs can learn an efficient internal reasoning language** (more compact than natural language CoT)
2. **Discrete codes naturally cluster** into interpretable reasoning primitives
3. **Token efficiency improves** without sacrificing accuracy

### 13.2 Critical Success Factors

**For this to be successful, we need:**

✅ Codebook utilization >50%
✅ Token reduction >20%
✅ Accuracy within 5% of baseline
✅ Some interpretable structure in codes

**If we achieve 3/4, it's a strong result.**

### 13.3 Next Steps (Immediate)

1. Set up environment and dependencies
2. Run baseline evaluations
3. Implement VQ bottleneck
4. Train first model (Pythia-160M, 512 codes)
5. Analyze results

**Start here:** `scripts/train_vq.py --config configs/vq_small.yaml`

### 13.4 Final Thoughts

This research direction addresses a fundamental question: **Is natural language the right substrate for machine reasoning?**

The answer likely isn't binary. Even if discrete IR doesn't outperform CoT, understanding *why* will deepen our knowledge of how these models think.

And if it *does* work—if models discover their own efficient reasoning languages—it could change how we approach:

- Reasoning dataset design
- Model architecture
- Interpretability
- Alignment (can we monitor reasoning in IR space?)

**This PoC is tractable, scientifically interesting, and potentially high-impact.**

I recommend proceeding.

---

## Appendix: Quick Reference

### A. Command Cheat Sheet

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train baseline
python scripts/train_vq.py --config configs/baseline.yaml

# Train VQ model
python scripts/train_vq.py --config configs/vq_small.yaml

# Evaluate
python scripts/evaluate.py --checkpoint outputs/vq_512codes/checkpoint-best/model.pt

# Analyze codebook
python scripts/analyze_codes.py --checkpoint outputs/vq_512codes/checkpoint-best/model.pt

# Monitor training
wandb login
wandb online  # View at wandb.ai
```

### B. Key Hyperparameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| num_codes | 512 | 128-1024 | Capacity vs sparsity |
| num_reasoning_tokens | 8 | 4-32 | Reasoning budget |
| commitment_weight | 0.25 | 0.1-0.5 | Encoder commitment |
| learning_rate | 5e-5 | 1e-5 to 1e-4 | Convergence speed |

### C. Expected Timeline

- **Day 1:** Setup + baselines (4 hrs)
- **Day 2-3:** Implementation (8 hrs)
- **Day 4-6:** Training (12 hrs compute, 4 hrs monitoring)
- **Day 7-9:** Evaluation + analysis (6 hrs)
- **Day 10-12:** Writeup (6 hrs)

**Total: ~30 active hours, ~1.5 weeks elapsed**

---

**End of Implementation Guide**

*This document is designed to be self-contained. You should be able to go from zero to working PoC using only this guide.*

*Good luck, and please share your results!*
