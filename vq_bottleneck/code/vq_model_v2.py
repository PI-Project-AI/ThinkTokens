#!/usr/bin/env python3
"""Simplified Vector Quantized model using adapter approach."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Tuple, Dict, Any

class SimpleVectorQuantizer(nn.Module):
    """Simplified Vector Quantizer."""

    def __init__(self, num_codes: int = 512, embedding_dim: int = 1024):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim

        # Codebook
        self.register_buffer('_embedding_initialized', torch.tensor(False))
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_codes, 1.0 / num_codes)

        # Track usage
        self.register_buffer('cluster_size', torch.zeros(num_codes))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize inputs.

        Args:
            inputs: [batch, seq, dim]

        Returns:
            quantized: quantized values
            loss: vq loss
            indices: code indices used
        """
        # Flatten
        flat = inputs.reshape(-1, self.embedding_dim)

        # Ensure embedding weights match input dtype
        embedding_weight = self.embedding.weight.to(flat.dtype)

        # L2 distances to codes
        distances = (
            torch.sum(flat ** 2, dim=1, keepdim=True)
            + torch.sum(embedding_weight ** 2, dim=1)
            - 2 * torch.matmul(flat, embedding_weight.t())
        )

        # Find nearest codes
        indices = distances.argmin(dim=1)
        indices_oh = F.one_hot(indices, self.num_codes).float()

        # Quantize - ensure output matches input dtype
        quantized_flat = F.embedding(indices, embedding_weight)
        quantized = quantized_flat.reshape_as(inputs)

        # Loss: commitment + codebook
        e_loss = F.mse_loss(quantized.detach(), inputs)
        q_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_loss + 0.25 * e_loss

        # Straight through
        quantized = inputs + (quantized - inputs).detach()

        # Track usage
        with torch.no_grad():
            cluster_updated = self.cluster_size + indices_oh.sum(dim=0)
            self.cluster_size.data = cluster_updated

        return quantized, loss, indices.reshape(inputs.shape[0], inputs.shape[1])


class VQLanguageModel(nn.Module):
    """Language model with VQ bottleneck enforced on hidden states.

    This implements Option A: Single-model with discrete bottleneck using forward hooks.
    The model is split at a middle layer, and ALL information MUST pass through
    the VQ bottleneck to reach the output - this is a HARD bottleneck.

    Implementation: Uses a forward hook to intercept hidden states at the bottleneck
    layer, quantize them, and replace them with quantized versions. This ensures
    that all downstream layers only see discrete codes.
    """

    def __init__(self, base_model_name: str = "EleutherAI/pythia-410m", num_codes: int = 512,
                 use_gradient_checkpointing: bool = False):
        super().__init__()

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16 if use_gradient_checkpointing else torch.float32
        )
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.num_codes = num_codes
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
            print("  Gradient checkpointing: ENABLED")
            print("  Mixed precision: bfloat16")

        # VQ module
        self.vq = SimpleVectorQuantizer(num_codes, self.hidden_size)

        # Bottleneck layer (middle of network)
        self.bottleneck_layer = self.config.num_hidden_layers // 2

        # Storage for VQ metrics (populated by hook)
        self.vq_loss = None
        self.vq_indices = None

        # Register the bottleneck hook
        self._register_bottleneck_hook()

        print(f"VQ Language Model initialized (Option A - Hard Bottleneck via Hooks):")
        print(f"  Model: {base_model_name}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Codes: {num_codes}")
        print(f"  Bottleneck layer: {self.bottleneck_layer}/{self.config.num_hidden_layers}")
        print(f"  Architecture: Layers[0:{self.bottleneck_layer}] → VQ (HARD) → Layers[{self.bottleneck_layer}:{self.config.num_hidden_layers}]")
        print(f"  ✓ Hard bottleneck enforced via forward hook")

    def _register_bottleneck_hook(self):
        """Register a forward hook that enforces VQ bottleneck.

        This hook intercepts the output of the bottleneck layer,
        quantizes it, and returns the quantized version. This creates
        a hard information bottleneck - all downstream layers only
        see discrete codes.
        """
        layers = self.base_model.gpt_neox.layers

        def bottleneck_hook(module, input, output):
            """
            Hook that quantizes hidden states.

            Args:
                output: Tuple (hidden_states, ...) from transformer layer

            Returns:
                Modified tuple with quantized hidden_states
            """
            # Extract hidden states (first element of output tuple)
            hidden_states = output[0]

            # Apply VQ - this is where the magic happens!
            quantized, vq_loss, indices = self.vq(hidden_states)

            # Store metrics for later retrieval
            self.vq_loss = vq_loss
            self.vq_indices = indices

            # Return quantized states in place of original
            # This ensures downstream layers only see discrete codes
            return (quantized,) + output[1:]

        # Register hook on the bottleneck layer
        layers[self.bottleneck_layer].register_forward_hook(bottleneck_hook)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with HARD VQ bottleneck enforced via hook.

        The hook registered in __init__ automatically:
        1. Intercepts hidden states at bottleneck_layer
        2. Quantizes them through VQ
        3. Replaces them with quantized versions
        4. Stores VQ loss and indices

        This creates a true hard bottleneck - downstream layers
        only see discrete codes, not continuous representations.
        """
        # Reset VQ metrics
        self.vq_loss = None
        self.vq_indices = None

        # Run model - the hook will intercept at bottleneck layer
        # and enforce quantization
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        # Get logits (these were computed using quantized hidden states!)
        logits = outputs.logits
        lm_loss = outputs.loss

        # Retrieve VQ loss from hook
        # If hook didn't fire (shouldn't happen), use zero loss
        vq_loss = self.vq_loss if self.vq_loss is not None else torch.tensor(0.0, device=logits.device)
        indices = self.vq_indices

        # Total loss: language modeling + VQ objectives
        # VQ weight balances reconstruction vs compression
        total_loss = lm_loss + 0.25 * vq_loss if lm_loss is not None else vq_loss

        # Get codebook stats
        code_usage = (self.vq.cluster_size > 0).sum().item()

        return {
            'loss': total_loss,
            'logits': logits,
            'lm_loss': lm_loss,
            'vq_loss': vq_loss,
            'indices': indices,
            'code_usage': code_usage,
            'code_usage_pct': 100 * code_usage / self.num_codes
        }

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate text with VQ bottleneck enforced.

        The forward hook is active during generation, so the bottleneck
        IS enforced! Each forward pass during autoregressive generation
        goes through the quantization bottleneck.

        This means generated text truly reflects the bottleneck constraint.
        """
        return self.base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def get_codebook_usage(self):
        """Get codebook usage stats."""
        return {
            'cluster_size': self.vq.cluster_size.cpu().numpy(),
            'num_codes_used': (self.vq.cluster_size > 0).sum().item(),
            'num_codes_total': self.num_codes
        }
