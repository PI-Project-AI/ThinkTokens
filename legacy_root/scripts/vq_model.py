#!/usr/bin/env python3
"""Vector Quantized bottleneck model for intermediate reasoning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Tuple, Dict, Any
import math

class VectorQuantizer(nn.Module):
    """Vector Quantizer module using straight-through estimator."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        """
        Initialize VQ module.

        Args:
            num_embeddings: Number of discrete codes in codebook
            embedding_dim: Dimension of each code
            commitment_cost: Weight for commitment loss
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # Track codebook usage for monitoring
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('w', torch.randn(embedding_dim, num_embeddings))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize inputs and return quantized outputs.

        Args:
            inputs: [batch_size, seq_len, embedding_dim]

        Returns:
            quantized: Quantized embeddings [batch_size, seq_len, embedding_dim]
            loss: VQ loss
            indices: Codebook indices used [batch_size, seq_len]
            metrics: Dictionary of metrics
        """
        # Flatten spatial dimensions
        flat_inputs = inputs.view(-1, self.embedding_dim)  # [batch_size * seq_len, embedding_dim]

        # Calculate distances to all codes
        # d(x, c) = ||x - c||^2 = ||x||^2 + ||c||^2 - 2 <x, c>
        distances = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_inputs, self.embedding.weight.t())
        )

        # Get nearest code indices
        indices = torch.argmin(distances, dim=1)  # [batch_size * seq_len]
        indices = indices.view(inputs.shape[0], inputs.shape[1])  # [batch_size, seq_len]

        # Quantize
        quantized = self.embedding(indices)  # [batch_size, seq_len, embedding_dim]

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Update cluster size (for monitoring)
        with torch.no_grad():
            flat_indices = indices.view(-1)
            updated_cluster_size = (
                self.cluster_size +
                torch.sum(
                    F.one_hot(flat_indices, num_classes=self.num_embeddings).float(),
                    dim=0
                )
            )

            # Laplace smoothing
            n = updated_cluster_size.sum()
            updated_cluster_size = (
                (updated_cluster_size + 1e-5) /
                (n + self.num_embeddings * 1e-5) * n
            )

            self.cluster_size.data = updated_cluster_size

        # Calculate metrics
        with torch.no_grad():
            code_usage = (self.cluster_size > 0).sum().item()
            perplexity = torch.exp(-torch.sum(self.cluster_size * torch.log(self.cluster_size + 1e-10)))

        metrics = {
            'vq_loss': loss.detach(),
            'e_latent_loss': e_latent_loss.detach(),
            'q_latent_loss': q_latent_loss.detach(),
            'code_usage': torch.tensor(code_usage, dtype=torch.float32),
            'code_usage_pct': torch.tensor(100 * code_usage / self.num_embeddings, dtype=torch.float32),
            'perplexity': perplexity.detach()
        }

        return quantized, loss, indices, metrics


class VQReasoningModel(nn.Module):
    """Transformer with VQ bottleneck for intermediate reasoning."""

    def __init__(
        self,
        base_model_name: str,
        num_codes: int = 512,
        bottleneck_position: str = "middle",
        commitment_cost: float = 0.25
    ):
        """
        Initialize VQ Reasoning Model.

        Args:
            base_model_name: HuggingFace model identifier
            num_codes: Number of discrete reasoning codes
            bottleneck_position: 'middle' (default) or specific layer number
            commitment_cost: VQ commitment loss weight
        """
        super().__init__()

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size

        # Determine bottleneck layer
        num_hidden_layers = self.config.num_hidden_layers
        if bottleneck_position == "middle":
            self.bottleneck_layer = num_hidden_layers // 2
        else:
            self.bottleneck_layer = int(bottleneck_position)

        # VQ Module
        self.vq = VectorQuantizer(
            num_embeddings=num_codes,
            embedding_dim=self.hidden_size,
            commitment_cost=commitment_cost
        )

        # Optional projection layers for bottleneck
        self.pre_vq_proj = nn.Identity()  # Can add projection if needed
        self.post_vq_proj = nn.Identity()

        print(f"VQ Reasoning Model initialized:")
        print(f"  Base model: {base_model_name}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Number of codes: {num_codes}")
        print(f"  Bottleneck layer: {self.bottleneck_layer}/{num_hidden_layers}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with VQ bottleneck.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] for training
            use_cache: Whether to return cache

        Returns:
            Dictionary with loss, logits, and metrics
        """
        # Get the transformer layers
        transformer = self.base_model.gpt_neox

        # Embed input
        hidden_states = transformer.embed_in(input_ids)

        # First half of transformer layers (before bottleneck)
        for i in range(self.bottleneck_layer):
            layer = transformer.layers[i]
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=False
            )
            hidden_states = layer_outputs[0]

        # Apply VQ bottleneck
        hidden_states = self.pre_vq_proj(hidden_states)
        quantized_states, vq_loss, indices, vq_metrics = self.vq(hidden_states)
        hidden_states = self.post_vq_proj(quantized_states)

        # Second half of transformer layers (after bottleneck)
        for i in range(self.bottleneck_layer, len(transformer.layers)):
            layer = transformer.layers[i]
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=False
            )
            hidden_states = layer_outputs[0]

        # Final layer norm and output projection
        hidden_states = transformer.final_layer_norm(hidden_states)
        logits = self.base_model.embed_out(hidden_states)

        # Calculate language modeling loss
        lm_loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss()
            lm_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Total loss
        total_loss = lm_loss + 0.25 * vq_loss if lm_loss is not None else vq_loss

        return {
            'loss': total_loss,
            'logits': logits,
            'lm_loss': lm_loss,
            'vq_loss': vq_loss,
            'indices': indices,
            'vq_metrics': vq_metrics,
            'hidden_states': hidden_states
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate text using the model with VQ bottleneck."""
        # Use the parent class generate method
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs
        )

    def get_codebook_usage(self):
        """Get codebook usage statistics."""
        return {
            'cluster_size': self.vq.cluster_size.cpu().numpy(),
            'num_codes_used': (self.vq.cluster_size > 0).sum().item(),
            'num_codes_total': self.vq.num_embeddings
        }
