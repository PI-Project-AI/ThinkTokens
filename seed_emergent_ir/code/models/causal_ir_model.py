"""
Causal IR Model - Two-pass architecture with forced IR dependency.

Architecture:
  Pass 1: Input → IR Buffer (autoregressive, with VQ codes)
  Stop-Grad: Detach IR buffer
  Pass 2: IR Buffer only → Answer (no HL bypass)

This ensures the answer MUST depend on the IR buffer.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
from transformers import GPTNeoXForCausalLM

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ir_generator import IRBufferGenerator
from local_cycle import LocalCycleHead, compute_local_cycle_loss


class CausalIRModel(nn.Module):
    """
    Two-pass decoder-only model with enforced IR causality.

    No cross-attention modules needed - we simply exclude HL from Pass 2 context.
    """

    def __init__(
        self,
        base_model_name: str,
        ir_token_ids: Dict,
        num_codes: int = 512,
        code_dim: int = 128,
        snippet_length: int = 10,
        use_local_cycle: bool = True,
        cycle_weight: float = 0.05,
        vq_weight: float = 0.1,
        coverage_weight: float = 0.02
    ):
        """
        Args:
            base_model_name: HuggingFace model (e.g., 'EleutherAI/pythia-70m')
            ir_token_ids: Dict with token IDs for tags and codes
            num_codes: Number of VQ codes
            code_dim: Code embedding dimension
            snippet_length: Length of HL snippets for local cycle
            use_local_cycle: Whether to use local cycle loss
            cycle_weight: Weight for local cycle loss
            vq_weight: Weight for VQ loss
            coverage_weight: Weight for coverage/diversity loss
        """
        super().__init__()

        # Load base model
        self.base_model = GPTNeoXForCausalLM.from_pretrained(base_model_name)
        self.config = self.base_model.config

        # Resize embeddings to accommodate new tokens
        self.base_model.resize_token_embeddings(len(ir_token_ids['codes']) + 1000)  # Add buffer

        self.ir_token_ids = ir_token_ids
        self.use_local_cycle = use_local_cycle
        self.cycle_weight = cycle_weight
        self.vq_weight = vq_weight
        self.coverage_weight = coverage_weight

        # IR Buffer Generator (Pass 1)
        self.ir_generator = IRBufferGenerator(
            base_model=self.base_model,
            ir_token_ids=ir_token_ids,
            num_codes=num_codes,
            code_dim=code_dim
        )

        # Local Cycle Head (optional, for anchoring)
        if use_local_cycle:
            self.local_cycle_head = LocalCycleHead(
                hidden_dim=self.config.hidden_size,
                vocab_size=self.base_model.config.vocab_size,
                snippet_length=snippet_length
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        answer_ids: Optional[torch.Tensor] = None,
        target_ir_ids: Optional[torch.Tensor] = None,
        input_snippets: Optional[torch.Tensor] = None,
        mode: str = 'train'
    ) -> Dict:
        """
        Two-pass forward.

        Args:
            input_ids: Input problem tokens (batch, input_len)
            attention_mask: Attention mask for input
            answer_ids: Target answer tokens (batch, answer_len)
            target_ir_ids: Ground truth IR (for teacher forcing)
            input_snippets: Target snippets for local cycle (batch, snippet_len)
            mode: 'train' or 'inference'

        Returns:
            Dict with:
                - answer_logits: Logits for answer (batch, answer_len, vocab)
                - ir_token_ids: Generated IR buffer
                - total_loss: Combined loss
                - loss_breakdown: Individual loss components
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # ============= PASS 1: Generate IR Buffer =============
        ir_output = self.ir_generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ir_ids=target_ir_ids if mode == 'train' else None
        )

        ir_token_ids = ir_output['ir_token_ids']  # (batch, ir_len)
        vq_loss = ir_output['vq_loss']
        ir_lm_loss = ir_output['lm_loss'] if ir_output['lm_loss'] is not None else torch.tensor(0.0, device=device)

        # ============= STOP GRADIENT =============
        # CRITICAL: Detach IR buffer so answer decoder can't backprop through it
        ir_token_ids_detached = ir_token_ids.detach()

        # ============= PASS 2: Generate Answer from IR Only =============
        # Context = IR buffer only (NO input tokens - this enforces 0% bypass)
        answer_logits = None
        answer_loss = torch.tensor(0.0, device=device)

        if answer_ids is not None:
            # Create combined sequence: IR + Answer
            # Model will predict answer_ids conditioned on IR only
            full_sequence = torch.cat([ir_token_ids_detached, answer_ids[:, :-1]], dim=1)

            # Forward through model
            outputs = self.base_model(
                input_ids=full_sequence,
                output_hidden_states=True
            )

            # Get logits for answer positions
            ir_len = ir_token_ids_detached.shape[1]
            answer_logits = outputs.logits[:, ir_len-1:-1, :]  # (batch, answer_len, vocab)

            # Compute CE loss on answer
            answer_loss = torch.nn.functional.cross_entropy(
                answer_logits.reshape(-1, answer_logits.shape[-1]),
                answer_ids.reshape(-1),
                ignore_index=-100
            )

        # ============= Auxiliary Losses =============
        coverage_loss = self._compute_coverage_loss(ir_output)
        diversity_loss = self._compute_diversity_loss(ir_output)

        # Local cycle loss (if enabled)
        cycle_loss = torch.tensor(0.0, device=device)
        if self.use_local_cycle and mode == 'train' and input_snippets is not None:
            # Get IR hidden states for cycle loss
            ir_outputs = self.base_model(
                input_ids=ir_token_ids_detached,
                output_hidden_states=True
            )
            ir_hidden = ir_outputs.hidden_states[-1]

            cycle_loss = compute_local_cycle_loss(
                ir_token_ids=ir_token_ids_detached,
                ir_hidden_states=ir_hidden,
                input_snippets=input_snippets,
                ir_token_id_dict=self.ir_token_ids,
                local_cycle_head=self.local_cycle_head,
                sample_ratio=0.1
            )

        # ============= Total Loss =============
        total_loss = (
            answer_loss +
            ir_lm_loss * 0.5 +  # Weight IR LM loss lower than answer
            self.vq_weight * vq_loss +
            self.cycle_weight * cycle_loss +
            self.coverage_weight * (coverage_loss + diversity_loss)
        )

        loss_breakdown = {
            'answer_loss': answer_loss.item(),
            'ir_lm_loss': ir_lm_loss.item() if isinstance(ir_lm_loss, torch.Tensor) else ir_lm_loss,
            'vq_loss': vq_loss.item() if isinstance(vq_loss, torch.Tensor) else 0.0,
            'cycle_loss': cycle_loss.item() if isinstance(cycle_loss, torch.Tensor) else 0.0,
            'coverage_loss': coverage_loss.item() if isinstance(coverage_loss, torch.Tensor) else 0.0,
            'diversity_loss': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0
        }

        return {
            'answer_logits': answer_logits,
            'ir_token_ids': ir_token_ids_detached,
            'total_loss': total_loss,
            'loss_breakdown': loss_breakdown,
            'metadata': ir_output['metadata']
        }

    def _compute_coverage_loss(self, ir_output: Dict) -> torch.Tensor:
        """
        Compute coverage loss to encourage using all span types.

        Penalizes if certain tags are never used.
        """
        # Simplified: count unique tags in IR buffer
        # TODO: Implement proper per-tag coverage tracking
        return torch.tensor(0.0, device=ir_output['ir_token_ids'].device)

    def _compute_diversity_loss(self, ir_output: Dict) -> torch.Tensor:
        """
        Compute diversity loss to encourage codebook utilization.

        Encourages using diverse codes (target 50-70% utilization).
        """
        if 'vq_indices' not in ir_output['metadata'] or ir_output['metadata']['vq_indices'] is None:
            return torch.tensor(0.0, device=ir_output['ir_token_ids'].device)

        vq_indices = ir_output['metadata']['vq_indices']
        if vq_indices is None or vq_indices.numel() == 0:
            return torch.tensor(0.0, device=ir_output['ir_token_ids'].device)

        # Compute entropy of code distribution
        flat_indices = vq_indices.view(-1)
        unique_codes = torch.unique(flat_indices)
        utilization = len(unique_codes) / self.ir_generator.num_codes

        # Target utilization: 0.5 - 0.7
        target_util = 0.6
        util_penalty = (utilization - target_util) ** 2

        return util_penalty

    def generate_answer(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_answer_length: int = 20
    ) -> Dict:
        """
        Generate answer for given input (inference mode).

        Args:
            input_ids: Input problem tokens
            attention_mask: Attention mask
            max_answer_length: Maximum answer length

        Returns:
            Dict with generated answer and IR buffer
        """
        self.eval()
        with torch.no_grad():
            # Generate IR buffer (Pass 1)
            ir_output = self.ir_generator(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_ir_ids=None
            )

            ir_token_ids = ir_output['ir_token_ids']

            # Generate answer from IR (Pass 2)
            # Start with IR buffer, autoregressively generate answer
            current_seq = ir_token_ids

            for _ in range(max_answer_length):
                outputs = self.base_model(input_ids=current_seq)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                current_seq = torch.cat([current_seq, next_token], dim=1)

                # Check for EOS token
                if (next_token == self.base_model.config.eos_token_id).all():
                    break

            # Extract answer (tokens after IR buffer)
            ir_len = ir_token_ids.shape[1]
            answer_ids = current_seq[:, ir_len:]

            return {
                'answer_ids': answer_ids,
                'ir_token_ids': ir_token_ids,
                'metadata': ir_output['metadata']
            }


if __name__ == "__main__":
    print("Testing CausalIRModel initialization...")

    # This is just a structure test - actual testing requires tokenizer setup
    from tokenizer_utils import extend_tokenizer_for_ir

    tokenizer, ir_token_ids = extend_tokenizer_for_ir(base_model_name="EleutherAI/pythia-70m")

    model = CausalIRModel(
        base_model_name="EleutherAI/pythia-70m",
        ir_token_ids=ir_token_ids,
        num_codes=512,
        code_dim=128
    )

    print(f"Model initialized successfully")
    print(f"Base model vocab size: {model.base_model.config.vocab_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
