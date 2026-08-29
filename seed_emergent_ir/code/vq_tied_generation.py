"""
VQ-tied code token generation.

CRITICAL: At code positions, output logits must come from VQ codebook distances,
not free softmax over vocabulary. This ensures code semantics come from quantization.

Implementation:
- At code positions: project hidden → VQ codebook → get nearest code
- Use negative L2 distance as logits (closer = higher prob)
- Straight-through estimator for gradients
- All other token positions: normal LM logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class VQTiedCodeGenerator(nn.Module):
    """
    Generates code tokens with logits tied to VQ codebook.

    At code positions:
    - Logits for code tokens = -distance to codebook vectors
    - Logits for other tokens = masked to -inf
    - Uses straight-through estimator

    At non-code positions:
    - Normal LM logits over full vocabulary
    """

    def __init__(
        self,
        vq_module,
        ir_token_ids: Dict,
        vocab_size: int,
        temperature: float = 1.0
    ):
        """
        Args:
            vq_module: ProjectionVQ instance with codebook
            ir_token_ids: Dict with code token ID ranges
            vocab_size: Total vocabulary size
            temperature: Softmax temperature for code selection
        """
        super().__init__()
        self.vq = vq_module
        self.ir_token_ids = ir_token_ids
        self.vocab_size = vocab_size
        self.temperature = temperature

        self.code_start = ir_token_ids['code_start']
        self.code_end = ir_token_ids['code_end']
        self.num_codes = self.code_end - self.code_start + 1

    def compute_code_logits(
        self,
        hidden_states: torch.Tensor,
        is_code_position: torch.Tensor,
        current_temperature: float = None,
        training_step: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute logits for code positions tied to VQ codebook.

        Args:
            hidden_states: Hidden states (batch, seq_len, hidden_dim)
            is_code_position: Boolean mask (batch, seq_len)
            current_temperature: Override temperature (for annealing)
            training_step: Global training step (for Gumbel warm-start)

        Returns:
            code_logits: Logits for code tokens (batch, seq_len, num_codes)
            vq_indices: Selected code indices (batch, seq_len)
            vq_loss: VQ commitment loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        temp = current_temperature if current_temperature is not None else self.temperature

        # Project hidden states through VQ
        # This gives us quantized representations and distances
        z_q, vq_loss, vq_indices = self.vq(hidden_states, training_step=training_step)

        # Get VQ codebook
        codebook = self.vq.vq.codebook.weight  # (num_codes, code_dim)

        # Project hidden states to code dimension
        z_proj = self.vq.projection(hidden_states)  # (batch, seq_len, code_dim)

        # Unit-normalize if VQ uses it
        if self.vq.vq.use_unit_norm:
            z_flat = F.normalize(z_proj.view(-1, z_proj.shape[-1]), dim=1)
            codebook_norm = F.normalize(codebook, dim=1)
        else:
            z_flat = z_proj.view(-1, z_proj.shape[-1])
            codebook_norm = codebook

        # Compute L2 distances or cosine similarity
        # If unit-normalized: distance ≈ 2 - 2*cos(θ) = 2(1 - cos)
        # For numerical stability with unit norm, use dot product directly
        if self.vq.vq.use_unit_norm:
            # Logits = (z · codebook^T) / temp (cosine similarity)
            similarity = torch.matmul(z_flat, codebook_norm.t())  # (batch*seq, num_codes)
            code_logits = similarity / temp
        else:
            # Standard L2 distance
            distances = (
                torch.sum(z_flat ** 2, dim=1, keepdim=True) +
                torch.sum(codebook_norm ** 2, dim=1) -
                2 * torch.matmul(z_flat, codebook_norm.t())
            )
            code_logits = -distances / temp

        code_logits = code_logits.view(batch_size, seq_len, self.num_codes)

        return code_logits, vq_indices, vq_loss

    def merge_logits(
        self,
        lm_logits: torch.Tensor,
        code_logits: torch.Tensor,
        is_code_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge LM logits with VQ-derived code logits.

        At code positions: use code_logits for code tokens, -inf for others
        At non-code positions: use normal LM logits

        Args:
            lm_logits: Standard LM logits (batch, seq_len, vocab_size)
            code_logits: VQ-derived code logits (batch, seq_len, num_codes)
            is_code_position: Boolean mask (batch, seq_len)

        Returns:
            merged_logits: Combined logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len, vocab_size = lm_logits.shape
        device = lm_logits.device

        # Start with LM logits
        merged = lm_logits.clone()

        # At code positions: mask out non-code tokens, insert VQ logits for codes
        if is_code_position.any():
            # Create mask for code positions
            code_mask = is_code_position.unsqueeze(-1)  # (batch, seq_len, 1)

            # Mask: at code positions, set all non-code tokens to -inf
            non_code_mask = torch.ones(batch_size, seq_len, vocab_size, device=device)
            non_code_mask[:, :, self.code_start:self.code_end+1] = 0
            non_code_mask = non_code_mask * code_mask.float() * float('-inf')

            merged = merged + non_code_mask

            # Insert VQ-derived logits for code tokens
            # At code positions, replace code token logits with VQ distances
            code_token_mask = code_mask.expand(-1, -1, self.num_codes)  # (batch, seq_len, num_codes)

            merged[:, :, self.code_start:self.code_end+1] = torch.where(
                code_token_mask,
                code_logits,
                merged[:, :, self.code_start:self.code_end+1]
            )

        return merged

    def sample_codes(
        self,
        hidden_states: torch.Tensor,
        is_code_position: torch.Tensor,
        use_argmax: bool = True,
        training_step: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample code tokens from VQ-tied logits.

        Args:
            hidden_states: Hidden states (batch, seq_len, hidden_dim)
            is_code_position: Boolean mask (batch, seq_len)
            use_argmax: If True, use argmax; else sample
            training_step: Global training step (for Gumbel warm-start)

        Returns:
            code_token_ids: Selected code token IDs (batch, seq_len)
            vq_loss: VQ commitment loss
        """
        code_logits, vq_indices, vq_loss = self.compute_code_logits(
            hidden_states, is_code_position, training_step=training_step
        )

        if use_argmax:
            # Greedy selection
            code_indices = torch.argmax(code_logits, dim=-1)
        else:
            # Sample from distribution
            probs = F.softmax(code_logits, dim=-1)
            code_indices = torch.multinomial(
                probs.view(-1, self.num_codes),
                num_samples=1
            ).view(code_logits.shape[0], code_logits.shape[1])

        # Convert code indices to token IDs
        code_token_ids = code_indices + self.code_start

        # Only apply at code positions
        code_token_ids = code_token_ids * is_code_position.long()

        return code_token_ids, vq_loss


def identify_code_positions(
    token_ids: torch.Tensor,
    code_start: int,
    code_end: int
) -> torch.Tensor:
    """
    Identify which positions contain code tokens.

    Args:
        token_ids: Token sequence (batch, seq_len)
        code_start: First code token ID
        code_end: Last code token ID

    Returns:
        Boolean mask (batch, seq_len)
    """
    return (token_ids >= code_start) & (token_ids <= code_end)


def should_emit_code_next(
    current_tokens: torch.Tensor,
    ir_token_ids: Dict,
    max_codes_per_span: int = 6
) -> torch.Tensor:
    """
    Determine if next position should emit a code token.

    Heuristic:
    - After open tag: emit code
    - After code (if < max_codes_per_span): emit code or close tag
    - After close tag: emit open tag
    - Other positions: emit non-code token

    Args:
        current_tokens: Current sequence (batch, seq_len)
        ir_token_ids: Dict with token IDs
        max_codes_per_span: Maximum codes per span

    Returns:
        Boolean mask (batch,) indicating if next should be code
    """
    batch_size = current_tokens.shape[0]
    device = current_tokens.device

    # Get last token for each sequence
    last_tokens = current_tokens[:, -1]

    # Check if last token is an open tag
    open_tags = torch.tensor([
        ir_token_ids['goal'],
        ir_token_ids['assume'],
        ir_token_ids['step'],
        ir_token_ids['check'],
        ir_token_ids['branch']
    ], device=device)

    is_after_open = torch.isin(last_tokens, open_tags)

    # Check if last token is a code
    is_after_code = (last_tokens >= ir_token_ids['code_start']) & \
                    (last_tokens <= ir_token_ids['code_end'])

    # Simplified: emit code if after open tag or after another code
    # (More sophisticated logic would track span lengths)
    should_emit_code = is_after_open | is_after_code

    return should_emit_code


if __name__ == "__main__":
    print("VQ-tied code generation module loaded.")
    print("\nKey concept: Code token logits derived from VQ codebook distances.")
    print("This ensures code semantics come from quantization, not free embeddings.")
