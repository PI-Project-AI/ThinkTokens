"""
Local Cycle Loss for IR-CoT.

Implements light local cycle: IR span → HL snippet reconstruction.
This anchors IR codes to task content without requiring long CoT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LocalCycleHead(nn.Module):
    """
    Reconstructs short HL snippets from IR spans.

    Takes an IR span (e.g., <STEP> c047 c089 </STEP>) and tries to
    reconstruct a 5-10 token snippet from the original input that
    corresponds to what this step is about.

    This provides weak supervision to anchor IR codes to task semantics.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        snippet_length: int = 10,
        num_layers: int = 2
    ):
        """
        Args:
            hidden_dim: Hidden dimension of the model
            vocab_size: Size of the vocabulary
            snippet_length: Length of HL snippets to reconstruct
            num_layers: Number of transformer layers for reconstruction
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.snippet_length = snippet_length

        # Small transformer decoder for reconstruction
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Positional embeddings for snippet positions
        self.pos_emb = nn.Parameter(torch.randn(1, snippet_length, hidden_dim) * 0.02)

    def forward(
        self,
        ir_span_hidden: torch.Tensor,
        target_snippet_ids: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Reconstruct HL snippet from IR span.

        Args:
            ir_span_hidden: Hidden states from IR span (batch, span_len, hidden_dim)
            target_snippet_ids: Target HL snippet tokens (batch, snippet_len)
                                If provided, compute loss; else generate

        Returns:
            Dict with:
                - snippet_logits: Logits for snippet tokens (batch, snippet_len, vocab)
                - cycle_loss: Reconstruction loss (if target provided)
        """
        batch_size = ir_span_hidden.shape[0]
        device = ir_span_hidden.device

        # Initialize decoder input with learned positional embeddings
        decoder_input = self.pos_emb.expand(batch_size, -1, -1)  # (batch, snippet_len, hidden_dim)

        # Use IR span as memory for cross-attention
        memory = ir_span_hidden  # (batch, span_len, hidden_dim)

        # Decode snippet
        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=memory
        )  # (batch, snippet_len, hidden_dim)

        # Project to vocabulary
        snippet_logits = self.output_proj(decoder_output)  # (batch, snippet_len, vocab)

        # Compute loss if target provided
        cycle_loss = None
        if target_snippet_ids is not None:
            cycle_loss = F.cross_entropy(
                snippet_logits.reshape(-1, self.vocab_size),
                target_snippet_ids.reshape(-1),
                ignore_index=-100
            )

        return {
            'snippet_logits': snippet_logits,
            'cycle_loss': cycle_loss
        }

    def sample_snippet(self, ir_span_hidden: torch.Tensor) -> torch.Tensor:
        """
        Generate snippet tokens from IR span (inference mode).

        Args:
            ir_span_hidden: Hidden states from IR span (batch, span_len, hidden_dim)

        Returns:
            Generated snippet token IDs (batch, snippet_len)
        """
        output = self.forward(ir_span_hidden, target_snippet_ids=None)
        snippet_logits = output['snippet_logits']

        # Greedy decode
        snippet_ids = torch.argmax(snippet_logits, dim=-1)
        return snippet_ids


def extract_ir_spans(
    ir_token_ids: torch.Tensor,
    ir_hidden_states: torch.Tensor,
    ir_token_id_dict: dict
) -> list:
    """
    Extract individual IR spans from full IR buffer.

    Args:
        ir_token_ids: IR token sequence (batch, ir_len)
        ir_hidden_states: Hidden states for IR tokens (batch, ir_len, hidden_dim)
        ir_token_id_dict: Dict with token IDs for tags

    Returns:
        List of dicts, each containing:
            - span_type: 'goal', 'step', 'check', etc.
            - span_hidden: Hidden states for this span (batch, span_len, hidden_dim)
            - span_tokens: Token IDs for this span
    """
    batch_size, ir_len = ir_token_ids.shape

    # Define open/close tag pairs
    tag_pairs = [
        (ir_token_id_dict['goal'], ir_token_id_dict['goal_end'], 'goal'),
        (ir_token_id_dict['assume'], ir_token_id_dict['assume_end'], 'assume'),
        (ir_token_id_dict['step'], ir_token_id_dict['step_end'], 'step'),
        (ir_token_id_dict['check'], ir_token_id_dict['check_end'], 'check'),
        (ir_token_id_dict['branch'], ir_token_id_dict['branch_end'], 'branch')
    ]

    spans = []

    # Extract spans for each example in batch
    for b in range(batch_size):
        tokens = ir_token_ids[b].tolist()

        for open_tag, close_tag, span_type in tag_pairs:
            # Find all spans of this type
            i = 0
            while i < len(tokens):
                if tokens[i] == open_tag:
                    # Find matching close tag
                    try:
                        close_idx = tokens.index(close_tag, i + 1)
                        # Extract span (including tags)
                        span_tokens = ir_token_ids[b, i:close_idx+1]
                        span_hidden = ir_hidden_states[b, i:close_idx+1, :]

                        spans.append({
                            'batch_idx': b,
                            'span_type': span_type,
                            'span_hidden': span_hidden,
                            'span_tokens': span_tokens
                        })

                        i = close_idx + 1
                    except ValueError:
                        # No matching close tag found
                        break
                else:
                    i += 1

    return spans


def compute_local_cycle_loss(
    ir_token_ids: torch.Tensor,
    ir_hidden_states: torch.Tensor,
    input_snippets: torch.Tensor,
    ir_token_id_dict: dict,
    local_cycle_head: LocalCycleHead,
    sample_ratio: float = 0.1
) -> torch.Tensor:
    """
    Compute local cycle loss on a subset of IR spans.

    Args:
        ir_token_ids: IR token sequence (batch, ir_len)
        ir_hidden_states: Hidden states for IR (batch, ir_len, hidden_dim)
        input_snippets: Target HL snippets for each span (N, snippet_len)
        ir_token_id_dict: Dict with token IDs
        local_cycle_head: LocalCycleHead module
        sample_ratio: Fraction of spans to compute loss on (to keep it light)

    Returns:
        Cycle loss (scalar tensor)
    """
    # Extract spans
    spans = extract_ir_spans(ir_token_ids, ir_hidden_states, ir_token_id_dict)

    if len(spans) == 0:
        return torch.tensor(0.0, device=ir_token_ids.device)

    # Sample subset of spans
    num_samples = max(1, int(len(spans) * sample_ratio))
    import random
    sampled_spans = random.sample(spans, num_samples)

    # Compute cycle loss on sampled spans
    total_loss = 0.0
    for span_data in sampled_spans:
        span_hidden = span_data['span_hidden'].unsqueeze(0)  # (1, span_len, hidden_dim)
        batch_idx = span_data['batch_idx']

        # Get corresponding target snippet
        target_snippet = input_snippets[batch_idx].unsqueeze(0)  # (1, snippet_len)

        # Compute reconstruction loss
        output = local_cycle_head(span_hidden, target_snippet)
        if output['cycle_loss'] is not None:
            total_loss += output['cycle_loss']

    # Average over sampled spans
    avg_loss = total_loss / num_samples if num_samples > 0 else torch.tensor(0.0)

    return avg_loss


if __name__ == "__main__":
    # Test LocalCycleHead
    print("Testing LocalCycleHead...")

    batch_size = 2
    span_len = 5
    hidden_dim = 256
    vocab_size = 50000
    snippet_length = 10

    # Create dummy IR span hidden states
    ir_span_hidden = torch.randn(batch_size, span_len, hidden_dim)

    # Create dummy target snippet
    target_snippet = torch.randint(0, vocab_size, (batch_size, snippet_length))

    # Initialize module
    cycle_head = LocalCycleHead(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        snippet_length=snippet_length
    )

    # Forward pass
    output = cycle_head(ir_span_hidden, target_snippet)

    print(f"Snippet logits shape: {output['snippet_logits'].shape}")
    print(f"Cycle loss: {output['cycle_loss'].item():.4f}")

    # Test sampling
    sampled_snippet = cycle_head.sample_snippet(ir_span_hidden)
    print(f"Sampled snippet shape: {sampled_snippet.shape}")
