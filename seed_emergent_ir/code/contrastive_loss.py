"""
InfoNCE Contrastive Loss for HL-IR Alignment (V7-Lite)

Forces IR embeddings to align with high-level problem embeddings using
contrastive learning. Prevents IR collapse by ensuring IR captures input info.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class InfoNCEContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for HL-IR alignment.

    Given batch of (HL_emb, IR_emb) pairs, maximize agreement between
    positive pairs while minimizing agreement with negatives (other batch elements).
    """

    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 0.07,
        projection_dim: int = 128
    ):
        """
        Args:
            hidden_dim: Hidden dimension of model
            temperature: Softmax temperature for contrastive loss
            projection_dim: Dimension of projection head
        """
        super().__init__()
        self.temperature = temperature

        # Projection heads to normalize embeddings
        self.hl_proj = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        self.ir_proj = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(
        self,
        hl_embeddings: torch.Tensor,
        ir_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute InfoNCE contrastive loss.

        Args:
            hl_embeddings: High-level problem embeddings (batch, hidden_dim)
            ir_embeddings: IR embeddings (batch, hidden_dim)

        Returns:
            loss: InfoNCE loss
            metrics: Dict with diagnostic metrics
        """
        batch_size = hl_embeddings.shape[0]
        device = hl_embeddings.device

        # Project and normalize
        hl_proj = F.normalize(self.hl_proj(hl_embeddings), dim=-1)  # (batch, proj_dim)
        ir_proj = F.normalize(self.ir_proj(ir_embeddings), dim=-1)  # (batch, proj_dim)

        # Compute similarity matrix: (batch, batch)
        # sim[i,j] = similarity between HL_i and IR_j
        similarity = torch.matmul(hl_proj, ir_proj.T) / self.temperature

        # Positive pairs are on diagonal (HL_i matched with IR_i)
        labels = torch.arange(batch_size, device=device)

        # InfoNCE loss: cross-entropy with positive on diagonal
        # Each row: IR_i should match HL_i (maximize diagonal)
        loss_ir_to_hl = F.cross_entropy(similarity, labels)

        # Symmetric loss: HL_i should match IR_i
        loss_hl_to_ir = F.cross_entropy(similarity.T, labels)

        loss = (loss_ir_to_hl + loss_hl_to_ir) / 2.0

        # Diagnostic metrics
        with torch.no_grad():
            # Diagonal vs off-diagonal similarity
            diag_sim = torch.diag(similarity).mean().item()
            off_diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            off_diag_sim = similarity[off_diag_mask].mean().item()

            # Nearest neighbor accuracy: does argmax match diagonal?
            nn_correct = (similarity.argmax(dim=1) == labels).float().mean().item()

        metrics = {
            'contrastive_loss': loss.item(),
            'diag_similarity': diag_sim,
            'offdiag_similarity': off_diag_sim,
            'diag_minus_offdiag': diag_sim - off_diag_sim,
            'nn_accuracy': nn_correct
        }

        return loss, metrics


def extract_hl_embedding(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract high-level problem embedding from encoder hidden states.

    Uses mean pooling over valid tokens.

    Args:
        hidden_states: Hidden states (batch, seq_len, hidden_dim)
        attention_mask: Attention mask (batch, seq_len)

    Returns:
        hl_embedding: Pooled embedding (batch, hidden_dim)
    """
    # Mask and pool
    masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
    sum_hidden = masked_hidden.sum(dim=1)
    lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)

    hl_embedding = sum_hidden / lengths
    return hl_embedding


def extract_ir_embedding(
    ir_token_ids: torch.Tensor,
    code_embeddings: torch.Tensor,
    code_token_range: Tuple[int, int]
) -> torch.Tensor:
    """
    Extract IR embedding from generated IR tokens.

    Uses mean pooling over code embeddings only (ignore tags).

    Args:
        ir_token_ids: Generated IR token IDs (batch, ir_len)
        code_embeddings: VQ code embeddings (num_codes, code_dim)
        code_token_range: (code_start, code_end) token ID range

    Returns:
        ir_embedding: Pooled IR embedding (batch, code_dim)
    """
    batch_size = ir_token_ids.shape[0]
    device = ir_token_ids.device

    code_start, code_end = code_token_range

    # Mask for code tokens
    code_mask = (ir_token_ids >= code_start) & (ir_token_ids <= code_end)

    # Extract code indices
    code_indices = torch.where(
        code_mask,
        ir_token_ids - code_start,
        torch.tensor(0, device=device)
    )

    # Get code embeddings
    ir_embeddings = code_embeddings[code_indices]  # (batch, ir_len, code_dim)

    # Mean pool over code positions only
    masked_ir = ir_embeddings * code_mask.unsqueeze(-1)
    sum_ir = masked_ir.sum(dim=1)
    code_counts = code_mask.sum(dim=1, keepdim=True).clamp(min=1)

    ir_embedding = sum_ir / code_counts
    return ir_embedding


if __name__ == "__main__":
    print("InfoNCE Contrastive Loss module loaded.")
    print("Features: HL-IR alignment, projection heads, symmetric loss")
