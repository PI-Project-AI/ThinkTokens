"""
IR→Value Auxiliary Head

Predicts the numeric answer value directly from IR code embeddings to ground
the IR representation to task semantics.

This head encourages the model to encode answer-relevant information in the IR,
improving task accuracy while maintaining structural diversity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IRValueHead(nn.Module):
    """
    Tiny MLP that predicts the numeric answer from mean-pooled IR code embeddings.

    Can operate in two modes:
    1. Regression: Predict continuous value (for arithmetic)
    2. Classification: Predict discrete answer class (for larger answer spaces)

    For arithmetic, we'll use regression with MSE loss.
    """

    def __init__(
        self,
        code_dim: int,
        hidden_dim: int = 128,
        max_answer_value: int = 100,
        mode: str = 'regression'
    ):
        """
        Args:
            code_dim: Dimension of code embeddings from VQ
            hidden_dim: Hidden dimension for MLP
            max_answer_value: Maximum expected answer value (for normalization)
            mode: 'regression' or 'classification'
        """
        super().__init__()

        self.mode = mode
        self.max_answer_value = max_answer_value

        # Tiny 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1 if mode == 'regression' else max_answer_value)
        )

    def forward(self, ir_code_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ir_code_embeddings: [batch, num_codes, code_dim] - code embeddings from VQ

        Returns:
            predictions: [batch] - predicted answer values (regression)
                        or [batch, num_classes] - logits (classification)
        """
        # Mean-pool over code sequence dimension
        # [batch, num_codes, code_dim] -> [batch, code_dim]
        pooled = ir_code_embeddings.mean(dim=1)

        # Pass through MLP
        output = self.mlp(pooled)  # [batch, 1] or [batch, num_classes]

        if self.mode == 'regression':
            return output.squeeze(-1)  # [batch]
        else:
            return output  # [batch, num_classes]

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_strings: list = None
    ) -> tuple:
        """
        Compute loss and MAE metric.

        Args:
            predictions: [batch] - predicted values (regression)
                        or [batch, num_classes] - logits (classification)
            targets: [batch] - ground truth values (regression)
                    or [batch] - ground truth class indices (classification)
            target_strings: Optional list of target string answers for parsing

        Returns:
            loss: scalar loss tensor
            mae: mean absolute error (for logging)
        """
        if self.mode == 'regression':
            # MSE loss for regression
            loss = F.mse_loss(predictions, targets)

            # MAE for interpretability
            mae = (predictions - targets).abs().mean().item()

        else:
            # Cross-entropy for classification
            loss = F.cross_entropy(predictions, targets.long())

            # Accuracy as metric
            pred_classes = predictions.argmax(dim=-1)
            mae = (pred_classes != targets).float().mean().item()  # Error rate

        return loss, mae


def parse_numeric_answer(answer_string: str) -> float:
    """
    Parse a numeric answer from string.

    Handles:
    - Integers: "42" -> 42.0
    - Decimals: "3.14" -> 3.14
    - Negative: "-5" -> -5.0
    - Invalid: "abc" -> 0.0 (fallback)

    Args:
        answer_string: Answer string from dataset

    Returns:
        Parsed numeric value
    """
    try:
        return float(answer_string.strip())
    except ValueError:
        # Fallback for non-numeric answers
        return 0.0


def prepare_answer_targets(
    answer_strings: list,
    max_value: int = 100
) -> torch.Tensor:
    """
    Convert answer strings to numeric targets for IR→value head.

    Args:
        answer_strings: List of answer strings (e.g., ["42", "7", "-3"])
        max_value: Maximum expected value for normalization

    Returns:
        targets: [batch] tensor of parsed numeric values
    """
    targets = []
    for ans_str in answer_strings:
        value = parse_numeric_answer(ans_str)
        # Clamp to reasonable range to avoid extreme values
        value = max(-max_value, min(max_value, value))
        targets.append(value)

    return torch.tensor(targets, dtype=torch.float32)
