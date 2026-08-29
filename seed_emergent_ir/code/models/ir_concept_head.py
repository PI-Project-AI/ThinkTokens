#!/usr/bin/env python3
"""
IR→Concept Head for V9: Multi-label classification of arithmetic concepts from IR codes.

Predicts semantic features of the problem from mean-pooled IR embeddings:
- num_terms (scalar regression)
- operation_types (3-class multi-hot)
- max_operand_magnitude (4-class classification)
- depth (2-class classification)
- has_carry_addition (binary classification)
- difficulty (3-class classification)
- parity (binary classification)
- sign (binary classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class IRConceptHead(nn.Module):
    """
    Concept bottleneck head for IR representations.

    Maps mean-pooled IR code embeddings to semantic concept predictions.
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Concept prediction heads
        # 1. num_terms: regression (1-5 range, but typically 2-3)
        self.num_terms_head = nn.Linear(hidden_dim, 1)

        # 2. operation_types: 3-class multi-hot [add, sub, mul]
        self.operation_types_head = nn.Linear(hidden_dim, 3)

        # 3. max_operand_magnitude: 4-class (buckets: <10, <20, <50, >=50)
        self.max_operand_magnitude_head = nn.Linear(hidden_dim, 4)

        # 4. depth: 2-class (1=flat, 2=has_parens)
        self.depth_head = nn.Linear(hidden_dim, 2)

        # 5. has_carry_addition: binary
        self.has_carry_head = nn.Linear(hidden_dim, 1)

        # 6. difficulty: 3-class (easy, medium, hard)
        self.difficulty_head = nn.Linear(hidden_dim, 3)

        # 7. parity: binary (even/odd)
        self.parity_head = nn.Linear(hidden_dim, 1)

        # 8. sign: binary (positive/negative)
        self.sign_head = nn.Linear(hidden_dim, 1)

        print(f"[IRConceptHead] in_dim={in_dim}, hidden_dim={hidden_dim}")
        print(f"[IRConceptHead] Concept outputs: num_terms(1), op_types(3), magnitude(4), "
              f"depth(2), carry(1), difficulty(3), parity(1), sign(1)")

    def forward(self, ir_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to predict all concepts.

        Args:
            ir_embeddings: Mean-pooled IR code embeddings (batch_size, in_dim)

        Returns:
            Dictionary of concept predictions (logits/values)
        """
        # Shared encoding
        h = self.encoder(ir_embeddings)  # (B, hidden_dim)

        # Predict all concepts
        predictions = {
            'num_terms': self.num_terms_head(h).squeeze(-1),  # (B,)
            'operation_types': self.operation_types_head(h),  # (B, 3)
            'max_operand_magnitude': self.max_operand_magnitude_head(h),  # (B, 4)
            'depth': self.depth_head(h),  # (B, 2)
            'has_carry_addition': self.has_carry_head(h).squeeze(-1),  # (B,)
            'difficulty': self.difficulty_head(h),  # (B, 3)
            'parity': self.parity_head(h).squeeze(-1),  # (B,)
            'sign': self.sign_head(h).squeeze(-1),  # (B,)
        }

        return predictions

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute concept prediction losses.

        Args:
            predictions: Output from forward()
            targets: Ground-truth concept labels

        Returns:
            total_loss: Weighted sum of all concept losses
            loss_dict: Individual losses for logging
        """
        losses = {}

        # 1. num_terms: MSE regression
        losses['num_terms'] = F.mse_loss(
            predictions['num_terms'],
            targets['num_terms'].float()
        )

        # 2. operation_types: multi-label BCE (3 binary classifiers)
        losses['operation_types'] = F.binary_cross_entropy_with_logits(
            predictions['operation_types'],
            targets['operation_types'].float()
        )

        # 3. max_operand_magnitude: 4-class CE
        losses['max_operand_magnitude'] = F.cross_entropy(
            predictions['max_operand_magnitude'],
            targets['max_operand_magnitude']
        )

        # 4. depth: 2-class CE (convert 1/2 to 0/1)
        depth_targets = (targets['depth'] - 1).clamp(0, 1)  # Map 1→0, 2→1
        losses['depth'] = F.cross_entropy(
            predictions['depth'],
            depth_targets
        )

        # 5. has_carry_addition: binary BCE
        losses['has_carry_addition'] = F.binary_cross_entropy_with_logits(
            predictions['has_carry_addition'],
            targets['has_carry_addition'].float()
        )

        # 6. difficulty: 3-class CE
        losses['difficulty'] = F.cross_entropy(
            predictions['difficulty'],
            targets['difficulty']
        )

        # 7. parity: binary BCE
        losses['parity'] = F.binary_cross_entropy_with_logits(
            predictions['parity'],
            targets['parity'].float()
        )

        # 8. sign: binary BCE
        losses['sign'] = F.binary_cross_entropy_with_logits(
            predictions['sign'],
            targets['sign'].float()
        )

        # Total loss (equal weighting for now)
        total_loss = sum(losses.values())

        # Convert to dict of floats for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def compute_accuracy(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute per-concept accuracy metrics.

        Args:
            predictions: Output from forward()
            targets: Ground-truth concept labels

        Returns:
            Dictionary of accuracy metrics
        """
        accs = {}

        with torch.no_grad():
            # 1. num_terms: MAE (not accuracy)
            accs['num_terms_mae'] = (predictions['num_terms'] - targets['num_terms'].float()).abs().mean().item()

            # 2. operation_types: multi-label accuracy (all 3 must match)
            op_preds = (torch.sigmoid(predictions['operation_types']) > 0.5).float()
            accs['operation_types'] = (op_preds == targets['operation_types'].float()).all(dim=1).float().mean().item()

            # 3. max_operand_magnitude: classification accuracy
            mag_preds = predictions['max_operand_magnitude'].argmax(dim=1)
            accs['max_operand_magnitude'] = (mag_preds == targets['max_operand_magnitude']).float().mean().item()

            # 4. depth: classification accuracy
            depth_targets = (targets['depth'] - 1).clamp(0, 1)
            depth_preds = predictions['depth'].argmax(dim=1)
            accs['depth'] = (depth_preds == depth_targets).float().mean().item()

            # 5. has_carry_addition: binary accuracy
            carry_preds = (torch.sigmoid(predictions['has_carry_addition']) > 0.5).float()
            accs['has_carry_addition'] = (carry_preds == targets['has_carry_addition'].float()).float().mean().item()

            # 6. difficulty: classification accuracy
            diff_preds = predictions['difficulty'].argmax(dim=1)
            accs['difficulty'] = (diff_preds == targets['difficulty']).float().mean().item()

            # 7. parity: binary accuracy
            parity_preds = (torch.sigmoid(predictions['parity']) > 0.5).float()
            accs['parity'] = (parity_preds == targets['parity'].float()).float().mean().item()

            # 8. sign: binary accuracy
            sign_preds = (torch.sigmoid(predictions['sign']) > 0.5).float()
            accs['sign'] = (sign_preds == targets['sign'].float()).float().mean().item()

            # Overall accuracy (average of all concept accuracies, excluding num_terms MAE)
            concept_accs = [accs[k] for k in ['operation_types', 'max_operand_magnitude', 'depth',
                                               'has_carry_addition', 'difficulty', 'parity', 'sign']]
            accs['overall'] = sum(concept_accs) / len(concept_accs)

        return accs
