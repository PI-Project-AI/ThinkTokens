"""
Vector Quantizer for IR-CoT.

Maps continuous hidden states to discrete code IDs from a learned codebook.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with straight-through estimator.

    Takes continuous hidden states and quantizes them to discrete codes.
    Uses L2-nearest neighbor for code selection and straight-through estimator
    for gradients.
    """

    def __init__(
        self,
        num_codes: int = 512,
        code_dim: int = 128,
        beta: float = 0.25,
        use_unit_norm: bool = True,
        use_gumbel_warmstart: bool = False,
        gumbel_tau: float = 0.6,  # Deprecated: use tau_init
        tau_init: float = 1.2,
        tau_final: float = 0.6,
        gumbel_steps: int = 3000
    ):
        """
        Args:
            num_codes: Number of discrete codes in codebook
            code_dim: Dimension of each code embedding
            beta: Commitment loss weight (encourages encoder to commit)
            use_unit_norm: If True, unit-normalize codebook and inputs
            use_gumbel_warmstart: If True, use Gumbel-Softmax during warm-start
            gumbel_tau: DEPRECATED - use tau_init instead (kept for backward compat)
            tau_init: Initial Gumbel temperature (1.2 = more exploration)
            tau_final: Final Gumbel temperature (0.6 = less exploration)
            gumbel_steps: Number of steps to use Gumbel before switching to VQ
        """
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.use_unit_norm = use_unit_norm

        # V7-Lite: Gumbel-Softmax warm-start with tau annealing
        self.use_gumbel_warmstart = use_gumbel_warmstart
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.gumbel_steps = gumbel_steps
        self.current_step = 0

        # Logit debias for code collapse (warm-start only)
        self.gamma0 = 3.0  # Initial debias strength (increased from 1.0)
        self.freq_beta = 0.99  # EMA decay for frequency tracking
        self.register_buffer('code_freq_ema', torch.full((num_codes,), 1.0 / num_codes))

        # Learned codebook: (num_codes, code_dim)
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

        if use_unit_norm:
            # Initialize with unit norm
            self.codebook.weight.data = F.normalize(self.codebook.weight.data, dim=1)

        # Store soft probabilities for diversity loss (Gumbel warm-start only)
        self._last_y_soft = None

    def forward(self, z: torch.Tensor, use_gumbel_override: bool = None, training_step: int = None) -> tuple:
        """
        Quantize hidden states to discrete codes.

        Args:
            z: Hidden states (batch, seq_len, hidden_dim)
               Must be projected to code_dim before calling this
            use_gumbel_override: Override use_gumbel_warmstart (for testing)
            training_step: Global training step (for Gumbel warm-start schedule)

        Returns:
            z_q: Quantized vectors (same shape as z)
            vq_loss: VQ commitment loss
            indices: Discrete code indices (batch, seq_len)
        """
        # Ensure input matches code dimension
        assert z.shape[-1] == self.code_dim, \
            f"Input dim {z.shape[-1]} != code_dim {self.code_dim}. Add projection layer."

        # Flatten for quantization
        batch_size, seq_len = z.shape[0], z.shape[1]
        z_flat = z.view(-1, self.code_dim)  # (B*S, D)

        # Unit-normalize if enabled
        if self.use_unit_norm:
            z_flat = F.normalize(z_flat, dim=1)
            codebook_weight = F.normalize(self.codebook.weight, dim=1)
        else:
            codebook_weight = self.codebook.weight

        # Compute L2 distances to all codebook entries
        # dist = ||z - e||^2 = ||z||^2 - 2*z*e + ||e||^2
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(codebook_weight ** 2, dim=1) -
            2 * torch.matmul(z_flat, codebook_weight.t())
        )  # (B*S, num_codes)

        # Use training_step if provided, otherwise fall back to self.current_step
        current_step = training_step if training_step is not None else self.current_step

        # V7-Lite: Use Gumbel-Softmax during warm-start phase
        use_gumbel = use_gumbel_override if use_gumbel_override is not None else \
                     (self.use_gumbel_warmstart and current_step < self.gumbel_steps and self.training)

        if use_gumbel:
            # Convert distances to logits (negative distance = similarity)
            logits = -distances  # (B*S, num_codes)

            # ============= LOGIT DEBIAS (warm-start only) =============
            # Penalize frequently-selected codes to prevent collapse
            if self.training and current_step < self.gumbel_steps:
                # Linear decay: gamma = gamma0 * (1 - step/gumbel_steps)
                gamma = self.gamma0 * max(0.0, 1.0 - current_step / self.gumbel_steps)

                # Clamp EMA frequencies to avoid NaN
                freq_clamped = self.code_freq_ema.clamp(1e-6, 1.0)

                # Log-prior: higher for rare codes, lower for frequent ones
                log_prior = -torch.log(freq_clamped + 1e-3)  # (num_codes,)

                # Apply debias: boost rare codes, penalize frequent ones
                debias = gamma * log_prior  # (num_codes,)
                logits = logits + debias.unsqueeze(0)  # Broadcast to (B*S, num_codes)

            # ============= TAU ANNEALING =============
            # Linear anneal: tau_init (1.2) -> tau_final (0.6) over gumbel_steps
            progress = min(1.0, current_step / self.gumbel_steps)
            tau = self.tau_init + (self.tau_final - self.tau_init) * progress

            # Apply Gumbel-Softmax with annealed tau and straight-through
            y_soft = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)  # (B*S, num_codes)
            indices_flat = y_soft.argmax(dim=-1)  # (B*S,)
            indices = indices_flat.view(batch_size, seq_len)  # (B, S)

            # Store y_soft for diversity loss (training only)
            self._last_y_soft = y_soft if self.training else None

            # ============= UPDATE EMA FREQUENCY (detached) =============
            if self.training:
                with torch.no_grad():
                    batch_hist = torch.bincount(indices_flat.detach(), minlength=self.num_codes).float()
                    batch_hist = batch_hist / batch_hist.sum().clamp_min(1.0)
                    self.code_freq_ema = self.freq_beta * self.code_freq_ema + (1 - self.freq_beta) * batch_hist

            # P5 FIX: Use actual parameter for gradient flow (not normalized copy)
            # y_soft @ self.codebook.weight ensures gradients flow to the nn.Parameter
            E = self.codebook.weight  # nn.Parameter, requires_grad=True
            z_q_flat = y_soft @ E  # (B*S, D) - gradient flows to both y_soft and E
            z_q = z_q_flat.view(batch_size, seq_len, self.code_dim)
        else:
            # Standard VQ: argmin distance
            indices_flat = torch.argmin(distances, dim=1)  # (B*S,)
            indices = indices_flat.view(batch_size, seq_len)  # (B, S)

            # Look up quantized vectors
            z_q = self.codebook(indices_flat).view(batch_size, seq_len, self.code_dim)

        # VQ loss: commitment loss (encourages encoder to commit to codes)
        # Loss = ||sg[z_q] - z||^2 + beta * ||z_q - sg[z]||^2
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = commitment_loss + self.beta * codebook_loss

        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()

        return z_q, vq_loss, indices

    def compute_diversity_loss(self) -> torch.Tensor:
        """
        Compute entropy-based diversity loss from last forward pass.
        Encourages uniform code distribution during Gumbel warm-start.

        Returns:
            diversity_loss: (max_entropy - actual_entropy)
        """
        if self._last_y_soft is None or not self.training:
            return torch.tensor(0.0, device=self.codebook.weight.device)

        # Batch-averaged soft distribution: (num_codes,)
        code_dist = self._last_y_soft.mean(dim=0)  # Average over batch

        # Compute entropy: H = -sum(p * log(p))
        entropy = -(code_dist * (code_dist + 1e-9).log()).sum()

        # Maximum entropy (uniform distribution)
        max_entropy = torch.log(torch.tensor(float(self.num_codes), device=entropy.device))

        # Loss = (max_H - H), so lower entropy = higher loss
        diversity_loss = max_entropy - entropy

        return diversity_loss

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Get discrete code indices without full forward pass (for inference)."""
        z_flat = z.view(-1, self.code_dim)

        # Unit-normalize if enabled
        if self.use_unit_norm:
            z_flat = F.normalize(z_flat, dim=1)
            codebook_weight = F.normalize(self.codebook.weight, dim=1)
        else:
            codebook_weight = self.codebook.weight

        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(codebook_weight ** 2, dim=1) -
            2 * torch.matmul(z_flat, codebook_weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        return indices.view(z.shape[0], z.shape[1])

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up code embeddings from indices."""
        return self.codebook(indices)

    def compute_utilization(self, indices: torch.Tensor) -> dict:
        """
        Compute codebook utilization statistics.

        Args:
            indices: Code indices from a batch (batch, seq_len)

        Returns:
            Dictionary with utilization metrics
        """
        flat_indices = indices.view(-1)
        unique_codes = torch.unique(flat_indices)
        num_unique = len(unique_codes)
        utilization = num_unique / self.num_codes

        # Per-code frequency
        code_counts = torch.bincount(flat_indices, minlength=self.num_codes)

        return {
            'num_unique_codes': num_unique,
            'total_codes': self.num_codes,
            'utilization': utilization,
            'used_codes': sorted(unique_codes.tolist()),
            'code_frequencies': code_counts.cpu().numpy()
        }


class ProjectionVQ(nn.Module):
    """
    VectorQuantizer with built-in projection layer.

    Automatically projects hidden states to code dimension before quantization.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_codes: int = 512,
        code_dim: int = 128,
        beta: float = 0.25,
        use_unit_norm: bool = True,
        use_gumbel_warmstart: bool = False,
        gumbel_tau: float = 0.6,  # Deprecated
        tau_init: float = 1.2,
        tau_final: float = 0.6,
        gumbel_steps: int = 3000
    ):
        """
        Args:
            hidden_dim: Input hidden dimension (e.g., Pythia hidden size)
            num_codes: Number of discrete codes
            code_dim: Code embedding dimension
            beta: VQ commitment loss weight
            use_unit_norm: If True, unit-normalize codebook and inputs
            use_gumbel_warmstart: If True, use Gumbel-Softmax during warm-start
            gumbel_tau: DEPRECATED - use tau_init instead
            tau_init: Initial Gumbel temperature (1.2)
            tau_final: Final Gumbel temperature (0.6)
            gumbel_steps: Number of steps to use Gumbel before switching to VQ
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim

        # Projection layer: hidden_dim → code_dim
        self.projection = nn.Linear(hidden_dim, code_dim, bias=False)

        # Vector quantizer with Gumbel support and tau annealing
        self.vq = VectorQuantizer(
            num_codes, code_dim, beta, use_unit_norm,
            use_gumbel_warmstart, gumbel_tau, tau_init, tau_final, gumbel_steps
        )

    def forward(self, hidden_states: torch.Tensor, training_step: int = None) -> tuple:
        """
        Project and quantize hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            training_step: Global training step (passed to VQ)

        Returns:
            z_q: Quantized vectors (batch, seq_len, code_dim)
            vq_loss: VQ loss
            indices: Code indices (batch, seq_len)
        """
        # Project to code dimension
        z = self.projection(hidden_states)  # (B, S, code_dim)

        # Quantize
        z_q, vq_loss, indices = self.vq(z, training_step=training_step)

        return z_q, vq_loss, indices

    def quantize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get code indices without full forward pass."""
        z = self.projection(hidden_states)
        return self.vq.quantize(z)

    def compute_utilization(self, indices: torch.Tensor) -> dict:
        """Compute codebook utilization."""
        return self.vq.compute_utilization(indices)


if __name__ == "__main__":
    # Test VQ module
    print("Testing VectorQuantizer...")

    batch_size = 4
    seq_len = 10
    code_dim = 128
    num_codes = 512

    # Create dummy hidden states
    z = torch.randn(batch_size, seq_len, code_dim)

    # Initialize VQ
    vq = VectorQuantizer(num_codes=num_codes, code_dim=code_dim)

    # Forward pass
    z_q, vq_loss, indices = vq(z)

    print(f"Input shape: {z.shape}")
    print(f"Output shape: {z_q.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Unique codes used: {len(torch.unique(indices))}/{num_codes}")

    # Test utilization
    util = vq.compute_utilization(indices)
    print(f"Codebook utilization: {util['utilization']:.2%}")

    # Test ProjectionVQ
    print("\nTesting ProjectionVQ...")
    hidden_dim = 512
    z_hidden = torch.randn(batch_size, seq_len, hidden_dim)

    proj_vq = ProjectionVQ(hidden_dim=hidden_dim, num_codes=num_codes, code_dim=code_dim)
    z_q, vq_loss, indices = proj_vq(z_hidden)

    print(f"Input shape: {z_hidden.shape}")
    print(f"Output shape: {z_q.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
