"""
Balanced Code Sampler for Phase 0 VQ Bootstrap

Generates diverse code sequences to prime the VQ codebook during early training.
Ensures broad coverage of the codebook space before transitioning to emergent regime.
"""
import torch
import numpy as np
from typing import List, Tuple


class BalancedCodeSampler:
    """
    Samples balanced, diverse code sequences for VQ bootstrap phase.

    Strategy:
    - Rotating reservoir ensures all codes get used
    - Track coverage to verify we're exposing model to full codebook
    - Sample without replacement within spans for diversity
    """

    def __init__(
        self,
        num_codes: int = 512,
        min_codes_per_span: int = 3,
        max_codes_per_span: int = 6,
        device: str = "cuda"
    ):
        """
        Args:
            num_codes: Total codebook size
            min_codes_per_span: Minimum codes per span
            max_codes_per_span: Maximum codes per span
            device: Device for tensors
        """
        self.num_codes = num_codes
        self.min_codes = min_codes_per_span
        self.max_codes = max_codes_per_span
        self.device = device

        # Tracking for coverage stats
        self.codes_used = set()
        self.total_codes_sampled = 0

        # Reservoir for uniform sampling
        self.reservoir = list(range(num_codes))
        self.reservoir_idx = 0

    def sample_span_codes(self, span_length: int) -> List[int]:
        """
        Sample codes for a single span with no consecutive repeats.

        Args:
            span_length: Number of codes to sample

        Returns:
            List of code indices
        """
        codes = []
        prev_code = None

        for _ in range(span_length):
            # Refill reservoir if exhausted
            if self.reservoir_idx >= len(self.reservoir):
                np.random.shuffle(self.reservoir)
                self.reservoir_idx = 0

            # Sample next code from reservoir
            code = self.reservoir[self.reservoir_idx]
            self.reservoir_idx += 1

            # Ensure no consecutive repeats
            if code == prev_code and self.reservoir_idx < len(self.reservoir):
                # Swap with next in reservoir to avoid repeat
                next_code = self.reservoir[self.reservoir_idx]
                self.reservoir_idx += 1
                code = next_code

            codes.append(code)
            prev_code = code

        return codes

    def sample_batch(
        self,
        batch_size: int,
        num_spans_per_example: int
    ) -> torch.Tensor:
        """
        Sample balanced codes for a batch of IR sequences.

        Each example gets num_spans_per_example spans, each with 3-6 codes.

        Args:
            batch_size: Number of examples in batch
            num_spans_per_example: Number of spans per example

        Returns:
            Tensor of shape (batch_size, total_codes) with code indices
        """
        batch_codes = []

        for _ in range(batch_size):
            example_codes = []

            for _ in range(num_spans_per_example):
                # Random span length
                span_length = np.random.randint(self.min_codes, self.max_codes + 1)

                # Sample codes for this span
                span_codes = self.sample_span_codes(span_length)
                example_codes.extend(span_codes)

                # Track coverage
                self.codes_used.update(span_codes)
                self.total_codes_sampled += len(span_codes)

            batch_codes.append(example_codes)

        # Pad to max length
        max_len = max(len(codes) for codes in batch_codes)
        padded_codes = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)

        for i, codes in enumerate(batch_codes):
            padded_codes[i, :len(codes)] = torch.tensor(codes, dtype=torch.long, device=self.device)

        return padded_codes

    def get_coverage_stats(self) -> dict:
        """
        Get statistics on codebook coverage.

        Returns:
            Dict with coverage metrics
        """
        coverage = len(self.codes_used) / self.num_codes
        avg_usage = self.total_codes_sampled / max(len(self.codes_used), 1)

        return {
            'codes_covered': len(self.codes_used),
            'coverage_pct': coverage * 100,
            'total_sampled': self.total_codes_sampled,
            'avg_usage_per_code': avg_usage
        }

    def reset_coverage(self):
        """Reset coverage tracking (e.g., for new epoch)."""
        self.codes_used = set()
        self.total_codes_sampled = 0


if __name__ == "__main__":
    # Test sampler
    sampler = BalancedCodeSampler(num_codes=512, min_codes_per_span=3, max_codes_per_span=6)

    # Sample 100 batches
    for _ in range(100):
        batch = sampler.sample_batch(batch_size=8, num_spans_per_example=4)
        print(f"Batch shape: {batch.shape}")

    # Check coverage
    stats = sampler.get_coverage_stats()
    print(f"\nCoverage stats after 100 batches:")
    print(f"  Codes covered: {stats['codes_covered']}/512 ({stats['coverage_pct']:.1f}%)")
    print(f"  Total sampled: {stats['total_sampled']}")
    print(f"  Avg usage per code: {stats['avg_usage_per_code']:.1f}")
