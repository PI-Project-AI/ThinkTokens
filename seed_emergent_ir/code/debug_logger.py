"""
Debug logger for diagnosing training collapse.

Implements:
A) Single-batch step dumps (every 500 steps)
B) Distribution & health metrics (every epoch)
C) Hard assertions (fail fast)
D) Control experiment toggles
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class DebugLogger:
    """Comprehensive debug logging for IR-CoT training."""

    def __init__(
        self,
        log_dir: str,
        tokenizer,
        ir_token_ids: Dict,
        num_samples: int = 5,
        step_frequency: int = 500
    ):
        """
        Args:
            log_dir: Directory to save debug logs
            tokenizer: Tokenizer for decoding
            ir_token_ids: IR token ID mapping
            num_samples: Number of examples to dump per step
            step_frequency: Log step dumps every N steps
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = tokenizer
        self.ir_token_ids = ir_token_ids
        self.num_samples = num_samples
        self.step_frequency = step_frequency

        # IR error rate tracking (for abort condition)
        self.ir_error_violations = 0  # Count consecutive batches with IR error > 20%

        # Early diversity guards (activate after step 500)
        self.low_utilization_violations = 0  # Count consecutive steps with util < 5%
        self.high_top1_violations = 0  # Count consecutive steps with top-1 > 50%

        # PHASE 0 CATASTROPHIC GUARDS: Only fail on extreme collapse
        self.phase0_catastrophic_util_violations = 0  # util < 1% for 200 steps
        self.phase0_catastrophic_top1_violations = 0  # top-1 > 70% for 50 steps

        # Epoch-level metrics accumulator
        self.epoch_metrics = {
            'answer_ce_values': [],
            'vq_loss_values': [],
            'tag_loss_values': [],
            'gradient_norms': [],
            'ir_integrity_rates': [],
            'codebook_utilizations': [],
            'coverage_losses': [],
            'temperatures': []
        }

    def should_log_step(self, step: int) -> bool:
        """Check if we should log this step."""
        return step % self.step_frequency == 0

    def log_step_dump(
        self,
        epoch: int,
        step: int,
        batch: Dict,
        outputs: Dict,
        model,
        device: str
    ):
        """
        A) Single-batch step dump with detailed diagnostics.

        Logs 5 random examples with:
        - HL input (truncated)
        - IR structure (target if available, generated)
        - IR validity flags
        - Answer generation (greedy)
        - Top-k predictions at answer/code positions
        - EOS probabilities
        - Token counts
        """
        batch_size = batch['input_ids'].shape[0]
        num_log = min(self.num_samples, batch_size)

        # Select random indices
        indices = torch.randperm(batch_size)[:num_log].tolist()

        dump_data = {
            'epoch': epoch,
            'step': step,
            'examples': []
        }

        for idx in indices:
            example_data = self._dump_single_example(
                idx, batch, outputs, model, device
            )
            dump_data['examples'].append(example_data)

        # Save to JSON
        dump_path = self.log_dir / f"debug_epoch{epoch}_step{step}.json"
        with open(dump_path, 'w') as f:
            json.dump(dump_data, f, indent=2)

        print(f"\n[DEBUG] Step dump saved to {dump_path}")

    def _dump_single_example(
        self,
        idx: int,
        batch: Dict,
        outputs: Dict,
        model,
        device: str
    ) -> Dict:
        """Dump detailed info for a single example."""

        # 1. HL Input (truncated to 128 chars)
        input_ids = batch['input_ids'][idx:idx+1]
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        input_truncated = input_text[:128] + "..." if len(input_text) > 128 else input_text

        # 2. IR Buffer (generated)
        ir_token_ids = outputs['ir_token_ids'][idx]
        ir_text = self.tokenizer.decode(ir_token_ids, skip_special_tokens=False)
        ir_structure = self._parse_ir_structure(ir_token_ids)

        # 3. IR Validity Flags
        ir_validity = self._check_ir_validity(ir_token_ids, ir_structure)

        # 4. Answer Generation (greedy)
        with torch.no_grad():
            gen_outputs = model.generate_answer(
                input_ids.to(device),
                attention_mask=batch['attention_mask'][idx:idx+1].to(device),
                temperature=0.1,  # Greedy
                max_answer_length=20
            )
            answer_ids = gen_outputs['answer_ids'][0]
            answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        # 5. Top-k at answer positions (using full forward)
        answer_logits = outputs['answer_logits']
        if answer_logits is not None and answer_logits.shape[0] > idx:
            answer_topk = self._get_topk_at_positions(
                answer_logits[idx], positions=[0, 1, 2], k=5
            )
        else:
            answer_topk = {"error": "No answer logits available"}

        # 6. Top-k at code positions (from IR generation)
        # We need to re-run IR generation to get code logits
        with torch.no_grad():
            ir_output = model.ir_generator(
                input_ids=input_ids.to(device),
                attention_mask=batch['attention_mask'][idx:idx+1].to(device),
                target_ir_ids=None,
                temperature=0.1
            )
            # Get code logits from metadata
            code_topk = self._extract_code_topk(ir_output, positions=[0, 1, 2])

        # 7. EOS probabilities at answer positions
        eos_probs = self._get_eos_probs(answer_logits[idx] if answer_logits is not None else None)

        # 8. Support size (non-masked logits) at answer positions
        support_sizes = self._compute_support_size(answer_logits[idx] if answer_logits is not None else None)

        # 9. Top-1 code frequency (for single-code collapse detection)
        top1_code_freq = self._compute_top1_code_frequency(
            ir_token_ids,
            model.ir_generator.code_start,
            model.ir_generator.code_end
        )

        # 10. Codebook utilization
        code_mask = (ir_token_ids >= model.ir_generator.code_start) & \
                    (ir_token_ids <= model.ir_generator.code_end)
        if code_mask.any():
            code_tokens = ir_token_ids[code_mask]
            unique_codes = torch.unique(code_tokens)
            codebook_util = len(unique_codes) / model.ir_generator.num_codes
        else:
            codebook_util = 0.0

        # 11. Counts
        answer_targets = batch['answer_ids'][idx].clone()
        answer_targets[answer_targets == model.pad_token_id] = -100
        num_answer_tokens_with_loss = (answer_targets != -100).sum().item()

        return {
            'index': idx,
            'hl_input': input_truncated,
            'ir_generated': {
                'text': ir_text,
                'structure': ir_structure,
                'validity': ir_validity
            },
            'answer_generated': answer_text,
            'answer_topk_positions': answer_topk,
            'code_topk_positions': code_topk,
            'eos_probs': eos_probs,
            'support_sizes': support_sizes,
            'codebook_metrics': {
                'utilization': codebook_util,
                'top1_code_frequency': top1_code_freq,
                'debias_gamma': self._compute_debias_gamma(model),
                'freq_ema_stats': self._compute_freq_ema_stats(model)
            },
            'counts': {
                'num_answer_tokens_with_loss': num_answer_tokens_with_loss,
                'answer_len': batch['answer_ids'][idx].shape[0],
                'ir_len': ir_token_ids.shape[0]
            }
        }

    def _parse_ir_structure(self, ir_token_ids: torch.Tensor) -> List[Dict]:
        """Parse IR into spans with tag, num_codes, codes."""
        structure = []
        current_span = None

        ir_ids = ir_token_ids.cpu().tolist()

        # Build reverse mapping for tags
        tag_names = {
            self.ir_token_ids['ir_start']: 'IR_START',
            self.ir_token_ids['ir_end']: 'IR_END',
            self.ir_token_ids['goal']: 'GOAL',
            self.ir_token_ids['goal_end']: '/GOAL',
            self.ir_token_ids['assume']: 'ASSUME',
            self.ir_token_ids['assume_end']: '/ASSUME',
            self.ir_token_ids['step']: 'STEP',
            self.ir_token_ids['step_end']: '/STEP',
            self.ir_token_ids['check']: 'CHECK',
            self.ir_token_ids['check_end']: '/CHECK',
            self.ir_token_ids['branch']: 'BRANCH',
            self.ir_token_ids['branch_end']: '/BRANCH'
        }

        code_start = min(self.ir_token_ids['codes'])
        code_end = max(self.ir_token_ids['codes'])

        for token_id in ir_ids:
            if token_id in tag_names:
                tag_name = tag_names[token_id]

                # End tags close current span
                if tag_name.startswith('/'):
                    if current_span is not None:
                        structure.append(current_span)
                        current_span = None
                else:
                    # Start new span
                    current_span = {
                        'tag': tag_name,
                        'num_codes': 0,
                        'codes': []
                    }
            elif code_start <= token_id <= code_end:
                if current_span is not None:
                    current_span['num_codes'] += 1
                    current_span['codes'].append(token_id)

        # Close final span if open
        if current_span is not None:
            structure.append(current_span)

        return structure

    def _check_ir_validity(
        self,
        ir_token_ids: torch.Tensor,
        structure: List[Dict]
    ) -> Dict:
        """Check IR validity flags."""
        ir_ids = ir_token_ids.cpu().tolist()

        # Check starts with IR_START
        starts_with_ir_start = (
            len(ir_ids) > 0 and
            ir_ids[0] == self.ir_token_ids['ir_start']
        )

        # Check balanced tags
        open_tags = []
        balanced = True
        for token_id in ir_ids:
            tag_name = None
            for key, val in self.ir_token_ids.items():
                if val == token_id and key in ['goal', 'assume', 'step', 'check', 'branch']:
                    tag_name = key
                    break
                elif val == token_id and key in ['goal_end', 'assume_end', 'step_end', 'check_end', 'branch_end']:
                    tag_name = key
                    break

            if tag_name and not tag_name.endswith('_end'):
                open_tags.append(tag_name)
            elif tag_name and tag_name.endswith('_end'):
                expected = tag_name.replace('_end', '')
                if open_tags and open_tags[-1] == expected:
                    open_tags.pop()
                else:
                    balanced = False
                    break

        if open_tags:
            balanced = False

        # Check codes per step (3-6 expected)
        codes_per_step = [span['num_codes'] for span in structure if span['tag'] in ['GOAL', 'STEP', 'ASSUME']]

        return {
            'starts_with_ir_start': starts_with_ir_start,
            'balanced_tags': balanced,
            'codes_per_step': codes_per_step,
            'spans_count': len(structure)
        }

    def _get_topk_at_positions(
        self,
        logits: torch.Tensor,
        positions: List[int],
        k: int = 5
    ) -> Dict:
        """Get top-k tokens at specific positions."""
        topk_data = {}

        for pos in positions:
            if pos < logits.shape[0]:
                pos_logits = logits[pos]
                topk_vals, topk_ids = torch.topk(pos_logits, k=k)

                tokens = [self.tokenizer.decode([tid.item()]) for tid in topk_ids]

                topk_data[f"pos_{pos}"] = {
                    'tokens': tokens,
                    'logits': topk_vals.cpu().tolist(),
                    'probs': F.softmax(pos_logits, dim=-1)[topk_ids].cpu().tolist()
                }

        return topk_data

    def _extract_code_topk(
        self,
        ir_output: Dict,
        positions: List[int]
    ) -> Dict:
        """Extract top-k code predictions from IR generation metadata."""
        # This would require modifying ir_generator_v2.py to store code logits
        # For now, return placeholder
        return {
            'note': 'Code topk requires IR generator modification',
            'positions': positions
        }

    def _get_eos_probs(self, answer_logits: Optional[torch.Tensor]) -> List[float]:
        """Get EOS probability at first 3 answer positions."""
        if answer_logits is None:
            return []

        eos_id = self.tokenizer.eos_token_id
        eos_probs = []

        for pos in [0, 1, 2]:
            if pos < answer_logits.shape[0]:
                probs = F.softmax(answer_logits[pos], dim=-1)
                eos_prob = probs[eos_id].item()
                eos_probs.append(eos_prob)

        return eos_probs

    def _compute_support_size(self, answer_logits: Optional[torch.Tensor]) -> List[int]:
        """
        Compute support size (# of logits > -1e9) at first 3 answer positions.
        Support size = number of non-masked logits. Should be >> 1 (e.g., >= 15).
        If support_size = 1, we accidentally masked everything except target.
        """
        if answer_logits is None:
            return []

        support_sizes = []
        for pos in [0, 1, 2]:
            if pos < answer_logits.shape[0]:
                pos_logits = answer_logits[pos]
                support_size = (pos_logits > -1e9).sum().item()
                support_sizes.append(support_size)

        return support_sizes

    def _compute_top1_code_frequency(self, ir_token_ids: torch.Tensor, code_start: int, code_end: int) -> float:
        """
        Compute top-1 code frequency across all codes in IR.
        Returns fraction of most common code. Should be < 0.3 (30%).
        If > 0.5, single code dominates (collapse).
        """
        code_mask = (ir_token_ids >= code_start) & (ir_token_ids <= code_end)
        if not code_mask.any():
            return 0.0

        code_tokens = ir_token_ids[code_mask]
        unique, counts = torch.unique(code_tokens, return_counts=True)

        if len(counts) == 0:
            return 0.0

        max_count = counts.max().item()
        total_count = counts.sum().item()

        return max_count / total_count if total_count > 0 else 0.0

    def _compute_debias_gamma(self, model) -> float:
        """Compute current debias gamma (decays during warm-start)."""
        if not hasattr(model.ir_generator.vq_tied_gen, 'vq'):
            return 0.0

        vq = model.ir_generator.vq_tied_gen.vq
        if not hasattr(vq, 'current_step') or not hasattr(vq, 'gumbel_steps'):
            return 0.0

        gamma = vq.gamma0 * max(0.0, 1.0 - vq.current_step / vq.gumbel_steps)
        return gamma

    def _compute_freq_ema_stats(self, model) -> Dict:
        """Compute min/max/mean of code frequency EMA."""
        if not hasattr(model.ir_generator.vq_tied_gen, 'vq'):
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0}

        vq = model.ir_generator.vq_tied_gen.vq
        if not hasattr(vq, 'code_freq_ema'):
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0}

        freq_ema = vq.code_freq_ema
        return {
            'min': freq_ema.min().item(),
            'max': freq_ema.max().item(),
            'mean': freq_ema.mean().item()
        }

    def accumulate_batch_metrics(
        self,
        loss_breakdown: Dict,
        ir_token_ids: torch.Tensor,
        temperature: float,
        model
    ):
        """B) Accumulate metrics for epoch-level summary."""
        # Answer CE
        self.epoch_metrics['answer_ce_values'].append(loss_breakdown['answer_loss'])

        # VQ loss
        self.epoch_metrics['vq_loss_values'].append(loss_breakdown['vq_loss'])

        # Tag loss
        self.epoch_metrics['tag_loss_values'].append(loss_breakdown['tag_lm_loss'])

        # Coverage loss
        self.epoch_metrics['coverage_losses'].append(loss_breakdown['diversity_loss'])

        # Temperature
        self.epoch_metrics['temperatures'].append(temperature)

        # Gradient norms (compute once per batch)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.epoch_metrics['gradient_norms'].append(total_norm)

        # Codebook utilization
        code_mask = (ir_token_ids >= model.ir_generator.code_start) & \
                    (ir_token_ids <= model.ir_generator.code_end)
        if code_mask.any():
            code_tokens = ir_token_ids[code_mask]
            unique_codes = torch.unique(code_tokens)
            utilization = len(unique_codes) / model.ir_generator.num_codes
            self.epoch_metrics['codebook_utilizations'].append(utilization)

    def log_epoch_metrics(self, epoch: int, ir_error_rate: float, val_accuracy: float = 0.0):
        """B) Save epoch-level distribution & health metrics."""

        def safe_mean(values):
            return np.mean(values) if values else 0.0

        def safe_std(values):
            return np.std(values) if values else 0.0

        metrics = {
            'epoch': epoch,
            'answer_ce': {
                'mean': safe_mean(self.epoch_metrics['answer_ce_values']),
                'std': safe_std(self.epoch_metrics['answer_ce_values']),
                'min': min(self.epoch_metrics['answer_ce_values']) if self.epoch_metrics['answer_ce_values'] else 0,
                'max': max(self.epoch_metrics['answer_ce_values']) if self.epoch_metrics['answer_ce_values'] else 0
            },
            'vq_loss': {
                'mean': safe_mean(self.epoch_metrics['vq_loss_values']),
                'std': safe_std(self.epoch_metrics['vq_loss_values'])
            },
            'tag_loss': {
                'mean': safe_mean(self.epoch_metrics['tag_loss_values']),
                'std': safe_std(self.epoch_metrics['tag_loss_values'])
            },
            'gradient_norms': {
                'mean': safe_mean(self.epoch_metrics['gradient_norms']),
                'max': max(self.epoch_metrics['gradient_norms']) if self.epoch_metrics['gradient_norms'] else 0
            },
            'ir_integrity_pct': (1 - ir_error_rate) * 100,
            'val_accuracy_pct': val_accuracy * 100,
            'codebook_utilization': {
                'mean': safe_mean(self.epoch_metrics['codebook_utilizations']),
                'std': safe_std(self.epoch_metrics['codebook_utilizations'])
            },
            'coverage_loss': {
                'mean': safe_mean(self.epoch_metrics['coverage_losses']),
                'std': safe_std(self.epoch_metrics['coverage_losses'])
            },
            'temperature': {
                'mean': safe_mean(self.epoch_metrics['temperatures']),
                'final': self.epoch_metrics['temperatures'][-1] if self.epoch_metrics['temperatures'] else 0
            }
        }

        # Save to JSON
        metrics_path = self.log_dir / f"epoch{epoch}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n[DEBUG] Epoch metrics saved to {metrics_path}")
        print(f"  Answer CE: {metrics['answer_ce']['mean']:.4f} ± {metrics['answer_ce']['std']:.4f}")
        print(f"  Codebook util: {metrics['codebook_utilization']['mean']:.2%}")
        print(f"  Gradient norm: {metrics['gradient_norms']['mean']:.2f}")

        # Clear for next epoch
        for key in self.epoch_metrics:
            self.epoch_metrics[key] = []

    def assert_training_health(
        self,
        batch_idx: int,
        tokens_with_loss: int,
        ir_error_rate: float,
        ir_token_ids: torch.Tensor,
        answer_logits: Optional[torch.Tensor],
        model,
        guards_mode: str = "FAIL"
    ):
        """
        C) Hard assertions to fail fast.

        Args:
            guards_mode: "WARN" (Phase 0, soft guards) or "FAIL" (Phase 1, hard guards)
        """

        # 1. Must have answer tokens with loss
        assert tokens_with_loss > 0, \
            f"[FAIL FAST] Batch {batch_idx}: ZERO answer tokens with loss (all PAD)"

        # 2. Answer support size check (fail fast if < 10)
        if answer_logits is not None and batch_idx > 100:  # Allow warm-up
            support_sizes = self._compute_support_size(answer_logits[0:1])  # Check first example
            if support_sizes and support_sizes[0] < 10:
                print(f"\n[CRITICAL] Batch {batch_idx}: Support size at answer pos 0 = {support_sizes[0]}")
                print(f"[CRITICAL] This means we masked the distribution to gold token → CE collapses to 0")
                assert False, \
                    f"[FAIL FAST] Batch {batch_idx}: Answer support size {support_sizes[0]} < 10 (masked to gold)"

        # 3. IR error rate tracking (abort if violated 10 batches in a row)
        # Only check after warm-start period (epoch >= 3) to allow IR learning
        if batch_idx > 100 and model.current_epoch >= 3:
            if ir_error_rate >= 0.2:
                self.ir_error_violations += 1
                if self.ir_error_violations >= 10:
                    assert False, \
                        f"[FAIL FAST] Batch {batch_idx}: IR error rate > 20% for 10 consecutive batches"
            else:
                self.ir_error_violations = 0  # Reset counter

        # 4. PHASE-AWARE DIVERSITY GUARDS
        code_mask = (ir_token_ids >= model.ir_generator.code_start) & \
                    (ir_token_ids <= model.ir_generator.code_end)
        if code_mask.any() and batch_idx >= 100:
            code_tokens = ir_token_ids[code_mask]
            unique_codes = torch.unique(code_tokens)
            utilization = len(unique_codes) / model.ir_generator.num_codes

            # Compute top-1 code frequency
            unique, counts = torch.unique(code_tokens, return_counts=True)
            if len(counts) > 0:
                max_count = counts.max().item()
                total_count = counts.sum().item()
                top1_freq = max_count / total_count if total_count > 0 else 0.0
            else:
                top1_freq = 0.0

            if guards_mode == "WARN":
                # ===== PHASE 0: Soft guards (WARN only) + catastrophic guards =====
                # Soft warnings for util < 5% or top-1 > 50%
                if utilization < 0.05:
                    print(f"\n[PHASE 0 WARN] Batch {batch_idx}: Util {utilization:.2%} < 5% (monitoring, not failing)")
                if top1_freq > 0.50:
                    print(f"\n[PHASE 0 WARN] Batch {batch_idx}: Top-1 freq {top1_freq:.2%} > 50% (monitoring, not failing)")

                # Catastrophic guards: FAIL only on extreme collapse
                # Util < 1% for 200 steps
                if utilization < 0.01:
                    self.phase0_catastrophic_util_violations += 1
                    if self.phase0_catastrophic_util_violations >= 200:
                        print(f"\n[PHASE 0 CATASTROPHIC] Batch {batch_idx}: Util {utilization:.2%} < 1% for 200 steps")
                        assert False, \
                            f"[FAIL FAST] Batch {batch_idx}: CATASTROPHIC util {utilization:.2%} < 1% for 200 steps"
                else:
                    self.phase0_catastrophic_util_violations = 0

                # Top-1 > 70% for 50 steps
                if top1_freq > 0.70:
                    self.phase0_catastrophic_top1_violations += 1
                    if self.phase0_catastrophic_top1_violations >= 50:
                        print(f"\n[PHASE 0 CATASTROPHIC] Batch {batch_idx}: Top-1 freq {top1_freq:.2%} > 70% for 50 steps")
                        assert False, \
                            f"[FAIL FAST] Batch {batch_idx}: CATASTROPHIC top-1 {top1_freq:.2%} > 70% for 50 steps"
                else:
                    self.phase0_catastrophic_top1_violations = 0

            else:  # guards_mode == "FAIL"
                # ===== PHASE 1: Hard guards (standard enforcement) =====
                # Track violations for 50 consecutive steps
                if utilization < 0.10:  # Phase 1: stricter threshold (10%)
                    self.low_utilization_violations += 1
                    if self.low_utilization_violations >= 50:
                        print(f"\n[CRITICAL] Batch {batch_idx}: Codebook utilization {utilization:.2%} < 10% for 50 steps")
                        assert False, \
                            f"[FAIL FAST] Batch {batch_idx}: Codebook utilization {utilization:.2%} < 10% for 50 steps (single-code collapse)"
                else:
                    self.low_utilization_violations = 0  # Reset counter

                if top1_freq > 0.50:
                    self.high_top1_violations += 1
                    if self.high_top1_violations >= 50:
                        print(f"\n[CRITICAL] Batch {batch_idx}: Top-1 code frequency {top1_freq:.2%} > 50% for 50 steps")
                        assert False, \
                            f"[FAIL FAST] Batch {batch_idx}: Top-1 code frequency {top1_freq:.2%} > 50% for 50 steps (single-code collapse)"
                else:
                    self.high_top1_violations = 0  # Reset counter


def run_control_experiment_random_ir(
    model,
    dataloader,
    device,
    tokenizer,
    temperature: float
):
    """D) Control: Force random IR and check accuracy drop."""
    print("\n[CONTROL] Running Random-IR experiment...")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            answer_ids = batch['answer_ids'].to(device)

            # Generate random IR
            batch_size = input_ids.shape[0]
            ir_len = 20  # Fixed length
            ir_token_ids = torch.randint(
                model.ir_generator.code_start,
                model.ir_generator.code_end + 1,
                (batch_size, ir_len),
                device=device
            )

            # Generate answer from random IR
            for j in range(batch_size):
                current_seq = ir_token_ids[j:j+1]

                for _ in range(20):
                    outputs = model.base_model(input_ids=current_seq)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    current_seq = torch.cat([current_seq, next_token], dim=1)

                    if (next_token == model.base_model.config.eos_token_id).all():
                        break

                # Extract answer
                pred_answer_ids = current_seq[:, ir_len:]
                pred_text = tokenizer.decode(pred_answer_ids[0], skip_special_tokens=True).strip()
                true_text = tokenizer.decode(answer_ids[j], skip_special_tokens=True).strip()

                if pred_text == true_text:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"[CONTROL] Random-IR Accuracy: {accuracy:.2%}")
    print(f"[CONTROL] Expected: <10% if IR is genuinely used")

    return accuracy
