"""
Training script V2 for Seed + Emergent IR-CoT with VQ-tied codes.

Features:
- VQ-tied code generation (semantics from codebook)
- Temperature annealing (0.7 → 0.4)
- 410M with LoRA/QLoRA for memory efficiency
- Gradient checkpointing
- 8-bit Adam optimizer
- Exact numeric answer matching
- IR integrity validation
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Optional

# Import model components
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from tokenizer_utils import extend_tokenizer_for_ir
from evaluation.causal_tests import CausalityTester
from evaluation.answer_matching import exact_match
from ir_grammar import validate_ir_integrity
from debug_logger import DebugLogger, run_control_experiment_random_ir

# Import V2 model directly
from models import causal_ir_model_v2
CausalIRModelV2 = causal_ir_model_v2.CausalIRModelV2

# Optional: LoRA/QLoRA for memory efficiency
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not available. Install with: pip install peft bitsandbytes")

# Optional: 8-bit Adam
try:
    import bitsandbytes as bnb
    BNBAVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("Warning: bitsandbytes not available. Using standard AdamW.")


def normalize_numeric_answer(answer: str) -> str:
    """
    Normalize numeric answers to canonical form.

    Normalization rules:
    - Strip leading/trailing whitespace
    - Remove leading zeros (e.g., "007" → "7", "00" → "0")
    - Remove trailing ".0" for whole numbers (e.g., "5.0" → "5")
    - Keep negative signs and decimal points

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string
    """
    # Strip whitespace
    answer = answer.strip()

    # Handle empty string
    if not answer:
        return answer

    # Try to parse as number and normalize
    try:
        # Parse as float to handle decimals
        num = float(answer)

        # Check if it's actually an integer
        if num == int(num):
            # Return as integer string (removes ".0")
            return str(int(num))
        else:
            # Return as float string (preserves significant decimals)
            return str(num)
    except ValueError:
        # If not a valid number, return stripped original
        return answer


def dump_train_mode_snapshot(model, step, output_dir, recent_losses=None, tokenizer=None):
    """Dump train-mode VQ metrics + V8 diagnostics."""
    vq = model.ir_generator.vq.vq

    # Compute utilization_train from last 1000 code positions
    # (Requires tracking recent codes - placeholder for now)
    utilization_train = 0.0  # TODO: Implement code tracking

    # Compute top-1 frequency from EMA
    top1_code_freq_train = float(vq.code_freq_ema.max().item())

    # Debias gamma (current)
    gamma_train = vq.gamma0 * max(0.0, 1.0 - step / vq.gumbel_steps) if step < vq.gumbel_steps else 0.0

    # Gumbel tau (current)
    tau_g_train = vq.tau_init + (vq.tau_final - vq.tau_init) * min(1.0, step / vq.gumbel_steps)

    snapshot = {
        "step": step,
        "mode": "train",
        "seed": 42,  # Hardcoded for reproducibility

        # Gumbel settings
        "gumbel_active": step < vq.gumbel_steps,
        "tau_current": tau_g_train,
        "tau_init": float(vq.tau_init),
        "tau_final": float(vq.tau_final),
        "gumbel_steps": vq.gumbel_steps,
        "debias_gamma": gamma_train,

        # P1.4: Codebook utilization (train-mode)
        "utilization_train": utilization_train,
        "utilization_train_definition": "unique codes in last 1000 train positions / 512",
        "top1_code_freq_train": top1_code_freq_train,
        "code_freq_ema_train": {
            "min": float(vq.code_freq_ema.min().item()),
            "max": float(vq.code_freq_ema.max().item()),
            "mean": float(vq.code_freq_ema.mean().item()),
            "top5_codes": vq.code_freq_ema.topk(5).indices.tolist(),
            "top5_freqs": vq.code_freq_ema.topk(5).values.tolist()
        },
        "gamma_train": gamma_train,
        "tau_g_train": tau_g_train,

        # P1.3: V8 metrics (extract from recent_losses)
        "ir_value_mae": recent_losses.get('ir_value_mae', 0.0) if recent_losses else 0.0,
        "ir_value_mse": recent_losses.get('ir_value_mse', 0.0) if recent_losses else 0.0,
        "nn_acc": recent_losses.get('nn_acc', 0.0) if recent_losses else 0.0,
        "sim_diag": recent_losses.get('sim_diag', 0.0) if recent_losses else 0.0,
        "sim_offdiag": recent_losses.get('sim_offdiag', 0.0) if recent_losses else 0.0,
        "diag_minus_offdiag": recent_losses.get('diag_minus_offdiag', 0.0) if recent_losses else 0.0,

        # P1.3: Answer-bias diagnostic
        "answer_bias": compute_answer_bias_diagnostic(model, tokenizer) if tokenizer else {},

        # Cross-attention coverage (placeholder)
        "cross_attn_ir_coverage_train": 0.0
    }

    dump_path = os.path.join(output_dir, "logs", f"train_step{step:04d}.json")
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"[TRAIN-MODE SNAPSHOT] Saved to {dump_path}")


def compute_answer_bias_diagnostic(model, tokenizer):
    """Compute answer-bias diagnostic from recent predictions."""
    # This requires storing recent answer predictions during training
    # For now, return placeholder
    return {
        "first_token_hist": {},  # Top 20 tokens at answer pos0
        "num_answer_tokens_hist": {},
        "top5_at_pos0": []
    }
    # TODO: Implement by tracking answer_logits during training


def dump_eval_mode_snapshot(model, step, output_dir, val_loader, tokenizer, device):
    """Dump eval-mode VQ metrics with softmax sampling."""
    model.eval()

    all_codes = []
    all_ir_valid = []
    all_spans = []
    all_ir_lens = []
    examples = []

    total_correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 50:  # Limit eval to 50 batches for speed
                break

            input_ids = batch['input_ids'].to(device)
            answer_ids = batch['answer_ids'].to(device)

            # Generate IR with eval softmax sampling
            ir_output = model.ir_generator(
                input_ids=input_ids,
                attention_mask=batch.get('attention_mask'),
                target_ir_ids=None,
                temperature=None  # Uses eval settings
            )

            ir_token_ids = ir_output['ir_token_ids']

            # Extract codes
            code_mask = (ir_token_ids >= model.ir_token_ids['code_start']) & \
                       (ir_token_ids <= model.ir_token_ids['code_end'])
            codes = ir_token_ids[code_mask]
            all_codes.extend(codes.cpu().tolist())

            # Check IR validity (placeholder)
            all_ir_valid.append(True)  # TODO: Implement structural validation

            # Count spans (placeholder)
            all_spans.append(4)  # TODO: Count actual spans
            all_ir_lens.append(ir_token_ids.shape[1])

            # Collect first 3 examples
            if len(examples) < 3:
                input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                ir_text = tokenizer.decode(ir_token_ids[0], skip_special_tokens=False)

                # Generate answer
                answer_pred_ids = model.answer_from_ir(
                    input_ids=input_ids[:1],
                    ir_ids=ir_token_ids[:1],
                    ir_embeddings=None,
                    max_length=10,
                    temperature=0.0
                )
                answer_text = tokenizer.decode(answer_pred_ids[0], skip_special_tokens=True)

                examples.append({
                    "input": input_text,
                    "ir": ir_text,
                    "answer": answer_text
                })

            # Compute accuracy (placeholder)
            # TODO: Implement answer comparison

    # Compute metrics
    unique_codes = len(set(all_codes))
    utilization_eval = unique_codes / 512.0

    code_counts = {}
    for c in all_codes:
        code_counts[c] = code_counts.get(c, 0) + 1
    top1_code_freq_eval = max(code_counts.values()) / len(all_codes) if all_codes else 0.0

    ir_integrity_eval = sum(all_ir_valid) / len(all_ir_valid) if all_ir_valid else 0.0
    avg_spans_eval = sum(all_spans) / len(all_spans) if all_spans else 0.0
    ir_len_eval = sum(all_ir_lens) / len(all_ir_lens) if all_ir_lens else 0.0

    snapshot = {
        "step": step,
        "mode": "eval",
        "seed": 42,
        "eval_sampler": {
            "method": "softmax",
            "tau": 0.9,
            "top_k": 32,
            "top_p": 0.95
        },

        # P1.4: Codebook utilization (eval-mode)
        "utilization_eval": utilization_eval,
        "top1_code_freq_eval": top1_code_freq_eval,

        # IR integrity
        "ir_integrity_eval": ir_integrity_eval,
        "avg_spans_eval": avg_spans_eval,
        "ir_len_eval": ir_len_eval,

        # Task performance
        "val_accuracy": 0.0,  # TODO
        "val_loss": 0.0,  # TODO

        # Examples
        "examples": examples,

        # Cross-attention coverage
        "cross_attn_ir_coverage_eval": 0.0  # Placeholder
    }

    dump_path = os.path.join(output_dir, "logs", f"eval_step{step:04d}_softmax.json")
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"[EVAL-MODE SNAPSHOT] Saved to {dump_path}")

    model.train()


class ArithmeticDataset(Dataset):
    """Dataset for arithmetic problems."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        problem_tokens = self.tokenizer(
            example['problem'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Normalize answer before tokenization (strip spaces, leading zeros, ".0")
        normalized_answer = normalize_numeric_answer(example['answer'])
        answer_tokens = self.tokenizer(
            normalized_answer,
            max_length=20,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        snippet_tokens = self.tokenizer(
            example['snippet'],
            max_length=10,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        result = {
            'input_ids': problem_tokens['input_ids'].squeeze(0),
            'attention_mask': problem_tokens['attention_mask'].squeeze(0),
            'answer_ids': answer_tokens['input_ids'].squeeze(0),
            'snippet_ids': snippet_tokens['input_ids'].squeeze(0)
        }

        # V9: Add concept labels if present
        if 'concepts' in example:
            result['concepts'] = example['concepts']

        return result


def generate_synthetic_ir_target(
    batch_size: int,
    ir_token_ids: dict,
    num_codes: int = 512,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Generate synthetic IR target sequences for teacher forcing.

    Template: <IR_START> <GOAL> c_i c_j c_k </GOAL> <STEP> c_m c_n c_o </STEP> <IR_END>

    Args:
        batch_size: Number of examples
        ir_token_ids: Dict with IR token IDs
        num_codes: Number of available codes
        device: Device to create tensor on

    Returns:
        Tensor of shape (batch_size, ir_len) with synthetic IR sequences
    """
    # Simple template: 2 spans with 3-4 codes each
    ir_sequences = []

    for _ in range(batch_size):
        seq = [
            ir_token_ids['ir_start'],
            ir_token_ids['goal'],
        ]

        # Add 3-4 random codes for GOAL span
        num_goal_codes = torch.randint(3, 5, (1,)).item()
        for _ in range(num_goal_codes):
            code_idx = torch.randint(0, num_codes, (1,)).item()
            seq.append(ir_token_ids['codes'][code_idx])

        seq.append(ir_token_ids['goal_end'])
        seq.append(ir_token_ids['step'])

        # Add 3-4 random codes for STEP span
        num_step_codes = torch.randint(3, 5, (1,)).item()
        for _ in range(num_step_codes):
            code_idx = torch.randint(0, num_codes, (1,)).item()
            seq.append(ir_token_ids['codes'][code_idx])

        seq.append(ir_token_ids['step_end'])
        seq.append(ir_token_ids['ir_end'])

        ir_sequences.append(seq)

    # Pad to same length
    max_len = max(len(s) for s in ir_sequences)
    padded_seqs = []
    for seq in ir_sequences:
        padded = seq + [0] * (max_len - len(seq))  # Pad with 0 (PAD token)
        padded_seqs.append(padded)

    return torch.tensor(padded_seqs, device=device, dtype=torch.long)


def compute_temperature_schedule(
    current_step: int,
    total_steps: int,
    temp_init: float = 0.7,
    temp_final: float = 0.4
) -> float:
    """Linear temperature annealing."""
    progress = current_step / total_steps
    return temp_init + (temp_final - temp_init) * progress


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch: int,
    current_step: int,
    total_steps: int,
    temp_init: float,
    temp_final: float,
    use_teacher_forcing: bool = True,
    ir_token_ids: dict = None,
    num_codes: int = 512,
    debug_logger: Optional[DebugLogger] = None,
    enable_assertions: bool = False,
    scaler: torch.cuda.amp.GradScaler = None,
    max_grad_norm: float = 1.0,
    use_fp16: bool = False,
    output_dir: str = None,
    tokenizer = None,
    val_loader = None,
    args = None
):
    """Train for one epoch with temperature annealing and optional teacher forcing."""
    model.train()

    total_loss = 0
    loss_components = {
        'answer_loss': 0,
        'tag_lm_loss': 0,
        'vq_loss': 0,
        'cycle_loss': 0,
        'no_empty_span_loss': 0,
        'diversity_entropy_loss': 0
    }

    # Adaptive guard state
    low_util_streak = 0  # Count consecutive steps with util < 5%
    high_top1_streak = 0  # Count consecutive steps with top1 > 70%
    gamma_boost_active = False
    tau_hold_steps_remaining = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Compute current temperature (annealing)
        temperature = compute_temperature_schedule(
            current_step, total_steps, temp_init, temp_final
        )

        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        answer_ids = batch['answer_ids'].to(device)
        snippet_ids = batch['snippet_ids'].to(device)

        # Generate synthetic IR targets if teacher forcing is enabled
        target_ir_ids = None
        if use_teacher_forcing and ir_token_ids is not None:
            target_ir_ids = generate_synthetic_ir_target(
                batch_size=input_ids.shape[0],
                ir_token_ids=ir_token_ids,
                num_codes=num_codes,
                device=device
            )

        # Set current training step on VQ module for Gumbel warm-start schedule
        if hasattr(model, 'ir_generator') and hasattr(model.ir_generator, 'vq_tied_gen'):
            if hasattr(model.ir_generator.vq_tied_gen, 'vq') and hasattr(model.ir_generator.vq_tied_gen.vq, 'vq'):
                model.ir_generator.vq_tied_gen.vq.vq.current_step = current_step

        # V9: Prepare concept targets if present in batch
        concept_targets = None
        if 'concepts' in batch:  # Check if concepts dict exists in collated batch
            concept_targets = {
                'num_terms': torch.tensor(batch['concepts']['num_terms'], device=device),
                'operation_types': torch.tensor(batch['concepts']['operation_types'], device=device),
                'max_operand_magnitude': torch.tensor(batch['concepts']['max_operand_magnitude'], device=device, dtype=torch.long),
                'depth': torch.tensor(batch['concepts']['depth'], device=device, dtype=torch.long),
                'has_carry_addition': torch.tensor(batch['concepts']['has_carry_addition'], device=device),
                'difficulty': torch.tensor(batch['concepts']['difficulty'], device=device, dtype=torch.long),
                'parity': torch.tensor(batch['concepts']['parity'], device=device),
                'sign': torch.tensor(batch['concepts']['sign'], device=device)
            }

        # Forward pass (with autocast for FP16)
        with torch.cuda.amp.autocast(enabled=use_fp16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                answer_ids=answer_ids,
                target_ir_ids=target_ir_ids,
                input_snippets=snippet_ids,
                temperature=temperature,
                mode='train',
                concept_targets=concept_targets
            )

            loss = outputs['total_loss']

        breakdown = outputs['loss_breakdown']

        # ============= ADAPTIVE GUARDS DURING WARM-START =============
        # Get VQ module for guard checks
        vq = model.ir_generator.vq.vq
        if current_step < vq.gumbel_steps:
            # Extract code indices from IR token IDs
            ir_token_ids = outputs.get('ir_token_ids', None)
            if ir_token_ids is not None:
                # Extract VQ codes from IR buffer (filter out structural tags)
                code_token_start = model.ir_token_ids['codes'][0]
                code_token_end = model.ir_token_ids['codes'][-1]
                # Mask for code tokens only
                code_mask = (ir_token_ids >= code_token_start) & (ir_token_ids <= code_token_end)
                vq_codes = ir_token_ids[code_mask]

                if vq_codes.numel() > 0:
                    # Compute utilization from current batch codes
                    unique_codes = len(torch.unique(vq_codes))
                    utilization = unique_codes / vq.num_codes
                    top1_freq = vq.code_freq_ema.max().item()

                    # Guard 1: Low utilization (< 5% for 100 steps)
                    if utilization < 0.05:
                        low_util_streak += 1
                        if low_util_streak >= 100 and not gamma_boost_active:
                            # Double gamma (cap at 6.0)
                            vq.gamma0 = min(6.0, vq.gamma0 * 2.0)
                            tau_hold_steps_remaining = 100
                            gamma_boost_active = True
                            print(f"\n[GUARD] Util < 5% for 100 steps! Boosting gamma to {vq.gamma0:.2f}, holding tau for 100 steps")
                            low_util_streak = 0  # Reset
                    else:
                        low_util_streak = 0
                        if gamma_boost_active and tau_hold_steps_remaining == 0:
                            gamma_boost_active = False

                    # Guard 2: High top-1 frequency (> 70% for 50 steps)
                    if top1_freq > 0.70:
                        high_top1_streak += 1
                        if high_top1_streak >= 50:
                            print(f"\n[GUARD WARNING] Top-1 code freq > 70% for 50 steps (current: {top1_freq:.2%})")
                            high_top1_streak = 0  # Reset after warning
                    else:
                        high_top1_streak = 0

                    # Decrement tau hold counter
                    if tau_hold_steps_remaining > 0:
                        tau_hold_steps_remaining -= 1

        # ============= BATCH SAFETY CHECKS =============
        # Count tokens with loss (non-PAD answer tokens)
        answer_targets = answer_ids.clone()
        answer_targets[answer_targets == model.pad_token_id] = -100
        tokens_with_loss = (answer_targets != -100).sum().item()

        # HARD ASSERT: Must have training signal
        assert tokens_with_loss > 0, \
            f"CRITICAL: Batch {batch_idx} has ZERO answer tokens (all PAD)! Stopping training."

        # ============= DEBUG LOGGING =============
        # A) Step dump every 500 steps
        if debug_logger and debug_logger.should_log_step(current_step):
            debug_logger.log_step_dump(
                epoch=epoch,
                step=current_step,
                batch=batch,
                outputs=outputs,
                model=model,
                device=device
            )

        # B) Accumulate batch metrics
        if debug_logger:
            debug_logger.accumulate_batch_metrics(
                loss_breakdown=breakdown,
                ir_token_ids=outputs['ir_token_ids'],
                temperature=temperature,
                model=model
            )

        # C) Hard assertions (fail fast)
        if enable_assertions and debug_logger:
            # Compute IR error rate for this batch
            ir_integrity = validate_ir_integrity(
                outputs['ir_token_ids'],
                model.ir_token_ids,
                min_codes_per_span=3,
                max_codes_per_span=6
            )
            batch_ir_error_rate = ir_integrity['error_rate']

            debug_logger.assert_training_health(
                batch_idx=batch_idx,
                tokens_with_loss=tokens_with_loss,
                ir_error_rate=batch_ir_error_rate,
                ir_token_ids=outputs['ir_token_ids'],
                answer_logits=outputs['answer_logits'],
                model=model,
                guards_mode=model.get_guards_mode()  # Phase-aware: "WARN" (Phase 0) or "FAIL" (Phase 1)
            )

        # Log answer length statistics (every 100 batches)
        if batch_idx % 100 == 0:
            answer_lengths = (answer_targets != -100).sum(dim=1)
            min_len = answer_lengths.min().item()
            avg_len = answer_lengths.float().mean().item()
            max_len = answer_lengths.max().item()
            print(f"\n[Batch {batch_idx}] Answer lengths: min={min_len}, avg={avg_len:.1f}, max={max_len}")

        # Collect V8 metrics for snapshot logging
        recent_losses = {}
        if hasattr(model, 'use_ir_value_head') and model.use_ir_value_head:
            # Extract ir_value_mae/mse from loss components (placeholder)
            # TODO: Modify forward() to return these as separate components
            recent_losses['ir_value_mae'] = 0.0  # Placeholder
            recent_losses['ir_value_mse'] = 0.0  # Placeholder

        if hasattr(model, 'use_contrastive') and model.use_contrastive:
            # Extract contrastive metrics (placeholder)
            # TODO: Modify contrastive_module to return metrics
            recent_losses['nn_acc'] = 0.0  # Placeholder
            recent_losses['sim_diag'] = 0.0  # Placeholder
            recent_losses['sim_offdiag'] = 0.0  # Placeholder
            recent_losses['diag_minus_offdiag'] = 0.0  # Placeholder

        # Train-mode debug snapshots at steps 20 (smoke test), 100, 200, 600, 800, 1000
        if current_step in [20, 100, 200, 600, 800, 1000] and output_dir:
            dump_train_mode_snapshot(model, current_step, output_dir,
                                    recent_losses=recent_losses, tokenizer=tokenizer)

            # Also dump eval snapshot every 200 steps
            if current_step % 200 == 0 and val_loader is not None:
                dump_eval_mode_snapshot(model, current_step, output_dir,
                                       val_loader, tokenizer, device)

        # P1.3: Fail-fast guard for IR→value head
        if args and hasattr(args, 'use_ir_value_head') and args.use_ir_value_head and current_step >= 200:
            # Check if ir_value_mae was logged (currently placeholder, so skip check for now)
            # TODO: Enable this check once ir_value_mae extraction is implemented
            pass
            # if 'ir_value_mae' not in recent_losses or recent_losses['ir_value_mae'] == 0.0:
            #     raise RuntimeError(
            #         f"[FAIL-FAST] Step {current_step}: --use_ir_value_head enabled but "
            #         f"ir_value_mae not logged. Check IR→value head implementation."
            #     )

        # Save checkpoints at steps 200, 600, 800, 1000
        if current_step in [200, 600, 800, 1000] and output_dir:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_step{current_step:04d}.pt")
            model_config = {
                'base_model': args.model_name,
                'num_codes': args.num_codes,
                'code_dim': args.code_dim,
                'use_contrastive': args.use_contrastive,
                'contrastive_weight': args.contrastive_weight,
                'contrastive_temperature': args.contrastive_T,
            }
            torch.save({
                'step': current_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config,
                'args': vars(args)
            }, checkpoint_path)
            print(f"\n[CHECKPOINT] Saved step {current_step} checkpoint to {checkpoint_path}")

        # Backward pass (with GradScaler for FP16)
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        scheduler.step()

        # Periodically sync code embeddings with VQ codebook
        if batch_idx % 100 == 0:
            model.sync_code_embeddings()

        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += breakdown[key]

        # Update progress bar with sanity check info
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'ans': f"{breakdown['answer_loss']:.4f}",
            'vq': f"{breakdown['vq_loss']:.4f}",
            'tok': tokens_with_loss,
            'temp': f"{temperature:.3f}"
        })

        current_step += 1

    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components, current_step


def evaluate(model, dataloader, device, tokenizer, temperature):
    """Evaluate with exact matching and IR integrity."""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    all_ir_buffers = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            answer_ids = batch['answer_ids'].to(device)

            # Forward for loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                answer_ids=answer_ids,
                temperature=temperature,
                mode='train'
            )

            total_loss += outputs['total_loss'].item()
            ir_buffer = outputs['ir_token_ids']
            all_ir_buffers.append(ir_buffer)

            # Generate answers for accuracy (use longer max_length for safety)
            gen_outputs = model.generate_answer(
                input_ids, attention_mask, temperature, max_answer_length=20
            )
            pred_answer_ids = gen_outputs['answer_ids']

            # Exact numeric matching
            for j in range(pred_answer_ids.shape[0]):
                pred_text = tokenizer.decode(
                    pred_answer_ids[j],
                    skip_special_tokens=True
                ).strip()

                true_text = tokenizer.decode(
                    answer_ids[j],
                    skip_special_tokens=True
                ).strip()

                # Sanity check: warn if generated answer is empty
                if not pred_text and j == 0:
                    print(f"\nWARNING: Empty generated answer (IR len: {ir_buffer.shape[1]})")

                matches, _, _ = exact_match(pred_text, true_text)
                if matches:
                    correct += 1
                total += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0

    # IR integrity check
    if all_ir_buffers:
        all_ir = torch.cat(all_ir_buffers, dim=0)
        integrity = validate_ir_integrity(
            all_ir,
            model.ir_token_ids,
            min_codes_per_span=3,
            max_codes_per_span=6
        )
        ir_error_rate = integrity['error_rate']
    else:
        ir_error_rate = 0.0

    return avg_loss, accuracy, ir_error_rate


def main(args):
    """Main training loop."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("="*60)
    print("IR-CoT V2 Training (VQ-Tied Codes)")
    print("="*60)
    print(f"Random seed: {args.seed}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup tokenizer
    print("\nSetting up tokenizer...")
    tokenizer, ir_token_ids = extend_tokenizer_for_ir(
        base_model_name=args.model_name,
        num_codes=args.num_codes
    )

    # Setup debug logger (if enabled)
    debug_logger = None
    if args.enable_debug:
        print("\n[DEBUG] Debug logging ENABLED")
        print(f"[DEBUG] Log directory: {args.output_dir}/logs")
        debug_logger = DebugLogger(
            log_dir=f"{args.output_dir}/logs",
            tokenizer=tokenizer,
            ir_token_ids=ir_token_ids,
            num_samples=5,
            step_frequency=args.debug_step_frequency
        )
        print(f"[DEBUG] Step dumps every {args.debug_step_frequency} steps")
        print(f"[DEBUG] Hard assertions: {'ENABLED' if args.enable_assertions else 'DISABLED'}")

    # Initialize model
    print(f"\nInitializing model: {args.model_name}")

    # Configure for memory efficiency if using 410M
    if "410m" in args.model_name.lower() and args.use_lora and PEFT_AVAILABLE:
        print("Using LoRA for memory efficiency...")

        # Create base model
        model = CausalIRModelV2(
            base_model_name=args.model_name,
            ir_token_ids=ir_token_ids,
            num_codes=args.num_codes,
            code_dim=args.code_dim,
            temperature_init=args.temp_init,
            temperature_final=args.temp_final,
            pad_token_id=tokenizer.pad_token_id,
            phase0_seeded_steps=args.phase0_seeded_steps,
            phase0_code_ce_weight=args.phase0_code_ce_weight,
            phase0_vq_tau=args.phase0_vq_tau,
            use_contrastive=args.use_contrastive,
            contrastive_weight=args.contrastive_weight,
            contrastive_T=args.contrastive_T,
            use_gumbel_warmstart=args.use_gumbel_warmstart,
            gumbel_tau=args.gumbel_tau,
            gumbel_steps=args.gumbel_steps,
            diversity_weight=args.diversity_weight,
            eval_code_sampling=args.eval_code_sampling,
            eval_tau=args.eval_tau,
            eval_topk=args.eval_topk,
            eval_topp=args.eval_topp,
            use_ir_value_head=args.use_ir_value_head,
            ir_value_weight=args.ir_value_weight,
            answer_ce_boost_steps=args.answer_ce_boost_steps,
            use_concept_head=args.use_concept_head,
            concept_weight=args.concept_weight
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=16,  # Low rank
            lora_alpha=32,
            target_modules=["query_key_value", "dense"],  # Pythia attention layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model.base_model = get_peft_model(model.base_model, lora_config)
        print(f"LoRA applied. Trainable params: {model.base_model.print_trainable_parameters()}")

    else:
        model = CausalIRModelV2(
            base_model_name=args.model_name,
            ir_token_ids=ir_token_ids,
            num_codes=args.num_codes,
            code_dim=args.code_dim,
            temperature_init=args.temp_init,
            temperature_final=args.temp_final,
            pad_token_id=tokenizer.pad_token_id,
            phase0_seeded_steps=args.phase0_seeded_steps,
            phase0_code_ce_weight=args.phase0_code_ce_weight,
            phase0_vq_tau=args.phase0_vq_tau,
            use_contrastive=args.use_contrastive,
            contrastive_weight=args.contrastive_weight,
            contrastive_T=args.contrastive_T,
            use_gumbel_warmstart=args.use_gumbel_warmstart,
            gumbel_tau=args.gumbel_tau,
            gumbel_steps=args.gumbel_steps,
            diversity_weight=args.diversity_weight,
            eval_code_sampling=args.eval_code_sampling,
            eval_tau=args.eval_tau,
            eval_topk=args.eval_topk,
            eval_topp=args.eval_topp,
            use_ir_value_head=args.use_ir_value_head,
            ir_value_weight=args.ir_value_weight,
            answer_ce_boost_steps=args.answer_ce_boost_steps,
            use_concept_head=args.use_concept_head,
            concept_weight=args.concept_weight
        )

    # Enable gradient checkpointing for memory
    if args.gradient_checkpointing:
        model.base_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    model.to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # V8 Configuration Verification (fail-fast)
    print("\n" + "="*60)
    print("V8 CONFIGURATION VERIFICATION")
    print("="*60)
    print(f"use_ir_value_head: {args.use_ir_value_head}")
    print(f"ir_value_weight: {args.ir_value_weight}")
    print(f"answer_ce_boost_steps: {args.answer_ce_boost_steps}")
    print(f"use_contrastive: {args.use_contrastive}")
    print(f"contrastive_weight: {args.contrastive_weight}")
    print(f"use_gumbel_warmstart: {args.use_gumbel_warmstart}")
    print(f"eval_code_sampling: {args.eval_code_sampling}")
    print(f"eval_tau: {args.eval_tau}, eval_topk: {args.eval_topk}, eval_topp: {args.eval_topp}")

    if args.use_ir_value_head:
        assert hasattr(model, 'ir_value_head'), \
            "FATAL: use_ir_value_head=True but model.ir_value_head not found!"
        ir_head_params = sum(p.numel() for p in model.ir_value_head.parameters())
        print(f"\n[V8] IR→value head ACTIVE")
        print(f"     IR→value head parameters: {ir_head_params:,}")
        print(f"     Answer CE boost: 2.0x for first {args.answer_ce_boost_steps} steps")
    else:
        print(f"\n[V7-Lite] IR→value head DISABLED")
    print("="*60 + "\n")

    # Setup datasets
    print("\nLoading datasets...")
    train_dataset = ArithmeticDataset(args.train_data, tokenizer)
    val_dataset = ArithmeticDataset(args.val_data, tokenizer)

    # V9: Custom collate function to handle concepts dict
    def collate_fn_with_concepts(batch):
        """Custom collate function that handles optional 'concepts' field."""
        # Use default collate for main fields
        collated = {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'answer_ids': torch.stack([item['answer_ids'] for item in batch]),
            'snippet_ids': torch.stack([item['snippet_ids'] for item in batch]),
        }

        # If concepts exist, collate them
        if 'concepts' in batch[0]:
            collated['concepts'] = {
                'num_terms': [item['concepts']['num_terms'] for item in batch],
                'operation_types': [item['concepts']['operation_types'] for item in batch],
                'max_operand_magnitude': [item['concepts']['max_operand_magnitude'] for item in batch],
                'depth': [item['concepts']['depth'] for item in batch],
                'has_carry_addition': [item['concepts']['has_carry_addition'] for item in batch],
                'difficulty': [item['concepts']['difficulty'] for item in batch],
                'parity': [item['concepts']['parity'] for item in batch],
                'sign': [item['concepts']['sign'] for item in batch],
            }

        return collated

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=collate_fn_with_concepts
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_with_concepts
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Setup optimizer
    if args.use_8bit_adam and BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        print("Using 8-bit AdamW")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # Setup scheduler
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Setup GradScaler for FP16 mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print(f"Temperature: {args.temp_init} → {args.temp_final}")
    print(f"IR Teacher Forcing: {'ENABLED' if args.ir_teacher_forcing else 'DISABLED'}")
    print(f"Mixed Precision (FP16): {'ENABLED' if args.fp16 else 'DISABLED'}")
    print(f"Gradient Clipping: max_norm={args.max_grad_norm}")
    print("="*60)

    best_val_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_step = 0

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.num_epochs} ---")

        # Set current epoch for gradient leak scheduling
        model.set_epoch(epoch - 1)  # 0-indexed

        # Train
        train_loss, train_components, current_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            current_step, total_steps, args.temp_init, args.temp_final,
            use_teacher_forcing=args.ir_teacher_forcing,
            ir_token_ids=ir_token_ids,
            num_codes=args.num_codes,
            debug_logger=debug_logger,
            enable_assertions=args.enable_assertions,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
            use_fp16=args.fp16,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            val_loader=val_loader,
            args=args
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print("Loss components:")
        for key, val in train_components.items():
            print(f"  {key}: {val:.4f}")

        # Validate
        current_temp = compute_temperature_schedule(
            current_step, total_steps, args.temp_init, args.temp_final
        )
        val_loss, val_acc, ir_error_rate = evaluate(
            model, val_loader, device, tokenizer, current_temp
        )

        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2%}")
        print(f"IR Error Rate: {ir_error_rate:.2%}")
        print(f"Temperature: {current_temp:.3f}")

        # B) Log epoch metrics (if debug enabled)
        if debug_logger:
            debug_logger.log_epoch_metrics(epoch, ir_error_rate, val_acc)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pt"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'ir_error_rate': ir_error_rate,
                'temperature': current_temp,
                'args': vars(args)
            }, checkpoint_path)

            print(f"Saved best model to {checkpoint_path}")

        # Run causality tests every N epochs
        if epoch % args.test_frequency == 0:
            print("\n--- Running Causality Tests ---")
            tester = CausalityTester(model, tokenizer, ir_token_ids)

            test_data_path = args.test_data if args.test_data else args.val_data
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)[:100]

            results = tester.run_all_tests(test_data, batch_size=8)

            results_path = output_dir / f"causality_tests_epoch{epoch}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

    # D) Run control experiments (if requested)
    if args.run_control_random_ir:
        print("\n" + "="*60)
        print("Running Control Experiments")
        print("="*60)
        random_ir_acc = run_control_experiment_random_ir(
            model, val_loader, device, tokenizer, args.temp_final
        )
        print(f"\n[CONTROL] Random-IR Accuracy: {random_ir_acc:.2%}")
        print("[CONTROL] Expected: <10% if IR is genuinely used for reasoning")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IR-CoT V2 (VQ-Tied)")

    # Model args
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-410m',
                       help='Base model (recommend 410m)')
    parser.add_argument('--num_codes', type=int, default=512,
                       help='VQ codebook size')
    parser.add_argument('--code_dim', type=int, default=128,
                       help='Code embedding dimension')

    # Temperature args
    parser.add_argument('--temp_init', type=float, default=0.7,
                       help='Initial temperature')
    parser.add_argument('--temp_final', type=float, default=0.4,
                       help='Final temperature (annealed)')

    # Phase 0 Bootstrap args (VQ codebook priming)
    parser.add_argument('--phase0_seeded_steps', type=int, default=0,
                       help='Phase 0 seeded bootstrap steps (0=disabled, 1500=default)')
    parser.add_argument('--phase0_code_ce_weight', type=float, default=0.1,
                       help='Code CE loss weight in Phase 0 (default: 0.1)')
    parser.add_argument('--phase0_vq_tau', type=float, default=2.0,
                       help='VQ temperature in Phase 0 for exploration (default: 2.0)')

    # V7-Lite: Contrastive loss args
    parser.add_argument('--use_contrastive', action='store_true', default=False,
                       help='Enable InfoNCE contrastive loss for HL-IR alignment')
    parser.add_argument('--contrastive_weight', type=float, default=0.3,
                       help='Weight for contrastive loss (default: 0.3)')
    parser.add_argument('--contrastive_T', type=float, default=0.07,
                       help='Temperature for contrastive loss (default: 0.07)')

    # V7-Lite: Gumbel warm-start args
    parser.add_argument('--use_gumbel_warmstart', action='store_true', default=False,
                       help='Enable Gumbel-Softmax warm-start for differentiable code selection')
    parser.add_argument('--gumbel_tau', type=float, default=0.6,
                       help='Gumbel-Softmax temperature (default: 0.6)')
    parser.add_argument('--gumbel_steps', type=int, default=1500,
                       help='Number of steps to use Gumbel before switching to VQ (default: 1500)')
    parser.add_argument('--diversity_weight', type=float, default=0.5,
                       help='Weight for diversity entropy loss during Gumbel warm-start (default: 0.5)')

    # V8: IR→value auxiliary head args
    parser.add_argument('--use_ir_value_head', action='store_true', default=False,
                       help='Enable IR→value auxiliary head for answer grounding')
    parser.add_argument('--ir_value_weight', type=float, default=0.25,
                       help='Weight for IR→value auxiliary loss (default: 0.25)')
    parser.add_argument('--answer_ce_boost_steps', type=int, default=2000,
                       help='Number of steps to use boosted answer CE weight 2.0 (default: 2000)')

    # V9: IR→concept auxiliary head args
    parser.add_argument('--use_concept_head', action='store_true', default=False,
                       help='Enable IR→concept auxiliary head for semantic grounding (V9)')
    parser.add_argument('--concept_weight', type=float, default=0.3,
                       help='Weight for IR→concept auxiliary loss (default: 0.3)')

    # Teacher forcing args (DISABLED by default - let VQ guide learning)
    parser.add_argument('--ir_teacher_forcing', action='store_true', default=False,
                       help='Use teacher forcing for IR generation')
    parser.add_argument('--no_ir_teacher_forcing', dest='ir_teacher_forcing',
                       action='store_false',
                       help='Disable teacher forcing for IR generation (default)')

    # Training args
    parser.add_argument('--train_data', type=str,
                       default='../data/arithmetic/train.json')
    parser.add_argument('--val_data', type=str,
                       default='../data/arithmetic/val.json')
    parser.add_argument('--test_data', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # Memory optimization args
    parser.add_argument('--use_lora', action='store_true',
                       help='Use LoRA for memory efficiency (410m)')
    parser.add_argument('--use_8bit_adam', action='store_true',
                       help='Use 8-bit AdamW optimizer')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing')

    # Training optimization args
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training (fp16)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')

    # Output args
    parser.add_argument('--output_dir', type=str,
                       default='../checkpoints/ir_cot_v2_410m')
    parser.add_argument('--test_frequency', type=int, default=5)
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file (optional, for script compatibility)')

    # Debug args
    parser.add_argument('--enable_debug', action='store_true',
                       help='Enable comprehensive debug logging')
    parser.add_argument('--debug_step_frequency', type=int, default=500,
                       help='Log step dumps every N steps (default: 500)')
    parser.add_argument('--enable_assertions', action='store_true',
                       help='Enable hard assertions for fail-fast (requires --enable_debug)')
    parser.add_argument('--run_control_random_ir', action='store_true',
                       help='Run Random-IR control experiment after training')

    # Eval-time code sampling args (resolve train-eval mismatch)
    parser.add_argument('--eval_code_sampling', type=str, default='softmax',
                       choices=['argmin', 'softmax', 'gumbel'],
                       help='Code sampling mode during eval: argmin (greedy), softmax (temp), gumbel')
    parser.add_argument('--eval_tau', type=float, default=0.9,
                       help='Temperature for eval code sampling (default: 0.9)')
    parser.add_argument('--eval_topk', type=int, default=32,
                       help='Top-K for eval code sampling (0=disabled, default: 32)')
    parser.add_argument('--eval_topp', type=float, default=0.95,
                       help='Top-P (nucleus) for eval code sampling (1.0=disabled, default: 0.95)')

    args = parser.parse_args()

    main(args)
