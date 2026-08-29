"""
IR Buffer Generator V2 - Full VQ-tied code generation with grammar enforcement.

Key features:
- VQ-tied logits at code positions (semantics from codebook, not embeddings)
- Grammar masks enforce tag structure (3-6 codes/span, 4-12 spans/buffer)
- Temperature annealing for code selection
- CE loss only on tags, not codes (VQ handles code learning)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from vq import ProjectionVQ
from vq_tied_generation import VQTiedCodeGenerator
from ir_grammar import IRGrammarEnforcer


def top_k_top_p_filter(logits: torch.Tensor, topk: int = 0, topp: float = 1.0) -> torch.Tensor:
    """
    Filter logits using top-k and/or top-p (nucleus) filtering.

    Args:
        logits: (batch, vocab_size) or (vocab_size,)
        topk: Keep only top k tokens with highest probability (0 = disabled)
        topp: Keep top tokens with cumulative probability >= topp (1.0 = disabled)

    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    if topk > 0:
        # Top-K filtering
        top_k = min(topk, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    if topp < 1.0:
        # Top-P (nucleus) filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > topp
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted indices to original indices
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    return logits


class IRBufferGeneratorV2(nn.Module):
    """
    Generates structured IR buffer with VQ-tied codes and grammar enforcement.

    Architecture:
    1. Tags: predicted via standard LM head (CE loss)
    2. Codes: logits from VQ codebook distances (no CE loss, VQ commitment only)
    3. Grammar: masks enforce valid sequences during generation
    """

    def __init__(
        self,
        base_model,
        ir_token_ids: Dict,
        num_codes: int = 512,
        code_dim: int = 128,
        beta: float = 0.25,
        temperature_init: float = 0.7,
        temperature_final: float = 0.4,
        min_codes_per_span: int = 3,
        max_codes_per_span: int = 6,
        min_spans: int = 4,
        max_spans: int = 12,
        use_grammar_masks: bool = True,
        pad_token_id: int = 0,
        use_gumbel_warmstart: bool = False,
        gumbel_tau: float = 0.6,
        gumbel_steps: int = 1500,
        eval_code_sampling: str = 'softmax',
        eval_tau: float = 0.9,
        eval_topk: int = 32,
        eval_topp: float = 0.95
    ):
        """
        Args:
            base_model: Pythia model (GPTNeoXForCausalLM)
            ir_token_ids: Dict with token IDs
            num_codes: VQ codebook size
            code_dim: Code embedding dimension
            beta: VQ commitment loss weight
            temperature_init: Initial temperature for code sampling
            temperature_final: Final temperature (annealed)
            min_codes_per_span: Minimum codes per span
            max_codes_per_span: Maximum codes per span
            min_spans: Minimum spans
            max_spans: Maximum spans
            use_grammar_masks: Apply grammar masks during generation
        """
        super().__init__()
        self.base_model = base_model
        self.ir_token_ids = ir_token_ids
        self.num_codes = num_codes
        self.temperature_init = temperature_init
        self.temperature_final = temperature_final
        self.use_grammar_masks = use_grammar_masks
        self.pad_token_id = pad_token_id

        hidden_dim = base_model.config.hidden_size

        # VQ module for code generation (V7-Lite: with Gumbel warm-start)
        self.vq = ProjectionVQ(
            hidden_dim=hidden_dim,
            num_codes=num_codes,
            code_dim=code_dim,
            beta=beta,
            use_unit_norm=True,  # Enable unit normalization
            use_gumbel_warmstart=use_gumbel_warmstart,
            gumbel_tau=gumbel_tau,
            gumbel_steps=gumbel_steps
        )

        # VQ-tied code generator
        self.vq_tied_gen = VQTiedCodeGenerator(
            vq_module=self.vq,
            ir_token_ids=ir_token_ids,
            vocab_size=base_model.config.vocab_size,
            temperature=temperature_init
        )

        # Grammar enforcer (with tightened constraints)
        # CRITICAL FIX: min_spans=3 (was 4), requires ≥2 distinct codes/span, bans consecutive codes
        self.grammar = IRGrammarEnforcer(
            ir_token_ids=ir_token_ids,
            min_codes_per_span=min_codes_per_span,
            max_codes_per_span=max_codes_per_span,
            min_spans=max(3, min_spans),  # Enforce minimum of 3 spans
            max_spans=max_spans
        )

        self.code_start = ir_token_ids['code_start']
        self.code_end = ir_token_ids['code_end']

        # Code blacklisting for anti-reuse (temporary, epochs 1-2 only)
        self.blacklisted_codes = set()  # Set of code indices to mask out

        # Eval-time code sampling parameters
        self.eval_code_sampling = eval_code_sampling
        self.eval_tau = eval_tau
        self.eval_topk = eval_topk
        self.eval_topp = eval_topp

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ir_ids: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        seeded_code_ids: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Generate or evaluate IR buffer.

        Args:
            input_ids: Input problem tokens (batch, seq_len)
            attention_mask: Attention mask
            target_ir_ids: Ground truth IR for teacher forcing (training)
            temperature: Current temperature for code sampling
            seeded_code_ids: Phase 0 seeded codes (batch, num_codes) for bootstrap

        Returns:
            Dict with ir_token_ids, vq_loss, tag_lm_loss, metadata
        """
        if target_ir_ids is not None:
            # Teacher forcing mode
            return self._teacher_forced_forward(
                input_ids, attention_mask, target_ir_ids, temperature, seeded_code_ids
            )
        else:
            # Free generation mode
            return self._generate_ir(
                input_ids, attention_mask, temperature
            )

    def _teacher_forced_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        target_ir_ids: torch.Tensor,
        temperature: Optional[float],
        seeded_code_ids: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Teacher forcing: compute losses over provided IR sequence.

        Key: Only compute CE loss on TAG tokens, not code tokens.
        Code tokens are learned via VQ commitment loss only.

        Args:
            seeded_code_ids: Phase 0 seeded codes (batch, num_codes) to replace target codes
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # PHASE 0: Replace code tokens with seeded codes if provided
        if seeded_code_ids is not None:
            target_ir_ids = self._inject_seeded_codes(target_ir_ids, seeded_code_ids)

        # Concatenate input + target IR (shifted for prediction)
        full_input = torch.cat([input_ids, target_ir_ids[:, :-1]], dim=1)

        if attention_mask is None:
            full_mask = None
        else:
            ir_mask = torch.ones(
                batch_size, target_ir_ids.shape[1] - 1,
                device=device, dtype=attention_mask.dtype
            )
            full_mask = torch.cat([attention_mask, ir_mask], dim=1)

        # Forward through model
        outputs = self.base_model(
            input_ids=full_input,
            attention_mask=full_mask,
            output_hidden_states=True
        )

        input_len = input_ids.shape[1]
        ir_hidden = outputs.hidden_states[-1][:, input_len-1:-1, :]  # (batch, ir_len, hidden)
        ir_logits = outputs.logits[:, input_len-1:-1, :]  # (batch, ir_len, vocab)

        # Identify code positions in target (align with shifted sequence)
        # We predict target_ir_ids[1:] from positions corresponding to target_ir_ids[:-1]
        target_shifted = target_ir_ids[:, 1:]  # What we're predicting
        code_mask = (target_shifted >= self.code_start) & (target_shifted <= self.code_end)

        # Compute CE loss ONLY on tag positions (not codes)
        tag_mask = ~code_mask  # Everything except codes
        tag_loss = torch.tensor(0.0, device=device)

        if tag_mask.any():
            # Compute CE only on tag positions
            tag_logits = ir_logits[tag_mask]
            tag_targets = target_shifted[tag_mask].clone()

            # P6 FIX: Mask PAD tokens in tag targets
            tag_targets[tag_targets == self.pad_token_id] = -100

            # P6 FIX: Assert we have non-PAD tag targets
            num_tag_targets = (tag_targets != -100).sum().item()
            if num_tag_targets > 0:
                tag_loss = F.cross_entropy(
                    tag_logits,
                    tag_targets,
                    ignore_index=-100
                )
                # P6 FIX: Assert tag loss is non-zero when we have supervision
                assert tag_loss.item() > 0, f"Tag CE is zero despite {num_tag_targets} supervised tag tokens!"

        # Compute VQ loss on code positions
        vq_loss = torch.tensor(0.0, device=device)
        if code_mask.any():
            # Get hidden states at code positions (need to slice ir_hidden to match)
            # ir_hidden corresponds to predicting target_shifted, so masks align
            code_hidden = ir_hidden[code_mask].unsqueeze(1)  # (N, 1, hidden)

            # VQ forward (commitment loss only, no CE)
            _, vq_loss_step, _ = self.vq(code_hidden)
            vq_loss = vq_loss_step

        # ENTROPY-BASED DIVERSITY: Collect VQ logits from code positions for diversity loss
        vq_code_logits = None
        if code_mask.any():
            # Get code logits for entropy computation
            code_hidden = ir_hidden[code_mask].unsqueeze(1)  # (N, 1, hidden)
            code_is_code = torch.ones(code_hidden.shape[0], 1, dtype=torch.bool, device=device)
            # Get training step from VQ module for logit debias
            training_step = self.vq.vq.current_step if hasattr(self.vq, 'vq') else None
            code_logits_out, _, _ = self.vq_tied_gen.compute_code_logits(
                code_hidden, code_is_code, current_temperature=temperature, training_step=training_step
            )
            vq_code_logits = code_logits_out.squeeze(1)  # (N, num_codes)

        return {
            'ir_token_ids': target_ir_ids,
            'tag_lm_loss': tag_loss,
            'vq_loss': vq_loss,
            'vq_code_logits': vq_code_logits,  # For entropy-based diversity
            'metadata': {
                'num_codes': code_mask.sum().item(),
                'num_tags': tag_mask.sum().item()
            }
        }

    def _generate_ir(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        temperature: Optional[float],
        max_length: int = 50
    ) -> Dict:
        """
        Generate IR buffer autoregressively with grammar enforcement.

        P3 FIX: SLOT-TYPED SELECTION
        - Code slots: Select ONLY from VQ codebook (512 codes) - EOS can't appear
        - Tag/structural slots: Select from full vocab with grammar masks + hard ban PAD/EOS
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Start with <IR_START>
        ir_buffer = torch.full(
            (batch_size, 1),
            self.ir_token_ids['ir_start'],
            dtype=torch.long,
            device=device
        )

        vq_loss = torch.tensor(0.0, device=device)
        all_vq_code_logits = []  # Collect code logits for entropy diversity

        # EOS and PAD tokens for banning
        eos_token_id = self.base_model.config.eos_token_id
        pad_token_id = self.pad_token_id

        for step in range(max_length):
            # Concatenate input + IR so far
            full_input = torch.cat([input_ids, ir_buffer], dim=1)

            # Forward through model
            outputs = self.base_model(
                input_ids=full_input,
                output_hidden_states=True
            )

            last_hidden = outputs.hidden_states[-1][:, -1:, :]  # (batch, 1, hidden)
            last_logits = outputs.logits[:, -1, :]  # (batch, vocab)

            # ============= P3 FIX: SLOT-TYPED SELECTION =============
            next_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)

            for b in range(batch_size):
                ir_seq = ir_buffer[b].tolist()

                # Get grammar-allowed tokens
                if self.use_grammar_masks:
                    valid_mask = self.grammar.get_valid_next_tokens(
                        ir_seq,
                        last_logits.shape[-1]
                    )
                else:
                    valid_mask = torch.ones(last_logits.shape[-1], dtype=torch.bool, device=device)

                # Determine if ANY code tokens are allowed
                code_mask_valid = valid_mask[self.code_start:self.code_end+1].any()

                # Check if only non-code tokens are allowed (structural tags)
                non_code_allowed = valid_mask.clone()
                non_code_allowed[self.code_start:self.code_end+1] = False
                only_tags_allowed = non_code_allowed.any() and not code_mask_valid

                if code_mask_valid and not only_tags_allowed:
                    # ============= CODE SLOT: Select ONLY from VQ codebook =============
                    # Compute VQ logits for this position (codebook space only)
                    # Get training step from VQ module for logit debias
                    training_step = self.vq.vq.current_step if hasattr(self.vq, 'vq') else None
                    code_logits_b, vq_indices_b, vq_loss_step = self.vq_tied_gen.compute_code_logits(
                        last_hidden[b:b+1, :, :],  # (1, 1, hidden)
                        torch.tensor([[True]], device=device),  # (1, 1)
                        current_temperature=temperature,
                        training_step=training_step
                    )

                    # Apply code blacklisting (mask out blacklisted codes)
                    if len(self.blacklisted_codes) > 0:
                        for code_idx in self.blacklisted_codes:
                            code_logits_b[:, :, code_idx] = float('-inf')

                    # Mask out codes that aren't allowed by grammar
                    grammar_code_mask = valid_mask[self.code_start:self.code_end+1]  # (512,)
                    code_logits_b.squeeze()[~grammar_code_mask] = float('-inf')

                    vq_loss = vq_loss + vq_loss_step

                    # Collect code logits for entropy-based diversity
                    all_vq_code_logits.append(code_logits_b.squeeze())  # (num_codes,)

                    # Select code index with train/eval mode switching
                    code_logits_flat = code_logits_b[0, 0, :]  # (num_codes,)

                    if self.training:
                        # Training: always use argmax (Gumbel is in forward pass)
                        code_idx = torch.argmax(code_logits_flat)
                    else:
                        # Eval: use configured sampling mode
                        if self.eval_code_sampling == 'argmin':
                            code_idx = torch.argmax(code_logits_flat)
                        elif self.eval_code_sampling == 'gumbel':
                            # Gumbel-softmax sampling
                            code_probs = F.gumbel_softmax(code_logits_flat.unsqueeze(0),
                                                           tau=self.eval_tau,
                                                           hard=True,
                                                           dim=-1)
                            code_idx = torch.argmax(code_probs)
                        else:  # softmax
                            # Temperature softmax sampling with top-k/top-p
                            logits_temp = code_logits_flat / self.eval_tau
                            logits_filtered = top_k_top_p_filter(logits_temp.unsqueeze(0),
                                                                  topk=self.eval_topk,
                                                                  topp=self.eval_topp)
                            probs = F.softmax(logits_filtered, dim=-1)
                            code_idx = torch.multinomial(probs, num_samples=1).squeeze()

                    # Convert code index to token ID
                    next_token_id = code_idx + self.code_start
                    next_tokens[b] = next_token_id

                else:
                    # ============= TAG/STRUCTURAL SLOT: Full vocab with hard masks =============
                    tag_logits = last_logits[b].clone()  # (vocab,)

                    # Hard ban PAD everywhere inside IR
                    tag_logits[pad_token_id] = -1e9

                    # Hard ban EOS inside IR (until IR_END is emitted)
                    if self.ir_token_ids['ir_end'] not in ir_seq:
                        tag_logits[eos_token_id] = -1e9

                    # Apply grammar masks
                    tag_logits[~valid_mask] = -1e9

                    # Check if any valid tokens remain after masking
                    max_logit = tag_logits.max().item()
                    if max_logit <= -1e8:
                        # All tokens masked - force IR_END to terminate gracefully
                        chosen_token = self.ir_token_ids['ir_end']
                    else:
                        # Select token (argmax over allowed vocab)
                        chosen_token = torch.argmax(tag_logits).item()

                        # Safety assert: verify chosen token is legal
                        allowed_set = set(torch.where(valid_mask)[0].tolist())
                        # Remove banned tokens from allowed_set for assertion
                        allowed_set.discard(pad_token_id)
                        if self.ir_token_ids['ir_end'] not in ir_seq:
                            allowed_set.discard(eos_token_id)

                        assert chosen_token in allowed_set, \
                            f"Illegal token {chosen_token} at IR step {step}, batch {b}"

                    next_tokens[b] = chosen_token

            # Append to IR buffer
            ir_buffer = torch.cat([ir_buffer, next_tokens.unsqueeze(1)], dim=1)

            # Check termination
            if (next_tokens == self.ir_token_ids['ir_end']).all():
                break

        # Concatenate all collected code logits for entropy diversity
        vq_code_logits = None
        if len(all_vq_code_logits) > 0:
            vq_code_logits = torch.stack(all_vq_code_logits, dim=0)  # (total_codes, num_codes)

        return {
            'ir_token_ids': ir_buffer,
            'tag_lm_loss': None,  # Not computed during generation
            'vq_loss': vq_loss,
            'vq_code_logits': vq_code_logits,  # For entropy-based diversity
            'metadata': {
                'ir_length': ir_buffer.shape[1]
            }
        }

    def _inject_seeded_codes(
        self,
        target_ir_ids: torch.Tensor,
        seeded_code_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Replace code tokens in target_ir_ids with seeded codes for Phase 0 bootstrap.

        Args:
            target_ir_ids: Target IR sequence (batch, seq_len)
            seeded_code_ids: Balanced seeded code indices (batch, num_codes)

        Returns:
            Modified target_ir_ids with seeded codes injected
        """
        batch_size = target_ir_ids.shape[0]
        device = target_ir_ids.device

        # Clone to avoid modifying original
        modified_ir = target_ir_ids.clone()

        # Find code positions in each batch element
        for b in range(batch_size):
            code_mask = (target_ir_ids[b] >= self.code_start) & (target_ir_ids[b] <= self.code_end)
            code_positions = torch.where(code_mask)[0]

            if len(code_positions) > 0:
                # Get seeded codes for this batch element
                num_seeded = min(len(code_positions), seeded_code_ids.shape[1])
                seeded_codes = seeded_code_ids[b, :num_seeded]

                # Convert code indices to token IDs
                seeded_tokens = seeded_codes + self.code_start

                # Replace code positions with seeded tokens
                modified_ir[b, code_positions[:num_seeded]] = seeded_tokens

        return modified_ir

    def _should_emit_code(self, current_seq: torch.Tensor) -> torch.Tensor:
        """
        Determine if next position should emit a code token.

        Heuristic: emit code if last token was open tag or code token.
        """
        batch_size = current_seq.shape[0]
        device = current_seq.device

        last_tokens = current_seq[:, -1]

        # Open tags
        open_tags = torch.tensor([
            self.ir_token_ids['goal'],
            self.ir_token_ids['assume'],
            self.ir_token_ids['step'],
            self.ir_token_ids['check'],
            self.ir_token_ids['branch']
        ], device=device)

        is_after_open = torch.isin(last_tokens, open_tags)

        # Code tokens
        is_after_code = (last_tokens >= self.code_start) & (last_tokens <= self.code_end)

        # Emit code if after open tag or after another code
        return is_after_open | is_after_code


if __name__ == "__main__":
    print("IR Buffer Generator V2 loaded.")
    print("Features: VQ-tied codes, grammar masks, temperature annealing")
