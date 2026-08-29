"""
Causal IR Model V2 - Full VQ-tied implementation with proper loss scoping.

Key changes from V1:
- Uses IRBufferGeneratorV2 with VQ-tied code generation
- CE loss only on tags (not codes)
- Code embeddings tied to VQ codebook
- Temperature annealing support
- Grammar enforcement during generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import GPTNeoXForCausalLM

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ir_generator_v2 import IRBufferGeneratorV2
from local_cycle import LocalCycleHead, compute_local_cycle_loss
from tied_embeddings import tie_code_embeddings_to_codebook, CodeEmbeddingTier
from ir_grammar import IRGrammarEnforcer


class CausalIRModelV2(nn.Module):
    """
    Two-pass decoder-only model with enforced IR causality and VQ-tied codes.

    Loss structure:
    - Answer CE: 1.0 (primary)
    - Tag CE: 0.5 (structure learning)
    - VQ commitment: 0.1 (code learning via quantization, NOT CE)
    - Local cycle: 0.05 (anchoring)
    - No-empty-span: 0.02 (grammar)
    - Diversity: 0.02 (codebook utilization)
    """

    def __init__(
        self,
        base_model_name: str,
        ir_token_ids: Dict,
        num_codes: int = 512,
        code_dim: int = 128,
        snippet_length: int = 10,
        use_local_cycle: bool = True,
        temperature_init: float = 0.7,
        temperature_final: float = 0.4,
        cycle_weight: float = 0.1,
        vq_weight: float = 0.5,
        coverage_weight: float = 0.05,
        no_empty_span_weight: float = 0.02,
        pad_token_id: int = 0,
        gradient_leak_lambda: float = 0.1,
        gradient_leak_epochs: int = 3,
        phase0_seeded_steps: int = 1500,
        phase0_code_ce_weight: float = 0.1,
        phase0_vq_tau: float = 2.0,
        use_contrastive: bool = False,
        contrastive_weight: float = 0.3,
        contrastive_T: float = 0.07,
        use_gumbel_warmstart: bool = False,
        gumbel_tau: float = 0.6,
        gumbel_steps: int = 1500,
        diversity_weight: float = 0.5,
        eval_code_sampling: str = 'softmax',
        eval_tau: float = 0.9,
        eval_topk: int = 32,
        eval_topp: float = 0.95,
        use_ir_value_head: bool = True,
        ir_value_weight: float = 0.25,
        answer_ce_boost_steps: int = 2000,
        hl_residual_in_pass2: float = 0.0,
        use_concept_head: bool = False,
        concept_weight: float = 0.3
    ):
        """
        Args:
            base_model_name: HuggingFace model (e.g., 'EleutherAI/pythia-410m')
            ir_token_ids: Dict with token IDs
            num_codes: VQ codebook size
            code_dim: Code embedding dimension
            snippet_length: HL snippet length for local cycle
            use_local_cycle: Whether to use local cycle loss
            temperature_init: Initial temperature for code sampling
            temperature_final: Final temperature (annealed)
            cycle_weight: Weight for cycle loss
            vq_weight: Weight for VQ loss
            coverage_weight: Weight for diversity loss
            no_empty_span_weight: Weight for no-empty-span penalty
        """
        super().__init__()

        # Load base model
        self.base_model = GPTNeoXForCausalLM.from_pretrained(base_model_name)
        self.config = self.base_model.config

        # Resize embeddings for new tokens
        new_vocab_size = len(ir_token_ids['codes']) + self.config.vocab_size + 100  # Buffer
        self.base_model.resize_token_embeddings(new_vocab_size)

        self.ir_token_ids = ir_token_ids
        self.use_local_cycle = use_local_cycle
        self.cycle_weight = cycle_weight
        self.vq_weight = vq_weight
        self.coverage_weight = coverage_weight
        self.no_empty_span_weight = no_empty_span_weight

        # VQ STABILIZATION: Slower temperature annealing (1.0→0.7 over 3 epochs) for exploration
        self.temperature_init = 1.0
        self.temperature_final = 0.7
        self.pad_token_id = pad_token_id
        # CRITICAL FIX: Disable gradient leak (λ=0) to prevent single-code attractor during warm-start
        self.gradient_leak_lambda = 0.0  # Was: gradient_leak_lambda
        self.gradient_leak_epochs = 0  # Was: gradient_leak_epochs
        self.current_epoch = 0
        self.current_step = 0

        # V5: EMA frequency tracking for logit debias (lightweight diversity nudge)
        self.register_buffer('ema_code_freq', torch.ones(num_codes) / num_codes)  # Start uniform
        self.ema_alpha = 0.05  # EMA momentum for frequency tracking (K≈500 steps)

        # Logit debias schedule: gamma = 1.0 (steps 0-1500) → 0.0 (epoch 4+)
        self.debias_gamma_init = 1.0
        self.debias_gamma_final = 0.0
        self.debias_ramp_start = 0
        self.debias_ramp_end = 1500  # Start ramping down after step 1500

        # PHASE 0 BOOTSTRAP: Seeded code usage to prime VQ codebook
        self.phase0_seeded_steps = phase0_seeded_steps
        self.phase0_code_ce_weight = phase0_code_ce_weight
        self.phase0_vq_tau = phase0_vq_tau
        self.num_codes = num_codes

        # Balanced code sampler for Phase 0
        from balanced_code_sampler import BalancedCodeSampler
        self.balanced_code_sampler = BalancedCodeSampler(
            num_codes=num_codes,
            min_codes_per_span=3,
            max_codes_per_span=6,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # V7-Lite: Gumbel warm-start parameters
        self.use_gumbel_warmstart = use_gumbel_warmstart
        self.gumbel_tau = gumbel_tau
        self.gumbel_steps = gumbel_steps
        self.diversity_weight = diversity_weight

        # V7-Lite: Contrastive loss for HL-IR alignment
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        if use_contrastive:
            from contrastive_loss import InfoNCEContrastiveLoss
            self.contrastive_module = InfoNCEContrastiveLoss(
                hidden_dim=self.config.hidden_size,
                temperature=contrastive_T
            )

        # V8: IR→value auxiliary head for answer grounding
        self.use_ir_value_head = use_ir_value_head
        self.ir_value_weight = ir_value_weight
        self.answer_ce_boost_steps = answer_ce_boost_steps
        self.hl_residual_in_pass2 = hl_residual_in_pass2

        if use_ir_value_head:
            from ir_value_head import IRValueHead
            self.ir_value_head = IRValueHead(
                code_dim=code_dim,
                hidden_dim=128,
                max_answer_value=100,
                mode='regression'
            )

        # V9: IR→concept head for semantic concept supervision
        self.use_concept_head = use_concept_head
        self.concept_weight = concept_weight
        self.concept_zero_count = 0  # Fail-fast guard counter

        if use_concept_head:
            from .ir_concept_head import IRConceptHead
            self.concept_head = IRConceptHead(
                in_dim=code_dim,
                hidden_dim=128,
                dropout=0.1
            )
            print(f"[V9] Concept head enabled (weight={concept_weight})")

        # Log V8 architecture settings
        print(f"[V8] hl_residual_in_pass2={hl_residual_in_pass2:.1f} (0.0 = no input leakage to Pass-2)")

        # Sanity check: PAD should not equal EOS
        assert pad_token_id != self.base_model.config.eos_token_id, \
            f"PAD token ({pad_token_id}) must differ from EOS ({self.base_model.config.eos_token_id})"

        # IR Buffer Generator V2 (with VQ-tied codes + V7-Lite Gumbel warm-start)
        self.ir_generator = IRBufferGeneratorV2(
            base_model=self.base_model,
            ir_token_ids=ir_token_ids,
            num_codes=num_codes,
            code_dim=code_dim,
            beta=0.25,  # VQ commitment weight
            temperature_init=temperature_init,
            temperature_final=temperature_final,
            use_grammar_masks=True,
            pad_token_id=pad_token_id,
            use_gumbel_warmstart=use_gumbel_warmstart,
            gumbel_tau=gumbel_tau,
            gumbel_steps=gumbel_steps,
            eval_code_sampling=eval_code_sampling,
            eval_tau=eval_tau,
            eval_topk=eval_topk,
            eval_topp=eval_topp
        )

        # Tie code embeddings to VQ codebook
        tie_code_embeddings_to_codebook(
            self.base_model,
            self.ir_generator.vq,
            ir_token_ids
        )

        # Embedding tier for periodic re-sync
        self.embedding_tier = CodeEmbeddingTier(
            self.base_model,
            self.ir_generator.vq,
            ir_token_ids
        )

        # Local Cycle Head (optional)
        if use_local_cycle:
            self.local_cycle_head = LocalCycleHead(
                hidden_dim=self.config.hidden_size,
                vocab_size=new_vocab_size,
                snippet_length=snippet_length
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        answer_ids: Optional[torch.Tensor] = None,
        target_ir_ids: Optional[torch.Tensor] = None,
        input_snippets: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        mode: str = 'train',
        concept_targets: Optional[Dict[str, torch.Tensor]] = None  # V9: concept labels
    ) -> Dict:
        """
        Two-pass forward.

        Args:
            input_ids: Input problem tokens (batch, input_len)
            attention_mask: Attention mask
            answer_ids: Target answer tokens (batch, answer_len)
            target_ir_ids: Ground truth IR for teacher forcing
            input_snippets: Snippets for local cycle
            temperature: Current temperature (for annealing)
            concept_targets: V9 concept labels (dict with 8 concept keys)
            mode: 'train' or 'inference'

        Returns:
            Dict with logits, losses, IR buffer
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # V7-Lite: Sync VQ step counter for Gumbel warm-start
        if self.use_gumbel_warmstart:
            self.ir_generator.vq.vq.current_step = self.current_step

        # ============= PASS 1: Generate IR Buffer =============
        # PHASE 0: Use phase-aware VQ temperature
        current_vq_tau = self.get_current_vq_tau()

        # PHASE 0: Generate balanced seeded codes for VQ bootstrap
        seeded_code_ids = None
        if self.is_in_phase0() and mode == 'train':
            # Sample balanced codes from sampler (batch_size, num_codes)
            # Each example gets 4 spans of 3-6 codes = ~18 codes total
            seeded_code_ids = self.generate_seeded_codes(batch_size, num_spans=4)

        ir_output = self.ir_generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ir_ids=target_ir_ids if mode == 'train' else None,
            temperature=current_vq_tau,  # Phase-aware temperature
            seeded_code_ids=seeded_code_ids  # Phase 0 seeded codes for bootstrap
        )

        ir_token_ids = ir_output['ir_token_ids']  # (batch, ir_len)
        tag_lm_loss = ir_output['tag_lm_loss']  # CE on tags only
        vq_loss = ir_output['vq_loss']  # VQ commitment
        vq_code_logits = ir_output.get('vq_code_logits', None)  # For Phase 0 code CE

        # Handle None tag_lm_loss
        if tag_lm_loss is None:
            tag_lm_loss = torch.tensor(0.0, device=device)

        # ============= STRICT IR-ONLY ENFORCEMENT (NO GRADIENT LEAK) =============
        # CRITICAL FIX: Always use λ=0 (no gradient leak from answer to IR)
        # Previous warm-start (λ=0.1→0.05→0) created single-code attractor basin
        leak_lambda = 0.0  # Strict enforcement from start

        # Always detach token IDs (they're discrete)
        ir_token_ids_detached = ir_token_ids.detach()

        # ============= PASS 2: Generate Answer from IR Only =============
        answer_logits = None
        answer_loss = torch.tensor(0.0, device=device)

        if answer_ids is not None:
            # Create sequence: IR + Answer (for teacher forcing)
            # Use full answer_ids (model will autoregressively predict each token)
            full_sequence = torch.cat([ir_token_ids_detached, answer_ids], dim=1)

            # Forward through model
            outputs = self.base_model(
                input_ids=full_sequence,
                output_hidden_states=True
            )

            # Get logits for answer positions
            # The logits are shifted by 1 (predicting next token at each position)
            ir_len = ir_token_ids_detached.shape[1]
            answer_len = answer_ids.shape[1]

            # Extract logits for answer prediction positions
            # Logits at positions [ir_len-1:ir_len+answer_len-1] predict the answer tokens
            # (We predict answer_ids from the last IR token onward)
            answer_logits = outputs.logits[:, ir_len-1:ir_len+answer_len-1, :]

            # ============= P4 FIX: Hard ban PAD/EOS in answer logits =============
            eos_id = self.base_model.config.eos_token_id

            # Ban PAD everywhere in answer generation
            answer_logits[:, :, self.pad_token_id] = -1e9

            # Ban EOS on first 2 answer positions (enforce min answer length)
            for pos in range(min(2, answer_logits.shape[1])):
                answer_logits[:, pos, eos_id] = -1e9

            # Targets are the actual answer_ids
            # CRITICAL: Mask PAD tokens to prevent collapse
            answer_targets = answer_ids.clone()
            answer_targets[answer_targets == self.pad_token_id] = -100

            # P4 FIX: Assert non-empty supervision
            answer_mask = (answer_targets != -100)
            num_answer_tokens = answer_mask.sum().item()
            assert num_answer_tokens > 0, "No supervised answer tokens - all PAD!"

            # Compute CE loss on answer (PAD tokens ignored)
            # V8: Temporary boost for answer CE weight (2.0 for first ~2k steps, then 1.0)
            if self.current_step < self.answer_ce_boost_steps:
                answer_ce_weight = 2.0
            else:
                answer_ce_weight = 1.0

            answer_loss = torch.nn.functional.cross_entropy(
                answer_logits.reshape(-1, answer_logits.shape[-1]),
                answer_targets.reshape(-1),
                ignore_index=-100
            ) * answer_ce_weight

            # P4 FIX: Assert answer loss is non-zero
            assert answer_loss.item() > 0, f"Answer CE is zero despite {num_answer_tokens} supervised tokens!"

        # ============= Auxiliary Losses =============

        # No-empty-span penalty
        no_empty_span_loss = self.ir_generator.grammar.compute_no_empty_span_penalty(
            ir_token_ids_detached
        )

        # Diversity loss (codebook utilization)
        diversity_loss = self._compute_diversity_loss(ir_output)

        # Gumbel-specific entropy diversity loss (warm-start only)
        diversity_entropy_loss = self.ir_generator.vq.vq.compute_diversity_loss()

        # Local cycle loss (if enabled)
        cycle_loss = torch.tensor(0.0, device=device)
        if self.use_local_cycle and mode == 'train' and input_snippets is not None:
            ir_outputs = self.base_model(
                input_ids=ir_token_ids_detached,
                output_hidden_states=True
            )
            ir_hidden = ir_outputs.hidden_states[-1]

            cycle_loss = compute_local_cycle_loss(
                ir_token_ids=ir_token_ids_detached,
                ir_hidden_states=ir_hidden,
                input_snippets=input_snippets,
                ir_token_id_dict=self.ir_token_ids,
                local_cycle_head=self.local_cycle_head,
                sample_ratio=0.1
            )

        # ============= V7-Lite: Contrastive Loss (HL-IR Alignment) =============
        contrastive_loss = torch.tensor(0.0, device=device)
        contrastive_metrics = {}
        if self.use_contrastive and mode == 'train':
            # Extract HL embeddings: mean-pool hidden states over input tokens
            hl_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hl_hidden = hl_outputs.hidden_states[-1]  # (batch, input_len, hidden_dim)

            # Mean-pool over input sequence (excluding padding)
            if attention_mask is not None:
                # Expand attention mask to match hidden dim
                mask_expanded = attention_mask.unsqueeze(-1).expand(hl_hidden.size())
                sum_hidden = (hl_hidden * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                hl_emb = sum_hidden / sum_mask  # (batch, hidden_dim)
            else:
                hl_emb = hl_hidden.mean(dim=1)  # (batch, hidden_dim)

            # Extract IR embeddings: mean-pool code embeddings from IR
            # Find code positions in IR
            code_start = self.ir_token_ids['code_start']
            code_end = self.ir_token_ids['code_end']
            code_mask = (ir_token_ids_detached >= code_start) & (ir_token_ids_detached <= code_end)

            # Get code embeddings
            ir_code_embs = []
            for b in range(batch_size):
                if code_mask[b].any():
                    code_tokens = ir_token_ids_detached[b][code_mask[b]]
                    code_embs = self.base_model.get_input_embeddings()(code_tokens)  # (num_codes, hidden_dim)
                    ir_code_embs.append(code_embs.mean(dim=0))
                else:
                    # No codes in IR, use zero embedding
                    ir_code_embs.append(torch.zeros(self.config.hidden_size, device=device))

            ir_emb = torch.stack(ir_code_embs)  # (batch, hidden_dim)

            # Compute contrastive loss
            contrastive_loss, contrastive_metrics = self.contrastive_module(hl_emb, ir_emb)

            # Rename metrics to match expected format
            contrastive_metrics = {
                'sim_diag': contrastive_metrics['diag_similarity'],
                'sim_offdiag': contrastive_metrics['offdiag_similarity'],
                'diag_minus_offdiag': contrastive_metrics['diag_minus_offdiag'],
                'nn_acc': contrastive_metrics['nn_accuracy']
            }

        # ============= V8: IR→Value Auxiliary Loss (Answer Grounding) =============
        ir_value_loss = torch.tensor(0.0, device=device)
        ir_value_mae = 0.0
        concept_loss = torch.tensor(0.0, device=device)
        concept_accs = {}
        if self.use_ir_value_head and mode == 'train' and answer_ids is not None:
            # Get IR code embeddings from VQ codebook
            # Extract code indices from quantized codes
            if hasattr(ir_output, 'code_indices') and ir_output.code_indices is not None:
                # CRITICAL: Detach code_indices to prevent gradient bypass from IR→value back to pass-1
                code_indices = ir_output.code_indices.detach()  # (batch, seq_len)

                # Get code embeddings from VQ codebook (stop-grad enforced)
                # (batch, seq_len, code_dim)
                code_embeddings = self.ir_generator.vq.vq.embedding(code_indices)

                # Predict answer value from IR codes
                predictions = self.ir_value_head(code_embeddings)  # (batch,)

                # Prepare targets from answer strings
                # We need to extract the numeric answer from answer_ids
                from ir_value_head import prepare_answer_targets

                # Decode answer_ids to strings for parsing
                answer_strings = []
                for b in range(batch_size):
                    # Get non-pad tokens
                    non_pad_mask = answer_ids[b] != self.pad_token_id
                    ans_tokens = answer_ids[b][non_pad_mask]
                    # Decode to string (simplified - assumes tokenizer available)
                    # For now, just convert first token to string as approximation
                    if len(ans_tokens) > 0:
                        answer_strings.append(str(ans_tokens[0].item()))
                    else:
                        answer_strings.append("0")

                # Prepare numeric targets
                targets = prepare_answer_targets(answer_strings, max_value=100).to(device)

                # Compute loss and MAE
                ir_value_loss, ir_value_mae = self.ir_value_head.compute_loss(
                    predictions, targets
                )

        # ============= V9: IR→Concept Auxiliary Loss (Semantic Grounding) =============
        if self.use_concept_head and concept_targets is not None:
            # Concept head should ONLY see IR code embeddings (no HL leakage)
            if hasattr(ir_output, 'code_indices') and ir_output.code_indices is not None:
                # CRITICAL: Detach code_indices to prevent gradient bypass
                code_indices = ir_output.code_indices.detach()  # (batch, seq_len)

                # Get code embeddings from VQ codebook (stop-grad enforced)
                code_embeddings = self.ir_generator.vq.vq.embedding(code_indices)  # (batch, seq_len, code_dim)

                # Mean-pool across sequence length to get fixed-size representation
                code_embeddings_pooled = code_embeddings.mean(dim=1)  # (batch, code_dim)

                # Predict concepts from IR codes
                concept_predictions = self.concept_head(code_embeddings_pooled)

                # Compute concept loss and accuracy
                concept_loss, concept_loss_dict = self.concept_head.compute_loss(
                    concept_predictions, concept_targets
                )
                concept_accs = self.concept_head.compute_accuracy(
                    concept_predictions, concept_targets
                )

                # Fail-fast guard: track consecutive zero/NaN losses
                if concept_loss.item() < 1e-6 or torch.isnan(concept_loss):
                    self.concept_zero_count += 1
                    if self.concept_zero_count > 20:
                        raise RuntimeError(
                            f"[V9 Fail-Fast] Concept loss has been zero/NaN for {self.concept_zero_count} "
                            f"consecutive steps! Check concept_targets are being passed correctly."
                        )
                else:
                    self.concept_zero_count = 0  # Reset counter on valid loss

        # ============= PHASE 0: Code CE Loss (Bootstrap) =============
        code_ce_loss = torch.tensor(0.0, device=device)
        if self.is_in_phase0() and vq_code_logits is not None and target_ir_ids is not None:
            # Compute code CE loss to teach projection head all codes
            # Identify code positions in target_ir_ids
            code_start = self.ir_token_ids['code_start']
            code_end = self.ir_token_ids['code_end']

            # Find code positions in target_ir (excluding first token which is IR_START)
            target_flat = target_ir_ids[:, 1:].reshape(-1)
            code_mask = (target_flat >= code_start) & (target_flat <= code_end)

            if code_mask.sum() > 0 and vq_code_logits.shape[0] >= code_mask.sum():
                # Get code targets (convert token IDs to code indices)
                code_targets = target_flat[code_mask] - code_start

                # Get code logits for these positions
                code_logits_for_ce = vq_code_logits[:code_mask.sum()]

                # Compute CE loss
                code_ce_loss = F.cross_entropy(code_logits_for_ce, code_targets)

        # Get phase-aware code CE weight (with early exit support)
        code_ce_weight = self.get_current_code_ce_weight()

        # ============= Total Loss with Epoch-Dependent & Phase-Dependent Weights =============
        # PHASE 0: Higher coverage weight for exploration
        if self.is_in_phase0():
            coverage_weight = 0.75  # Phase 0: AGGRESSIVE diversity push (increased from 0.5)
            vq_weight = 0.5  # Phase 0: Strong commitment
        elif self.current_epoch < 2:
            coverage_weight = 0.75  # Epochs 1-2: AGGRESSIVE diversity (increased from 0.5 to combat collapse)
            vq_weight = 0.5  # Warm-start: strong commitment
        else:
            coverage_weight = 0.5  # Epochs 3+: maintain diversity (increased from 0.25)
            vq_weight = 0.25  # Normal: reduced commitment

        total_loss = (
            answer_loss +  # Already weighted by answer_ce_weight (2.0 boost → 1.0)
            tag_lm_loss * 0.5 +  # Tags only, not codes
            vq_weight * vq_loss +  # Code learning via VQ (epoch-dependent)
            self.cycle_weight * cycle_loss +
            self.no_empty_span_weight * no_empty_span_loss +
            coverage_weight * diversity_loss +  # Epoch-dependent coverage
            code_ce_weight * code_ce_loss +  # Phase 0 only: Code CE bootstrap
            self.contrastive_weight * contrastive_loss +  # V7-Lite: HL-IR alignment
            self.diversity_weight * diversity_entropy_loss +  # Gumbel entropy diversity
            self.ir_value_weight * ir_value_loss +  # V8: IR→value answer grounding
            self.concept_weight * concept_loss  # V9: IR→concept semantic grounding
        )

        loss_breakdown = {
            'answer_loss': answer_loss.item(),
            'tag_lm_loss': tag_lm_loss.item() if isinstance(tag_lm_loss, torch.Tensor) else 0.0,
            'vq_loss': vq_loss.item() if isinstance(vq_loss, torch.Tensor) else 0.0,
            'cycle_loss': cycle_loss.item() if isinstance(cycle_loss, torch.Tensor) else 0.0,
            'no_empty_span_loss': no_empty_span_loss.item() if isinstance(no_empty_span_loss, torch.Tensor) else 0.0,
            'diversity_loss': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else 0.0,
            'diversity_entropy_loss': diversity_entropy_loss.item() if isinstance(diversity_entropy_loss, torch.Tensor) else 0.0,
            'contrastive_loss': contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0,
            'ir_value_loss': ir_value_loss.item() if isinstance(ir_value_loss, torch.Tensor) else 0.0,
            'ir_value_mae': ir_value_mae,
            'concept_loss': concept_loss.item() if isinstance(concept_loss, torch.Tensor) else 0.0,
            'concept_accuracy': concept_accs.get('overall', 0.0) if concept_accs else 0.0,
            'concept_num_terms_mae': concept_accs.get('num_terms_mae', 0.0) if concept_accs else 0.0,
            'concept_operation_types_acc': concept_accs.get('operation_types', 0.0) if concept_accs else 0.0,
            'concept_magnitude_acc': concept_accs.get('max_operand_magnitude', 0.0) if concept_accs else 0.0,
            'concept_depth_acc': concept_accs.get('depth', 0.0) if concept_accs else 0.0,
            'concept_carry_acc': concept_accs.get('has_carry_addition', 0.0) if concept_accs else 0.0,
            'concept_difficulty_acc': concept_accs.get('difficulty', 0.0) if concept_accs else 0.0,
            'concept_parity_acc': concept_accs.get('parity', 0.0) if concept_accs else 0.0,
            'concept_sign_acc': concept_accs.get('sign', 0.0) if concept_accs else 0.0,
            'answer_ce_weight': answer_ce_weight,  # V8: Track dynamic weight
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0,
            **contrastive_metrics  # V7-Lite: Add contrastive diagnostics
        }

        return {
            'answer_logits': answer_logits,
            'ir_token_ids': ir_token_ids_detached,
            'total_loss': total_loss,
            'loss_breakdown': loss_breakdown,
            'metadata': ir_output['metadata']
        }

    def _compute_diversity_loss(self, ir_output: Dict) -> torch.Tensor:
        """
        V5: ENTROPY-BASED diversity loss (V3 baseline, no global KL).

        Local loss only: log(C) - H(avg_batch_dist) - encourages uniform code usage within batch

        This provides differentiable gradients through VQ logits, unlike
        the previous utilization-based approach which had no gradient path.
        """
        device = ir_output['ir_token_ids'].device

        # Get VQ code logits from IR generation
        vq_code_logits = ir_output.get('vq_code_logits', None)

        if vq_code_logits is None or vq_code_logits.numel() == 0:
            # No code logits available (no codes generated)
            return torch.tensor(0.0, device=device, requires_grad=False)

        # Compute soft probability distribution via temperature-scaled softmax
        # Lower temperature (0.2) for smoother gradients
        temperature = 0.2
        code_probs = F.softmax(vq_code_logits / temperature, dim=-1)  # (num_codes_positions, num_codes)

        # Average distribution across all code positions in this batch
        avg_code_dist = code_probs.mean(dim=0)  # (num_codes,)

        # ============= LOCAL ENTROPY LOSS (within-batch diversity) =============
        # Compute Shannon entropy: H = -sum(p * log(p))
        entropy = -(avg_code_dist * torch.log(avg_code_dist + 1e-10)).sum()

        # Maximum possible entropy for uniform distribution
        num_codes = torch.tensor(self.ir_generator.num_codes, dtype=torch.float32, device=device)
        max_entropy = torch.log(num_codes)

        # Local loss: penalize deviation from maximum entropy
        local_loss = max_entropy - entropy

        # ============= UPDATE EMA FREQUENCY (for logit debias, not loss) =============
        # Track EMA of code frequency to debias logits at generation time
        with torch.no_grad():
            self.ema_code_freq = (1 - self.ema_alpha) * self.ema_code_freq + self.ema_alpha * avg_code_dist

        # V5: Return only local entropy loss (no global KL term)
        return local_loss

    def generate_ir(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        max_length: int = 50,
        do_sample: bool = True,
        top_k: int = 32,
        top_p: float = 0.95
    ) -> Dict:
        """
        Generate IR buffer only (Pass 1) - intervention-friendly.

        Args:
            input_ids: Input problem tokens (B, L_input)
            attention_mask: Attention mask
            temperature: Temperature for code sampling
            max_length: Maximum IR length
            do_sample: Whether to sample (vs argmax)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Dict with:
                - 'ir_ids': Generated IR token IDs (B, L_ir)
                - 'ir_embeddings': IR embeddings (B, L_ir, D) for cross-attention
                - 'metadata': Generation metadata
        """
        self.eval()
        with torch.no_grad():
            # Generate IR buffer using IR generator
            ir_output = self.ir_generator(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_ir_ids=None,
                temperature=temperature
            )

            ir_token_ids = ir_output['ir_token_ids']

            # Get embeddings for the generated IR
            ir_embeddings = self.base_model.get_input_embeddings()(ir_token_ids)

            return {
                'ir_ids': ir_token_ids,
                'ir_embeddings': ir_embeddings,
                'metadata': ir_output.get('metadata', {})
            }

    def answer_from_ir(
        self,
        input_ids: torch.Tensor,
        ir_ids: torch.Tensor,
        ir_embeddings: Optional[torch.Tensor] = None,
        max_length: int = 20,
        temperature: float = 0.0
    ) -> torch.Tensor:
        """
        Generate answer from provided IR (Pass 2) - intervention-friendly.

        Args:
            input_ids: Input problem tokens (B, L_input) [not used but kept for API consistency]
            ir_ids: IR token IDs (B, L_ir) - can be modified/intervened
            ir_embeddings: IR embeddings (B, L_ir, D) - if None, will re-embed from ir_ids
            max_length: Maximum answer length
            temperature: Temperature for answer generation (0.0 = greedy)

        Returns:
            Generated answer token IDs (B, L_answer)
        """
        self.eval()
        with torch.no_grad():
            # Re-embed if embeddings not provided (e.g., after intervention on ir_ids)
            if ir_embeddings is None:
                ir_embeddings = self.base_model.get_input_embeddings()(ir_ids)

            # Generate answer from IR (Pass 2)
            current_seq = ir_ids
            answer_pos = 0  # Track position in answer for EOS banning

            for _ in range(max_length):
                outputs = self.base_model(input_ids=current_seq)
                next_token_logits = outputs.logits[:, -1, :]

                # ============= HARD GUARD: Ban EOS on first 2 answer tokens =============
                if answer_pos < 2:
                    next_token_logits[:, self.base_model.config.eos_token_id] = -1e10

                # Sample or argmax based on temperature
                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                current_seq = torch.cat([current_seq, next_token], dim=1)
                answer_pos += 1

                # Check for EOS (only after first 2 tokens)
                if answer_pos >= 2 and (next_token == self.base_model.config.eos_token_id).all():
                    break

            # Extract answer
            ir_len = ir_ids.shape[1]
            answer_ids = current_seq[:, ir_len:]

            return answer_ids

    def generate_answer(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        max_answer_length: int = 20
    ) -> Dict:
        """
        Generate answer for given input (inference mode).

        This is a convenience wrapper around generate_ir() + answer_from_ir().

        Args:
            input_ids: Input problem tokens
            attention_mask: Attention mask
            temperature: Temperature for code sampling
            max_answer_length: Maximum answer length

        Returns:
            Dict with generated answer and IR buffer
        """
        self.eval()
        with torch.no_grad():
            # Pass 1: Generate IR
            ir_output = self.generate_ir(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature
            )

            # Pass 2: Generate answer from IR
            answer_ids = self.answer_from_ir(
                input_ids=input_ids,
                ir_ids=ir_output['ir_ids'],
                ir_embeddings=ir_output['ir_embeddings'],
                max_length=max_answer_length,
                temperature=0.0  # Greedy for answer generation
            )

            return {
                'answer_ids': answer_ids,
                'ir_token_ids': ir_output['ir_ids'],
                'metadata': ir_output.get('metadata', {})
            }

    def sync_code_embeddings(self):
        """Periodically re-sync code embeddings with VQ codebook."""
        self.embedding_tier.sync_embeddings()

    def set_epoch(self, epoch: int):
        """Set current epoch for gradient leak scheduling."""
        self.current_epoch = epoch

    def set_step(self, step: int):
        """Set current training step for temperature annealing and debias gamma."""
        self.current_step = step

    def get_debias_gamma(self) -> float:
        """
        V5: Compute current logit debias gamma based on step.

        Schedule: gamma = 1.0 (steps 0-1500) → 0.0 (by epoch 4)
        """
        if self.current_step <= self.debias_ramp_start:
            return self.debias_gamma_init
        elif self.current_step >= self.debias_ramp_end:
            # Linear ramp down from debias_ramp_end to end of epoch 3
            # Assume ~875 steps/epoch for 70M model, batch size 8
            epoch_4_start = 875 * 3  # Step 2625
            if self.current_step >= epoch_4_start:
                return self.debias_gamma_final
            else:
                # Linear interpolation from ramp_end to epoch 4 start
                progress = (self.current_step - self.debias_ramp_end) / (epoch_4_start - self.debias_ramp_end)
                return self.debias_gamma_init * (1 - progress) + self.debias_gamma_final * progress
        else:
            return self.debias_gamma_init

    def is_in_phase0(self) -> bool:
        """Check if we're in Phase 0 (seeded bootstrap)."""
        return self.current_step < self.phase0_seeded_steps

    def get_current_vq_tau(self) -> float:
        """
        Get phase-aware VQ temperature.

        Phase 0: 2.0 (exploration)
        Phase 1: 1.0 → 0.8 (standard annealing)
        """
        if self.is_in_phase0():
            return self.phase0_vq_tau
        else:
            # Phase 1: Normal annealing (1.0 → 0.8 over epochs)
            # Temperature annealing starts after Phase 0
            steps_since_phase1 = self.current_step - self.phase0_seeded_steps
            # Assume ~875 steps/epoch, anneal over epochs 2-4 (1750 steps)
            epoch_3_end = 875 * 3  # Step 2625 from start
            steps_for_annealing = max(1, epoch_3_end - self.phase0_seeded_steps)

            if steps_since_phase1 <= 0:
                return 1.0
            elif steps_since_phase1 >= steps_for_annealing:
                return 0.8
            else:
                # Linear interpolation
                progress = steps_since_phase1 / steps_for_annealing
                return 1.0 * (1 - progress) + 0.8 * progress

    def get_current_code_ce_weight(self, util: float = 0.0) -> float:
        """
        Get phase-aware code CE weight with early exit ramp-down.

        Phase 0 normal: 0.1
        Phase 0 early exit (util ≥ 20% at step 600): ramp 0.1 → 0 over 200 steps
        Phase 1: 0.0 (no code CE)
        """
        if not self.is_in_phase0():
            return 0.0

        # Check for early exit condition (util ≥ 20% by step 600)
        if util >= 0.20 and 600 <= self.current_step < 800:
            # Linear ramp down over 200 steps
            progress = (self.current_step - 600) / 200
            return self.phase0_code_ce_weight * (1 - progress)
        elif self.current_step >= 800 and util >= 0.20:
            # Early exit completed
            return 0.0
        else:
            # Normal Phase 0
            return self.phase0_code_ce_weight

    def get_guards_mode(self) -> str:
        """
        Get phase-aware guards mode.

        Phase 0: "WARN" (soft guards)
        Phase 1: "FAIL" (hard guards)
        """
        return "WARN" if self.is_in_phase0() else "FAIL"

    def generate_seeded_codes(self, batch_size: int, num_spans: int = 4) -> torch.Tensor:
        """
        Generate balanced seeded codes for Phase 0 bootstrap.

        Args:
            batch_size: Number of examples in batch
            num_spans: Number of spans per example

        Returns:
            Tensor of code indices (batch_size, total_codes)
        """
        return self.balanced_code_sampler.sample_batch(batch_size, num_spans)


if __name__ == "__main__":
    print("CausalIRModelV2 loaded.")
    print("Features: VQ-tied codes, grammar enforcement, proper loss scoping")
