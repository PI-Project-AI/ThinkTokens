"""
IR Buffer Generator for IR-CoT.

Autoregressively generates structured IR buffer with tags and VQ codes.
Pass 1: Input → IR buffer (constrained autoregressive generation)
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from vq import ProjectionVQ


class IRBufferGenerator(nn.Module):
    """
    Generates structured IR buffer autoregressively.

    The IR buffer has format:
    <IR_START> <GOAL> c047 c089 </GOAL> <STEP> c201 c033 </STEP> ... <IR_END>

    Strategy:
    - Use base LM to autoregressively generate sequence
    - At code positions: quantize hidden state → emit code token
    - At tag positions: emit tag token
    - Enforce span budgets via masking/penalties
    """

    def __init__(
        self,
        base_model,
        ir_token_ids: Dict,
        num_codes: int = 512,
        code_dim: int = 128,
        max_spans: int = 12,
        min_spans: int = 4,
        min_codes_per_span: int = 3,
        max_codes_per_span: int = 6,
        use_vq: bool = True
    ):
        """
        Args:
            base_model: Pythia model (GPTNeoXForCausalLM)
            ir_token_ids: Dict with token IDs for tags and codes
            num_codes: Number of VQ codes
            code_dim: Code embedding dimension
            max_spans: Maximum number of spans per IR buffer
            min_spans: Minimum number of spans
            min_codes_per_span: Minimum codes per span
            max_codes_per_span: Maximum codes per span
            use_vq: If True, use VQ to determine codes; else let model predict directly
        """
        super().__init__()
        self.base_model = base_model
        self.ir_token_ids = ir_token_ids
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.max_spans = max_spans
        self.min_spans = min_spans
        self.min_codes_per_span = min_codes_per_span
        self.max_codes_per_span = max_codes_per_span
        self.use_vq = use_vq

        hidden_dim = base_model.config.hidden_size

        # VQ module for quantizing hidden states to codes
        if use_vq:
            self.vq = ProjectionVQ(
                hidden_dim=hidden_dim,
                num_codes=num_codes,
                code_dim=code_dim
            )

        # Tag types (open tags only)
        self.open_tags = [
            ir_token_ids['goal'],
            ir_token_ids['assume'],
            ir_token_ids['step'],
            ir_token_ids['check'],
            ir_token_ids['branch']
        ]

        # Closing tag for each open tag
        self.close_tags = {
            ir_token_ids['goal']: ir_token_ids['goal_end'],
            ir_token_ids['assume']: ir_token_ids['assume_end'],
            ir_token_ids['step']: ir_token_ids['step_end'],
            ir_token_ids['check']: ir_token_ids['check_end'],
            ir_token_ids['branch']: ir_token_ids['branch_end']
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ir_ids: Optional[torch.Tensor] = None,
        max_ir_length: int = 50
    ) -> Dict:
        """
        Generate IR buffer autoregressively.

        Args:
            input_ids: Input problem tokens (batch, seq_len)
            attention_mask: Attention mask for input
            target_ir_ids: Ground truth IR tokens for teacher forcing (training)
            max_ir_length: Maximum IR buffer length

        Returns:
            Dict with:
                - ir_token_ids: Generated IR tokens (batch, ir_len)
                - vq_loss: VQ commitment loss (if use_vq=True)
                - lm_loss: LM loss over IR tokens (if target provided)
                - metadata: Span counts, code counts, etc.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize IR buffer with <IR_START>
        ir_buffer = torch.full(
            (batch_size, 1),
            self.ir_token_ids['ir_start'],
            dtype=torch.long,
            device=device
        )

        vq_loss = torch.tensor(0.0, device=device)
        all_vq_indices = []
        span_count = 0

        # Track state for each example in batch
        # State: 'start', 'in_tag', 'in_codes', 'closing_tag'
        states = ['start'] * batch_size
        current_tags = [None] * batch_size
        codes_in_span = [0] * batch_size

        if target_ir_ids is not None:
            # Teacher forcing: use provided IR sequence
            return self._teacher_forced_forward(
                input_ids, attention_mask, target_ir_ids
            )

        # Autoregressive generation
        for step in range(max_ir_length):
            # Concatenate input + IR buffer so far
            full_input = torch.cat([input_ids, ir_buffer], dim=1)

            # Get model outputs
            outputs = self.base_model(
                input_ids=full_input,
                attention_mask=attention_mask if attention_mask is None
                             else torch.cat([attention_mask,
                                           torch.ones(batch_size, ir_buffer.shape[1],
                                                     device=device)], dim=1),
                output_hidden_states=True
            )

            # Get hidden state at last position
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # (batch, hidden_dim)
            logits = outputs.logits[:, -1, :]  # (batch, vocab_size)

            # Determine next token based on state machine
            next_tokens = self._get_next_tokens(
                logits, last_hidden, states, current_tags,
                codes_in_span, span_count, batch_size, device
            )

            # If using VQ for code positions, quantize hidden states
            if self.use_vq:
                # Identify code positions
                code_mask = self._is_code_position(next_tokens)
                if code_mask.any():
                    # Quantize hidden states at code positions
                    code_hidden = last_hidden[code_mask].unsqueeze(1)  # (N, 1, hidden_dim)
                    _, vq_loss_step, vq_indices = self.vq(code_hidden)
                    vq_loss = vq_loss + vq_loss_step

                    # Replace with quantized code tokens
                    code_tokens = vq_indices.squeeze(1) + self.ir_token_ids['code_start']
                    next_tokens[code_mask] = code_tokens
                    all_vq_indices.append(vq_indices)

            # Append to IR buffer
            ir_buffer = torch.cat([ir_buffer, next_tokens.unsqueeze(1)], dim=1)

            # Update states
            states, current_tags, codes_in_span, span_count = self._update_states(
                next_tokens, states, current_tags, codes_in_span, span_count
            )

            # Check termination
            if self._should_terminate(next_tokens, states):
                break

        # Compute metadata
        metadata = {
            'num_spans': span_count,
            'ir_length': ir_buffer.shape[1],
            'vq_indices': torch.cat(all_vq_indices, dim=1) if all_vq_indices else None
        }

        return {
            'ir_token_ids': ir_buffer,
            'vq_loss': vq_loss,
            'lm_loss': None,
            'metadata': metadata
        }

    def _teacher_forced_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        target_ir_ids: torch.Tensor
    ) -> Dict:
        """
        Teacher forcing: compute loss over provided IR sequence.

        Args:
            input_ids: Input tokens (batch, input_len)
            attention_mask: Attention mask
            target_ir_ids: Target IR sequence (batch, ir_len)

        Returns:
            Dict with losses and generated IR
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Concatenate input + target IR (shifted right for teacher forcing)
        # We predict target_ir_ids from input + target_ir_ids[:-1]
        full_input = torch.cat([input_ids, target_ir_ids[:, :-1]], dim=1)

        # Create attention mask
        if attention_mask is None:
            full_mask = None
        else:
            ir_mask = torch.ones(batch_size, target_ir_ids.shape[1] - 1,
                                device=device, dtype=attention_mask.dtype)
            full_mask = torch.cat([attention_mask, ir_mask], dim=1)

        # Forward through model
        outputs = self.base_model(
            input_ids=full_input,
            attention_mask=full_mask,
            output_hidden_states=True
        )

        # Get logits for IR positions only
        input_len = input_ids.shape[1]
        ir_logits = outputs.logits[:, input_len-1:-1, :]  # (batch, ir_len, vocab)

        # Compute LM loss on IR tokens
        lm_loss = torch.nn.functional.cross_entropy(
            ir_logits.reshape(-1, ir_logits.shape[-1]),
            target_ir_ids.reshape(-1),
            ignore_index=-100
        )

        # Compute VQ loss if using VQ
        vq_loss = torch.tensor(0.0, device=device)
        if self.use_vq:
            # Get hidden states at code positions
            code_positions = self._get_code_positions(target_ir_ids)
            if code_positions.any():
                ir_hidden = outputs.hidden_states[-1][:, input_len-1:-1, :]
                code_hidden = ir_hidden[code_positions]
                if code_hidden.shape[0] > 0:
                    _, vq_loss, vq_indices = self.vq(code_hidden.unsqueeze(1))

        return {
            'ir_token_ids': target_ir_ids,
            'vq_loss': vq_loss,
            'lm_loss': lm_loss,
            'metadata': {}
        }

    def _get_next_tokens(
        self, logits, hidden_states, states, current_tags,
        codes_in_span, span_count, batch_size, device
    ):
        """Determine next tokens based on state machine and constraints."""
        next_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            if states[i] == 'start':
                # Must emit an open tag
                next_tokens[i] = self.open_tags[span_count % len(self.open_tags)]

            elif states[i] == 'in_codes':
                # Emit code token (will be quantized if use_vq=True)
                # For now, sample from code range
                code_idx = torch.argmax(
                    logits[i, self.ir_token_ids['code_start']:self.ir_token_ids['code_end']+1]
                )
                next_tokens[i] = self.ir_token_ids['code_start'] + code_idx

            elif states[i] == 'closing_tag':
                # Emit closing tag
                next_tokens[i] = self.close_tags[current_tags[i]]

        return next_tokens

    def _is_code_position(self, tokens):
        """Check which tokens are code tokens."""
        return (tokens >= self.ir_token_ids['code_start']) & \
               (tokens <= self.ir_token_ids['code_end'])

    def _get_code_positions(self, ir_ids):
        """Get boolean mask of code positions in IR sequence."""
        return (ir_ids >= self.ir_token_ids['code_start']) & \
               (ir_ids <= self.ir_token_ids['code_end'])

    def _update_states(self, tokens, states, current_tags, codes_in_span, span_count):
        """Update state machine based on emitted tokens."""
        # Simplified state transitions
        for i in range(len(states)):
            token = tokens[i].item()

            if token in self.open_tags:
                states[i] = 'in_codes'
                current_tags[i] = token
                codes_in_span[i] = 0

            elif self._is_code_position(tokens)[i]:
                codes_in_span[i] += 1
                if codes_in_span[i] >= self.max_codes_per_span:
                    states[i] = 'closing_tag'

            elif token in self.close_tags.values():
                states[i] = 'start'
                span_count += 1

        return states, current_tags, codes_in_span, span_count

    def _should_terminate(self, tokens, states):
        """Check if generation should terminate."""
        # Terminate if all examples emit <IR_END>
        return all(t.item() == self.ir_token_ids['ir_end'] for t in tokens)
