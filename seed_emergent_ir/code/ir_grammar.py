"""
IR grammar enforcement and integrity checking.

Enforces:
- Open/close tag pairing
- 3-6 codes per <STEP> span
- 4-12 spans per IR buffer
- No empty spans (must have at least min_codes)

Provides:
- Grammar masks for valid next tokens
- No-empty-span penalty
- IR integrity validation
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class IRState:
    """Tracks state during IR generation for grammar enforcement."""
    in_span: bool = False
    current_span_type: Optional[int] = None  # Token ID of open tag
    codes_in_current_span: int = 0
    unique_codes_in_span: set = None  # Track unique codes for diversity
    total_spans: int = 0
    last_token: Optional[int] = None
    last_code: Optional[int] = None  # Track last code to ban consecutive duplicates

    def __post_init__(self):
        if self.unique_codes_in_span is None:
            self.unique_codes_in_span = set()


class IRGrammarEnforcer:
    """
    Enforces IR grammar rules during generation.

    Rules:
    1. IR must start with <IR_START>
    2. After <IR_START>: must emit open tag
    3. After open tag: must emit codes (min 3, max 6)
    4. After codes: can emit more codes (if < max) or close tag
    5. After close tag: can emit another open tag or <IR_END>
    6. IR must end with <IR_END>
    7. Total spans: min 4, max 12
    """

    def __init__(
        self,
        ir_token_ids: Dict,
        min_codes_per_span: int = 3,
        max_codes_per_span: int = 6,
        min_spans: int = 4,
        max_spans: int = 12
    ):
        """
        Args:
            ir_token_ids: Dict with token IDs
            min_codes_per_span: Minimum codes per span
            max_codes_per_span: Maximum codes per span
            min_spans: Minimum spans per IR buffer
            max_spans: Maximum spans per IR buffer
        """
        self.ir_token_ids = ir_token_ids
        self.min_codes_per_span = min_codes_per_span
        self.max_codes_per_span = max_codes_per_span
        self.min_spans = min_spans
        self.max_spans = max_spans

        # Token sets
        self.open_tags = [
            ir_token_ids['goal'],
            ir_token_ids['assume'],
            ir_token_ids['step'],
            ir_token_ids['check'],
            ir_token_ids['branch']
        ]

        self.close_tags = {
            ir_token_ids['goal']: ir_token_ids['goal_end'],
            ir_token_ids['assume']: ir_token_ids['assume_end'],
            ir_token_ids['step']: ir_token_ids['step_end'],
            ir_token_ids['check']: ir_token_ids['check_end'],
            ir_token_ids['branch']: ir_token_ids['branch_end']
        }

        self.code_start = ir_token_ids['code_start']
        self.code_end = ir_token_ids['code_end']

    def get_valid_next_tokens(
        self,
        current_sequence: List[int],
        vocab_size: int
    ) -> torch.Tensor:
        """
        Get mask of valid next tokens based on grammar.

        Args:
            current_sequence: List of token IDs generated so far
            vocab_size: Total vocabulary size

        Returns:
            Boolean mask (vocab_size,) where True = valid token
        """
        mask = torch.zeros(vocab_size, dtype=torch.bool)

        if len(current_sequence) == 0:
            # Must start with <IR_START>
            mask[self.ir_token_ids['ir_start']] = True
            return mask

        state = self._compute_state(current_sequence)

        # Rule 1: After IR_START, must emit open tag
        if state.last_token == self.ir_token_ids['ir_start'] and not state.in_span:
            for tag in self.open_tags:
                mask[tag] = True
            return mask

        # Rule 2: After open tag, must emit codes
        if state.last_token in self.open_tags:
            mask[self.code_start:self.code_end+1] = True
            return mask

        # Rule 3: After code, can emit more codes or close tag
        if self.code_start <= state.last_token <= self.code_end:
            # Can emit more codes if under max
            if state.codes_in_current_span < self.max_codes_per_span:
                mask[self.code_start:self.code_end+1] = True

                # CRITICAL FIX: Ban consecutive identical codes to force diversity
                if state.last_code is not None:
                    mask[state.last_code] = False

            # Can emit close tag if at least min codes AND at least 2 distinct codes
            if (state.codes_in_current_span >= self.min_codes_per_span and
                len(state.unique_codes_in_span) >= 2 and
                state.current_span_type is not None):  # Safety: check span_type not None
                close_tag = self.close_tags[state.current_span_type]
                mask[close_tag] = True

            return mask

        # Rule 4: After close tag, emit open tag or IR_END
        if state.last_token in self.close_tags.values():
            # Can start new span if under max spans
            if state.total_spans < self.max_spans:
                for tag in self.open_tags:
                    mask[tag] = True

            # Can end IR if at least min spans
            if state.total_spans >= self.min_spans:
                mask[self.ir_token_ids['ir_end']] = True

            return mask

        # Default: allow all (shouldn't reach here if grammar is followed)
        mask[:] = True
        return mask

    def _compute_state(self, sequence: List[int]) -> IRState:
        """Compute current IR generation state from sequence."""
        state = IRState()

        for token in sequence:
            if token in self.open_tags:
                state.in_span = True
                state.current_span_type = token
                state.codes_in_current_span = 0
                state.unique_codes_in_span = set()  # Reset for new span
                state.last_code = None

            elif self.code_start <= token <= self.code_end:
                state.codes_in_current_span += 1
                state.unique_codes_in_span.add(token)
                state.last_code = token  # Track for consecutive ban

            elif token in self.close_tags.values():
                state.in_span = False
                state.total_spans += 1
                state.codes_in_current_span = 0
                state.unique_codes_in_span = set()
                state.current_span_type = None
                state.last_code = None

            state.last_token = token

        return state

    def compute_no_empty_span_penalty(
        self,
        ir_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute penalty for spans with too few codes.

        Args:
            ir_sequence: IR token sequence (batch, seq_len)

        Returns:
            Penalty loss (scalar)
        """
        batch_size = ir_sequence.shape[0]
        device = ir_sequence.device

        total_penalty = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            seq = ir_sequence[b].tolist()
            spans = self._extract_spans(seq)

            for span in spans:
                num_codes = len(span['codes'])
                if num_codes < self.min_codes_per_span:
                    # Penalty proportional to deficit
                    deficit = self.min_codes_per_span - num_codes
                    total_penalty += deficit ** 2

        # Average over batch
        return total_penalty / batch_size if batch_size > 0 else total_penalty

    def _extract_spans(self, sequence: List[int]) -> List[Dict]:
        """Extract spans from IR sequence."""
        spans = []
        i = 0

        while i < len(sequence):
            token = sequence[i]

            if token in self.open_tags:
                # Found open tag, collect codes until close tag
                codes = []
                i += 1

                while i < len(sequence):
                    if self.code_start <= sequence[i] <= self.code_end:
                        codes.append(sequence[i])
                        i += 1
                    elif sequence[i] in self.close_tags.values():
                        # Found close tag
                        spans.append({
                            'open_tag': token,
                            'codes': codes,
                            'close_tag': sequence[i]
                        })
                        i += 1
                        break
                    else:
                        # Malformed span
                        i += 1
                        break
            else:
                i += 1

        return spans


def validate_ir_integrity(
    ir_sequence: torch.Tensor,
    ir_token_ids: Dict,
    min_codes_per_span: int = 3,
    max_codes_per_span: int = 6
) -> Dict:
    """
    Validate IR buffer integrity.

    Checks:
    - Starts with IR_START
    - Ends with IR_END
    - All spans are balanced (open/close tags match)
    - All spans have correct number of codes
    - No orphan tokens

    Args:
        ir_sequence: IR tokens (batch, seq_len)
        ir_token_ids: Token ID mappings
        min_codes_per_span: Minimum codes per span
        max_codes_per_span: Maximum codes per span

    Returns:
        Dict with:
            - is_valid: Boolean (True if all checks pass)
            - error_rate: Fraction of malformed examples
            - errors: List of error descriptions
    """
    batch_size = ir_sequence.shape[0]
    code_start = ir_token_ids['code_start']
    code_end = ir_token_ids['code_end']

    open_tags = [
        ir_token_ids['goal'],
        ir_token_ids['assume'],
        ir_token_ids['step'],
        ir_token_ids['check'],
        ir_token_ids['branch']
    ]

    close_tags_map = {
        ir_token_ids['goal']: ir_token_ids['goal_end'],
        ir_token_ids['assume']: ir_token_ids['assume_end'],
        ir_token_ids['step']: ir_token_ids['step_end'],
        ir_token_ids['check']: ir_token_ids['check_end'],
        ir_token_ids['branch']: ir_token_ids['branch_end']
    }

    num_errors = 0
    all_errors = []

    for b in range(batch_size):
        seq = ir_sequence[b].tolist()
        errors = []

        # Check 1: Starts with IR_START
        if len(seq) == 0 or seq[0] != ir_token_ids['ir_start']:
            errors.append("Missing IR_START")

        # Check 2: Ends with IR_END
        if len(seq) == 0 or seq[-1] != ir_token_ids['ir_end']:
            errors.append("Missing IR_END")

        # Check 3: Extract and validate spans
        tag_stack = []
        current_codes = 0
        i = 0

        while i < len(seq):
            token = seq[i]

            if token in open_tags:
                tag_stack.append(token)
                current_codes = 0

            elif code_start <= token <= code_end:
                current_codes += 1

            elif token in close_tags_map.values():
                if len(tag_stack) == 0:
                    errors.append(f"Unmatched close tag at pos {i}")
                else:
                    open_tag = tag_stack.pop()
                    expected_close = close_tags_map[open_tag]

                    if token != expected_close:
                        errors.append(f"Mismatched tags at pos {i}")

                    if current_codes < min_codes_per_span:
                        errors.append(f"Too few codes ({current_codes}) at pos {i}")
                    elif current_codes > max_codes_per_span:
                        errors.append(f"Too many codes ({current_codes}) at pos {i}")

                    current_codes = 0

            i += 1

        # Check 4: All tags balanced
        if len(tag_stack) > 0:
            errors.append(f"Unclosed tags: {len(tag_stack)}")

        if errors:
            num_errors += 1
            all_errors.append(f"Example {b}: " + "; ".join(errors))

    error_rate = num_errors / batch_size if batch_size > 0 else 0

    return {
        'is_valid': num_errors == 0,
        'error_rate': error_rate,
        'num_errors': num_errors,
        'total_examples': batch_size,
        'errors': all_errors
    }


if __name__ == "__main__":
    print("IR grammar enforcement module loaded.")
    print("\nKey features:")
    print("- Enforces tag pairing and code counts")
    print("- No-empty-span penalty")
    print("- IR integrity validation")
