"""
Tokenizer utilities for IR-CoT with structural tags and VQ codes.

Extends Pythia tokenizer with:
- Structural tags: <IR_START>, <IR_END>, <GOAL>, </GOAL>, <ASSUME>, etc.
- VQ code tokens: c000, c001, ..., c511 (512 codes)
"""
from transformers import AutoTokenizer
from typing import List, Dict


def extend_tokenizer_for_ir(
    base_model_name: str = "EleutherAI/pythia-70m",
    num_codes: int = 512
) -> tuple:
    """
    Extend Pythia tokenizer with IR special tokens.

    Args:
        base_model_name: HuggingFace model identifier
        num_codes: Number of VQ codes to add (default 512)

    Returns:
        tokenizer: Extended tokenizer
        ir_token_ids: Dict mapping token names to IDs
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Define structural tags
    structural_tags = [
        "<IR_START>", "<IR_END>",
        "<GOAL>", "</GOAL>",
        "<ASSUME>", "</ASSUME>",
        "<STEP>", "</STEP>",
        "<CHECK>", "</CHECK>",
        "<BRANCH>", "</BRANCH>",
        "<PAD>"  # Add dedicated PAD token
    ]

    # Define VQ code tokens: c000, c001, ..., c511
    code_tokens = [f"c{i:03d}" for i in range(num_codes)]

    # Combine all new tokens
    new_tokens = structural_tags + code_tokens

    # Add tokens to tokenizer
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})

    # Set padding token to dedicated <PAD> token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<PAD>"

    print(f"Added {num_added} special tokens to tokenizer")
    print(f"New vocab size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token}")

    # Create mapping for easy access
    ir_token_ids = {
        'ir_start': tokenizer.convert_tokens_to_ids('<IR_START>'),
        'ir_end': tokenizer.convert_tokens_to_ids('<IR_END>'),
        'goal': tokenizer.convert_tokens_to_ids('<GOAL>'),
        'goal_end': tokenizer.convert_tokens_to_ids('</GOAL>'),
        'assume': tokenizer.convert_tokens_to_ids('<ASSUME>'),
        'assume_end': tokenizer.convert_tokens_to_ids('</ASSUME>'),
        'step': tokenizer.convert_tokens_to_ids('<STEP>'),
        'step_end': tokenizer.convert_tokens_to_ids('</STEP>'),
        'check': tokenizer.convert_tokens_to_ids('<CHECK>'),
        'check_end': tokenizer.convert_tokens_to_ids('</CHECK>'),
        'branch': tokenizer.convert_tokens_to_ids('<BRANCH>'),
        'branch_end': tokenizer.convert_tokens_to_ids('</BRANCH>'),
    }

    # Add code token IDs
    code_token_ids = [tokenizer.convert_tokens_to_ids(f"c{i:03d}") for i in range(num_codes)]
    ir_token_ids['code_start'] = code_token_ids[0]
    ir_token_ids['code_end'] = code_token_ids[-1]
    ir_token_ids['codes'] = code_token_ids

    return tokenizer, ir_token_ids


def get_tag_pairs() -> List[tuple]:
    """Return list of (open_tag, close_tag) pairs."""
    return [
        ('<GOAL>', '</GOAL>'),
        ('<ASSUME>', '</ASSUME>'),
        ('<STEP>', '</STEP>'),
        ('<CHECK>', '</CHECK>'),
        ('<BRANCH>', '</BRANCH>')
    ]


def decode_ir_buffer(tokenizer, token_ids: List[int]) -> str:
    """
    Decode IR buffer tokens to human-readable string.

    Args:
        tokenizer: Extended tokenizer
        token_ids: List of token IDs from IR buffer

    Returns:
        Decoded string with tags and codes
    """
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return ' '.join(tokens)


def is_code_token(token_id: int, ir_token_ids: Dict) -> bool:
    """Check if token ID is a VQ code token."""
    return ir_token_ids['code_start'] <= token_id <= ir_token_ids['code_end']


def is_tag_token(token_id: int, ir_token_ids: Dict) -> bool:
    """Check if token ID is a structural tag."""
    tag_ids = [
        ir_token_ids['ir_start'], ir_token_ids['ir_end'],
        ir_token_ids['goal'], ir_token_ids['goal_end'],
        ir_token_ids['assume'], ir_token_ids['assume_end'],
        ir_token_ids['step'], ir_token_ids['step_end'],
        ir_token_ids['check'], ir_token_ids['check_end'],
        ir_token_ids['branch'], ir_token_ids['branch_end']
    ]
    return token_id in tag_ids


if __name__ == "__main__":
    # Test tokenizer extension
    tokenizer, ir_token_ids = extend_tokenizer_for_ir()

    print("\nStructural tag IDs:")
    for key, val in ir_token_ids.items():
        if key not in ['codes', 'code_start', 'code_end']:
            print(f"  {key}: {val}")

    print(f"\nCode token range: {ir_token_ids['code_start']} - {ir_token_ids['code_end']}")
    print(f"Total codes: {len(ir_token_ids['codes'])}")

    # Test encoding/decoding
    test_sequence = [
        ir_token_ids['ir_start'],
        ir_token_ids['goal'],
        ir_token_ids['codes'][47],  # c047
        ir_token_ids['codes'][89],  # c089
        ir_token_ids['goal_end'],
        ir_token_ids['step'],
        ir_token_ids['codes'][201],  # c201
        ir_token_ids['step_end'],
        ir_token_ids['ir_end']
    ]

    decoded = decode_ir_buffer(tokenizer, test_sequence)
    print(f"\nTest IR buffer: {decoded}")
