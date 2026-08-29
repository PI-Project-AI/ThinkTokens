"""
Exact numeric answer matching for arithmetic evaluation.

Handles normalization: trim whitespace, signs, leading zeros, decimal points.
"""
import re
from typing import Optional, Tuple


def extract_number(text: str) -> Optional[str]:
    """
    Extract first numeric value from text.

    Args:
        text: Text potentially containing a number

    Returns:
        Normalized numeric string or None if no number found
    """
    # Remove common answer prefixes
    text = text.lower()
    text = re.sub(r'(the answer is|answer:|=|result:|output:)', '', text)

    # Find first number (including negative, decimals)
    match = re.search(r'-?\d+\.?\d*', text.strip())

    if match:
        return match.group(0)
    return None


def normalize_number(num_str: str) -> Optional[str]:
    """
    Normalize numeric string for comparison.

    Handles:
    - Leading zeros: "007" → "7"
    - Trailing zeros after decimal: "5.00" → "5.0" or "5"
    - Sign normalization: "-0" → "0"
    - Whitespace trimming

    Args:
        num_str: Raw numeric string

    Returns:
        Normalized string or None if invalid
    """
    if num_str is None:
        return None

    try:
        # Parse as float to handle all cases
        num = float(num_str)

        # Check if it's effectively an integer
        if num == int(num):
            return str(int(num))
        else:
            # Keep decimal but remove unnecessary trailing zeros
            return str(num).rstrip('0').rstrip('.')

    except (ValueError, TypeError):
        return None


def exact_match(pred_text: str, true_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if predicted answer exactly matches ground truth.

    Args:
        pred_text: Generated text from model
        true_text: Ground truth answer text

    Returns:
        Tuple of (matches: bool, pred_normalized: str, true_normalized: str)
    """
    # Extract numbers
    pred_num = extract_number(pred_text)
    true_num = extract_number(true_text)

    # Normalize
    pred_normalized = normalize_number(pred_num) if pred_num else None
    true_normalized = normalize_number(true_num) if true_num else None

    # Compare
    if pred_normalized is None or true_normalized is None:
        return False, pred_normalized, true_normalized

    matches = (pred_normalized == true_normalized)

    return matches, pred_normalized, true_normalized


# Test cases
if __name__ == "__main__":
    test_cases = [
        # (pred, true, expected_match)
        ("The answer is 42", "42", True),
        ("42", "42", True),
        ("007", "7", True),
        ("-5", "-5", True),
        ("5.0", "5", True),
        ("5.00", "5.0", True),
        ("-0", "0", True),
        ("  12  ", "12", True),
        ("The result is 123", "123", True),
        ("42", "43", False),
        ("-5", "5", False),
        ("5.5", "5", False),
        ("no number here", "5", False),
        ("12 and 34", "12", True),  # Takes first number
    ]

    print("Testing exact_match()...\n")

    passed = 0
    failed = 0

    for pred, true, expected in test_cases:
        matches, pred_norm, true_norm = exact_match(pred, true)

        status = "✓" if matches == expected else "✗"

        if matches == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status} pred='{pred}' true='{true}'")
        print(f"   → pred_norm={pred_norm}, true_norm={true_norm}, match={matches}")

        if matches != expected:
            print(f"   EXPECTED: {expected}")
        print()

    print(f"\nResults: {passed} passed, {failed} failed")
