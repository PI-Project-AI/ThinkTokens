#!/usr/bin/env python3
"""
V9 Concept Extraction: Add semantic concept labels to arithmetic dataset.

Extracts deterministic features from problem expressions and answers:
- num_terms: Count of numeric operands
- operation_types: Multi-hot [has_add, has_sub, has_mul]
- max_operand_magnitude: Bucketed magnitude of largest operand
- depth: 1 (flat) or 2 (has parentheses)
- has_carry_addition: Whether any addition requires carrying
- difficulty: 0 (easy), 1 (medium), 2 (hard)
- parity: 0 (even answer), 1 (odd answer)
- sign: 0 (positive answer), 1 (negative answer)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def extract_expression(problem: str) -> str:
    """Extract arithmetic expression from 'What is X?' format."""
    match = re.search(r'What is (.+)\?', problem)
    if match:
        return match.group(1).strip()
    return problem.strip()


def parse_expression(expr: str) -> Tuple[List[int], List[str], bool]:
    """
    Parse expression to extract operands, operators, and depth.

    Returns:
        operands: List of integer operands
        operators: List of operators ('+', '-', '*')
        has_parens: Whether expression contains parentheses
    """
    has_parens = '(' in expr

    # Remove parentheses for parsing
    expr_clean = expr.replace('(', '').replace(')', '')

    # Extract operands and operators
    # Split on operators while keeping them
    tokens = re.split(r'(\+|\-|\*)', expr_clean)

    operands = []
    operators = []

    for i, token in enumerate(tokens):
        token = token.strip()
        if not token:
            continue
        if token in ['+', '-', '*']:
            operators.append(token)
        else:
            try:
                operands.append(int(token))
            except ValueError:
                # Handle edge cases like negative numbers
                if token.startswith('-'):
                    try:
                        operands.append(int(token))
                    except:
                        pass

    return operands, operators, has_parens


def has_carry_in_addition(operands: List[int], operators: List[str]) -> bool:
    """
    Check if any addition operation would require carrying.

    Simple heuristic: For addition a + b, check if any digit position sums to ≥10.
    """
    for i, op in enumerate(operators):
        if op == '+' and i < len(operands) - 1:
            a, b = operands[i], operands[i + 1]
            # Convert to strings and check digit-wise
            str_a, str_b = str(abs(a)), str(abs(b))
            max_len = max(len(str_a), len(str_b))
            str_a = str_a.zfill(max_len)
            str_b = str_b.zfill(max_len)

            carry = 0
            for j in range(max_len - 1, -1, -1):
                digit_sum = int(str_a[j]) + int(str_b[j]) + carry
                if digit_sum >= 10:
                    return True
                carry = 1 if digit_sum >= 10 else 0

    return False


def extract_concepts(problem: str, answer: str, existing_ops: List[str], existing_difficulty: str) -> Dict:
    """
    Extract all concept features from a problem.

    Args:
        problem: Problem text (e.g., "What is 3 + 5?")
        answer: Answer string (e.g., "8")
        existing_ops: Existing operations list from data
        existing_difficulty: Existing difficulty from data

    Returns:
        Dictionary of concept features
    """
    expr = extract_expression(problem)
    operands, operators, has_parens = parse_expression(expr)

    # Parse answer
    try:
        answer_value = int(answer)
    except ValueError:
        answer_value = 0

    # 1. num_terms: count of numeric operands
    num_terms = len(operands)

    # 2. operation_types: multi-hot [has_add, has_sub, has_mul]
    has_add = 1 if '+' in operators else 0
    has_sub = 1 if '-' in operators else 0
    has_mul = 1 if '*' in operators else 0
    operation_types = [has_add, has_sub, has_mul]

    # 3. max_operand_magnitude: bucketed (0: <10, 1: <20, 2: <50, 3: >=50)
    max_operand = max([abs(op) for op in operands]) if operands else 0
    if max_operand < 10:
        max_operand_magnitude = 0
    elif max_operand < 20:
        max_operand_magnitude = 1
    elif max_operand < 50:
        max_operand_magnitude = 2
    else:
        max_operand_magnitude = 3

    # 4. depth: 1 (flat) or 2 (has parentheses)
    depth = 2 if has_parens else 1

    # 5. has_carry_addition: Whether any addition requires carrying
    has_carry = 1 if has_carry_in_addition(operands, operators) else 0

    # 6. difficulty: map existing difficulty to 0/1/2
    difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
    difficulty = difficulty_map.get(existing_difficulty, 1)

    # 7. parity: 0 (even), 1 (odd)
    parity = answer_value % 2

    # 8. sign: 0 (positive or zero), 1 (negative)
    sign = 1 if answer_value < 0 else 0

    return {
        'num_terms': num_terms,
        'operation_types': operation_types,
        'max_operand_magnitude': max_operand_magnitude,
        'depth': depth,
        'has_carry_addition': has_carry,
        'difficulty': difficulty,
        'parity': parity,
        'sign': sign
    }


def process_dataset(input_path: str, output_path: str):
    """Add concept labels to dataset."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Processing {len(data)} examples from {input_path}...")

    for i, example in enumerate(data):
        concepts = extract_concepts(
            problem=example['problem'],
            answer=example['answer'],
            existing_ops=example.get('operations', []),
            existing_difficulty=example.get('difficulty', 'medium')
        )
        example['concepts'] = concepts

    # Write output
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Wrote {len(data)} examples to {output_path}")

    # Print statistics
    print("\nConcept Statistics:")
    print(f"  num_terms: min={min(ex['concepts']['num_terms'] for ex in data)}, "
          f"max={max(ex['concepts']['num_terms'] for ex in data)}")

    op_types_counts = {
        'add_only': sum(1 for ex in data if ex['concepts']['operation_types'] == [1,0,0]),
        'sub_only': sum(1 for ex in data if ex['concepts']['operation_types'] == [0,1,0]),
        'mul_only': sum(1 for ex in data if ex['concepts']['operation_types'] == [0,0,1]),
        'mixed': sum(1 for ex in data if sum(ex['concepts']['operation_types']) > 1)
    }
    print(f"  operation_types: {op_types_counts}")

    difficulty_counts = {
        'easy': sum(1 for ex in data if ex['concepts']['difficulty'] == 0),
        'medium': sum(1 for ex in data if ex['concepts']['difficulty'] == 1),
        'hard': sum(1 for ex in data if ex['concepts']['difficulty'] == 2)
    }
    print(f"  difficulty: {difficulty_counts}")

    has_carry_count = sum(1 for ex in data if ex['concepts']['has_carry_addition'] == 1)
    print(f"  has_carry_addition: {has_carry_count}/{len(data)} ({100*has_carry_count/len(data):.1f}%)")

    parity_counts = {
        'even': sum(1 for ex in data if ex['concepts']['parity'] == 0),
        'odd': sum(1 for ex in data if ex['concepts']['parity'] == 1)
    }
    print(f"  parity: {parity_counts}")


def main():
    parser = argparse.ArgumentParser(description='Add V9 concept labels to arithmetic dataset')
    parser.add_argument('--input_dir', type=str, default='../data/arithmetic',
                        help='Input directory with train.json, val.json')
    parser.add_argument('--output_dir', type=str, default='../data/arithmetic_v9',
                        help='Output directory for V9 dataset')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Process train, val, and test (if exists)
    for split in ['train', 'val', 'test']:
        input_file = input_dir / f"{split}.json"
        if input_file.exists():
            output_file = output_dir / f"{split}_v9.json"
            process_dataset(str(input_file), str(output_file))
            print()

    print("✓ V9 dataset preprocessing complete!")


if __name__ == '__main__':
    main()
