"""
Sanity test: Train on 8 examples to verify fixes.

Acceptance criteria:
- Non-empty IR buffers generated
- Non-empty answers generated
- Val accuracy > 0%
- Grammar error rate < 20%
- Tokens with loss > 0 per batch
"""
import torch
import json
from pathlib import Path
import sys

from train_v2 import main
from tokenizer_utils import extend_tokenizer_for_ir
from models.causal_ir_model_v2 import CausalIRModelV2
from evaluation.answer_matching import exact_match
from ir_grammar import validate_ir_integrity
from tqdm import tqdm


def create_small_dataset(source_path: str, output_path: str, num_examples: int = 8):
    """Create a small dataset for sanity testing."""
    with open(source_path, 'r') as f:
        data = json.load(f)

    # Take first num_examples
    small_data = data[:num_examples]

    with open(output_path, 'w') as f:
        json.dump(small_data, f, indent=2)

    print(f"Created small dataset with {len(small_data)} examples at {output_path}")


def run_sanity_checks(checkpoint_path: str, test_data_path: str):
    """Run sanity checks on trained model."""
    print("\n" + "="*60)
    print("Running Sanity Checks")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tokenizer, ir_token_ids = extend_tokenizer_for_ir(
        base_model_name='EleutherAI/pythia-70m',
        num_codes=512
    )

    # Load model
    checkpoint = torch.load(checkpoint_path)
    model = CausalIRModelV2(
        base_model_name='EleutherAI/pythia-70m',
        ir_token_ids=ir_token_ids,
        num_codes=512,
        code_dim=128,
        pad_token_id=tokenizer.pad_token_id
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    print(f"\nTesting on {len(test_data)} examples...")

    # Run inference
    all_ir_buffers = []
    all_answers = []
    correct = 0
    total = 0
    empty_ir_count = 0
    empty_answer_count = 0

    with torch.no_grad():
        for i, example in enumerate(tqdm(test_data, desc="Inference")):
            # Tokenize input
            inputs = tokenizer(
                example['problem'],
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            input_ids = inputs['input_ids'].to(device)

            # Generate answer
            outputs = model.generate_answer(
                input_ids,
                temperature=0.5,
                max_answer_length=20
            )

            ir_buffer = outputs['ir_token_ids']
            answer_ids = outputs['answer_ids']

            # Decode
            ir_text = tokenizer.decode(ir_buffer[0], skip_special_tokens=False)
            answer_text = tokenizer.decode(answer_ids[0], skip_special_tokens=True).strip()

            # Check for empty
            if ir_buffer.shape[1] <= 2:  # Only start/end tokens
                empty_ir_count += 1
            if not answer_text:
                empty_answer_count += 1

            all_ir_buffers.append(ir_buffer)
            all_answers.append(answer_text)

            # Check accuracy
            true_answer = example['answer'].strip()
            matches, _, _ = exact_match(answer_text, true_answer)
            if matches:
                correct += 1
            total += 1

            # Print first few examples
            if i < 3:
                print(f"\nExample {i+1}:")
                print(f"  Problem: {example['problem']}")
                print(f"  IR: {ir_text[:100]}...")
                print(f"  Predicted: {answer_text}")
                print(f"  True: {true_answer}")
                print(f"  Match: {matches}")

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0

    # Check IR integrity
    all_ir = torch.cat(all_ir_buffers, dim=0)
    integrity = validate_ir_integrity(
        all_ir,
        ir_token_ids,
        min_codes_per_span=3,
        max_codes_per_span=6
    )

    # Print results
    print("\n" + "="*60)
    print("SANITY CHECK RESULTS")
    print("="*60)
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"IR Grammar Error Rate: {integrity['error_rate']:.2%}")
    print(f"Empty IR Buffers: {empty_ir_count}/{total}")
    print(f"Empty Answers: {empty_answer_count}/{total}")

    # Check acceptance criteria
    print("\n--- Acceptance Criteria ---")
    checks = {
        "Non-empty IR buffers": empty_ir_count == 0,
        "Non-empty answers": empty_answer_count == 0,
        "Accuracy > 0%": accuracy > 0,
        "Grammar error < 20%": integrity['error_rate'] < 0.20
    }

    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {check}")

    all_passed = all(checks.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL SANITY CHECKS PASSED")
    else:
        print("✗ SOME SANITY CHECKS FAILED")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity test on 8 examples")
    parser.add_argument('--create_dataset', action='store_true',
                       help='Create small test dataset')
    parser.add_argument('--source_data', type=str,
                       default='../data/arithmetic/train.json')
    parser.add_argument('--test_data', type=str,
                       default='../data/arithmetic/sanity_test.json')
    parser.add_argument('--checkpoint', type=str,
                       default='../checkpoints/sanity_test/best_model.pt')
    parser.add_argument('--train', action='store_true',
                       help='Run training on small dataset')
    parser.add_argument('--check', action='store_true',
                       help='Run sanity checks on checkpoint')

    args = parser.parse_args()

    if args.create_dataset:
        print("Creating small dataset...")
        create_small_dataset(args.source_data, args.test_data, num_examples=8)

    if args.train:
        print("\nTraining on small dataset...")
        # Import and run training with sanity test settings
        import sys
        sys.argv = [
            'train_v2.py',
            '--model_name', 'EleutherAI/pythia-70m',
            '--train_data', args.test_data,
            '--val_data', args.test_data,
            '--batch_size', '4',
            '--num_epochs', '5',
            '--gradient_checkpointing',
            '--output_dir', '../checkpoints/sanity_test',
            '--ir_teacher_forcing'
        ]
        from train_v2 import main
        import argparse as ap
        parser = ap.ArgumentParser()
        # Re-parse with training args
        exec(open('train_v2.py').read())

    if args.check:
        print("\nRunning sanity checks...")
        passed = run_sanity_checks(args.checkpoint, args.test_data)
        sys.exit(0 if passed else 1)
