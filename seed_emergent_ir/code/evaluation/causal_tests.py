"""
Causality Tests for IR-CoT.

These diagnostic tests verify that the IR buffer is genuinely used for reasoning
and not bypassed by the model.

Success criteria:
- Random-IR test: ≥70% relative accuracy drop
- Shuffle-IR test: ≥70% relative accuracy drop
- Drop-IR test: ≥70% relative accuracy drop (or near random performance)
"""
import torch
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.answer_matching import exact_match
from ir_grammar import validate_ir_integrity


class CausalityTester:
    """
    Implements three diagnostic tests to verify IR causality.

    Tests:
    1. Random-IR: Replace code tokens with random codes
    2. Shuffle-IR: Swap IR buffers between examples
    3. Drop-IR: Remove IR entirely, generate answer from empty context
    """

    def __init__(self, model, tokenizer, ir_token_ids: Dict):
        """
        Args:
            model: CausalIRModel instance
            tokenizer: Extended tokenizer with IR tokens
            ir_token_ids: Dict with token IDs for tags and codes
        """
        self.model = model
        self.tokenizer = tokenizer
        self.ir_token_ids = ir_token_ids

        self.code_start = ir_token_ids['code_start']
        self.code_end = ir_token_ids['code_end']

    def run_all_tests(
        self,
        test_data: List[Dict],
        batch_size: int = 8,
        verbose: bool = True
    ) -> Dict:
        """
        Run all three causality tests.

        Args:
            test_data: List of examples with 'problem' and 'answer' keys
            batch_size: Batch size for evaluation
            verbose: Print detailed results

        Returns:
            Dict with results for each test
        """
        results = {}

        # Baseline: normal accuracy
        print("Running baseline evaluation...")
        baseline_acc, ir_integrity = self.evaluate_baseline_with_integrity(test_data, batch_size)
        results['baseline_accuracy'] = baseline_acc
        results['ir_integrity'] = ir_integrity

        # Test 1: Random IR
        print("\nRunning Random-IR test...")
        random_ir_acc = self.test_random_ir(test_data, batch_size)
        results['random_ir_accuracy'] = random_ir_acc
        results['random_ir_drop'] = (baseline_acc - random_ir_acc) / baseline_acc if baseline_acc > 0 else 0

        # Test 2: Shuffle IR
        print("\nRunning Shuffle-IR test...")
        shuffle_ir_acc = self.test_shuffle_ir(test_data, batch_size)
        results['shuffle_ir_accuracy'] = shuffle_ir_acc
        results['shuffle_ir_drop'] = (baseline_acc - shuffle_ir_acc) / baseline_acc if baseline_acc > 0 else 0

        # Test 3: Drop IR
        print("\nRunning Drop-IR test...")
        drop_ir_acc = self.test_drop_ir(test_data, batch_size)
        results['drop_ir_accuracy'] = drop_ir_acc
        results['drop_ir_drop'] = (baseline_acc - drop_ir_acc) / baseline_acc if baseline_acc > 0 else 0

        # Print summary
        if verbose:
            self._print_results(results)

        # Check if tests pass
        results['all_tests_passed'] = self._check_tests_passed(results)

        return results

    def evaluate_baseline(self, test_data: List[Dict], batch_size: int) -> float:
        """
        Evaluate baseline accuracy (normal inference).

        Args:
            test_data: List of test examples
            batch_size: Batch size

        Returns:
            Accuracy (0-1)
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), batch_size), desc="Baseline"):
                batch = test_data[i:i + batch_size]

                # Tokenize inputs
                input_texts = [ex['problem'] for ex in batch]
                inputs = self.tokenizer(
                    input_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
                input_ids = inputs['input_ids'].to(self.model.base_model.device)

                # Generate answers
                outputs = self.model.generate_answer(input_ids, max_answer_length=10)
                pred_answer_ids = outputs['answer_ids']

                # Decode and compare
                for j, ex in enumerate(batch):
                    pred_text = self.tokenizer.decode(
                        pred_answer_ids[j],
                        skip_special_tokens=True
                    ).strip()

                    true_answer = ex['answer'].strip()

                    # Use exact numeric match
                    matches, _, _ = exact_match(pred_text, true_answer)

                    if matches:
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def evaluate_baseline_with_integrity(self, test_data: List[Dict], batch_size: int) -> tuple:
        """
        Evaluate baseline accuracy and IR integrity.

        Args:
            test_data: List of test examples
            batch_size: Batch size

        Returns:
            Tuple of (accuracy, integrity_dict)
        """
        self.model.eval()
        correct = 0
        total = 0
        all_ir_buffers = []

        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), batch_size), desc="Baseline + Integrity"):
                batch = test_data[i:i + batch_size]

                # Tokenize inputs
                input_texts = [ex['problem'] for ex in batch]
                inputs = self.tokenizer(
                    input_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
                input_ids = inputs['input_ids'].to(self.model.base_model.device)

                # Generate answers
                outputs = self.model.generate_answer(input_ids, max_answer_length=10)
                pred_answer_ids = outputs['answer_ids']
                ir_token_ids = outputs['ir_token_ids']

                # Collect IR buffers for integrity check
                all_ir_buffers.append(ir_token_ids)

                # Decode and compare
                for j, ex in enumerate(batch):
                    pred_text = self.tokenizer.decode(
                        pred_answer_ids[j],
                        skip_special_tokens=True
                    ).strip()

                    true_answer = ex['answer'].strip()

                    # Use exact numeric match
                    matches, _, _ = exact_match(pred_text, true_answer)

                    if matches:
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0

        # Check IR integrity on all collected buffers
        all_ir = torch.cat(all_ir_buffers, dim=0)
        integrity_results = validate_ir_integrity(
            all_ir,
            self.ir_token_ids,
            min_codes_per_span=3,
            max_codes_per_span=6
        )

        return accuracy, integrity_results

    def test_random_ir(self, test_data: List[Dict], batch_size: int) -> float:
        """
        Test with random IR: replace all code tokens with random codes.

        If model relies on IR, accuracy should crash.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), batch_size), desc="Random-IR"):
                batch = test_data[i:i + batch_size]

                # Tokenize inputs
                input_texts = [ex['problem'] for ex in batch]
                inputs = self.tokenizer(
                    input_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
                input_ids = inputs['input_ids'].to(self.model.base_model.device)

                # Generate IR buffer
                ir_output = self.model.ir_generator(
                    input_ids=input_ids,
                    target_ir_ids=None
                )
                ir_token_ids = ir_output['ir_token_ids']

                # REPLACE CODES WITH RANDOM
                randomized_ir = self._randomize_codes(ir_token_ids)

                # Generate answer from randomized IR
                pred_answer_ids = self._generate_from_ir(randomized_ir, max_length=10)

                # Decode and compare
                for j, ex in enumerate(batch):
                    pred_text = self.tokenizer.decode(
                        pred_answer_ids[j],
                        skip_special_tokens=True
                    ).strip()

                    true_answer = ex['answer'].strip()

                    matches, _, _ = exact_match(pred_text, true_answer)

                    if matches:
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def test_shuffle_ir(self, test_data: List[Dict], batch_size: int) -> float:
        """
        Test with shuffled IR: swap IR buffers between examples.

        If model relies on IR content, accuracy should crash.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), batch_size), desc="Shuffle-IR"):
                batch = test_data[i:i + batch_size]

                if len(batch) < 2:
                    continue  # Need at least 2 examples to shuffle

                # Tokenize inputs
                input_texts = [ex['problem'] for ex in batch]
                inputs = self.tokenizer(
                    input_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
                input_ids = inputs['input_ids'].to(self.model.base_model.device)

                # Generate IR buffers
                ir_output = self.model.ir_generator(
                    input_ids=input_ids,
                    target_ir_ids=None
                )
                ir_token_ids = ir_output['ir_token_ids']

                # SHUFFLE IR BUFFERS
                shuffled_ir = ir_token_ids[torch.randperm(ir_token_ids.shape[0])]

                # Generate answers from shuffled IR
                pred_answer_ids = self._generate_from_ir(shuffled_ir, max_length=10)

                # Decode and compare
                for j, ex in enumerate(batch):
                    pred_text = self.tokenizer.decode(
                        pred_answer_ids[j],
                        skip_special_tokens=True
                    ).strip()

                    true_answer = ex['answer'].strip()

                    matches, _, _ = exact_match(pred_text, true_answer)

                    if matches:
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def test_drop_ir(self, test_data: List[Dict], batch_size: int) -> float:
        """
        Test with no IR: generate answer from empty/minimal context.

        If model relies on IR, accuracy should crash to near-random.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), batch_size), desc="Drop-IR"):
                batch = test_data[i:i + batch_size]

                # Generate answer with NO IR context
                # Use only IR_START and IR_END tokens (empty IR)
                device = self.model.base_model.device
                empty_ir = torch.tensor(
                    [[self.ir_token_ids['ir_start'], self.ir_token_ids['ir_end']]]
                ).repeat(len(batch), 1).to(device)

                pred_answer_ids = self._generate_from_ir(empty_ir, max_length=10)

                # Decode and compare
                for j, ex in enumerate(batch):
                    pred_text = self.tokenizer.decode(
                        pred_answer_ids[j],
                        skip_special_tokens=True
                    ).strip()

                    true_answer = ex['answer'].strip()

                    matches, _, _ = exact_match(pred_text, true_answer)

                    if matches:
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def _randomize_codes(self, ir_token_ids: torch.Tensor) -> torch.Tensor:
        """Replace all code tokens with random codes."""
        randomized = ir_token_ids.clone()

        # Find code positions
        code_mask = (ir_token_ids >= self.code_start) & (ir_token_ids <= self.code_end)

        # Replace with random codes
        num_codes = self.code_end - self.code_start + 1
        random_codes = torch.randint(
            self.code_start,
            self.code_end + 1,
            size=(code_mask.sum().item(),),
            device=ir_token_ids.device
        )

        randomized[code_mask] = random_codes

        return randomized

    def _generate_from_ir(self, ir_token_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        """Generate answer tokens from IR buffer."""
        current_seq = ir_token_ids

        for _ in range(max_length):
            outputs = self.model.base_model(input_ids=current_seq)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            current_seq = torch.cat([current_seq, next_token], dim=1)

            # Check for EOS
            if (next_token == self.model.base_model.config.eos_token_id).all():
                break

        # Extract answer (after IR)
        ir_len = ir_token_ids.shape[1]
        answer_ids = current_seq[:, ir_len:]

        return answer_ids

    def _print_results(self, results: Dict):
        """Print formatted test results."""
        print("\n" + "="*60)
        print("CAUSALITY TEST RESULTS")
        print("="*60)

        baseline = results['baseline_accuracy']
        print(f"\nBaseline Accuracy: {baseline:.2%}")

        # Print IR integrity
        if 'ir_integrity' in results:
            integrity = results['ir_integrity']
            print(f"\nIR Integrity:")
            print(f"  Valid: {integrity['is_valid']}")
            print(f"  Error rate: {integrity['error_rate']:.2%}")
            print(f"  Malformed: {integrity['num_errors']}/{integrity['total_examples']}")
            if integrity['num_errors'] > 0 and len(integrity['errors']) > 0:
                print(f"  First error: {integrity['errors'][0]}")

        print("\n--- Test Results ---")
        for test_name in ['random_ir', 'shuffle_ir', 'drop_ir']:
            acc_key = f'{test_name}_accuracy'
            drop_key = f'{test_name}_drop'

            acc = results[acc_key]
            drop = results[drop_key]

            # Check if test passed (≥70% relative drop)
            passed = drop >= 0.70
            status = "✓ PASS" if passed else "✗ FAIL"

            print(f"\n{test_name.replace('_', ' ').title()}:")
            print(f"  Accuracy: {acc:.2%} (baseline: {baseline:.2%})")
            print(f"  Relative drop: {drop:.2%}")
            print(f"  Status: {status}")

        print("\n" + "="*60)

    def _check_tests_passed(self, results: Dict) -> bool:
        """Check if all causality tests passed."""
        threshold = 0.70

        random_passed = results['random_ir_drop'] >= threshold
        shuffle_passed = results['shuffle_ir_drop'] >= threshold
        drop_passed = results['drop_ir_drop'] >= threshold

        return random_passed and shuffle_passed and drop_passed


if __name__ == "__main__":
    print("Causality tests module loaded successfully.")
    print("\nUsage:")
    print("  tester = CausalityTester(model, tokenizer, ir_token_ids)")
    print("  results = tester.run_all_tests(test_data)")
