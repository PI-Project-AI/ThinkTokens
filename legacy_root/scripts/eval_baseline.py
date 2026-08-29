#!/usr/bin/env python3
"""Baseline evaluation on GSM8K dataset."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re
import json
from pathlib import Path

def extract_answer(text):
    """Extract numerical answer from GSM8K format."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    return None

def evaluate_model(model_name, num_samples=100, save_results=True):
    """Evaluate model on GSM8K."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")['test'].select(range(min(num_samples, len(load_dataset("gsm8k", "main")['test']))))

    correct = 0
    total_tokens = 0
    token_counts = []
    results = []

    print(f"\nEvaluating {model_name} on {len(dataset)} samples:")
    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        prompt = f"Question: {example['question']}\nAnswer:"

        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0
            )

            generated = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)

        predicted = extract_answer(generated)
        true_answer = extract_answer(example['answer'])

        is_correct = predicted == true_answer
        if is_correct:
            correct += 1

        num_tokens = outputs.size(1)
        total_tokens += num_tokens
        token_counts.append(num_tokens)

        results.append({
            'question': example['question'][:100],
            'predicted_answer': predicted,
            'true_answer': true_answer,
            'correct': is_correct,
            'num_tokens': num_tokens,
            'generated': generated[:200]
        })

    accuracy = correct / len(dataset)
    avg_tokens = total_tokens / len(dataset)

    print(f"\n{'='*60}")
    print(f"Baseline Results: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(dataset)})")
    print(f"Avg tokens: {avg_tokens:.1f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Min tokens: {min(token_counts)}")
    print(f"Max tokens: {max(token_counts)}")

    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        baseline_file = results_dir / "baseline.json"
        baseline_data = {
            'model_name': model_name,
            'num_samples': len(dataset),
            'accuracy': float(accuracy),
            'avg_tokens': float(avg_tokens),
            'total_tokens': int(total_tokens),
            'min_tokens': int(min(token_counts)),
            'max_tokens': int(max(token_counts)),
            'samples': results
        }

        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        print(f"\nResults saved to {baseline_file}")

    return {
        'accuracy': accuracy,
        'avg_tokens': avg_tokens,
        'total_tokens': total_tokens,
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts)
    }

if __name__ == '__main__':
    # Evaluate baseline model
    results = evaluate_model("EleutherAI/pythia-410m", num_samples=100)
    print(f"\nBaseline metrics saved. Ready for VQ implementation.")
