#!/usr/bin/env python3
"""Evaluation script for VQ model."""

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re
import json
from pathlib import Path
from vq_model_v2 import VQLanguageModel

def extract_answer(text):
    """Extract numerical answer from GSM8K format."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    return None

def evaluate_vq_model(
    model_name,
    checkpoint_path,
    num_samples=100,
    num_codes=512,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Evaluate VQ model on GSM8K."""
    device = torch.device(device)

    # Load model
    print(f"Loading VQ model from {checkpoint_path}...")
    model = VQLanguageModel(model_name, num_codes=num_codes).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main")['test'].select(range(min(num_samples, 1319)))

    correct = 0
    total_tokens = 0
    token_counts = []
    codebook_indices_all = []
    results = []

    print(f"\nEvaluating on {len(dataset)} samples:")
    for example in tqdm(dataset):
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

    # Get codebook statistics
    codebook_stats = model.get_codebook_usage()

    print(f"\n{'='*60}")
    print(f"VQ Model Results")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(dataset)})")
    print(f"Avg tokens: {avg_tokens:.1f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Min tokens: {min(token_counts)}")
    print(f"Max tokens: {max(token_counts)}")
    print(f"\nCodebook Statistics:")
    print(f"  Codes used: {codebook_stats['num_codes_used']}/{codebook_stats['num_codes_total']}")
    print(f"  Utilization: {100*codebook_stats['num_codes_used']/codebook_stats['num_codes_total']:.1f}%")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    vq_results = {
        'model_name': model_name,
        'checkpoint': str(checkpoint_path),
        'num_samples': len(dataset),
        'accuracy': float(accuracy),
        'avg_tokens': float(avg_tokens),
        'total_tokens': int(total_tokens),
        'min_tokens': int(min(token_counts)),
        'max_tokens': int(max(token_counts)),
        'codebook_stats': {
            'num_codes_used': int(codebook_stats['num_codes_used']),
            'num_codes_total': int(codebook_stats['num_codes_total']),
            'utilization_pct': 100*codebook_stats['num_codes_used']/codebook_stats['num_codes_total']
        },
        'samples': results
    }

    vq_file = results_dir / "vq_results.json"
    with open(vq_file, 'w') as f:
        json.dump(vq_results, f, indent=2)
    print(f"\nResults saved to {vq_file}")

    return vq_results

if __name__ == '__main__':
    # Find the latest checkpoint
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        final_model = checkpoints_dir / "final_model.pt"
        if final_model.exists():
            results = evaluate_vq_model(
                "EleutherAI/pythia-410m",
                final_model,
                num_samples=100,
                num_codes=512
            )
        else:
            print("No final model found. Run training first.")
    else:
        print("No checkpoints directory found. Run training first.")
