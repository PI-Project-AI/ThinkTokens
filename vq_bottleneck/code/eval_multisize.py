#!/usr/bin/env python3
"""Multi-scale evaluation script for VQ models."""

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re
import json
from pathlib import Path
from vq_model_v2 import VQLanguageModel
import argparse

def extract_answer(text):
    """Extract numerical answer from GSM8K format."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    return None

def evaluate_model(model_size, num_samples=100):
    """Evaluate VQ model at specified scale."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_dir = Path(f"checkpoints_{model_size}")
    results_dir = Path(f"results_{model_size}")
    results_dir.mkdir(exist_ok=True)

    if not checkpoints_dir.exists():
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        print(f"Please train the {model_size} model first using: python train_multisize.py --model {model_size}")
        return None

    # Load config to get model name
    config_path = results_dir / "training_config.json"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return None

    with open(config_path) as f:
        config = json.load(f)

    model_name = config['model_name']
    num_codes = config.get('num_codes', 512)

    print(f"\n{'='*80}")
    print(f"EVALUATING VQ MODEL - {model_size}")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Device: {device}\n")

    # Load model
    print(f"Loading VQ model from {checkpoints_dir}/final_model.pt...")
    # Check if model was trained with gradient checkpointing
    use_checkpointing = model_size in ["1.4B", "2.8B"]
    model = VQLanguageModel(
        model_name,
        num_codes=num_codes,
        use_gradient_checkpointing=use_checkpointing
    ).to(device)

    checkpoint = torch.load(checkpoints_dir / "final_model.pt", map_location=device)
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

    # Get codebook stats
    codebook_stats = model.get_codebook_usage()

    print(f"\n{'='*60}")
    print(f"VQ Model Results - {model_size}")
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
    vq_results = {
        'model_size': model_size,
        'model_name': model_name,
        'checkpoint': str(checkpoints_dir / "final_model.pt"),
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

def main():
    parser = argparse.ArgumentParser(description="Multi-scale VQ model evaluation")
    parser.add_argument("--model", type=str, default="1.4B",
                       choices=["160M", "410M", "1.4B", "2.8B", "all"],
                       help="Model size to evaluate")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of test samples")

    args = parser.parse_args()

    if args.model == "all":
        print("Evaluating all trained models...")
        for model_size in ["160M", "410M", "1.4B", "2.8B"]:
            checkpoints_dir = Path(f"checkpoints_{model_size}")
            if checkpoints_dir.exists():
                try:
                    evaluate_model(model_size, args.samples)
                except Exception as e:
                    print(f"Error evaluating {model_size}: {e}")
            else:
                print(f"Skipping {model_size} (not trained yet)")
    else:
        evaluate_model(args.model, args.samples)

if __name__ == '__main__':
    main()
