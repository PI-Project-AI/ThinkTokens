#!/usr/bin/env python3
"""
Causality Ablations for V8 (with IR→value head)

Tests informativeness of IR codes via interventions:
- intact: normal inference (baseline)
- random-IR: replace IR codes with random codes from codebook
- shuffle-IR: shuffle the order of IR codes within sequence
- drop-IR: zero out IR codes entirely

Expected: If IR is causally important, ablations should severely degrade accuracy.
V8 prediction: Strong drops >70% (IR grounded by value head supervision).
"""

import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.causal_ir_model_v2 import CausalIRModelV2

def normalize_answer(answer: str) -> str:
    """Normalize numeric answers (strip spaces, leading zeros, .0)"""
    answer = answer.strip()
    if not answer:
        return answer
    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer

def evaluate_with_intervention(model, tokenizer, examples, device, intervention='intact',
                               tau=0.9, topk=32, topp=0.95):
    """
    Run inference with specified IR intervention.

    Args:
        intervention: 'intact', 'random-IR', 'shuffle-IR', 'drop-IR'
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for ex in tqdm(examples, desc=f"{intervention:12s}"):
            # Tokenize input
            input_text = ex['question']
            input_ids = tokenizer(input_text, return_tensors='pt', padding=True).input_ids.to(device)

            # Generate IR (pass 1) with eval softmax sampling
            ir_output = model.generate_ir(
                input_ids,
                max_length=50,
                temperature=tau,
                top_k=topk,
                top_p=topp,
                do_sample=True
            )

            ir_ids = ir_output['ir_ids']
            ir_embeddings = ir_output['ir_embeddings']

            # Apply intervention to IR codes
            if intervention == 'random-IR':
                # Replace IR codes with random codes from codebook
                num_codes = (ir_ids >= 50290).sum()  # Count code tokens
                if num_codes > 0:
                    random_code_ids = torch.randint(50290, 50802, (num_codes.item(),), device=device)
                    code_mask = ir_ids >= 50290
                    ir_ids = ir_ids.clone()
                    ir_ids[code_mask] = random_code_ids
                    # Re-embed with random codes
                    ir_embeddings = model.ir_generator.model.get_input_embeddings()(ir_ids)

            elif intervention == 'shuffle-IR':
                # Shuffle the order of IR codes
                code_mask = ir_ids >= 50290
                if code_mask.sum() > 1:
                    code_positions = code_mask.nonzero(as_tuple=True)[1]
                    shuffled_positions = code_positions[torch.randperm(len(code_positions))]
                    ir_ids = ir_ids.clone()
                    ir_ids[0, code_positions] = ir_ids[0, shuffled_positions]
                    ir_embeddings = model.ir_generator.model.get_input_embeddings()(ir_ids)

            elif intervention == 'drop-IR':
                # Zero out IR embeddings (keep structural tokens)
                code_mask = ir_ids >= 50290
                ir_embeddings = ir_embeddings.clone()
                ir_embeddings[0, code_mask] = 0.0

            # Generate answer (pass 2) with intervened IR
            answer_ids = model.generate_answer(
                input_ids,
                ir_ids,
                ir_embeddings,
                max_length=10,
                temperature=0.0  # Greedy for evaluation
            )

            # Decode and normalize
            pred_answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
            pred_answer = normalize_answer(pred_answer)
            true_answer = normalize_answer(ex['answer'])

            if pred_answer == true_answer:
                correct += 1
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to V7-Lite checkpoint')
    parser.add_argument('--val_data', type=str, default='../data/arithmetic/val.json')
    parser.add_argument('--num_examples', type=int, default=500, help='Number of held-out examples')
    parser.add_argument('--eval_tau', type=float, default=0.9)
    parser.add_argument('--eval_topk', type=int, default=32)
    parser.add_argument('--eval_topp', type=float, default=0.95)
    parser.add_argument('--output', type=str, default='../checkpoints/ir_cot_70m_mini_sanity/logs/v7_lite_ablations.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Initialize model (V8 config with IR→value head)
    model = CausalIRModelV2(
        base_model_name="EleutherAI/pythia-70m",
        ir_token_ids=list(range(50290, 50802)),  # 512 code tokens
        num_codes=512,
        code_dim=128,
        use_contrastive=True,
        contrastive_weight=0.3,
        contrastive_T=0.07,
        use_gumbel_warmstart=True,
        gumbel_tau=0.6,
        gumbel_steps=3000,
        diversity_weight=0.5,
        use_ir_value_head=True,  # V8 HAS IR→value head
        ir_value_weight=0.25,
        pad_token_id=tokenizer.pad_token_id
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load validation data
    with open(args.val_data) as f:
        val_data = json.load(f)

    # Sample held-out examples
    if len(val_data) > args.num_examples:
        val_data = val_data[:args.num_examples]

    print(f"Running ablations on {len(val_data)} examples")
    print(f"Eval sampling: τ={args.eval_tau}, top-k={args.eval_topk}, top-p={args.eval_topp}")
    print("="*70)

    # Run 4 interventions
    results = {}

    for intervention in ['intact', 'random-IR', 'shuffle-IR', 'drop-IR']:
        acc, correct, total = evaluate_with_intervention(
            model, tokenizer, val_data, device,
            intervention=intervention,
            tau=args.eval_tau, topk=args.eval_topk, topp=args.eval_topp
        )
        results[intervention] = {
            'accuracy': acc,
            'correct': correct,
            'total': total
        }
        print(f"{intervention:12s}: {acc:.2f}% ({correct}/{total})")

    # Calculate relative drops
    intact_acc = results['intact']['accuracy']
    print("\n" + "="*70)
    print("RELATIVE DROPS vs INTACT:")
    print("="*70)

    for intervention in ['random-IR', 'shuffle-IR', 'drop-IR']:
        drop_pct = ((intact_acc - results[intervention]['accuracy']) / intact_acc * 100) if intact_acc > 0 else 0
        results[intervention]['relative_drop_pct'] = drop_pct
        print(f"{intervention:12s}: {drop_pct:.1f}% drop")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "="*70)
    print("CAUSALITY ABLATION TABLE:")
    print("="*70)
    print(f"{'Condition':<15} {'Accuracy (%)':<15} {'Relative Drop (%)':<20}")
    print("-"*70)
    print(f"{'intact':<15} {results['intact']['accuracy']:<15.2f} {'—':<20}")
    for intervention in ['random-IR', 'shuffle-IR', 'drop-IR']:
        print(f"{intervention:<15} {results[intervention]['accuracy']:<15.2f} {results[intervention]['relative_drop_pct']:<20.1f}")
    print("="*70)

    # Interpretation
    avg_drop = np.mean([results[k]['relative_drop_pct'] for k in ['random-IR', 'shuffle-IR', 'drop-IR']])
    print("\nINTERPRETATION:")
    if avg_drop >= 70:
        print(f"✅ IR is causally important (avg drop: {avg_drop:.1f}% ≥ 70%)")
    else:
        print(f"⚠ IR is weakly informative (avg drop: {avg_drop:.1f}% < 70%)")
        print("   → Supports V8's IR→value head for answer grounding")

if __name__ == '__main__':
    main()
