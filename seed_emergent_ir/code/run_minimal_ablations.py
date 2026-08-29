#!/usr/bin/env python3
"""
Minimal Ablation Harness for V7-Lite & V8
Aligned with current CausalIRModelV2 API

Tests IR causality via 4 interventions on held-out examples with eval softmax sampling.
"""

import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import sys

sys.path.insert(0, str(Path(__file__).parent))

def setup_ir_tokens_simple(tokenizer, num_codes=512):
    """Simple IR token setup matching tokenizer_utils.py"""
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
    tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})

    # Set padding token to dedicated <PAD> token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<PAD>"

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

    return ir_token_ids

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

def evaluate_with_intervention(checkpoint_path, val_data, device, intervention='intact',
                                tau=0.9, topk=32, topp=0.95, num_examples=500, seed=42):
    """
    Run inference with specified IR intervention using checkpoint directly.

    Args:
        checkpoint_path: Path to saved checkpoint (.pt file)
        val_data: Validation examples
        device: torch device
        intervention: 'intact', 'random-IR', 'shuffle-IR', 'drop-IR'
        tau, topk, topp: Eval softmax sampling params
        num_examples: Number of examples to evaluate
        seed: Random seed
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    # Setup IR tokens (includes PAD token setup)
    ir_token_ids = setup_ir_tokens_simple(tokenizer, num_codes=512)

    # Import model after ir_token_ids is set up
    from models.causal_ir_model_v2 import CausalIRModelV2

    # Initialize model - use checkpoint config if available
    model_config = checkpoint.get('config', {})

    model = CausalIRModelV2(
        base_model_name="EleutherAI/pythia-70m",
        ir_token_ids=ir_token_ids,
        num_codes=model_config.get('num_codes', 512),
        code_dim=model_config.get('code_dim', 128),
        temperature_init=1.0,
        temperature_final=0.8,
        pad_token_id=tokenizer.pad_token_id,
        use_contrastive=model_config.get('use_contrastive', True),
        contrastive_weight=model_config.get('contrastive_weight', 0.3),
        contrastive_T=model_config.get('contrastive_T', 0.07),
        use_gumbel_warmstart=model_config.get('use_gumbel_warmstart', True),
        gumbel_tau=model_config.get('gumbel_tau', 0.6),
        gumbel_steps=model_config.get('gumbel_steps', 3000),
        diversity_weight=model_config.get('diversity_weight', 0.5),
        use_ir_value_head=model_config.get('use_ir_value_head', True),
        ir_value_weight=model_config.get('ir_value_weight', 0.25),
        use_concept_head=model_config.get('use_concept_head', False),
        concept_weight=model_config.get('concept_weight', 0.3),
        eval_code_sampling='softmax',
        eval_tau=tau,
        eval_topk=topk,
        eval_topp=topp,
    )

    # Load state dict (strict=False to handle concept_head weights)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"IR→value head: {'ACTIVE' if model_config.get('use_ir_value_head') else 'DISABLED'}")
    print(f"IR→concept head: {'ACTIVE' if model_config.get('use_concept_head') else 'DISABLED'}")

    # Sample examples
    examples = val_data[:num_examples] if len(val_data) > num_examples else val_data

    correct = 0
    total = 0

    with torch.no_grad():
        for ex in tqdm(examples, desc=f"{intervention:12s}"):
            # Tokenize input
            input_text = ex['problem']
            input_ids = tokenizer(input_text, return_tensors='pt', padding=True).input_ids.to(device)

            # Generate IR (pass 1) with eval softmax sampling
            ir_output = model.generate_ir(
                input_ids,
                attention_mask=None,
                temperature=tau,
                max_length=50,
                do_sample=True,
                top_k=topk,
                top_p=topp
            )

            ir_ids = ir_output['ir_ids']
            ir_embeddings = ir_output['ir_embeddings']

            # Apply intervention to IR codes
            if intervention == 'random-IR':
                # Replace IR codes with random codes from codebook
                code_mask = (ir_ids >= ir_token_ids['code_start']) & (ir_ids <= ir_token_ids['code_end'])
                num_codes = code_mask.sum()
                if num_codes > 0:
                    random_code_ids = torch.randint(
                        ir_token_ids['code_start'],
                        ir_token_ids['code_end'] + 1,
                        (num_codes.item(),),
                        device=device
                    )
                    ir_ids = ir_ids.clone()
                    ir_ids[code_mask] = random_code_ids
                    # Set embeddings to None to trigger re-embedding in answer_from_ir
                    ir_embeddings = None

            elif intervention == 'shuffle-IR':
                # Shuffle the order of IR codes
                code_mask = (ir_ids >= ir_token_ids['code_start']) & (ir_ids <= ir_token_ids['code_end'])
                if code_mask.sum() > 1:
                    code_positions = code_mask.nonzero(as_tuple=True)[1]
                    shuffled_positions = code_positions[torch.randperm(len(code_positions))]
                    ir_ids = ir_ids.clone()
                    ir_ids[0, code_positions] = ir_ids[0, shuffled_positions]
                    # Set embeddings to None to trigger re-embedding in answer_from_ir
                    ir_embeddings = None

            elif intervention == 'drop-IR':
                # Drop IR completely: use BOS token to start answer generation
                # This removes all IR context while providing a valid starting token
                bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
                ir_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
                ir_embeddings = None
                # Log once for verification
                if total == 0:
                    print(f"[drop-IR] IR replaced with BOS token (token_id={bos_id})")

            # Generate answer (pass 2) with intervened IR
            answer_ids = model.answer_from_ir(
                input_ids=input_ids,
                ir_ids=ir_ids,
                ir_embeddings=ir_embeddings,
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
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--val_data', type=str, default='../data/arithmetic/val.json')
    parser.add_argument('--num_examples', type=int, default=500)
    parser.add_argument('--eval_tau', type=float, default=0.9)
    parser.add_argument('--eval_topk', type=int, default=32)
    parser.add_argument('--eval_topp', type=float, default=0.95)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Eval sampling: τ={args.eval_tau}, top-k={args.eval_topk}, top-p={args.eval_topp}")
    print("=" * 70)

    # Load validation data
    with open(args.val_data) as f:
        val_data = json.load(f)

    print(f"Running ablations on {min(len(val_data), args.num_examples)} examples\n")

    # Run 4 interventions
    results = {}

    for intervention in ['intact', 'random-IR', 'shuffle-IR', 'drop-IR']:
        acc, correct, total = evaluate_with_intervention(
            args.checkpoint, val_data, device,
            intervention=intervention,
            tau=args.eval_tau, topk=args.eval_topk, topp=args.eval_topp,
            num_examples=args.num_examples, seed=args.seed
        )
        results[intervention] = {
            'accuracy': acc,
            'correct': correct,
            'total': total
        }
        print(f"{intervention:12s}: {acc:.2f}% ({correct}/{total})")

    # Calculate relative drops
    intact_acc = results['intact']['accuracy']
    print("\n" + "=" * 70)
    print("RELATIVE DROPS vs INTACT:")
    print("=" * 70)

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
    print("\n" + "=" * 70)
    print("CAUSALITY ABLATION TABLE:")
    print("=" * 70)
    print(f"{'Condition':<15} {'Accuracy (%)':<15} {'Relative Drop (%)':<20}")
    print("-" * 70)
    print(f"{'intact':<15} {results['intact']['accuracy']:<15.2f} {'—':<20}")
    for intervention in ['random-IR', 'shuffle-IR', 'drop-IR']:
        print(f"{intervention:<15} {results[intervention]['accuracy']:<15.2f} {results[intervention]['relative_drop_pct']:<20.1f}")
    print("=" * 70)

    # Interpretation
    avg_drop = np.mean([results[k]['relative_drop_pct'] for k in ['random-IR', 'shuffle-IR', 'drop-IR']])
    print("\nINTERPRETATION:")
    if avg_drop >= 70:
        print(f"✅ IR is causally important (avg drop: {avg_drop:.1f}% ≥ 70%)")
    else:
        print(f"⚠ IR is weakly informative (avg drop: {avg_drop:.1f}% < 70%)")

if __name__ == '__main__':
    main()
