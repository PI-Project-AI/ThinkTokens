#!/usr/bin/env python3
"""
A/B Testing: Evaluate checkpoint with different code sampling modes.

Compares argmin (greedy) vs softmax (temperature) vs gumbel sampling at eval time.
"""
import os
import json
import torch
from pathlib import Path

def run_ab_eval(checkpoint_path, output_dir, step_num):
    """Run A/B evaluation with different sampling modes."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from models.causal_ir_model_v2 import CausalIRModelV2
    from transformers import AutoTokenizer
    import numpy as np

    print(f"\n{'='*70}")
    print(f"A/B EVALUATION: Step {step_num}")
    print(f"{'='*70}\n")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model config from checkpoint
    model_config = checkpoint.get('model_config', {})

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_config.get('base_model', 'EleutherAI/pythia-70m'))

    # Add special tokens (simplified version)
    special_tokens = {
        'pad_token': '<PAD>',
        'additional_special_tokens': [
            '<IR_START>', '<IR_END>',
            '<GOAL>', '</GOAL>',
            '<ASSUME>', '</ASSUME>',
            '<STEP>', '</STEP>',
            '<CHECK>', '</CHECK>',
            '<BRANCH>', '</BRANCH>'
        ] + [f'<CODE_{i}>' for i in range(512)]
    }
    tokenizer.add_special_tokens(special_tokens)

    # Test inputs
    test_inputs = [
        "What is 7 + 13?",
        "What is 1 - 1?",
        "What is 15 + 4 - 11?",
        "What is 6 + 5 + 11?",
        "What is (8 * 10) + 1?"
    ]

    # Sampling modes to test
    sampling_modes = ['argmin', 'softmax', 'gumbel']

    results = {}

    for mode in sampling_modes:
        print(f"\n--- Testing mode: {mode.upper()} ---")

        # Recreate model with specific sampling mode
        model = CausalIRModelV2(
            base_model_name=model_config.get('base_model', 'EleutherAI/pythia-70m'),
            tokenizer=tokenizer,
            num_codes=model_config.get('num_codes', 512),
            code_dim=model_config.get('code_dim', 128),
            use_contrastive=model_config.get('use_contrastive', True),
            contrastive_weight=model_config.get('contrastive_weight', 0.3),
            contrastive_temperature=model_config.get('contrastive_temperature', 0.07),
            # Eval sampling config
            eval_code_sampling=mode,
            eval_tau=0.9,
            eval_topk=32,
            eval_topp=0.95
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()

        # Generate IRs
        mode_results = {
            'mode': mode,
            'tau': 0.9 if mode != 'argmin' else None,
            'examples': []
        }

        all_codes = []
        for inp in test_inputs:
            with torch.no_grad():
                # Tokenize input
                input_ids = tokenizer.encode(inp, return_tensors='pt').to(device)

                # Generate IR
                ir_output = model.ir_generator.generate_ir(
                    input_ids=input_ids,
                    max_length=50,
                    temperature=0.8
                )

                # Decode
                ir_text = tokenizer.decode(ir_output[0], skip_special_tokens=False)

                # Extract codes
                ir_tokens = ir_output[0].cpu().tolist()
                code_start = model.ir_token_ids['code_start']
                code_end = model.ir_token_ids['code_end']
                codes_in_ir = [t for t in ir_tokens if code_start <= t <= code_end]
                all_codes.extend(codes_in_ir)

                mode_results['examples'].append({
                    'input': inp,
                    'ir_text': ir_text,
                    'codes': codes_in_ir,
                    'num_codes': len(codes_in_ir)
                })

        # Compute metrics
        unique_codes = len(set(all_codes))
        total_codes = len(all_codes)
        utilization = unique_codes / 512

        if all_codes:
            code_counts = {}
            for c in all_codes:
                code_counts[c] = code_counts.get(c, 0) + 1
            top1_freq = max(code_counts.values()) / total_codes if code_counts else 0
        else:
            top1_freq = 0

        mode_results['metrics'] = {
            'unique_codes': unique_codes,
            'total_codes': total_codes,
            'utilization': utilization,
            'top1_code_frequency': top1_freq
        }

        results[mode] = mode_results

        print(f"  Unique codes: {unique_codes}/512 ({utilization*100:.2f}%)")
        print(f"  Top-1 frequency: {top1_freq*100:.2f}%")
        print(f"  Total codes generated: {total_codes}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    for mode, mode_results in results.items():
        output_path = os.path.join(output_dir, f'eval_step{step_num:04d}_{mode}.json')
        with open(output_path, 'w') as f:
            json.dump(mode_results, f, indent=2)
        print(f"\nSaved {mode} results to: {output_path}")

    # Print comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY (Step {step_num})")
    print(f"{'='*70}")
    print(f"{'Mode':<12} {'Utilization':<15} {'Top-1 Freq':<15} {'Unique Codes':<15}")
    print("-" * 70)
    for mode in sampling_modes:
        m = results[mode]['metrics']
        print(f"{mode:<12} {m['utilization']*100:>6.2f}%         {m['top1_code_frequency']*100:>6.2f}%         {m['unique_codes']:>4}/512")
    print(f"{'='*70}\n")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for A/B results')
    parser.add_argument('--step', type=int, required=True,
                       help='Step number for naming')

    args = parser.parse_args()

    run_ab_eval(args.checkpoint, args.output_dir, args.step)
