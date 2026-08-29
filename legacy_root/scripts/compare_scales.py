#!/usr/bin/env python3
"""Compare results across different model scales."""

import json
from pathlib import Path
import argparse

def load_results(model_size):
    """Load baseline and VQ results for a model size."""
    results_dir = Path(f"results_{model_size}")

    if not results_dir.exists():
        return None

    results = {}

    # Load baseline
    baseline_file = Path("results") / "baseline.json"  # Shared baseline (410M)
    if baseline_file.exists():
        with open(baseline_file) as f:
            results['baseline'] = json.load(f)

    # Load VQ results
    vq_file = results_dir / "vq_results.json"
    if vq_file.exists():
        with open(vq_file) as f:
            results['vq'] = json.load(f)

    # Load training history
    history_file = results_dir / "training_history.json"
    if history_file.exists():
        with open(history_file) as f:
            results['training_history'] = json.load(f)

    return results if results else None

def generate_comparison_report(all_results):
    """Generate comparison across all scales."""
    report = []
    report.append("=" * 100)
    report.append("INTERMEDIATE REASONING LANGUAGE - MULTI-SCALE COMPARISON")
    report.append("=" * 100)

    # Summary table
    report.append("\n1. SUMMARY TABLE")
    report.append("-" * 100)
    report.append(f"{'Model':<10} {'Accuracy':<12} {'Avg Tokens':<14} {'Token Reduction':<18} {'Code Usage':<15}")
    report.append("-" * 100)

    baseline_accuracy = None
    baseline_tokens = None

    for model_size in sorted(all_results.keys()):
        results = all_results[model_size]
        if results is None:
            continue

        if 'baseline' in results:
            baseline = results['baseline']
            baseline_accuracy = baseline['accuracy']
            baseline_tokens = baseline['avg_tokens']
            report.append(
                f"{'Baseline':<10} {baseline['accuracy']*100:>10.1f}% {baseline['avg_tokens']:>12.1f} "
                f"{'—':<18} {'—':<15}"
            )
            report.append("")

        if 'vq' in results:
            vq = results['vq']
            acc_str = f"{vq['accuracy']*100:.1f}%"

            if baseline_accuracy is not None:
                acc_change = (vq['accuracy'] - baseline_accuracy) * 100
                acc_str += f" ({acc_change:+.1f}pp)"

            token_red = ""
            if baseline_tokens is not None:
                token_red_pct = ((baseline_tokens - vq['avg_tokens']) / baseline_tokens) * 100
                token_red = f"{token_red_pct:+.1f}%"

            code_usage = vq['codebook_stats']['utilization_pct']

            report.append(
                f"{model_size:<10} {acc_str:<12} {vq['avg_tokens']:>12.1f} "
                f"{token_red:>17} {code_usage:>13.1f}%"
            )

    report.append("-" * 100)

    # Detailed results per model
    report.append("\n2. DETAILED RESULTS PER MODEL")
    report.append("-" * 100)

    for model_size in sorted(all_results.keys()):
        results = all_results[model_size]
        if results is None:
            report.append(f"\n{model_size}: No results found")
            continue

        report.append(f"\n{model_size}")
        report.append("=" * 50)

        if 'baseline' in results and 'vq' in results:
            baseline = results['baseline']
            vq = results['vq']

            report.append(f"Baseline:")
            report.append(f"  Accuracy:     {baseline['accuracy']:.2%}")
            report.append(f"  Avg Tokens:   {baseline['avg_tokens']:.1f}")
            report.append(f"  Token Range:  {baseline['min_tokens']}-{baseline['max_tokens']}")

            report.append(f"\nVQ Model:")
            report.append(f"  Accuracy:     {vq['accuracy']:.2%}")
            report.append(f"  Avg Tokens:   {vq['avg_tokens']:.1f}")
            report.append(f"  Token Range:  {vq['min_tokens']}-{vq['max_tokens']}")

            report.append(f"\nComparison:")
            acc_change = (vq['accuracy'] - baseline['accuracy']) * 100
            report.append(f"  Accuracy Δ:   {acc_change:+.2f} pp")

            token_red = baseline['avg_tokens'] - vq['avg_tokens']
            token_red_pct = (token_red / baseline['avg_tokens']) * 100
            report.append(f"  Token Δ:      {token_red:+.1f} tokens ({token_red_pct:+.1f}%)")

            report.append(f"\nCodebook:")
            cb = vq['codebook_stats']
            report.append(f"  Codes Used:   {cb['num_codes_used']}/{cb['num_codes_total']}")
            report.append(f"  Utilization:  {cb['utilization_pct']:.1f}%")

    # Scaling trends
    report.append("\n3. SCALING TRENDS")
    report.append("-" * 100)

    accuracies = {}
    tokens = {}
    code_usage = {}

    for model_size in sorted(all_results.keys()):
        results = all_results[model_size]
        if results and 'vq' in results:
            vq = results['vq']
            accuracies[model_size] = vq['accuracy']
            tokens[model_size] = vq['avg_tokens']
            code_usage[model_size] = vq['codebook_stats']['utilization_pct']

    if len(accuracies) > 1:
        report.append("\nAccuracy Trend:")
        for size in sorted(accuracies.keys()):
            report.append(f"  {size:<10} {accuracies[size]:.2%}")

        report.append("\nToken Efficiency Trend:")
        for size in sorted(tokens.keys()):
            report.append(f"  {size:<10} {tokens[size]:.1f} avg tokens")

        report.append("\nCodebook Usage Trend:")
        for size in sorted(code_usage.keys()):
            report.append(f"  {size:<10} {code_usage[size]:.1f}%")

    # Key insights
    report.append("\n4. KEY INSIGHTS")
    report.append("-" * 100)

    if len(accuracies) > 1:
        acc_trend = "improving" if list(accuracies.values())[-1] > list(accuracies.values())[0] else "declining"
        report.append(f"• Accuracy trend: {acc_trend} with scale")

        if len(code_usage) > 1:
            cb_trend = "improving" if list(code_usage.values())[-1] > list(code_usage.values())[0] else "declining"
            report.append(f"• Codebook usage: {cb_trend} with scale")

    report.append("• Run: python compare_scales.py to regenerate this comparison")

    report.append("\n" + "=" * 100)

    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Compare results across scales")
    parser.add_argument("--models", type=str, default="410M,1.4B",
                       help="Comma-separated model sizes to compare (default: 410M,1.4B)")

    args = parser.parse_args()

    model_sizes = [m.strip() for m in args.models.split(",")]

    print("Loading results...")
    all_results = {}

    for model_size in model_sizes:
        results = load_results(model_size)
        all_results[model_size] = results
        if results:
            print(f"✓ Loaded {model_size}")
        else:
            print(f"✗ No results for {model_size}")

    if not any(all_results.values()):
        print("No results found. Please train and evaluate models first.")
        return

    # Generate report
    print("\nGenerating comparison report...")
    report = generate_comparison_report(all_results)
    print("\n" + report)

    # Save report
    report_path = Path("comparison_all_scales.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

if __name__ == '__main__':
    main()
