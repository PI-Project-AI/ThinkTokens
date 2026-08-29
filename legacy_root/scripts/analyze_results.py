#!/usr/bin/env python3
"""Analysis and visualization of results."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

def load_results(results_dir="results"):
    """Load all results from directory."""
    results_dir = Path(results_dir)
    results = {}

    if (results_dir / "baseline.json").exists():
        with open(results_dir / "baseline.json") as f:
            results['baseline'] = json.load(f)

    if (results_dir / "vq_results.json").exists():
        with open(results_dir / "vq_results.json") as f:
            results['vq'] = json.load(f)

    if (results_dir / "training_history.json").exists():
        with open(results_dir / "training_history.json") as f:
            results['training_history'] = json.load(f)

    return results

def generate_comparison_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive comparison report."""
    report = []
    report.append("=" * 80)
    report.append("INTERMEDIATE REASONING LANGUAGE - RESULTS ANALYSIS")
    report.append("=" * 80)

    # Baseline metrics
    if 'baseline' in results:
        baseline = results['baseline']
        report.append("\n1. BASELINE MODEL (Pythia-410M)")
        report.append("-" * 80)
        report.append(f"Model: {baseline['model_name']}")
        report.append(f"Accuracy: {baseline['accuracy']:.2%} ({baseline['accuracy']*100:.1f}%)")
        report.append(f"Avg Tokens: {baseline['avg_tokens']:.1f}")
        report.append(f"Total Tokens: {baseline['total_tokens']}")
        report.append(f"Token Range: {baseline['min_tokens']} - {baseline['max_tokens']}")

    # VQ metrics
    if 'vq' in results:
        vq = results['vq']
        report.append("\n2. VQ MODEL (With Intermediate Reasoning)")
        report.append("-" * 80)
        report.append(f"Model: {vq['model_name']}")
        report.append(f"Checkpoint: {Path(vq['checkpoint']).name}")
        report.append(f"Accuracy: {vq['accuracy']:.2%} ({vq['accuracy']*100:.1f}%)")
        report.append(f"Avg Tokens: {vq['avg_tokens']:.1f}")
        report.append(f"Total Tokens: {vq['total_tokens']}")
        report.append(f"Token Range: {vq['min_tokens']} - {vq['max_tokens']}")

        # Codebook stats
        cb_stats = vq['codebook_stats']
        report.append(f"\nCodebook Statistics:")
        report.append(f"  Codes Used: {cb_stats['num_codes_used']}/{cb_stats['num_codes_total']}")
        report.append(f"  Utilization: {cb_stats['utilization_pct']:.1f}%")

    # Comparison
    if 'baseline' in results and 'vq' in results:
        baseline = results['baseline']
        vq = results['vq']

        report.append("\n3. COMPARATIVE ANALYSIS")
        report.append("-" * 80)

        acc_diff = (vq['accuracy'] - baseline['accuracy']) * 100
        if baseline['accuracy'] > 0:
            acc_pct_change = ((vq['accuracy'] - baseline['accuracy']) / baseline['accuracy']) * 100
            report.append(f"Accuracy Change: {acc_diff:+.2f} percentage points ({acc_pct_change:+.1f}%)")
        else:
            report.append(f"Accuracy Change: {acc_diff:+.2f} percentage points (baseline is 0%, cannot compute %)")

        token_reduction = baseline['avg_tokens'] - vq['avg_tokens']
        token_reduction_pct = (token_reduction / baseline['avg_tokens']) * 100

        report.append(f"Token Reduction: {token_reduction:+.1f} tokens ({token_reduction_pct:+.1f}%)")
        report.append(f"  Baseline avg: {baseline['avg_tokens']:.1f}")
        report.append(f"  VQ avg: {vq['avg_tokens']:.1f}")

        # Success criteria
        report.append("\n4. SUCCESS CRITERIA")
        report.append("-" * 80)

        criteria = {
            "Codebook utilization >50%": cb_stats['utilization_pct'] > 50,
            "Token reduction >20%": token_reduction_pct > 20,
            "Accuracy within 5% of baseline": abs(acc_diff) <= 5,
            "No catastrophic accuracy loss": vq['accuracy'] >= baseline['accuracy'] * 0.8
        }

        for criterion, passed in criteria.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            report.append(f"{status}: {criterion}")

        # Overall assessment
        num_passed = sum(criteria.values())
        report.append(f"\nCriteria met: {num_passed}/{len(criteria)}")

    # Training history
    if 'training_history' in results:
        history = results['training_history']
        report.append("\n5. TRAINING PROGRESS")
        report.append("-" * 80)

        for epoch, loss_data in zip(history['epochs'], history['losses']):
            report.append(f"Epoch {epoch}:")
            report.append(f"  Total Loss: {loss_data['total_loss']:.4f}")
            report.append(f"  LM Loss: {loss_data['lm_loss']:.4f}")
            report.append(f"  VQ Loss: {loss_data['vq_loss']:.4f}")
            report.append(f"  Code Usage: {loss_data['code_usage']:.1f}%")

    # Insights and recommendations
    report.append("\n6. KEY INSIGHTS")
    report.append("-" * 80)

    if 'baseline' in results and 'vq' in results:
        vq = results['vq']
        cb_util = vq['codebook_stats']['utilization_pct']

        if cb_util < 20:
            report.append("⚠️  CODEBOOK COLLAPSE: Most codes unused. May need:")
            report.append("   - Lower commitment loss weight")
            report.append("   - Larger codebook (more diversity)")
            report.append("   - Code restart mechanism")
        elif cb_util < 50:
            report.append("⚠️  PARTIAL CODEBOOK USAGE: Room for improvement.")
        else:
            report.append("✓ Good codebook utilization. Codes being effectively used.")

        if token_reduction_pct < 0:
            report.append("⚠️  TOKEN INCREASE: VQ is not compressing tokens.")
        elif token_reduction_pct < 10:
            report.append("⚠️  Minimal compression. Bottleneck may not be effective.")
        elif token_reduction_pct < 20:
            report.append("✓ Modest compression achieved (~10-20%).")
        else:
            report.append("✓ Significant compression achieved (>20%).")

    report.append("\n7. RECOMMENDATIONS")
    report.append("-" * 80)
    report.append("• Validate scaling: Test on Pythia-1.4B to check if gains persist")
    report.append("• Test transfer: Evaluate on SVAMP (out-of-distribution test)")
    report.append("• Analyze codes: Visualize learned codebook structure")
    report.append("• Ablation studies: Try different bottleneck placements")
    report.append("• Interpretability: Probe codebook for semantic meaning")

    report.append("\n" + "=" * 80)

    return "\n".join(report)

def create_visualizations(results: Dict[str, Any], output_dir="results"):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Only create plots if we have data
    if 'baseline' not in results or 'vq' not in results:
        print("Insufficient data for visualizations")
        return

    baseline = results['baseline']
    vq = results['vq']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Intermediate Reasoning Language - Performance Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    models = ['Baseline', 'VQ Model']
    accuracies = [baseline['accuracy'] * 100, vq['accuracy'] * 100]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Token efficiency
    ax = axes[0, 1]
    tokens = [baseline['avg_tokens'], vq['avg_tokens']]
    bars = ax.bar(models, tokens, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Avg Tokens', fontsize=11, fontweight='bold')
    ax.set_title('Average Tokens Per Sample', fontsize=12, fontweight='bold')
    for bar, tok in zip(bars, tokens):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{tok:.0f}', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Codebook utilization
    ax = axes[1, 0]
    cb_util = vq['codebook_stats']['utilization_pct']
    cb_unused = 100 - cb_util
    sizes = [cb_util, cb_unused]
    labels = [f'Used\n({vq["codebook_stats"]["num_codes_used"]} codes)',
              f'Unused\n({vq["codebook_stats"]["num_codes_total"] - vq["codebook_stats"]["num_codes_used"]} codes)']
    colors_pie = ['#2ecc71', '#ecf0f1']
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90, textprops={'fontweight': 'bold'})
    ax.set_title('Codebook Utilization', fontsize=12, fontweight='bold')

    # Plot 4: Metrics summary
    ax = axes[1, 1]
    ax.axis('off')

    token_reduction = baseline['avg_tokens'] - vq['avg_tokens']
    token_reduction_pct = (token_reduction / baseline['avg_tokens']) * 100
    acc_change = (vq['accuracy'] - baseline['accuracy']) * 100

    summary_text = f"""
    KEY METRICS:

    Accuracy Change: {acc_change:+.2f} pp
    Token Reduction: {token_reduction_pct:+.1f}%
    Codebook Usage: {cb_util:.1f}%

    BASELINE:
    • Accuracy: {baseline['accuracy']:.1%}
    • Avg Tokens: {baseline['avg_tokens']:.0f}

    VQ MODEL:
    • Accuracy: {vq['accuracy']:.1%}
    • Avg Tokens: {vq['avg_tokens']:.0f}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontweight='bold')

    plt.tight_layout()
    plot_path = output_dir / "comparison_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {plot_path}")
    plt.close()

    # Training history plot
    if 'training_history' in results:
        history = results['training_history']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

        epochs = history['epochs']
        losses = history['losses']

        # Total loss
        ax = axes[0]
        total_losses = [l['total_loss'] for l in losses]
        lm_losses = [l['lm_loss'] for l in losses]
        vq_losses = [l['vq_loss'] for l in losses]

        ax.plot(epochs, total_losses, 'o-', label='Total Loss', linewidth=2, markersize=8)
        ax.plot(epochs, lm_losses, 's-', label='LM Loss', linewidth=2, markersize=8)
        ax.plot(epochs, vq_losses, '^-', label='VQ Loss', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title('Loss Convergence', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Code usage
        ax = axes[1]
        code_usage = [l['code_usage'] for l in losses]
        ax.plot(epochs, code_usage, 'go-', linewidth=2, markersize=8)
        ax.axhline(y=50, color='r', linestyle='--', label='50% threshold', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Codebook Usage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Codebook Utilization Over Training', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        history_plot_path = output_dir / "training_history.png"
        plt.savefig(history_plot_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved: {history_plot_path}")
        plt.close()

def main():
    """Generate comprehensive analysis report."""
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found. Run evaluation first.")
        return

    # Load results
    print("Loading results...")
    results = load_results()

    if not results:
        print("No results found.")
        return

    # Generate report
    print("\nGenerating analysis report...")
    report = generate_comparison_report(results)
    print(report)

    # Save report
    report_path = results_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results)

    print("\n✓ Analysis complete!")

if __name__ == '__main__':
    main()
