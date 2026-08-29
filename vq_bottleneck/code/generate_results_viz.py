#!/usr/bin/env python3
"""Generate visualizations for VQ bottleneck results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load results
results_410m = json.load(open('results_410M/vq_results.json'))
results_1_4b = json.load(open('results_1.4B/vq_results.json'))

# Create output directory
output_dir = Path('docs/results/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Figure 1: Codebook Utilization Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = ['410M', '1.4B']
codes_used = [results_410m['codebook_stats']['num_codes_used'],
              results_1_4b['codebook_stats']['num_codes_used']]
codes_total = [512, 512]
codes_unused = [total - used for total, used in zip(codes_total, codes_used)]

x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width/2, codes_used, width, label='Codes Used', color='#2ecc71')
axes[0].bar(x + width/2, codes_unused, width, label='Codes Unused', color='#e74c3c')
axes[0].set_xlabel('Model Size')
axes[0].set_ylabel('Number of Codes')
axes[0].set_title('Codebook Utilization (Total: 512 codes)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Add percentage labels
for i, (used, total) in enumerate(zip(codes_used, codes_total)):
    pct = 100 * used / total
    axes[0].text(i, used/2, f'{pct:.1f}%', ha='center', va='center',
                fontweight='bold', fontsize=12)

# Utilization percentage
utilization = [100 * used / total for used, total in zip(codes_used, codes_total)]
axes[1].bar(models, utilization, color=['#3498db', '#9b59b6'])
axes[1].set_ylabel('Utilization (%)')
axes[1].set_title('Codebook Utilization Rate')
axes[1].set_ylim([0, 100])
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (model, util) in enumerate(zip(models, utilization)):
    axes[1].text(i, util + 2, f'{util:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'codebook_utilization.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'codebook_utilization.png'}")
plt.close()

# Figure 2: Generation Length Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

avg_tokens = [results_410m['avg_tokens'], results_1_4b['avg_tokens']]
min_tokens = [results_410m['min_tokens'], results_1_4b['min_tokens']]
max_tokens = [results_410m['max_tokens'], results_1_4b['max_tokens']]

# Bar chart with error bars
x = np.arange(len(models))
axes[0].bar(x, avg_tokens, color=['#3498db', '#9b59b6'], alpha=0.7, edgecolor='black')
axes[0].errorbar(x, avg_tokens,
                yerr=[np.array(avg_tokens) - np.array(min_tokens),
                      np.array(max_tokens) - np.array(avg_tokens)],
                fmt='none', ecolor='black', capsize=5, capthick=2)
axes[0].set_xlabel('Model Size')
axes[0].set_ylabel('Number of Tokens')
axes[0].set_title('Average Generation Length (with min/max range)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, avg in enumerate(avg_tokens):
    axes[0].text(i, avg + 5, f'{avg:.1f}', ha='center', fontweight='bold')

# Token range comparison
token_stats = {
    '410M': [min_tokens[0], avg_tokens[0], max_tokens[0]],
    '1.4B': [min_tokens[1], avg_tokens[1], max_tokens[1]]
}

x_pos = np.arange(3)
width = 0.35

axes[1].bar(x_pos - width/2, token_stats['410M'], width, label='410M',
           color='#3498db', alpha=0.7)
axes[1].bar(x_pos + width/2, token_stats['1.4B'], width, label='1.4B',
           color='#9b59b6', alpha=0.7)
axes[1].set_xlabel('Statistic')
axes[1].set_ylabel('Number of Tokens')
axes[1].set_title('Token Generation Statistics')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(['Min', 'Avg', 'Max'])
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'generation_length.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'generation_length.png'}")
plt.close()

# Figure 3: Overall Performance Summary
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy\n(%)', 'Codebook\nUtilization (%)', 'Avg Tokens\n(normalized)']
scores_410m = [
    results_410m['accuracy'],
    results_410m['codebook_stats']['utilization_pct'],
    100 * results_410m['avg_tokens'] / max(results_410m['avg_tokens'], results_1_4b['avg_tokens'])
]
scores_1_4b = [
    results_1_4b['accuracy'],
    results_1_4b['codebook_stats']['utilization_pct'],
    100 * results_1_4b['avg_tokens'] / max(results_410m['avg_tokens'], results_1_4b['avg_tokens'])
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, scores_410m, width, label='410M', color='#3498db', alpha=0.7)
bars2 = ax.bar(x + width/2, scores_1_4b, width, label='1.4B', color='#9b59b6', alpha=0.7)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score (%)')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 100])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'performance_summary.png'}")
plt.close()

# Figure 4: Scaling Analysis
fig, ax = plt.subplots(figsize=(10, 6))

model_params = [410, 1400]  # millions
accuracies = [results_410m['accuracy'], results_1_4b['accuracy']]
utilizations = [results_410m['codebook_stats']['utilization_pct'],
                results_1_4b['codebook_stats']['utilization_pct']]

ax2 = ax.twinx()

line1 = ax.plot(model_params, accuracies, 'o-', linewidth=2, markersize=10,
               color='#e74c3c', label='Accuracy')
line2 = ax2.plot(model_params, utilizations, 's-', linewidth=2, markersize=10,
                color='#2ecc71', label='Codebook Utilization')

ax.set_xlabel('Model Parameters (Millions)')
ax.set_ylabel('Accuracy (%)', color='#e74c3c')
ax2.set_ylabel('Codebook Utilization (%)', color='#2ecc71')
ax.set_title('Scaling Behavior: Accuracy vs Codebook Utilization')
ax.tick_params(axis='y', labelcolor='#e74c3c')
ax2.tick_params(axis='y', labelcolor='#2ecc71')
ax.set_xticks(model_params)
ax.set_xticklabels(['410M', '1.4B'])
ax.grid(True, alpha=0.3)

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(output_dir / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'scaling_analysis.png'}")
plt.close()

print("\n✓ All visualizations generated successfully!")
print(f"  Output directory: {output_dir}")
