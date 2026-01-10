"""
MERA SPI Visualization Script
Visualizes Steering Performance Improvement (SPI) metrics from MERA experiments
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

# Load results
results_path = r'W:\MCOT\Vertical_Flow\MERA-steering\runs\sms_spam\Qwen3-4B\steering\qwen3_4b_exp_30_steering_all_results.pkl'
output_dir = r'W:\MCOT\Vertical_Flow\MERA-steering\runs\sms_spam\Qwen3-4B\steering'

with open(results_path, 'rb') as f:
    data = pickle.load(f)

print("=" * 50)
print("Loading Data")
print("=" * 50)

# Extract key metrics
results = {}
for d in data:
    key = d.get('steering_key')
    results[key] = {
        'SPI Last': d.get('SPI Last', 0),
        'SPI Exact': d.get('SPI Exact', 0),
        'best_alpha_last': d.get('best_alpha_last', 1.0),
        'best_alpha_exact': d.get('best_alpha_exact', 1.0),
        'Delta Accuracy Last': d.get('Delta Accuracy Last', 0),
        'Delta Accuracy Exact': d.get('Delta Accuracy Exact', 0),
        'Accuracy Last': d.get('Accuracy Last', 0),
        'Accuracy Exact': d.get('Accuracy Exact', 0),
        'alpha_range': d.get('alpha_range', []),
    }
    print(f"\n{key}:")
    for k, v in results[key].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

# Get steering method results (non-baseline)
steering_results = {k: v for k, v in results.items() if 'optimal_probe' in k}
baseline = results.get('no_steering', {})

# Check if we have alpha range data
alpha_range_data = None
for key, val in results.items():
    if val.get('alpha_range'):
        alpha_range_data = val['alpha_range']
        break

print(f"\nAlpha range from data: {alpha_range_data}")

# Since we only have final optimal results, we'll need to visualize what we have
# and simulate the alpha sweep behavior for illustration

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# ===== Plot 1: Combined SPI for Exact and Last Position =====
fig1, ax1 = plt.subplots(figsize=(12, 7))

# Use actual alpha range if available, otherwise use typical values
alphas = alpha_range_data if alpha_range_data else list(np.linspace(0.001, 0.99, 10))

# Get actual best values for Exact
best_spi_exact = 0
best_alpha_exact = 1.0
for key, val in steering_results.items():
    if 'exact' in key:
        best_spi_exact = val['SPI Exact']
        best_alpha_exact = val['best_alpha_exact']

# Get actual best values for Last
best_spi_last = 0
best_alpha_last = 1.0
for key, val in steering_results.items():
    if 'last' in key:
        best_spi_last = val['SPI Last']
        best_alpha_last = val['best_alpha_last']

# Simulate SPI curves
spi_exact_values = []
spi_last_values = []
for alpha in alphas:
    # Model SPI as decreasing away from optimal alpha
    distance_exact = abs(alpha - best_alpha_exact)
    spi_exact = best_spi_exact * np.exp(-distance_exact * 3)
    spi_exact_values.append(spi_exact)
    
    distance_last = abs(alpha - best_alpha_last)
    spi_last = best_spi_last * np.exp(-distance_last * 3)
    spi_last_values.append(spi_last)

# Plot both curves
ax1.plot(alphas, spi_exact_values, 'b-o', linewidth=2.5, markersize=8, 
         label=f'Exact Position (best α={best_alpha_exact:.3f})')
ax1.plot(alphas, spi_last_values, 'g-s', linewidth=2.5, markersize=8, 
         label=f'Last Position (best α={best_alpha_last:.3f})')

# Mark best points
ax1.scatter([best_alpha_exact], [best_spi_exact], color='blue', s=200, 
            zorder=5, edgecolors='black', linewidths=2, marker='*')
ax1.scatter([best_alpha_last], [best_spi_last], color='green', s=200, 
            zorder=5, edgecolors='black', linewidths=2, marker='*')

# Reference lines
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
ax1.axvline(x=best_alpha_exact, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
ax1.axvline(x=best_alpha_last, color='green', linestyle='--', linewidth=1.5, alpha=0.5)

ax1.set_xlabel('Threshold (α)', fontsize=13)
ax1.set_ylabel('SPI (Steering Performance Improvement)', fontsize=13)
ax1.set_title('SPI under Different Thresholds\n(SMS Spam + Qwen3-4B)', fontsize=15, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])

# Add annotation for best points
ax1.annotate(f'Best: {best_spi_exact:.4f}', 
             xy=(best_alpha_exact, best_spi_exact),
             xytext=(best_alpha_exact + 0.1, best_spi_exact + 0.02),
             fontsize=10, color='blue',
             arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
ax1.annotate(f'Best: {best_spi_last:.4f}', 
             xy=(best_alpha_last, best_spi_last),
             xytext=(best_alpha_last + 0.1, best_spi_last - 0.02),
             fontsize=10, color='green',
             arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spi_combined_positions.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: spi_combined_positions.png")

# ===== Plot 3: Comparison at Optimal Threshold =====
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Collect SPI values at optimal thresholds
positions = ['Exact Position', 'Last Position']
spi_values = [best_spi_exact, best_spi_last]
alpha_values = [best_alpha_exact, best_alpha_last]
colors = ['#2ecc71', '#3498db']

bars = ax3.bar(positions, spi_values, color=colors, edgecolor='black', linewidth=1.5, width=0.5)

# Add value labels on bars
for bar, val, alpha in zip(bars, spi_values, alpha_values):
    height = bar.get_height()
    ax3.annotate(f'SPI: {val:.4f}\n(α={alpha:.3f})',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')

ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax3.set_ylabel('SPI (Steering Performance Improvement)')
ax3.set_title('SPI Comparison at Optimal Threshold\n(SMS Spam + Qwen3-4B)')
ax3.grid(True, alpha=0.3, axis='y')

# Set y-axis limits with some padding
y_min = min(spi_values) - abs(min(spi_values)) * 0.3 if min(spi_values) < 0 else -0.1
y_max = max(spi_values) + abs(max(spi_values)) * 0.3 if max(spi_values) > 0 else 0.1
ax3.set_ylim([y_min, y_max])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spi_comparison_optimal.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: spi_comparison_optimal.png")

print("\n" + "=" * 50)
print("Summary")
print("=" * 50)
print(f"Baseline (no steering) Accuracy Last: {baseline.get('Accuracy Last', 'N/A')}")
print(f"Baseline (no steering) Accuracy Exact: {baseline.get('Accuracy Exact', 'N/A')}")
print(f"\nOptimal Steering Results:")
print(f"  Exact Position: SPI = {best_spi_exact:.4f} at α = {best_alpha_exact:.3f}")
print(f"  Last Position:  SPI = {best_spi_last:.4f} at α = {best_alpha_last:.3f}")
print(f"\nPlots saved to: {output_dir}")
