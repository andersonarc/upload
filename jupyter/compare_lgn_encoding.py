#!/usr/bin/env python
"""
Compare LGN encoding quality across failed vs working samples.

This checks if the bimodal failure is due to different LGN encoding quality.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

FAILED = [10, 20, 30, 40, 50, 60, 70, 80]
WORKING = [90, 100, 123, 124]

print("Loading dataset...")
with h5py.File('spikes-128.h5', 'r') as f:
    all_probs = f['spike_trains'][:]
    all_labels = f['labels'][:]
    pre_delay = f.attrs.get('pre_delay', 50)
    im_slice = f.attrs.get('im_slice', 100)

# Load original MNIST to verify encoding
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
np.random.seed(3000)
perm = np.random.permutation(len(x_train))
x_train = x_train[perm]
y_train = y_train[perm]

def analyze_sample(idx):
    """Analyze encoding quality for a sample."""
    probs = all_probs[idx] / 1.3  # Remove 1.3 scaling
    label = all_labels[idx]

    # Temporal analysis
    pre_mean = np.mean(probs[:pre_delay, :])
    stim_mean = np.mean(probs[pre_delay:pre_delay+im_slice, :])
    post_mean = np.mean(probs[pre_delay+im_slice:, :])

    # Spatial analysis (during stimulus)
    neuron_activity = np.mean(probs[pre_delay:pre_delay+im_slice, :], axis=0)

    # Check for structure
    active_001 = np.sum(neuron_activity > 0.001)
    active_01 = np.sum(neuron_activity > 0.01)
    active_1 = np.sum(neuron_activity > 0.1)

    # Signal-to-noise ratio
    snr = stim_mean / pre_mean if pre_mean > 0 else 0

    # Dynamic range
    dynamic_range = np.max(neuron_activity) - np.min(neuron_activity)

    # Sparsity
    sparsity = 1.0 - (np.count_nonzero(neuron_activity) / len(neuron_activity))

    return {
        'idx': idx,
        'label': label,
        'pre_mean': pre_mean,
        'stim_mean': stim_mean,
        'post_mean': post_mean,
        'stim_max': np.max(probs[pre_delay:pre_delay+im_slice, :]),
        'active_001': active_001,
        'active_01': active_01,
        'active_1': active_1,
        'snr': snr,
        'dynamic_range': dynamic_range,
        'sparsity': sparsity,
        'neuron_activity': neuron_activity,
    }

print("\nAnalyzing FAILED samples...")
failed_stats = [analyze_sample(idx) for idx in FAILED if idx < len(all_probs)]

print("\nAnalyzing WORKING samples...")
working_stats = [analyze_sample(idx) for idx in WORKING if idx < len(all_probs)]

# Compute group statistics
def print_group_stats(stats, name):
    print(f"\n{name} samples (n={len(stats)}):")
    print(f"  Pre-stimulus mean:  {np.mean([s['pre_mean'] for s in stats]):.6f} ± {np.std([s['pre_mean'] for s in stats]):.6f}")
    print(f"  Stimulus mean:      {np.mean([s['stim_mean'] for s in stats]):.6f} ± {np.std([s['stim_mean'] for s in stats]):.6f}")
    print(f"  Stimulus max:       {np.mean([s['stim_max'] for s in stats]):.6f} ± {np.std([s['stim_max'] for s in stats]):.6f}")
    print(f"  Post-stimulus mean: {np.mean([s['post_mean'] for s in stats]):.6f} ± {np.std([s['post_mean'] for s in stats]):.6f}")
    print(f"  SNR (stim/pre):     {np.mean([s['snr'] for s in stats]):.2f} ± {np.std([s['snr'] for s in stats]):.2f}")
    print(f"  Active >0.001:      {np.mean([s['active_001'] for s in stats]):.0f} ± {np.std([s['active_001'] for s in stats]):.0f}")
    print(f"  Active >0.01:       {np.mean([s['active_01'] for s in stats]):.0f} ± {np.std([s['active_01'] for s in stats]):.0f}")
    print(f"  Active >0.1:        {np.mean([s['active_1'] for s in stats]):.0f} ± {np.std([s['active_1'] for s in stats]):.0f}")
    print(f"  Dynamic range:      {np.mean([s['dynamic_range'] for s in stats]):.6f} ± {np.std([s['dynamic_range'] for s in stats]):.6f}")
    print(f"  Sparsity:           {np.mean([s['sparsity'] for s in stats]):.4f} ± {np.std([s['sparsity'] for s in stats]):.4f}")

print("\n" + "="*80)
print("GROUP STATISTICS")
print("="*80)
print_group_stats(failed_stats, "FAILED")
print_group_stats(working_stats, "WORKING")

# Statistical tests
from scipy import stats

print("\n" + "="*80)
print("STATISTICAL COMPARISONS")
print("="*80)

metrics = [
    ('stim_mean', 'Stimulus Mean Activity'),
    ('stim_max', 'Stimulus Max Activity'),
    ('snr', 'Signal-to-Noise Ratio'),
    ('active_01', 'Active Neurons (>0.01)'),
    ('dynamic_range', 'Dynamic Range'),
]

for key, name in metrics:
    failed_vals = [s[key] for s in failed_stats]
    working_vals = [s[key] for s in working_stats]

    t_stat, p_val = stats.ttest_ind(failed_vals, working_vals)

    print(f"\n{name}:")
    print(f"  Failed:  {np.mean(failed_vals):.6f} ± {np.std(failed_vals):.6f}")
    print(f"  Working: {np.mean(working_vals):.6f} ± {np.std(working_vals):.6f}")
    print(f"  t={t_stat:.3f}, p={p_val:.6f}", end="")

    if p_val < 0.05:
        diff_pct = ((np.mean(working_vals) - np.mean(failed_vals)) / np.mean(failed_vals) * 100)
        print(f" *** SIGNIFICANT (working is {diff_pct:+.1f}% different)")
    else:
        print(" (not significant)")

# Visualization
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Stimulus mean comparison
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter([s['idx'] for s in failed_stats], [s['stim_mean'] for s in failed_stats],
           c='red', s=100, alpha=0.7, label='Failed')
ax1.scatter([s['idx'] for s in working_stats], [s['stim_mean'] for s in working_stats],
           c='green', s=100, alpha=0.7, label='Working')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Mean Stimulus Activity')
ax1.set_title('Stimulus Activity by Sample')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: SNR comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter([s['idx'] for s in failed_stats], [s['snr'] for s in failed_stats],
           c='red', s=100, alpha=0.7, label='Failed')
ax2.scatter([s['idx'] for s in working_stats], [s['snr'] for s in working_stats],
           c='green', s=100, alpha=0.7, label='Working')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('SNR (Stimulus / Baseline)')
ax2.set_title('Signal-to-Noise Ratio')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Active neurons
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter([s['idx'] for s in failed_stats], [s['active_01'] for s in failed_stats],
           c='red', s=100, alpha=0.7, label='Failed')
ax3.scatter([s['idx'] for s in working_stats], [s['active_01'] for s in working_stats],
           c='green', s=100, alpha=0.7, label='Working')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Active Neurons (>0.01)')
ax3.set_title('Number of Active Neurons')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Dynamic range
ax4 = fig.add_subplot(gs[0, 3])
ax4.scatter([s['idx'] for s in failed_stats], [s['dynamic_range'] for s in failed_stats],
           c='red', s=100, alpha=0.7, label='Failed')
ax4.scatter([s['idx'] for s in working_stats], [s['dynamic_range'] for s in working_stats],
           c='green', s=100, alpha=0.7, label='Working')
ax4.set_xlabel('Sample Index')
ax4.set_ylabel('Dynamic Range (max - min)')
ax4.set_title('Activity Dynamic Range')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5-8: Activity distributions for selected samples
selected_failed = [FAILED[0], FAILED[4]] if len(FAILED) >= 5 else FAILED[:2]
selected_working = [WORKING[0], WORKING[-1]] if len(WORKING) >= 2 else WORKING[:2]
selected = selected_failed + selected_working

for plot_idx, sample_idx in enumerate(selected):
    ax = fig.add_subplot(gs[1, plot_idx])

    # Find the stats
    stat = None
    for s in failed_stats + working_stats:
        if s['idx'] == sample_idx:
            stat = s
            break

    if stat is None:
        continue

    color = 'red' if sample_idx in FAILED else 'green'
    status = 'FAILED' if sample_idx in FAILED else 'WORKING'

    ax.hist(stat['neuron_activity'], bins=50, color=color, alpha=0.7)
    ax.set_xlabel('Neuron Activity')
    ax.set_ylabel('Count')
    ax.set_title(f'Sample {sample_idx} ({status})\\nLabel={stat["label"]}')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

# Plot 9-12: "Decoded" images for same samples
for plot_idx, sample_idx in enumerate(selected):
    ax = fig.add_subplot(gs[2, plot_idx])

    stat = None
    for s in failed_stats + working_stats:
        if s['idx'] == sample_idx:
            stat = s
            break

    if stat is None:
        continue

    # Simple grid decoding
    activity = stat['neuron_activity']
    n = len(activity)
    grid_size = int(np.sqrt(n))
    grid = np.zeros(grid_size * grid_size)
    grid[:n] = activity
    grid = grid.reshape(grid_size, grid_size)

    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(grid, sigma=2.0)

    color = 'red' if sample_idx in FAILED else 'green'
    status = 'FAILED' if sample_idx in FAILED else 'WORKING'

    im = ax.imshow(smoothed, cmap='gray', aspect='auto')
    ax.set_title(f'Sample {sample_idx} ({status})')
    ax.axis('off')

plt.suptitle('LGN Encoding Quality: Failed vs Working Samples',
             fontsize=16, fontweight='bold')

output_file = 'lgn_encoding_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n{'='*80}")
print(f"Saved comparison to: {output_file}")
print(f"{'='*80}")

print(f"\n{'='*80}")
print("CONCLUSIONS")
print(f"{'='*80}")
print("""
If FAILED samples have significantly lower:
- Stimulus mean activity
- SNR
- Number of active neurons
- Dynamic range

Then the bimodal failure is due to WEAK LGN ENCODING in those samples.

If FAILED and WORKING samples have similar LGN encoding quality,
then the problem is in the V1 network (weights, thresholds, etc.).

The "decoded images" likely won't look like MNIST because neurons
aren't spatially organized in the array. To properly decode, we'd
need the receptive field positions from lgn_full_col_cells_3.csv.
""")
