#!/usr/bin/env python
"""
Compare multiple samples to understand why some spike and others don't.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Samples to compare (failed vs working based on your logs)
FAILED_SAMPLES = [10, 20, 30, 40, 50, 60, 70, 80]
WORKING_SAMPLES = [90, 100, 123, 124]

print("Loading dataset...")
with h5py.File('spikes-128.h5', 'r') as f:
    all_probs = f['spike_trains'][:]
    all_labels = f['labels'][:]

print(f"Dataset shape: {all_probs.shape}")
print(f"Analyzing {len(FAILED_SAMPLES)} failed + {len(WORKING_SAMPLES)} working samples")

# ============================================================================
# Compute statistics for each sample
# ============================================================================
def compute_stats(sample_idx):
    """Compute comprehensive statistics for a sample"""
    probs = all_probs[sample_idx]
    label = all_labels[sample_idx]

    stats = {
        'idx': sample_idx,
        'label': label,
        'mean': np.mean(probs),
        'std': np.std(probs),
        'max': np.max(probs),
        'median': np.median(probs),
        'p95': np.percentile(probs, 95),
        'p99': np.percentile(probs, 99),
        # By time window
        'pre_stim_mean': np.mean(probs[0:50, :]),
        'stim_onset_mean': np.mean(probs[50:100, :]),
        'stim_mid_mean': np.mean(probs[100:150, :]),
        'post_stim_mean': np.mean(probs[150:300, :]),
        # Per-neuron
        'neurons_active': np.sum(np.max(probs, axis=0) > 0.01),
        'neurons_max_over_05': np.sum(np.max(probs, axis=0) > 0.05),
        'neurons_silent': np.sum(np.max(probs, axis=0) == 0),
        # Expected spikes
        'expected_spikes': np.sum(probs / 1.3),
    }
    return stats

print("\nComputing statistics...")
failed_stats = [compute_stats(idx) for idx in FAILED_SAMPLES]
working_stats = [compute_stats(idx) for idx in WORKING_SAMPLES]

# ============================================================================
# Print comparison table
# ============================================================================
print("\n" + "="*100)
print("FAILED SAMPLES")
print("="*100)
print(f"{'Idx':<6} {'Label':<6} {'Mean':<10} {'Max':<10} {'Stim':<10} {'Active':<8} {'Expected':<10}")
print("-"*100)
for s in failed_stats:
    print(f"{s['idx']:<6} {s['label']:<6} {s['mean']:<10.6f} {s['max']:<10.6f} "
          f"{s['stim_onset_mean']:<10.6f} {s['neurons_active']:<8} {s['expected_spikes']:<10.0f}")

print("\n" + "="*100)
print("WORKING SAMPLES")
print("="*100)
print(f"{'Idx':<6} {'Label':<6} {'Mean':<10} {'Max':<10} {'Stim':<10} {'Active':<8} {'Expected':<10}")
print("-"*100)
for s in working_stats:
    print(f"{s['idx']:<6} {s['label']:<6} {s['mean']:<10.6f} {s['max']:<10.6f} "
          f"{s['stim_onset_mean']:<10.6f} {s['neurons_active']:<8} {s['expected_spikes']:<10.0f}")

# ============================================================================
# Statistical comparison
# ============================================================================
print("\n" + "="*100)
print("GROUP COMPARISON")
print("="*100)

def print_group_stats(stats_list, name):
    means = [s['mean'] for s in stats_list]
    maxs = [s['max'] for s in stats_list]
    stim = [s['stim_onset_mean'] for s in stats_list]
    expected = [s['expected_spikes'] for s in stats_list]

    print(f"\n{name}:")
    print(f"  Mean probability:     {np.mean(means):.6f} ± {np.std(means):.6f} "
          f"(range: [{np.min(means):.6f}, {np.max(means):.6f}])")
    print(f"  Max probability:      {np.mean(maxs):.6f} ± {np.std(maxs):.6f}")
    print(f"  Stim onset (50-100ms): {np.mean(stim):.6f} ± {np.std(stim):.6f}")
    print(f"  Expected spikes:      {np.mean(expected):.0f} ± {np.std(expected):.0f}")

print_group_stats(failed_stats, "FAILED SAMPLES")
print_group_stats(working_stats, "WORKING SAMPLES")

# ============================================================================
# Visualization
# ============================================================================
print("\nGenerating comparison visualization...")

fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

# Plot 1: Mean probability comparison
ax1 = fig.add_subplot(gs[0, 0])
failed_means = [s['mean'] for s in failed_stats]
working_means = [s['mean'] for s in working_stats]
ax1.scatter(FAILED_SAMPLES, failed_means, c='red', s=100, alpha=0.6, label='Failed')
ax1.scatter(WORKING_SAMPLES, working_means, c='green', s=100, alpha=0.6, label='Working')
ax1.axhline(np.mean(failed_means), color='red', linestyle='--', alpha=0.3)
ax1.axhline(np.mean(working_means), color='green', linestyle='--', alpha=0.3)
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Mean Probability')
ax1.set_title('Mean Spike Probability')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Max probability comparison
ax2 = fig.add_subplot(gs[0, 1])
failed_maxs = [s['max'] for s in failed_stats]
working_maxs = [s['max'] for s in working_stats]
ax2.scatter(FAILED_SAMPLES, failed_maxs, c='red', s=100, alpha=0.6, label='Failed')
ax2.scatter(WORKING_SAMPLES, working_maxs, c='green', s=100, alpha=0.6, label='Working')
ax2.axhline(np.mean(failed_maxs), color='red', linestyle='--', alpha=0.3)
ax2.axhline(np.mean(working_maxs), color='green', linestyle='--', alpha=0.3)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Max Probability')
ax2.set_title('Max Spike Probability')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Stimulus period mean
ax3 = fig.add_subplot(gs[0, 2])
failed_stim = [s['stim_onset_mean'] for s in failed_stats]
working_stim = [s['stim_onset_mean'] for s in working_stats]
ax3.scatter(FAILED_SAMPLES, failed_stim, c='red', s=100, alpha=0.6, label='Failed')
ax3.scatter(WORKING_SAMPLES, working_stim, c='green', s=100, alpha=0.6, label='Working')
ax3.axhline(np.mean(failed_stim), color='red', linestyle='--', alpha=0.3)
ax3.axhline(np.mean(working_stim), color='green', linestyle='--', alpha=0.3)
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Mean Probability (50-100ms)')
ax3.set_title('Stimulus Onset Period')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Expected spikes
ax4 = fig.add_subplot(gs[0, 3])
failed_expected = [s['expected_spikes'] for s in failed_stats]
working_expected = [s['expected_spikes'] for s in working_stats]
ax4.scatter(FAILED_SAMPLES, failed_expected, c='red', s=100, alpha=0.6, label='Failed')
ax4.scatter(WORKING_SAMPLES, working_expected, c='green', s=100, alpha=0.6, label='Working')
ax4.axhline(np.mean(failed_expected), color='red', linestyle='--', alpha=0.3)
ax4.axhline(np.mean(working_expected), color='green', linestyle='--', alpha=0.3)
ax4.set_xlabel('Sample Index')
ax4.set_ylabel('Expected Total Spikes')
ax4.set_title('Expected LGN Spikes')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5-8: Temporal profiles (mean over neurons) for selected samples
for plot_idx, (sample_idx, color, status) in enumerate([
    (FAILED_SAMPLES[0], 'red', 'Failed'),
    (FAILED_SAMPLES[-1], 'darkred', 'Failed'),
    (WORKING_SAMPLES[0], 'green', 'Working'),
    (WORKING_SAMPLES[-1], 'darkgreen', 'Working'),
]):
    ax = fig.add_subplot(gs[1, plot_idx])
    temporal_profile = np.mean(all_probs[sample_idx], axis=1)
    ax.plot(temporal_profile, color=color, linewidth=1)
    ax.axvline(50, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(150, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mean Prob')
    ax.set_title(f'Sample {sample_idx} ({status})\nLabel={all_labels[sample_idx]}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 300])

# Plot 9-12: Spatial profiles (mean over time) for selected samples
for plot_idx, (sample_idx, color, status) in enumerate([
    (FAILED_SAMPLES[0], 'red', 'Failed'),
    (FAILED_SAMPLES[-1], 'darkred', 'Failed'),
    (WORKING_SAMPLES[0], 'green', 'Working'),
    (WORKING_SAMPLES[-1], 'darkgreen', 'Working'),
]):
    ax = fig.add_subplot(gs[2, plot_idx])
    # Show spatial profile during stimulus (50-150ms)
    spatial_profile = np.mean(all_probs[sample_idx, 50:150, :], axis=0)
    # Downsample for visualization
    downsample = 20
    ax.plot(spatial_profile[::downsample], color=color, linewidth=0.5)
    ax.set_xlabel(f'Neuron ID (every {downsample}th)')
    ax.set_ylabel('Mean Prob (50-150ms)')
    ax.set_title(f'Spatial Pattern: Sample {sample_idx}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Failed vs Working Samples Comparison', fontsize=16, fontweight='bold')

output_file = 'samples_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved comparison to: {output_file}")

# ============================================================================
# Correlation analysis
# ============================================================================
print("\n" + "="*100)
print("CORRELATION ANALYSIS")
print("="*100)

# Compute correlations between failed samples
print("\nCorrelation between FAILED samples (stimulus period 50-150ms):")
failed_stim_patterns = [all_probs[idx, 50:150, :].flatten() for idx in FAILED_SAMPLES]
for i, idx1 in enumerate(FAILED_SAMPLES):
    for j, idx2 in enumerate(FAILED_SAMPLES):
        if i < j:
            corr = np.corrcoef(failed_stim_patterns[i], failed_stim_patterns[j])[0, 1]
            print(f"  Sample {idx1} vs {idx2}: {corr:.4f}")

print("\nCorrelation between WORKING samples (stimulus period 50-150ms):")
working_stim_patterns = [all_probs[idx, 50:150, :].flatten() for idx in WORKING_SAMPLES]
for i, idx1 in enumerate(WORKING_SAMPLES):
    for j, idx2 in enumerate(WORKING_SAMPLES):
        if i < j:
            corr = np.corrcoef(working_stim_patterns[i], working_stim_patterns[j])[0, 1]
            print(f"  Sample {idx1} vs {idx2}: {corr:.4f}")

print("\nCross-correlation (failed vs working):")
for idx1 in FAILED_SAMPLES[:2]:  # Just first 2 to avoid too much output
    for idx2 in WORKING_SAMPLES[:2]:
        pat1 = all_probs[idx1, 50:150, :].flatten()
        pat2 = all_probs[idx2, 50:150, :].flatten()
        corr = np.corrcoef(pat1, pat2)[0, 1]
        print(f"  Sample {idx1} (failed) vs {idx2} (working): {corr:.4f}")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
