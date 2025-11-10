#!/usr/bin/env python
"""
Detailed visualization of spike probability data to understand
why some samples produce network activity and others don't.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import sys

# Sample to analyze (set via environment or default)
SAMPLE_IDX = int(os.environ.get('SAMPLE_IDX', '10'))
print(f"Analyzing sample {SAMPLE_IDX}")

# Fixed random seed for reproducibility
np.random.seed(1)

# Load dataset
print("Loading dataset...")
with h5py.File('spikes-128.h5', 'r') as f:
    spike_probs = f['spike_trains'][SAMPLE_IDX]  # Shape: [seq_len, n_lgn]
    label = f['labels'][SAMPLE_IDX]
    response_window = f['response_window'][()]

print(f"Sample {SAMPLE_IDX}: Label = {label}")
print(f"Shape: {spike_probs.shape}")
print(f"Response window: {response_window}")

# ============================================================================
# 1. STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STATISTICS")
print("="*80)

print(f"\nOverall statistics:")
print(f"  Mean probability: {np.mean(spike_probs):.6f}")
print(f"  Std probability:  {np.std(spike_probs):.6f}")
print(f"  Min probability:  {np.min(spike_probs):.6f}")
print(f"  Max probability:  {np.max(spike_probs):.6f}")
print(f"  Median:           {np.median(spike_probs):.6f}")
print(f"  95th percentile:  {np.percentile(spike_probs, 95):.6f}")

# Statistics by time window
print(f"\nBy time window:")
windows = [
    ("Pre-stimulus (0-50ms)", 0, 50),
    ("Stimulus onset (50-100ms)", 50, 100),
    ("Stimulus mid (100-150ms)", 100, 150),
    ("Post-stimulus (150-300ms)", 150, 300),
    ("Late (300-600ms)", 300, 600),
]

for name, start, end in windows:
    window_data = spike_probs[start:end, :]
    print(f"  {name}:")
    print(f"    Mean: {np.mean(window_data):.6f}")
    print(f"    Max:  {np.max(window_data):.6f}")
    print(f"    Std:  {np.std(window_data):.6f}")

# Statistics by neuron (check if some neurons never fire)
neuron_means = np.mean(spike_probs, axis=0)
neuron_maxs = np.max(spike_probs, axis=0)
print(f"\nPer-neuron statistics:")
print(f"  Neurons with mean > 0.01: {np.sum(neuron_means > 0.01)}/{len(neuron_means)}")
print(f"  Neurons with max > 0.05:  {np.sum(neuron_maxs > 0.05)}/{len(neuron_maxs)}")
print(f"  Neurons completely silent: {np.sum(neuron_maxs == 0)}/{len(neuron_maxs)}")

# Check for NaN or invalid values
print(f"\nData validation:")
print(f"  Contains NaN: {np.any(np.isnan(spike_probs))}")
print(f"  Contains Inf: {np.any(np.isinf(spike_probs))}")
print(f"  Contains negative: {np.any(spike_probs < 0)}")
print(f"  Contains > 1.3: {np.any(spike_probs > 1.3)}")

# ============================================================================
# 2. GLOBAL DISTRIBUTION HEATMAP
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

# 2a. Full heatmap (downsampled for visibility)
print("\nCreating full probability heatmap...")
ax1 = fig.add_subplot(gs[0, :])
# Downsample neurons for visualization (show every 10th)
downsample_neurons = 50
heatmap_data = spike_probs[:, ::downsample_neurons].T
im1 = ax1.imshow(heatmap_data, aspect='auto', cmap='hot',
                 extent=[0, spike_probs.shape[0], 0, spike_probs.shape[1]],
                 vmin=0, vmax=np.percentile(spike_probs, 99))
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('LGN Neuron ID')
ax1.set_title(f'Sample {SAMPLE_IDX} (Label={label}): Spike Probabilities (every {downsample_neurons}th neuron)')
ax1.axvline(50, color='cyan', linestyle='--', alpha=0.5, label='Stimulus onset')
ax1.axvline(150, color='cyan', linestyle='--', alpha=0.5, label='Stimulus end')
plt.colorbar(im1, ax=ax1, label='Probability')

# 2b. Per-neuron statistics heatmap
print("Creating per-neuron statistics...")
ax2 = fig.add_subplot(gs[1, 0])
neuron_stats = np.vstack([
    neuron_means,
    neuron_maxs,
    np.std(spike_probs, axis=0),
    np.max(spike_probs, axis=0) - np.min(spike_probs, axis=0)  # Range
])
neuron_stats_downsampled = neuron_stats[:, ::downsample_neurons]
im2 = ax2.imshow(neuron_stats_downsampled, aspect='auto', cmap='viridis')
ax2.set_xlabel(f'LGN Neuron ID (every {downsample_neurons}th)')
ax2.set_yticks([0, 1, 2, 3])
ax2.set_yticklabels(['Mean', 'Max', 'Std', 'Range'])
ax2.set_title('Per-Neuron Statistics')
plt.colorbar(im2, ax=ax2)

# 2c. Temporal statistics
print("Creating temporal statistics...")
ax3 = fig.add_subplot(gs[1, 1])
time_stats = np.vstack([
    np.mean(spike_probs, axis=1),
    np.max(spike_probs, axis=1),
    np.std(spike_probs, axis=1),
])
im3 = ax3.imshow(time_stats, aspect='auto', cmap='plasma', extent=[0, spike_probs.shape[0], 0, 3])
ax3.set_xlabel('Time (ms)')
ax3.set_yticks([0.5, 1.5, 2.5])
ax3.set_yticklabels(['Mean', 'Max', 'Std'])
ax3.set_title('Per-Timestep Statistics')
ax3.axvline(50, color='white', linestyle='--', alpha=0.5)
ax3.axvline(150, color='white', linestyle='--', alpha=0.5)
plt.colorbar(im3, ax=ax3)

# ============================================================================
# 3. SPIKE VISUALIZATION
# ============================================================================

# Generate spikes using Poisson sampling (same as class.py)
def create_spike_times(spike_trains, timestep=1.0, scale=1.0):
    """Generate spike times from probabilities using Poisson sampling"""
    lgn_size = spike_trains.shape[1]
    spike_times = []

    for i in range(lgn_size):
        times = []
        for t in range(spike_trains.shape[0]):
            if np.clip((spike_trains[t, i] / 1.3) * scale, 0.0, 1.0) > np.random.rand():
                times.append(float(t * timestep))
        spike_times.append(times)

    return spike_times

print("\nGenerating spikes from probabilities...")
spike_times = create_spike_times(spike_probs)

# Count total spikes
total_spikes = sum(len(st) for st in spike_times)
print(f"Total spikes generated: {total_spikes}")

# 3a. Spike raster for random sample of neurons
print("Creating spike raster plots...")
ax4 = fig.add_subplot(gs[2, 0])
np.random.seed(42)  # Fixed seed for reproducible neuron selection
sample_neurons = np.random.choice(len(spike_times), size=min(100, len(spike_times)), replace=False)
sample_neurons = np.sort(sample_neurons)

for idx, neuron_idx in enumerate(sample_neurons):
    times = spike_times[neuron_idx]
    if len(times) > 0:
        ax4.plot(times, [idx] * len(times), 'k.', markersize=1)

ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Neuron index (sampled)')
ax4.set_title(f'Spike Raster: {len(sample_neurons)} Random Neurons')
ax4.set_xlim([0, spike_probs.shape[0]])
ax4.axvline(50, color='red', linestyle='--', alpha=0.3)
ax4.axvline(150, color='red', linestyle='--', alpha=0.3)

# 3b. Spike raster for first 100 neurons
ax5 = fig.add_subplot(gs[2, 1])
for neuron_idx in range(min(100, len(spike_times))):
    times = spike_times[neuron_idx]
    if len(times) > 0:
        ax5.plot(times, [neuron_idx] * len(times), 'k.', markersize=1)

ax5.set_xlabel('Time (ms)')
ax5.set_ylabel('Neuron ID')
ax5.set_title('Spike Raster: First 100 Neurons')
ax5.set_xlim([0, spike_probs.shape[0]])
ax5.axvline(50, color='red', linestyle='--', alpha=0.3)
ax5.axvline(150, color='red', linestyle='--', alpha=0.3)

# 3c. Population firing rate over time
print("Computing population firing rate...")
ax6 = fig.add_subplot(gs[3, 0])
bin_size = 10  # ms
n_bins = spike_probs.shape[0] // bin_size
spike_counts = np.zeros(n_bins)

for times in spike_times:
    for t in times:
        bin_idx = int(t / bin_size)
        if bin_idx < n_bins:
            spike_counts[bin_idx] += 1

time_bins = np.arange(n_bins) * bin_size
ax6.plot(time_bins, spike_counts / len(spike_times) / bin_size * 1000, 'k-', linewidth=1)
ax6.set_xlabel('Time (ms)')
ax6.set_ylabel('Population rate (Hz)')
ax6.set_title('Population Firing Rate')
ax6.axvline(50, color='red', linestyle='--', alpha=0.3, label='Stimulus')
ax6.axvline(150, color='red', linestyle='--', alpha=0.3)
ax6.grid(True, alpha=0.3)
ax6.legend()

# 3d. Histogram of spike counts per neuron
ax7 = fig.add_subplot(gs[3, 1])
spike_counts_per_neuron = [len(times) for times in spike_times]
ax7.hist(spike_counts_per_neuron, bins=50, color='black', alpha=0.7)
ax7.set_xlabel('Spikes per neuron')
ax7.set_ylabel('Count')
ax7.set_title(f'Distribution of Spike Counts (mean={np.mean(spike_counts_per_neuron):.1f})')
ax7.axvline(np.mean(spike_counts_per_neuron), color='red', linestyle='--',
            label=f'Mean={np.mean(spike_counts_per_neuron):.1f}')
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.suptitle(f'Sample {SAMPLE_IDX} Analysis (Label={label})', fontsize=16, fontweight='bold')

# Save figure
output_file = f'sample_{SAMPLE_IDX}_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: {output_file}")

# ============================================================================
# 4. ADDITIONAL DIAGNOSTICS
# ============================================================================
print("\n" + "="*80)
print("ADDITIONAL DIAGNOSTICS")
print("="*80)

# Find most active neurons
print("\nMost active neurons (by probability):")
most_active_idx = np.argsort(neuron_means)[-10:][::-1]
for idx in most_active_idx:
    print(f"  Neuron {idx}: mean={neuron_means[idx]:.6f}, max={neuron_maxs[idx]:.6f}, "
          f"spikes_generated={len(spike_times[idx])}")

# Find least active neurons
print("\nLeast active neurons (by probability):")
least_active_idx = np.argsort(neuron_means)[:10]
for idx in least_active_idx:
    print(f"  Neuron {idx}: mean={neuron_means[idx]:.6f}, max={neuron_maxs[idx]:.6f}, "
          f"spikes_generated={len(spike_times[idx])}")

# Check temporal clustering
print("\nTemporal clustering analysis:")
for name, start, end in windows[:3]:  # Just first 3 windows
    window_spikes = sum(sum(1 for t in times if start <= t < end) for times in spike_times)
    print(f"  {name}: {window_spikes} spikes ({window_spikes/total_spikes*100:.1f}% of total)"
          if total_spikes > 0 else f"  {name}: 0 spikes")

# Probability distribution
print("\nProbability distribution:")
hist, bins = np.histogram(spike_probs.flatten(), bins=20)
for i in range(len(hist)):
    if hist[i] > 0:
        print(f"  [{bins[i]:.4f}, {bins[i+1]:.4f}): {hist[i]} values "
              f"({hist[i]/np.prod(spike_probs.shape)*100:.2f}%)")

# Check if probabilities follow expected pattern
print("\nExpected vs actual spike generation:")
expected_spikes = np.sum(spike_probs / 1.3)  # Expected number after removing 1.3 scaling
print(f"  Expected spikes (from probabilities): {expected_spikes:.0f}")
print(f"  Actually generated: {total_spikes}")
print(f"  Ratio (actual/expected): {total_spikes/expected_spikes if expected_spikes > 0 else 0:.3f}")

# Compare to reference sample if specified
if 'REFERENCE_IDX' in os.environ:
    ref_idx = int(os.environ['REFERENCE_IDX'])
    print(f"\n" + "="*80)
    print(f"COMPARISON TO REFERENCE SAMPLE {ref_idx}")
    print("="*80)

    with h5py.File('spikes-128.h5', 'r') as f:
        ref_probs = f['spike_trains'][ref_idx]
        ref_label = f['labels'][ref_idx]

    print(f"Reference sample {ref_idx}: Label = {ref_label}")
    print(f"\nComparison:")
    print(f"  Mean prob: Current={np.mean(spike_probs):.6f}, Ref={np.mean(ref_probs):.6f}, "
          f"Ratio={np.mean(spike_probs)/np.mean(ref_probs):.3f}")
    print(f"  Max prob:  Current={np.max(spike_probs):.6f}, Ref={np.max(ref_probs):.6f}, "
          f"Ratio={np.max(spike_probs)/np.max(ref_probs):.3f}")
    print(f"  Std prob:  Current={np.std(spike_probs):.6f}, Ref={np.std(ref_probs):.6f}, "
          f"Ratio={np.std(spike_probs)/np.std(ref_probs):.3f}")

    # Stimulus window comparison
    stim_current = spike_probs[50:150, :]
    stim_ref = ref_probs[50:150, :]
    print(f"\nStimulus window (50-150ms):")
    print(f"  Mean prob: Current={np.mean(stim_current):.6f}, Ref={np.mean(stim_ref):.6f}, "
          f"Ratio={np.mean(stim_current)/np.mean(stim_ref):.3f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
