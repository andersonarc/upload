#!/usr/bin/env python
"""
Attempt to decode/reconstruct the original MNIST image from LGN spike probabilities.

This helps verify whether the LGN encoding is working correctly and whether
failed samples have corrupted/incorrect representations.

Usage:
    python decode_lgn.py --samples 10 100 123  # Compare specific samples
    SAMPLE_IDX=10 python decode_lgn.py         # Single sample
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

# Parse arguments
if '--samples' in sys.argv:
    idx = sys.argv.index('--samples')
    SAMPLES = [int(x) for x in sys.argv[idx+1:]]
    if not SAMPLES:
        SAMPLES = [10, 20, 50, 90, 100, 123]  # Default: mix of failed and working
else:
    SAMPLES = [int(os.environ.get('SAMPLE_IDX', '10'))]

print(f"Decoding samples: {SAMPLES}")

# Load dataset
print("\nLoading dataset...")
with h5py.File('spikes-128.h5', 'r') as f:
    all_probs = f['spike_trains'][:]
    all_labels = f['labels'][:]
    response_window = f['response_window'][()]

print(f"Dataset shape: {all_probs.shape}")
print(f"Response window: {response_window}")

# ============================================================================
# Decoding approach
# ============================================================================
"""
The LGN model has 17,400 neurons with Gaussian receptive fields at various
positions and scales. Without the full spatial parameters, we can still:

1. Visualize average activity patterns across all neurons
2. Reshape neuron activity into a 2D grid (assuming some spatial organization)
3. Apply Gaussian smoothing to approximate the original image
4. Compare decoded images between failed and working samples

The key insight: If failed samples have corrupted LGN encoding, the decoded
images should look significantly different or degraded.
"""

def decode_lgn_simple(spike_probs, stim_start=50, stim_end=150):
    """
    Simple decoding: average activity during stimulus period.

    Args:
        spike_probs: [time, n_neurons] spike probabilities
        stim_start, stim_end: stimulus window in ms

    Returns:
        neuron_activity: [n_neurons] average activity per neuron
    """
    # Remove the 1.3 scaling to get true probabilities
    true_probs = spike_probs / 1.3

    # Average activity during stimulus period
    stim_activity = np.mean(true_probs[stim_start:stim_end, :], axis=0)

    return stim_activity


def decode_lgn_spatial(spike_probs, image_size=(120, 240), stim_start=50, stim_end=150):
    """
    Attempt spatial reconstruction by reshaping neuron activity into 2D grid.

    Since we don't have exact receptive field positions, we'll:
    1. Reshape the neuron array into a 2D grid
    2. Apply Gaussian smoothing to approximate overlapping receptive fields
    3. Normalize to image range

    Args:
        spike_probs: [time, n_neurons] spike probabilities
        image_size: Expected spatial dimensions (LGN sees 120x240 image)
        stim_start, stim_end: stimulus window in ms

    Returns:
        decoded_image: 2D array approximating the original image
    """
    neuron_activity = decode_lgn_simple(spike_probs, stim_start, stim_end)
    n_neurons = len(neuron_activity)

    # Attempt to reshape into 2D
    # LGN has 17400 neurons, which factors as: 2^3 * 3 * 5^2 * 29
    # Try to find a reasonable 2D layout

    # Option 1: Approximate square grid
    grid_h = int(np.sqrt(n_neurons))
    grid_w = n_neurons // grid_h

    # Pad if needed
    if grid_h * grid_w < n_neurons:
        grid_w += 1

    padded_activity = np.zeros(grid_h * grid_w)
    padded_activity[:n_neurons] = neuron_activity

    # Reshape to 2D
    activity_2d = padded_activity.reshape(grid_h, grid_w)

    # Apply Gaussian smoothing to approximate overlapping receptive fields
    # LGN neurons have receptive fields with sigma ~ 3-5 pixels
    smoothed = gaussian_filter(activity_2d, sigma=2.0)

    # Normalize
    if smoothed.max() > 0:
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())

    return activity_2d, smoothed


def plot_decoding_analysis(samples, all_probs, all_labels):
    """
    Create comprehensive visualization of decoded images.
    """
    n_samples = len(samples)

    fig = plt.figure(figsize=(20, 6 * n_samples))
    gs = gridspec.GridSpec(n_samples, 5, figure=fig, hspace=0.3, wspace=0.3)

    for row_idx, sample_idx in enumerate(samples):
        spike_probs = all_probs[sample_idx]
        label = all_labels[sample_idx]

        # Compute statistics
        mean_prob = np.mean(spike_probs / 1.3)
        max_prob = np.max(spike_probs / 1.3)
        stim_mean = np.mean(spike_probs[50:150, :] / 1.3)
        expected_spikes = np.sum(spike_probs / 1.3)

        # Decode
        neuron_activity = decode_lgn_simple(spike_probs)
        activity_2d, decoded_smooth = decode_lgn_spatial(spike_probs)

        # Plot 1: Temporal profile (average over neurons)
        ax1 = fig.add_subplot(gs[row_idx, 0])
        temporal_profile = np.mean(spike_probs / 1.3, axis=1)
        ax1.plot(temporal_profile, 'k-', linewidth=0.5)
        ax1.axvline(50, color='red', linestyle='--', alpha=0.3)
        ax1.axvline(150, color='red', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Mean Probability')
        ax1.set_title(f'Sample {sample_idx} (Label={label})\\nTemporal Profile')
        ax1.grid(True, alpha=0.3)

        # Add statistics text
        ax1.text(0.98, 0.98,
                f'Mean: {mean_prob:.6f}\\n'
                f'Max: {max_prob:.6f}\\n'
                f'Stim: {stim_mean:.6f}\\n'
                f'Expected: {expected_spikes:.0f}',
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)

        # Plot 2: Neuron activity distribution
        ax2 = fig.add_subplot(gs[row_idx, 1])
        ax2.hist(neuron_activity, bins=50, color='black', alpha=0.7)
        ax2.set_xlabel('Average Activity (Stimulus Period)')
        ax2.set_ylabel('Neuron Count')
        ax2.set_title('Activity Distribution')
        ax2.axvline(np.mean(neuron_activity), color='red', linestyle='--',
                   label=f'Mean={np.mean(neuron_activity):.6f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Neuron activity in 1D
        ax3 = fig.add_subplot(gs[row_idx, 2])
        # Downsample for visibility
        downsample = 50
        ax3.plot(neuron_activity[::downsample], 'k-', linewidth=0.5)
        ax3.set_xlabel(f'Neuron ID (every {downsample}th)')
        ax3.set_ylabel('Average Activity')
        ax3.set_title('Spatial Pattern (1D)')
        ax3.grid(True, alpha=0.3)

        # Plot 4: 2D grid (raw)
        ax4 = fig.add_subplot(gs[row_idx, 3])
        im4 = ax4.imshow(activity_2d, cmap='hot', aspect='auto')
        ax4.set_title('2D Grid (Raw)')
        ax4.set_xlabel('Grid X')
        ax4.set_ylabel('Grid Y')
        plt.colorbar(im4, ax=ax4)

        # Plot 5: 2D grid (smoothed) - "Decoded Image"
        ax5 = fig.add_subplot(gs[row_idx, 4])
        im5 = ax5.imshow(decoded_smooth, cmap='gray', aspect='auto')
        ax5.set_title('Decoded (Smoothed)')
        ax5.set_xlabel('Grid X')
        ax5.set_ylabel('Grid Y')
        plt.colorbar(im5, ax=ax5)

    plt.suptitle('LGN Decoding Analysis', fontsize=16, fontweight='bold')

    output_file = f'lgn_decoding_{"_".join(map(str, samples))}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved decoding visualization to: {output_file}")


def compare_failed_vs_working(all_probs, all_labels):
    """
    Statistical comparison of decoded images between failed and working samples.
    """
    FAILED = [10, 20, 30, 40, 50, 60, 70, 80]
    WORKING = [90, 100, 123, 124]

    print("\n" + "="*80)
    print("DECODED IMAGE COMPARISON: FAILED vs WORKING")
    print("="*80)

    def analyze_decoded(sample_indices, name):
        activities = []
        means = []
        stds = []
        maxs = []

        for idx in sample_indices:
            if idx >= len(all_probs):
                continue
            activity = decode_lgn_simple(all_probs[idx])
            activities.append(activity)
            means.append(np.mean(activity))
            stds.append(np.std(activity))
            maxs.append(np.max(activity))

        activities = np.array(activities)

        print(f"\n{name} samples:")
        print(f"  Mean activity:     {np.mean(means):.6f} ± {np.std(means):.6f}")
        print(f"  Activity std:      {np.mean(stds):.6f} ± {np.std(stds):.6f}")
        print(f"  Max activity:      {np.mean(maxs):.6f} ± {np.std(maxs):.6f}")
        print(f"  Active neurons (>0.001): {np.mean(np.sum(activities > 0.001, axis=1)):.0f}")
        print(f"  Active neurons (>0.01):  {np.mean(np.sum(activities > 0.01, axis=1)):.0f}")

        return activities, means, stds, maxs

    failed_data = analyze_decoded(FAILED, "FAILED")
    working_data = analyze_decoded(WORKING, "WORKING")

    print(f"\n" + "="*80)
    print("STATISTICAL TESTS")
    print("="*80)

    failed_means = failed_data[1]
    working_means = working_data[1]

    # t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(failed_means, working_means)
    print(f"\nT-test (mean activity):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  ✓ SIGNIFICANT difference in decoded activity (p < 0.05)")
        if np.mean(failed_means) < np.mean(working_means):
            print(f"  → Failed samples have LOWER decoded activity")
        else:
            print(f"  → Failed samples have HIGHER decoded activity")
    else:
        print(f"  ✗ No significant difference (p >= 0.05)")

    # Correlation analysis
    print(f"\nCorrelation between decoded images:")
    print(f"  Within FAILED samples:")
    failed_activities = failed_data[0]
    failed_corrs = []
    for i in range(len(failed_activities)):
        for j in range(i+1, len(failed_activities)):
            corr = np.corrcoef(failed_activities[i], failed_activities[j])[0, 1]
            failed_corrs.append(corr)
    print(f"    Mean correlation: {np.mean(failed_corrs):.4f} ± {np.std(failed_corrs):.4f}")

    print(f"  Within WORKING samples:")
    working_activities = working_data[0]
    working_corrs = []
    for i in range(len(working_activities)):
        for j in range(i+1, len(working_activities)):
            corr = np.corrcoef(working_activities[i], working_activities[j])[0, 1]
            working_corrs.append(corr)
    print(f"    Mean correlation: {np.mean(working_corrs):.4f} ± {np.std(working_corrs):.4f}")

    print(f"  Between FAILED and WORKING:")
    cross_corrs = []
    for i in range(len(failed_activities)):
        for j in range(len(working_activities)):
            corr = np.corrcoef(failed_activities[i], working_activities[j])[0, 1]
            cross_corrs.append(corr)
    print(f"    Mean correlation: {np.mean(cross_corrs):.4f} ± {np.std(cross_corrs):.4f}")


# ============================================================================
# Main Analysis
# ============================================================================

print("\n" + "="*80)
print("DECODING ANALYSIS")
print("="*80)

# Decode and visualize requested samples
plot_decoding_analysis(SAMPLES, all_probs, all_labels)

# If not comparing specific samples, do group comparison
if len(SAMPLES) <= 3:
    compare_failed_vs_working(all_probs, all_labels)

print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)
print("""
The decoded images show the spatial pattern of LGN activity during the stimulus.

Key questions to answer:
1. Do failed samples have systematically lower decoded activity?
   → This would suggest the input encoding is weak/insufficient

2. Are decoded images qualitatively different between failed/working samples?
   → This would suggest different stimulus representations

3. Do decoded images correlate with the digit labels?
   → This would verify the LGN encoding preserves digit information

4. Are there spatial patterns (hot spots) in the decoded images?
   → This would verify the LGN receptive fields are spatially organized

Without the full LGN spatial parameters (receptive field positions), this
decoding is approximate. However, systematic differences between failed and
working samples would indicate an encoding problem.
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
