#!/usr/bin/env python
"""
Advanced LGN decoding using actual receptive field parameters.

This script attempts to load the LGN receptive field data (positions, sizes)
and perform proper inverse mapping to reconstruct the original image.

Usage:
    python decode_lgn_advanced.py --sample 10
    python decode_lgn_advanced.py --samples 10 100 --lgn-path /path/to/lgn_full_col_cells_3.csv
"""

import os
import sys
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

# Parse arguments
parser = argparse.ArgumentParser(description='Decode MNIST images from LGN spike probabilities')
parser.add_argument('--sample', type=int, help='Single sample index to decode')
parser.add_argument('--samples', type=int, nargs='+', help='Multiple sample indices')
parser.add_argument('--lgn-path', type=str, default=None,
                   help='Path to lgn_full_col_cells_3.csv')
parser.add_argument('--dataset', type=str, default='spikes-128.h5',
                   help='Path to H5 dataset')
parser.add_argument('--output', type=str, default=None,
                   help='Output filename for visualization')

args = parser.parse_args()

# Determine samples to analyze
if args.samples:
    SAMPLES = args.samples
elif args.sample is not None:
    SAMPLES = [args.sample]
else:
    SAMPLES = [10, 100]  # Default: one failed, one working

print(f"Decoding samples: {SAMPLES}")

# Load dataset
print(f"\nLoading dataset: {args.dataset}")
with h5py.File(args.dataset, 'r') as f:
    all_probs = f['spike_trains'][:]
    all_labels = f['labels'][:]
    response_window = f['response_window'][()]

print(f"Dataset shape: {all_probs.shape}")
print(f"Number of LGN neurons: {all_probs.shape[2]}")

# Try to load LGN receptive field parameters
lgn_params_available = False
if args.lgn_path is None:
    # Try default locations
    uid = os.getuid()
    if uid == 0:
        default_path = '/root/v1cortex/GLIF_network/lgn_full_col_cells_3.csv'
    else:
        default_path = '/home/uni/mouse/Training-data-driven-V1-model/GLIF_network/lgn_full_col_cells_3.csv'

    if os.path.exists(default_path):
        args.lgn_path = default_path

if args.lgn_path and os.path.exists(args.lgn_path):
    print(f"\nLoading LGN receptive field parameters from: {args.lgn_path}")
    import pandas as pd

    lgn_data = pd.read_csv(args.lgn_path, delimiter=' ')

    # Extract spatial parameters
    rf_x = lgn_data['x'].to_numpy()
    rf_y = lgn_data['y'].to_numpy()
    rf_sizes = lgn_data['spatial_size'].to_numpy()
    model_ids = lgn_data['model_id'].to_numpy()

    # Normalize coordinates to image space (120x240)
    rf_x = rf_x * 239 / 240
    rf_y = rf_y * 119 / 120
    rf_x = np.clip(rf_x, 0, 239)
    rf_y = np.clip(rf_y, 0, 119)

    print(f"Loaded {len(rf_x)} receptive field centers")
    print(f"  X range: [{rf_x.min():.1f}, {rf_x.max():.1f}]")
    print(f"  Y range: [{rf_y.min():.1f}, {rf_y.max():.1f}]")
    print(f"  Size range: [{rf_sizes.min():.1f}, {rf_sizes.max():.1f}]")

    lgn_params_available = True
else:
    print("\nWARNING: LGN receptive field parameters not available.")
    print("Will use simplified decoding without spatial information.")
    lgn_params_available = False


def decode_with_rf_params(spike_probs, rf_x, rf_y, rf_sizes,
                          image_shape=(120, 240),
                          stim_start=50, stim_end=150):
    """
    Proper decoding using receptive field parameters.

    Each LGN neuron has a Gaussian receptive field at position (x, y)
    with size (sigma). To decode:
    1. Average each neuron's activity during stimulus
    2. Reconstruct image by placing Gaussian at each RF position,
       weighted by neuron activity

    Args:
        spike_probs: [time, n_neurons] spike probabilities
        rf_x, rf_y: Receptive field centers
        rf_sizes: Receptive field sizes (diameter)
        image_shape: Output image dimensions
        stim_start, stim_end: Stimulus window

    Returns:
        decoded_image: [height, width] reconstructed image
    """
    # Remove 1.3 scaling
    true_probs = spike_probs / 1.3

    # Average activity during stimulus
    neuron_activity = np.mean(true_probs[stim_start:stim_end, :], axis=0)

    # Initialize output image
    height, width = image_shape
    decoded = np.zeros((height, width))
    weight_map = np.zeros((height, width))

    # Create coordinate grids
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    print(f"  Reconstructing from {len(neuron_activity)} neurons...")

    # For each neuron, add its contribution
    for i, activity in enumerate(neuron_activity):
        if activity < 1e-6:  # Skip silent neurons
            continue

        # Receptive field parameters
        cx, cy = rf_x[i], rf_y[i]
        sigma = rf_sizes[i] / 3.0  # Convert diameter to sigma (approx)

        # Gaussian receptive field
        gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

        # Add weighted contribution
        decoded += activity * gaussian
        weight_map += gaussian

    # Normalize by total weight to prevent bright spots where RFs overlap
    mask = weight_map > 1e-6
    decoded[mask] /= weight_map[mask]

    # Normalize to [0, 1]
    if decoded.max() > 0:
        decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min())

    return decoded, weight_map


def decode_simple_grid(spike_probs, stim_start=50, stim_end=150):
    """
    Fallback: simple 2D grid decoding without RF parameters.
    """
    true_probs = spike_probs / 1.3
    neuron_activity = np.mean(true_probs[stim_start:stim_end, :], axis=0)

    # Reshape to approximate square
    n = len(neuron_activity)
    h = int(np.sqrt(n))
    w = (n + h - 1) // h

    padded = np.zeros(h * w)
    padded[:n] = neuron_activity
    grid = padded.reshape(h, w)

    # Smooth
    smoothed = gaussian_filter(grid, sigma=2.0)

    # Normalize
    if smoothed.max() > 0:
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())

    return smoothed


def visualize_decoding(samples, all_probs, all_labels):
    """
    Create visualization of decoded images.
    """
    n_samples = len(samples)

    if lgn_params_available:
        fig = plt.figure(figsize=(24, 6 * n_samples))
        gs = gridspec.GridSpec(n_samples, 6, figure=fig, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(18, 6 * n_samples))
        gs = gridspec.GridSpec(n_samples, 4, figure=fig, hspace=0.3, wspace=0.3)

    for row_idx, sample_idx in enumerate(samples):
        print(f"\nProcessing sample {sample_idx}...")
        spike_probs = all_probs[sample_idx]
        label = all_labels[sample_idx]

        # Statistics
        true_probs = spike_probs / 1.3
        mean_prob = np.mean(true_probs)
        stim_mean = np.mean(true_probs[50:150, :])
        max_prob = np.max(true_probs)
        expected_spikes = np.sum(true_probs)

        # Decode
        if lgn_params_available:
            decoded, weight_map = decode_with_rf_params(spike_probs, rf_x, rf_y, rf_sizes)
        else:
            decoded = decode_simple_grid(spike_probs)

        # Plot 1: Temporal profile
        ax1 = fig.add_subplot(gs[row_idx, 0])
        temporal = np.mean(true_probs, axis=1)
        ax1.plot(temporal, 'k-', linewidth=1)
        ax1.axvline(50, color='red', linestyle='--', alpha=0.5, label='Stimulus')
        ax1.axvline(150, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Mean Probability')
        ax1.set_title(f'Sample {sample_idx} (Label={label})\\nTemporal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add stats
        stats_text = (f'Mean: {mean_prob:.6f}\\n'
                     f'Stim: {stim_mean:.6f}\\n'
                     f'Max: {max_prob:.6f}\\n'
                     f'Expected: {expected_spikes:.0f}')
        ax1.text(0.98, 0.98, stats_text,
                transform=ax1.transAxes, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=8, family='monospace')

        # Plot 2: Neuron activity distribution
        ax2 = fig.add_subplot(gs[row_idx, 1])
        neuron_activity = np.mean(true_probs[50:150, :], axis=0)
        ax2.hist(neuron_activity, bins=50, color='black', alpha=0.7, log=True)
        ax2.set_xlabel('Neuron Activity')
        ax2.set_ylabel('Count (log scale)')
        ax2.set_title('Activity Distribution')
        ax2.axvline(np.mean(neuron_activity), color='red', linestyle='--',
                   label=f'Mean={np.mean(neuron_activity):.6f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Neuron activity spatial
        ax3 = fig.add_subplot(gs[row_idx, 2])
        downsample = 50
        ax3.plot(neuron_activity[::downsample], 'k-', linewidth=0.8)
        ax3.set_xlabel(f'Neuron ID (/{downsample})')
        ax3.set_ylabel('Activity')
        ax3.set_title('Neuron Activity (Spatial)')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Decoded image
        ax4 = fig.add_subplot(gs[row_idx, 3])
        im4 = ax4.imshow(decoded, cmap='gray', aspect='auto')
        ax4.set_title('DECODED IMAGE')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        plt.colorbar(im4, ax=ax4)

        if lgn_params_available:
            # Plot 5: Weight map (RF coverage)
            ax5 = fig.add_subplot(gs[row_idx, 4])
            im5 = ax5.imshow(weight_map, cmap='viridis', aspect='auto')
            ax5.set_title('RF Coverage')
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            plt.colorbar(im5, ax=ax5)

            # Plot 6: RF centers colored by activity
            ax6 = fig.add_subplot(gs[row_idx, 5])
            scatter = ax6.scatter(rf_x, rf_y, c=neuron_activity,
                                 s=rf_sizes/2, alpha=0.5, cmap='hot')
            ax6.set_xlim([0, 240])
            ax6.set_ylim([0, 120])
            ax6.set_aspect('equal')
            ax6.invert_yaxis()
            ax6.set_title('RF Centers (colored by activity)')
            ax6.set_xlabel('X')
            ax6.set_ylabel('Y')
            plt.colorbar(scatter, ax=ax6)

    if lgn_params_available:
        plt.suptitle('LGN Decoding with Receptive Field Parameters',
                    fontsize=16, fontweight='bold')
    else:
        plt.suptitle('LGN Decoding (Simplified)',
                    fontsize=16, fontweight='bold')

    # Save
    if args.output:
        output_file = args.output
    else:
        output_file = f'lgn_decode_advanced_{"_".join(map(str, samples))}.png'

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_file}")


# ============================================================================
# Main
# ============================================================================

print("\n" + "="*80)
print("ADVANCED LGN DECODING")
print("="*80)

visualize_decoding(SAMPLES, all_probs, all_labels)

print("\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80)
print("""
The decoded images show the spatial pattern of LGN activity reconstructed
into image space using receptive field parameters.

What to look for:

1. **Decoded Image Quality**:
   - Do you see recognizable digit shapes?
   - Are failed samples' decoded images degraded or corrupted?
   - Are working samples' decoded images clearer?

2. **Activity Distribution**:
   - Failed samples with flat/narrow distributions → weak encoding
   - Working samples should have broader, multi-modal distributions

3. **RF Coverage** (if available):
   - Shows which parts of visual space are being sampled
   - Uniform coverage → good spatial encoding
   - Gaps or holes → potential encoding issues

4. **RF Centers**:
   - Hot spots should correspond to digit features
   - Failed samples might show random/diffuse activity patterns

If failed samples have systematically weaker or more uniform decoded images,
this suggests the LGN encoding itself is the problem (not the V1 network).
""")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
