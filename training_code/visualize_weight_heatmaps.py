#!/usr/bin/env python
"""
Weight Heatmap Comparison: TensorFlow checkpoint vs H5 vs Untrained

Creates side-by-side heatmaps to verify H5 weights match trained checkpoint.
All weights shown in same format (denormalized pA).

Usage:
    python visualize_weight_heatmaps.py --checkpoint /path/to/ckpt --h5 /path/to/file.h5
"""

import os
import sys
import argparse
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import dendrogram, linkage

# Import load_sparse to get network structure
sys.path.insert(0, os.path.dirname(__file__))
import load_sparse as ls

def load_tf_checkpoint_weights(checkpoint_path, network):
    """Load weights from TensorFlow checkpoint"""
    print(f"\n{'='*80}")
    print(f"LOADING TENSORFLOW CHECKPOINT")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not tf.train.latest_checkpoint(checkpoint_dir):
        print(f"⚠️  WARNING: No checkpoint found at {checkpoint_path}")
        return None

    # Create dummy model to restore checkpoint
    checkpoint = tf.train.Checkpoint()
    checkpoint.restore(checkpoint_path).expect_partial()

    # Extract weight variables
    all_vars = {v.name: v for v in checkpoint.save_counter.numpy() if hasattr(v, 'numpy')}

    weights = {}

    # Try to find recurrent weights
    for name, var in all_vars.items():
        if 'sparse_recurrent_weights' in name.lower() or 'recurrent' in name.lower():
            if hasattr(var, 'numpy'):
                weights['recurrent'] = var.numpy()
                print(f"✓ Found recurrent weights: {name}, shape={var.shape}")

        if 'input_weights' in name.lower() or 'input' in name.lower():
            if hasattr(var, 'numpy'):
                weights['input'] = var.numpy()
                print(f"✓ Found input weights: {name}, shape={var.shape}")

    if not weights:
        print("⚠️  WARNING: Could not extract weights from checkpoint")
        return None

    return weights

def load_h5_weights(h5_path):
    """Load weights from H5 file"""
    print(f"\n{'='*80}")
    print(f"LOADING H5 WEIGHTS")
    print(f"{'='*80}")
    print(f"H5 file: {h5_path}")

    weights = {}
    with h5py.File(h5_path, 'r') as f:
        if 'recurrent/weights' in f:
            weights['recurrent'] = np.array(f['recurrent/weights'])
            print(f"✓ Loaded recurrent weights: shape={weights['recurrent'].shape}")

        if 'input/weights' in f:
            weights['input'] = np.array(f['input/weights'])
            print(f"✓ Loaded input weights: shape={weights['input'].shape}")

    return weights

def generate_untrained_weights(network):
    """Generate untrained weights from network structure"""
    print(f"\n{'='*80}")
    print(f"GENERATING UNTRAINED WEIGHTS")
    print(f"{'='*80}")

    weights = {}

    # Use network structure to get weight dimensions
    if 'synapses' in network and 'weights' in network['synapses']:
        weights['recurrent'] = network['synapses']['weights'].copy()
        print(f"✓ Generated untrained recurrent weights: shape={weights['recurrent'].shape}")

    if 'input_weights' in network:
        weights['input'] = network['input_weights'].copy()
        print(f"✓ Generated untrained input weights: shape={weights['input'].shape}")

    return weights

def print_weight_stats(weights_dict, name):
    """Print statistics for weight set"""
    print(f"\n{'='*80}")
    print(f"{name} STATISTICS")
    print(f"{'='*80}")

    for key, w in weights_dict.items():
        if w is None:
            print(f"\n{key}: NOT AVAILABLE")
            continue

        print(f"\n{key} weights:")
        print(f"  Shape: {w.shape}")
        print(f"  Count: {len(w)}")
        print(f"  Mean: {np.mean(w):.6f}")
        print(f"  Std: {np.std(w):.6f}")
        print(f"  Min: {np.min(w):.6f}")
        print(f"  Max: {np.max(w):.6f}")
        print(f"  Median: {np.median(w):.6f}")
        print(f"  Positive: {np.sum(w > 0)} ({np.sum(w > 0)/len(w)*100:.1f}%)")
        print(f"  Negative: {np.sum(w < 0)} ({np.sum(w < 0)/len(w)*100:.1f}%)")
        print(f"  Zero: {np.sum(w == 0)} ({np.sum(w == 0)/len(w)*100:.1f}%)")

def create_heatmap_comparison(tf_weights, h5_weights, untrained_weights, output_path):
    """Create comprehensive heatmap comparison"""
    print(f"\n{'='*80}")
    print(f"CREATING HEATMAP COMPARISON")
    print(f"{'='*80}")

    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25)

    # Determine weight types available
    weight_types = ['recurrent', 'input']

    for idx, wtype in enumerate(weight_types):
        # Get weights
        tf_w = tf_weights.get(wtype) if tf_weights else None
        h5_w = h5_weights.get(wtype) if h5_weights else None
        ut_w = untrained_weights.get(wtype) if untrained_weights else None

        # Determine global color scale
        all_weights = [w for w in [tf_w, h5_w, ut_w] if w is not None]
        if not all_weights:
            continue

        vmax = np.percentile(np.abs(np.concatenate([w.flatten() for w in all_weights])), 99)
        vmin = -vmax

        # Sample for visualization (use up to 10k weights)
        sample_size = min(10000, min([len(w) for w in all_weights if w is not None]))

        # Row 1: TensorFlow checkpoint
        if tf_w is not None:
            ax = fig.add_subplot(gs[idx*2, 0])
            sample_idx = np.random.choice(len(tf_w), sample_size, replace=False)
            sample = tf_w[sample_idx].reshape(100, sample_size//100)
            im = ax.imshow(sample, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title(f'TensorFlow Checkpoint - {wtype.capitalize()}\n'
                        f'Mean={np.mean(tf_w):.4f}, Std={np.std(tf_w):.4f}',
                        fontweight='bold', fontsize=11)
            ax.set_ylabel('Block', fontweight='bold')
            ax.set_xlabel('Index', fontweight='bold')
            plt.colorbar(im, ax=ax, label='Weight (pA)')

        # Row 1: H5 file
        if h5_w is not None:
            ax = fig.add_subplot(gs[idx*2, 1])
            sample_idx = np.random.choice(len(h5_w), sample_size, replace=False)
            sample = h5_w[sample_idx].reshape(100, sample_size//100)
            im = ax.imshow(sample, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title(f'H5 File - {wtype.capitalize()}\n'
                        f'Mean={np.mean(h5_w):.4f}, Std={np.std(h5_w):.4f}',
                        fontweight='bold', fontsize=11)
            ax.set_ylabel('Block', fontweight='bold')
            ax.set_xlabel('Index', fontweight='bold')
            plt.colorbar(im, ax=ax, label='Weight (pA)')

        # Row 1: Untrained
        if ut_w is not None:
            ax = fig.add_subplot(gs[idx*2, 2])
            sample_idx = np.random.choice(len(ut_w), sample_size, replace=False)
            sample = ut_w[sample_idx].reshape(100, sample_size//100)
            im = ax.imshow(sample, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title(f'Untrained - {wtype.capitalize()}\n'
                        f'Mean={np.mean(ut_w):.4f}, Std={np.std(ut_w):.4f}',
                        fontweight='bold', fontsize=11)
            ax.set_ylabel('Block', fontweight='bold')
            ax.set_xlabel('Index', fontweight='bold')
            plt.colorbar(im, ax=ax, label='Weight (pA)')

        # Row 2: Distribution comparisons
        ax = fig.add_subplot(gs[idx*2+1, :])
        bins = np.linspace(vmin, vmax, 100)

        if tf_w is not None:
            ax.hist(tf_w, bins=bins, alpha=0.5, label='TensorFlow', color='blue', density=True)
        if h5_w is not None:
            ax.hist(h5_w, bins=bins, alpha=0.5, label='H5', color='green', density=True)
        if ut_w is not None:
            ax.hist(ut_w, bins=bins, alpha=0.5, label='Untrained', color='red', density=True)

        ax.set_xlabel('Weight Value (pA)', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title(f'{wtype.capitalize()} Weight Distribution Comparison', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.suptitle('Weight Heatmap Comparison: TensorFlow vs H5 vs Untrained',
                 fontsize=16, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

def create_difference_analysis(tf_weights, h5_weights, output_path):
    """Create difference analysis between TensorFlow and H5"""
    if tf_weights is None or h5_weights is None:
        print("⚠️  Cannot create difference analysis - missing weights")
        return

    print(f"\n{'='*80}")
    print(f"DIFFERENCE ANALYSIS")
    print(f"{'='*80}")

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    for idx, wtype in enumerate(['recurrent', 'input']):
        if wtype not in tf_weights or wtype not in h5_weights:
            continue

        tf_w = tf_weights[wtype]
        h5_w = h5_weights[wtype]

        if len(tf_w) != len(h5_w):
            print(f"⚠️  {wtype}: Shape mismatch TF={len(tf_w)} vs H5={len(h5_w)}")
            continue

        diff = h5_w - tf_w

        print(f"\n{wtype} difference:")
        print(f"  Mean difference: {np.mean(diff):.6f}")
        print(f"  Std difference: {np.std(diff):.6f}")
        print(f"  Max absolute difference: {np.max(np.abs(diff)):.6f}")
        print(f"  Correlation: {np.corrcoef(tf_w, h5_w)[0,1]:.6f}")

        # Scatter plot
        ax1 = fig.add_subplot(gs[idx, 0])
        sample_size = min(10000, len(tf_w))
        sample_idx = np.random.choice(len(tf_w), sample_size, replace=False)
        ax1.scatter(tf_w[sample_idx], h5_w[sample_idx], alpha=0.3, s=1)
        ax1.plot([tf_w.min(), tf_w.max()], [tf_w.min(), tf_w.max()], 'r--', linewidth=2, label='y=x')
        ax1.set_xlabel('TensorFlow Weights (pA)', fontweight='bold')
        ax1.set_ylabel('H5 Weights (pA)', fontweight='bold')
        ax1.set_title(f'{wtype.capitalize()} - TF vs H5\n'
                     f'Correlation={np.corrcoef(tf_w, h5_w)[0,1]:.4f}',
                     fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Difference histogram
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.hist(diff, bins=100, color='purple', alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
        ax2.set_xlabel('Difference (H5 - TensorFlow) [pA]', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title(f'{wtype.capitalize()} - Difference Distribution\n'
                     f'Mean={np.mean(diff):.6f}, Std={np.std(diff):.6f}',
                     fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.suptitle('TensorFlow vs H5 Difference Analysis', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare weight heatmaps')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to TensorFlow checkpoint')
    parser.add_argument('--h5', type=str, required=True,
                       help='Path to H5 file')
    parser.add_argument('--data_dir', type=str, default='v1cortex',
                       help='Path to network data directory')
    parser.add_argument('--output', type=str, default='weight_heatmap_comparison.png',
                       help='Output file for heatmap comparison')
    parser.add_argument('--output_diff', type=str, default='weight_difference_analysis.png',
                       help='Output file for difference analysis')

    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"WEIGHT HEATMAP COMPARISON")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"H5 file: {args.h5}")
    print(f"Data dir: {args.data_dir}")

    # Load network structure to get untrained weights
    print(f"\n{'='*80}")
    print(f"LOADING NETWORK STRUCTURE")
    print(f"{'='*80}")
    input_population, network, bkg_weights = ls.cached_load_billeh(
        n_input=17400,
        n_neurons=51978,
        core_only=False,
        data_dir=args.data_dir,
        n_output=10,
        neurons_per_output=30
    )

    # Load all weight sets
    tf_weights = load_tf_checkpoint_weights(args.checkpoint, network)
    h5_weights = load_h5_weights(args.h5)
    untrained_weights = generate_untrained_weights(network)

    # Print statistics
    if tf_weights:
        print_weight_stats(tf_weights, "TENSORFLOW CHECKPOINT")
    if h5_weights:
        print_weight_stats(h5_weights, "H5 FILE")
    if untrained_weights:
        print_weight_stats(untrained_weights, "UNTRAINED")

    # Create visualizations
    create_heatmap_comparison(tf_weights, h5_weights, untrained_weights, args.output)

    if tf_weights and h5_weights:
        create_difference_analysis(tf_weights, h5_weights, args.output_diff)

    print(f"\n{'='*80}")
    print(f"VERIFICATION COMPLETE")
    print(f"{'='*80}")

    # Final verdict
    if tf_weights and h5_weights:
        print("\n✓ Weight comparison complete")
        print("  Check heatmaps to verify H5 matches TensorFlow checkpoint")
    else:
        print("\n⚠️  Incomplete comparison - missing weight sources")

if __name__ == '__main__':
    main()
