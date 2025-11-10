#!/usr/bin/env python
"""
Diagnostic script to check if H5 weights are trained or untrained.

Based on c2.py analysis:
- Trained weights should have specific statistics (learned patterns)
- Untrained weights would match initial random distribution

This script checks weight statistics to determine status.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

H5_FILE = 'ckpt_51978-153.h5'

print("="*80)
print("H5 WEIGHT VALIDATION")
print("="*80)

try:
    with h5py.File(H5_FILE, 'r') as f:
        print("\n✓ H5 file found and opened successfully")

        print("\n" + "="*80)
        print("FILE STRUCTURE")
        print("="*80)

        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name:40s} shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group:   {name}")

        f.visititems(print_structure)

        print("\n" + "="*80)
        print("WEIGHT STATISTICS")
        print("="*80)

        # Load weights
        rec_weights = np.array(f['recurrent/weights'])
        inp_weights = np.array(f['input/weights'])

        print(f"\nRecurrent weights:")
        print(f"  Shape: {rec_weights.shape}")
        print(f"  Count: {len(rec_weights)}")
        print(f"  Mean: {np.mean(rec_weights):.6f}")
        print(f"  Std: {np.std(rec_weights):.6f}")
        print(f"  Min: {np.min(rec_weights):.6f}")
        print(f"  Max: {np.max(rec_weights):.6f}")
        print(f"  Positive: {np.sum(rec_weights > 0)} ({np.sum(rec_weights > 0)/len(rec_weights)*100:.1f}%)")
        print(f"  Negative: {np.sum(rec_weights < 0)} ({np.sum(rec_weights < 0)/len(rec_weights)*100:.1f}%)")
        print(f"  Zero: {np.sum(rec_weights == 0)} ({np.sum(rec_weights == 0)/len(rec_weights)*100:.1f}%)")

        print(f"\nInput weights:")
        print(f"  Shape: {inp_weights.shape}")
        print(f"  Count: {len(inp_weights)}")
        print(f"  Mean: {np.mean(inp_weights):.6f}")
        print(f"  Std: {np.std(inp_weights):.6f}")
        print(f"  Min: {np.min(inp_weights):.6f}")
        print(f"  Max: {np.max(inp_weights):.6f}")
        print(f"  Positive: {np.sum(inp_weights > 0)} ({np.sum(inp_weights > 0)/len(inp_weights)*100:.1f}%)")
        print(f"  Negative: {np.sum(inp_weights < 0)} ({np.sum(inp_weights < 0)/len(inp_weights)*100:.1f}%)")
        print(f"  Zero: {np.sum(inp_weights == 0)} ({np.sum(inp_weights == 0)/len(inp_weights)*100:.1f}%)")

        # Check for readout weights (only in trained networks)
        if 'readout' in f:
            readout_grp = f['readout']
            print(f"\nReadout group keys: {list(readout_grp.keys())}")

            if 'readout_weights' in readout_grp:
                readout_weights = np.array(readout_grp['readout_weights'])
                print(f"\n✓ READOUT WEIGHTS FOUND (indicates trained network)")
                print(f"  Shape: {readout_weights.shape}")
                print(f"  Mean: {np.mean(readout_weights):.6f}")
                print(f"  Std: {np.std(readout_weights):.6f}")
            else:
                print(f"\n⚠️  NO READOUT WEIGHTS (may indicate untrained network)")

        # Analyze weight distributions
        print("\n" + "="*80)
        print("DISTRIBUTION ANALYSIS")
        print("="*80)

        # Percentiles
        rec_percentiles = np.percentile(rec_weights, [1, 5, 25, 50, 75, 95, 99])
        inp_percentiles = np.percentile(inp_weights, [1, 5, 25, 50, 75, 95, 99])

        print("\nRecurrent weight percentiles:")
        print(f"  1%: {rec_percentiles[0]:.6f}")
        print(f"  5%: {rec_percentiles[1]:.6f}")
        print(f"  25%: {rec_percentiles[2]:.6f}")
        print(f"  50% (median): {rec_percentiles[3]:.6f}")
        print(f"  75%: {rec_percentiles[4]:.6f}")
        print(f"  95%: {rec_percentiles[5]:.6f}")
        print(f"  99%: {rec_percentiles[6]:.6f}")

        print("\nInput weight percentiles:")
        print(f"  1%: {inp_percentiles[0]:.6f}")
        print(f"  5%: {inp_percentiles[1]:.6f}")
        print(f"  25%: {inp_percentiles[2]:.6f}")
        print(f"  50% (median): {inp_percentiles[3]:.6f}")
        print(f"  75%: {inp_percentiles[4]:.6f}")
        print(f"  95%: {inp_percentiles[5]:.6f}")
        print(f"  99%: {inp_percentiles[6]:.6f}")

        # Check for patterns indicating trained vs untrained
        print("\n" + "="*80)
        print("TRAINED vs UNTRAINED INDICATORS")
        print("="*80)

        indicators = []

        # 1. Check if readout weights exist
        if 'readout' in f and 'readout_weights' in f['readout']:
            indicators.append("✓ Readout weights present (TRAINED)")
        else:
            indicators.append("✗ Readout weights missing (UNTRAINED?)")

        # 2. Check weight range
        rec_range = np.max(rec_weights) - np.min(rec_weights)
        if rec_range > 0.001:
            indicators.append(f"✓ Recurrent weights have range {rec_range:.6f} (likely trained)")
        else:
            indicators.append(f"✗ Recurrent weights have narrow range {rec_range:.6f} (uniform?)")

        # 3. Check for non-uniform distribution
        rec_hist, _ = np.histogram(rec_weights, bins=50)
        rec_uniformity = np.std(rec_hist) / np.mean(rec_hist)
        if rec_uniformity > 0.5:
            indicators.append(f"✓ Recurrent weights non-uniform (std/mean={rec_uniformity:.2f}, likely trained)")
        else:
            indicators.append(f"✗ Recurrent weights too uniform (std/mean={rec_uniformity:.2f}, untrained?)")

        # 4. Check if mean is near zero (initial weights often centered)
        if abs(np.mean(rec_weights)) < 0.01:
            indicators.append(f"⚠️  Recurrent weights mean ≈ 0 ({np.mean(rec_weights):.6f}, could be untrained)")
        else:
            indicators.append(f"✓ Recurrent weights mean ≠ 0 ({np.mean(rec_weights):.6f}, likely trained)")

        for indicator in indicators:
            print(f"  {indicator}")

        # Generate visualizations
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Plot 1: Recurrent weight histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(rec_weights, bins=100, color='blue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Weight Value', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.set_title(f'Recurrent Weights Distribution\nMean={np.mean(rec_weights):.6f}, Std={np.std(rec_weights):.6f}',
                     fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Input weight histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(inp_weights, bins=100, color='green', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Weight Value', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title(f'Input Weights Distribution\nMean={np.mean(inp_weights):.6f}, Std={np.std(inp_weights):.6f}',
                     fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Recurrent weight heatmap (sampled)
        ax3 = fig.add_subplot(gs[0, 2])
        sample_size = min(10000, len(rec_weights))
        sample_idx = np.random.choice(len(rec_weights), sample_size, replace=False)
        sample_weights = rec_weights[sample_idx].reshape(100, sample_size//100)
        im3 = ax3.imshow(sample_weights, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(sample_weights)), vmax=np.max(np.abs(sample_weights)))
        ax3.set_title('Recurrent Weights Heatmap\n(10k sample)', fontweight='bold')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Block')
        plt.colorbar(im3, ax=ax3)

        # Plot 4: Positive vs negative weights
        ax4 = fig.add_subplot(gs[1, 0])
        categories = ['Recurrent\nPositive', 'Recurrent\nNegative', 'Recurrent\nZero',
                     'Input\nPositive', 'Input\nNegative', 'Input\nZero']
        counts = [
            np.sum(rec_weights > 0), np.sum(rec_weights < 0), np.sum(rec_weights == 0),
            np.sum(inp_weights > 0), np.sum(inp_weights < 0), np.sum(inp_weights == 0)
        ]
        colors = ['blue', 'red', 'gray', 'green', 'orange', 'gray']
        ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Count', fontweight='bold')
        ax4.set_title('Weight Sign Distribution', fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=45)

        # Plot 5: Cumulative distribution
        ax5 = fig.add_subplot(gs[1, 1])
        rec_sorted = np.sort(rec_weights)
        inp_sorted = np.sort(inp_weights)
        ax5.plot(rec_sorted, np.linspace(0, 1, len(rec_sorted)), 'b-', linewidth=2, label='Recurrent', alpha=0.7)
        ax5.plot(inp_sorted, np.linspace(0, 1, len(inp_sorted)), 'g-', linewidth=2, label='Input', alpha=0.7)
        ax5.set_xlabel('Weight Value', fontweight='bold')
        ax5.set_ylabel('Cumulative Probability', fontweight='bold')
        ax5.set_title('Cumulative Distribution', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Plot 6: Weight magnitude distribution
        ax6 = fig.add_subplot(gs[1, 2])
        rec_abs = np.abs(rec_weights)
        inp_abs = np.abs(inp_weights)
        ax6.hist(rec_abs, bins=100, color='blue', alpha=0.5, label='Recurrent', edgecolor='black')
        ax6.hist(inp_abs, bins=100, color='green', alpha=0.5, label='Input', edgecolor='black')
        ax6.set_xlabel('Absolute Weight Value', fontweight='bold')
        ax6.set_ylabel('Count', fontweight='bold')
        ax6.set_title('Weight Magnitude Distribution', fontweight='bold')
        ax6.set_yscale('log')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Plot 7: Q-Q plot (check normality)
        ax7 = fig.add_subplot(gs[2, 0])
        from scipy import stats
        sample_size = min(5000, len(rec_weights))
        sample = np.random.choice(rec_weights, sample_size, replace=False)
        stats.probplot(sample, dist="norm", plot=ax7)
        ax7.set_title('Q-Q Plot (Recurrent Weights)', fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # Plot 8: Scatter plot (first 10k connections)
        ax8 = fig.add_subplot(gs[2, 1])
        plot_size = min(10000, len(rec_weights))
        ax8.scatter(range(plot_size), rec_weights[:plot_size], c='blue', s=1, alpha=0.5)
        ax8.set_xlabel('Connection Index', fontweight='bold')
        ax8.set_ylabel('Weight Value', fontweight='bold')
        ax8.set_title('Recurrent Weight Sequence (first 10k)', fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # Plot 9: Summary text
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_text = "WEIGHT VALIDATION SUMMARY\\n" + "="*40 + "\\n\\n"

        # Count indicators
        trained_indicators = sum(1 for i in indicators if i.startswith('✓'))
        untrained_indicators = sum(1 for i in indicators if i.startswith('✗'))
        uncertain_indicators = sum(1 for i in indicators if i.startswith('⚠️'))

        summary_text += f"Trained indicators: {trained_indicators}\\n"
        summary_text += f"Untrained indicators: {untrained_indicators}\\n"
        summary_text += f"Uncertain indicators: {uncertain_indicators}\\n\\n"

        if trained_indicators > untrained_indicators:
            verdict = "✓ LIKELY TRAINED"
            color = 'green'
        elif untrained_indicators > trained_indicators:
            verdict = "✗ LIKELY UNTRAINED"
            color = 'red'
        else:
            verdict = "⚠️  UNCERTAIN"
            color = 'orange'

        summary_text += f"\\nVerdict: {verdict}\\n\\n"
        summary_text += "="*40 + "\\n\\nIndicators:\\n"
        for ind in indicators:
            summary_text += f"  {ind}\\n"

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

        plt.suptitle(f'H5 Weight Analysis: {H5_FILE}', fontsize=16, fontweight='bold')

        output_file = 'h5_weight_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization: {output_file}")

        # Final verdict
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        print(f"\n{verdict}")
        print(f"\nEvidence:")
        for ind in indicators:
            print(f"  {ind}")

        print("\n" + "="*80)

except FileNotFoundError:
    print(f"\n✗ ERROR: H5 file not found: {H5_FILE}")
    print("\nThis script requires the H5 file to be present.")
    print("Expected location: ./ckpt_51978-153.h5")
    print("\nTo download:")
    print("  export HF_TOKEN=$(cat /tmp/hf_token)")
    print("  wget --header=\"Authorization: Bearer $HF_TOKEN\" \\")
    print("    https://huggingface.co/rmri/v1cortex-mnist_51978-ckpts-50-56/resolve/main/ckpt_51978-153.h5")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
