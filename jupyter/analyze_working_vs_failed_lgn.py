#!/usr/bin/env python
"""
Since LGN encoding is CORRECT for all samples (verified by visual decoding),
analyze what differs between samples that work (90+) vs fail (10-80).

The difference must be in the LGN activity LEVELS, not spatial structure.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FAILED = [10, 20, 30, 40, 50, 60, 70, 80]
WORKING = [90, 100, 123, 124]

print("="*80)
print("ANALYZING LGN ACTIVITY LEVELS: FAILED vs WORKING")
print("="*80)
print("\nKey insight: LGN encoding is spatially correct for ALL samples")
print("(verified by visual decoding with CSV)")
print("\nQuestion: What makes samples 90+ work while 10-80 fail?")
print("="*80)

with h5py.File('spikes-128.h5', 'r') as f:
    all_probs = f['spike_trains'][:]
    all_labels = f['labels'][:]
    pre_delay = f.attrs.get('pre_delay', 50)
    im_slice = f.attrs.get('im_slice', 100)

def analyze_sample_strength(idx):
    """Compute activity strength metrics."""
    probs = all_probs[idx] / 1.3  # True probabilities

    # Baseline (pre-stimulus)
    baseline = probs[:pre_delay, :]

    # Stimulus period
    stim = probs[pre_delay:pre_delay+im_slice, :]

    # Per-neuron average during stimulus
    neuron_activity = np.mean(stim, axis=0)

    return {
        'idx': idx,
        'label': all_labels[idx],

        # Overall activity
        'stim_mean': np.mean(stim),
        'stim_median': np.median(stim),
        'stim_max': np.max(stim),
        'stim_std': np.std(stim),

        # Baseline
        'baseline_mean': np.mean(baseline),
        'baseline_max': np.max(baseline),

        # SNR
        'snr': np.mean(stim) / np.mean(baseline) if np.mean(baseline) > 0 else np.inf,

        # Expected spikes (total across all neurons and time)
        'expected_total_spikes': np.sum(stim),

        # Expected spikes per neuron during stimulus
        'expected_spikes_per_neuron': np.sum(stim) / stim.shape[1],

        # Active neurons
        'active_neurons_001': np.sum(neuron_activity > 0.001),
        'active_neurons_01': np.sum(neuron_activity > 0.01),
        'active_neurons_05': np.sum(neuron_activity > 0.05),
        'active_neurons_1': np.sum(neuron_activity > 0.1),

        # Distribution metrics
        'activity_75th_percentile': np.percentile(neuron_activity, 75),
        'activity_90th_percentile': np.percentile(neuron_activity, 90),
        'activity_95th_percentile': np.percentile(neuron_activity, 95),
        'activity_99th_percentile': np.percentile(neuron_activity, 99),

        # Peak activity (how strong are the strongest neurons?)
        'top_100_mean': np.mean(np.sort(neuron_activity)[-100:]),
        'top_500_mean': np.mean(np.sort(neuron_activity)[-500:]),
        'top_1000_mean': np.mean(np.sort(neuron_activity)[-1000:]),

        # Temporal profile
        'stim_onset_mean': np.mean(stim[0:20, :]),  # First 20ms
        'stim_peak_mean': np.mean(stim[20:50, :]),  # Middle 30ms
        'stim_late_mean': np.mean(stim[50:100, :]),  # Last 50ms

        'neuron_activity': neuron_activity,
    }

print("\nAnalyzing samples...")
failed_stats = [analyze_sample_strength(idx) for idx in FAILED if idx < len(all_probs)]
working_stats = [analyze_sample_strength(idx) for idx in WORKING if idx < len(all_probs)]

# Print detailed comparison
print("\n" + "="*80)
print("DETAILED METRICS COMPARISON")
print("="*80)

metrics_to_compare = [
    ('stim_mean', 'Mean Activity During Stimulus', '.6f'),
    ('stim_max', 'Maximum Activity', '.6f'),
    ('expected_total_spikes', 'Expected Total Spikes', '.1f'),
    ('expected_spikes_per_neuron', 'Expected Spikes per Neuron', '.3f'),
    ('snr', 'Signal-to-Noise Ratio', '.2f'),
    ('active_neurons_01', 'Active Neurons (>0.01)', '.0f'),
    ('active_neurons_05', 'Active Neurons (>0.05)', '.0f'),
    ('active_neurons_1', 'Active Neurons (>0.1)', '.0f'),
    ('activity_95th_percentile', '95th Percentile Activity', '.6f'),
    ('top_100_mean', 'Mean of Top 100 Neurons', '.6f'),
    ('top_500_mean', 'Mean of Top 500 Neurons', '.6f'),
    ('stim_onset_mean', 'Stimulus Onset (0-20ms)', '.6f'),
]

from scipy import stats as scipy_stats

print("\nMetric                              Failed          Working         Ratio    p-value")
print("-" * 90)

significant_diffs = []

for key, name, fmt in metrics_to_compare:
    failed_vals = [s[key] for s in failed_stats]
    working_vals = [s[key] for s in working_stats]

    failed_mean = np.mean(failed_vals)
    working_mean = np.mean(working_vals)

    ratio = working_mean / failed_mean if failed_mean > 0 else np.inf

    t_stat, p_val = scipy_stats.ttest_ind(failed_vals, working_vals)

    sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

    print(f"{name:35} {failed_mean:{fmt}}      {working_mean:{fmt}}      {ratio:6.2f}x  {p_val:.4f} {sig_marker}")

    if p_val < 0.05 and abs(ratio - 1.0) > 0.1:
        significant_diffs.append((name, ratio, p_val))

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

if significant_diffs:
    print("\nSignificant differences found:")
    for name, ratio, p_val in significant_diffs:
        direction = "HIGHER" if ratio > 1 else "LOWER"
        print(f"  • {name}: Working samples are {abs(ratio):.2f}x {direction} (p={p_val:.4f})")
else:
    print("\n⚠️  WARNING: No significant differences in LGN activity levels!")
    print("    This suggests the problem may be elsewhere in the network.")

# Find the critical threshold
print("\n" + "="*80)
print("THRESHOLD ANALYSIS")
print("="*80)

# For each metric, find the threshold that best separates failed/working
print("\nLooking for activity thresholds that separate failed/working samples...")

for key, name, fmt in [
    ('stim_mean', 'Mean Activity', '.6f'),
    ('top_100_mean', 'Top 100 Neurons', '.6f'),
    ('expected_total_spikes', 'Expected Total Spikes', '.1f'),
]:
    failed_vals = sorted([s[key] for s in failed_stats])
    working_vals = sorted([s[key] for s in working_stats])

    print(f"\n{name}:")
    print(f"  Failed range:  [{min(failed_vals):{fmt}}, {max(failed_vals):{fmt}}]")
    print(f"  Working range: [{min(working_vals):{fmt}}, {max(working_vals):{fmt}}]")

    # Check for overlap
    if max(failed_vals) < min(working_vals):
        threshold = (max(failed_vals) + min(working_vals)) / 2
        print(f"  ✓ CLEAN SEPARATION at threshold ≈ {threshold:{fmt}}")
        print(f"    All failed samples below, all working samples above")
    elif min(working_vals) < max(failed_vals):
        overlap_start = min(working_vals)
        overlap_end = max(failed_vals)
        print(f"  ⚠️  OVERLAP: [{overlap_start:{fmt}}, {overlap_end:{fmt}}]")
        print(f"    Some failed samples have higher values than some working samples")
    else:
        print(f"  ✓ SEPARATION EXISTS")

# Visualization
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Mean activity
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter([s['idx'] for s in failed_stats], [s['stim_mean'] for s in failed_stats],
           c='red', s=150, alpha=0.7, marker='o', label='Failed', edgecolors='black', linewidths=1)
ax1.scatter([s['idx'] for s in working_stats], [s['stim_mean'] for s in working_stats],
           c='green', s=150, alpha=0.7, marker='s', label='Working', edgecolors='black', linewidths=1)
ax1.set_xlabel('Sample Index', fontweight='bold')
ax1.set_ylabel('Mean LGN Activity', fontweight='bold')
ax1.set_title('Mean Activity During Stimulus', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Expected spikes
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter([s['idx'] for s in failed_stats], [s['expected_total_spikes'] for s in failed_stats],
           c='red', s=150, alpha=0.7, marker='o', label='Failed', edgecolors='black', linewidths=1)
ax2.scatter([s['idx'] for s in working_stats], [s['expected_total_spikes'] for s in working_stats],
           c='green', s=150, alpha=0.7, marker='s', label='Working', edgecolors='black', linewidths=1)
ax2.set_xlabel('Sample Index', fontweight='bold')
ax2.set_ylabel('Expected Total Spikes', fontweight='bold')
ax2.set_title('Expected LGN Spike Count', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Top neuron activity
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter([s['idx'] for s in failed_stats], [s['top_100_mean'] for s in failed_stats],
           c='red', s=150, alpha=0.7, marker='o', label='Failed', edgecolors='black', linewidths=1)
ax3.scatter([s['idx'] for s in working_stats], [s['top_100_mean'] for s in working_stats],
           c='green', s=150, alpha=0.7, marker='s', label='Working', edgecolors='black', linewidths=1)
ax3.set_xlabel('Sample Index', fontweight='bold')
ax3.set_ylabel('Mean of Top 100 Neurons', fontweight='bold')
ax3.set_title('Peak Neuron Activity', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Active neurons
ax4 = fig.add_subplot(gs[0, 3])
ax4.scatter([s['idx'] for s in failed_stats], [s['active_neurons_05'] for s in failed_stats],
           c='red', s=150, alpha=0.7, marker='o', label='Failed', edgecolors='black', linewidths=1)
ax4.scatter([s['idx'] for s in working_stats], [s['active_neurons_05'] for s in working_stats],
           c='green', s=150, alpha=0.7, marker='s', label='Working', edgecolors='black', linewidths=1)
ax4.set_xlabel('Sample Index', fontweight='bold')
ax4.set_ylabel('Active Neurons (>0.05)', fontweight='bold')
ax4.set_title('Number of Strongly Active Neurons', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plots 5-8: Activity distributions for selected samples
selected = [FAILED[0], FAILED[-1], WORKING[0], WORKING[-1]]
colors = ['darkred', 'red', 'lightgreen', 'darkgreen']
labels_text = ['Failed (low idx)', 'Failed (high idx)', 'Working (low idx)', 'Working (high idx)']

for plot_idx, (sample_idx, color, label_text) in enumerate(zip(selected, colors, labels_text)):
    ax = fig.add_subplot(gs[1, plot_idx])

    stat = None
    for s in failed_stats + working_stats:
        if s['idx'] == sample_idx:
            stat = s
            break

    if stat is None:
        continue

    ax.hist(stat['neuron_activity'], bins=50, color=color, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Neuron Activity', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title(f'Sample {sample_idx}: {label_text}\\nLabel={stat["label"]}', fontweight='bold')
    ax.set_yscale('log')
    ax.axvline(0.01, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='0.01 threshold')
    ax.axvline(0.05, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='0.05 threshold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Plots 9-12: Cumulative distribution
ax9 = fig.add_subplot(gs[2, :])

for stat in failed_stats:
    sorted_activity = np.sort(stat['neuron_activity'])
    cumsum = np.arange(1, len(sorted_activity) + 1) / len(sorted_activity)
    ax9.plot(sorted_activity, cumsum, 'r-', alpha=0.3, linewidth=1)

for stat in working_stats:
    sorted_activity = np.sort(stat['neuron_activity'])
    cumsum = np.arange(1, len(sorted_activity) + 1) / len(sorted_activity)
    ax9.plot(sorted_activity, cumsum, 'g-', alpha=0.3, linewidth=1)

# Add representative curves
failed_median_activity = np.median([s['neuron_activity'] for s in failed_stats], axis=0)
working_median_activity = np.median([s['neuron_activity'] for s in working_stats], axis=0)

sorted_failed = np.sort(failed_median_activity)
cumsum_failed = np.arange(1, len(sorted_failed) + 1) / len(sorted_failed)
ax9.plot(sorted_failed, cumsum_failed, 'r-', linewidth=3, label='Failed (median)', alpha=0.8)

sorted_working = np.sort(working_median_activity)
cumsum_working = np.arange(1, len(sorted_working) + 1) / len(sorted_working)
ax9.plot(sorted_working, cumsum_working, 'g-', linewidth=3, label='Working (median)', alpha=0.8)

ax9.set_xlabel('Neuron Activity', fontweight='bold', fontsize=12)
ax9.set_ylabel('Cumulative Fraction', fontweight='bold', fontsize=12)
ax9.set_title('Cumulative Distribution of Neuron Activity', fontweight='bold', fontsize=14)
ax9.set_xscale('log')
ax9.axvline(0.01, color='blue', linestyle='--', alpha=0.5, linewidth=2)
ax9.axvline(0.05, color='orange', linestyle='--', alpha=0.5, linewidth=2)
ax9.legend(fontsize=10)
ax9.grid(True, alpha=0.3, which='both')

plt.suptitle('LGN Activity Level Analysis: Failed vs Working Samples\\n(LGN Encoding is Spatially Correct)',
             fontsize=16, fontweight='bold')

plt.savefig('lgn_activity_threshold_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization: lgn_activity_threshold_analysis.png")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
Since LGN encoding is spatially correct (verified by visual decoding),
the bimodal failure must be due to ACTIVITY LEVEL differences.

If working samples have significantly higher LGN activity:
→ V1 network needs a minimum input strength threshold to activate
→ Solution: Increase LGN→V1 connection weights or reduce V1 thresholds

If failed and working samples have similar LGN activity:
→ Problem is elsewhere (V1 connectivity, ASC scaling, etc.)
→ Need to investigate V1 network parameters more deeply

Check the statistics above to determine which case applies.
""")
