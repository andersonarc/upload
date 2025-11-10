#!/usr/bin/env python
"""
Check if the H5 file was generated with proper LGN receptive fields.

If LGN was generated correctly:
- Different spatial locations should produce different neuron patterns
- Nearby neurons (in RF space) should have correlated activity
- There should be spatial structure visible

If LGN was broken (missing/wrong CSV):
- Activity might be random/uniform
- No spatial correlation
- All samples look similar
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

print("Checking H5 file quality...")
print("="*80)

with h5py.File('spikes-128.h5', 'r') as f:
    all_probs = f['spike_trains'][:]
    all_labels = f['labels'][:]

    print("\nH5 File Metadata:")
    for key in f.attrs.keys():
        print(f"  {key}: {f.attrs[key]}")

    pre_delay = f.attrs.get('pre_delay', 50)
    im_slice = f.attrs.get('im_slice', 100)

print(f"\nDataset shape: {all_probs.shape}")
print(f"  Samples: {all_probs.shape[0]}")
print(f"  Time points: {all_probs.shape[1]}")
print(f"  Neurons: {all_probs.shape[2]}")

# Check spatial structure by comparing different samples
print("\n" + "="*80)
print("SPATIAL STRUCTURE CHECK")
print("="*80)

# Select samples with different labels
unique_labels = np.unique(all_labels)
sample_indices = []
for label in unique_labels[:5]:  # First 5 unique digits
    idx = np.where(all_labels == label)[0]
    if len(idx) > 0:
        sample_indices.append(idx[0])

print(f"\nAnalyzing {len(sample_indices)} samples with different labels:")
for idx in sample_indices:
    print(f"  Sample {idx}: Label {all_labels[idx]}")

# Compute neuron activity during stimulus for each sample
activities = []
for idx in sample_indices:
    probs = all_probs[idx] / 1.3
    activity = np.mean(probs[pre_delay:pre_delay+im_slice, :], axis=0)
    activities.append(activity)

activities = np.array(activities)

# Check if different samples have different spatial patterns
print("\n" + "="*80)
print("SPATIAL DIFFERENTIATION TEST")
print("="*80)

correlations = []
for i in range(len(activities)):
    for j in range(i+1, len(activities)):
        corr = np.corrcoef(activities[i], activities[j])[0, 1]
        correlations.append(corr)
        print(f"  Sample {sample_indices[i]} (label {all_labels[sample_indices[i]]}) vs "
              f"Sample {sample_indices[j]} (label {all_labels[sample_indices[j]]}): "
              f"correlation = {corr:.4f}")

mean_corr = np.mean(correlations)
print(f"\nMean correlation between different digits: {mean_corr:.4f}")

if mean_corr > 0.9:
    print("  ⚠️  WARNING: Very high correlation suggests weak spatial differentiation")
    print("      This could indicate LGN was generated incorrectly")
elif mean_corr > 0.7:
    print("  ⚠️  MODERATE: Some spatial structure but may not be optimal")
elif mean_corr > 0.5:
    print("  ✓  GOOD: Different digits have moderately different patterns")
else:
    print("  ✓  EXCELLENT: Strong spatial differentiation between digits")

# Check for spatial clustering
print("\n" + "="*80)
print("SPATIAL CLUSTERING TEST")
print("="*80)

# If LGN RFs are properly distributed, nearby neurons should be more correlated
# Sample neurons in chunks and check within-chunk vs between-chunk correlation

chunk_size = 1000
n_chunks = all_probs.shape[2] // chunk_size

within_chunk_corrs = []
between_chunk_corrs = []

for sample_idx in sample_indices[:3]:  # Check first 3 samples
    probs = all_probs[sample_idx] / 1.3
    activity = np.mean(probs[pre_delay:pre_delay+im_slice, :], axis=0)

    for chunk_i in range(min(3, n_chunks)):
        start_i = chunk_i * chunk_size
        end_i = start_i + chunk_size

        # Within-chunk correlation
        chunk_activity = activity[start_i:end_i]
        if len(chunk_activity) > 10:
            # Sample 10 random pairs within chunk
            for _ in range(10):
                i1, i2 = np.random.choice(len(chunk_activity), 2, replace=False)
                # Can't correlate single values, use surrounding time window instead
                pass  # Skip for now

        # Between-chunk correlation
        for chunk_j in range(chunk_i+1, min(3, n_chunks)):
            start_j = chunk_j * chunk_size
            end_j = start_j + chunk_size

            # This test isn't really valid without knowing actual RF positions
            # Skip for now

# Check diversity of activity patterns
print("\n" + "="*80)
print("ACTIVITY DIVERSITY TEST")
print("="*80)

for idx in sample_indices:
    probs = all_probs[idx] / 1.3
    activity = np.mean(probs[pre_delay:pre_delay+im_slice, :], axis=0)

    unique_activities = len(np.unique(np.round(activity, 6)))
    sparsity = np.sum(activity < 0.001) / len(activity)
    dynamic_range = activity.max() - activity.min()

    print(f"\nSample {idx} (Label {all_labels[idx]}):")
    print(f"  Unique activity levels: {unique_activities}/{len(activity)} ({unique_activities/len(activity)*100:.1f}%)")
    print(f"  Sparsity (< 0.001): {sparsity*100:.1f}%")
    print(f"  Dynamic range: {dynamic_range:.6f}")
    print(f"  Mean activity: {activity.mean():.6f}")
    print(f"  Active neurons (>0.01): {np.sum(activity > 0.01)}")

# Visualization
fig, axes = plt.subplots(2, len(sample_indices), figsize=(4*len(sample_indices), 8))

for col_idx, idx in enumerate(sample_indices):
    probs = all_probs[idx] / 1.3
    activity = np.mean(probs[pre_delay:pre_delay+im_slice, :], axis=0)

    # Top row: activity profile
    ax_top = axes[0, col_idx] if len(sample_indices) > 1 else axes[0]
    downsample = 50
    ax_top.plot(activity[::downsample], 'k-', linewidth=0.8)
    ax_top.set_title(f'Sample {idx} (Label {all_labels[idx]})')
    ax_top.set_xlabel(f'Neuron ID (/{downsample})')
    ax_top.set_ylabel('Activity')
    ax_top.grid(True, alpha=0.3)

    # Bottom row: activity distribution
    ax_bot = axes[1, col_idx] if len(sample_indices) > 1 else axes[1]
    ax_bot.hist(activity, bins=50, color='black', alpha=0.7, log=True)
    ax_bot.set_xlabel('Activity')
    ax_bot.set_ylabel('Count (log)')
    ax_bot.grid(True, alpha=0.3)

plt.suptitle('Spatial Activity Patterns Across Different Digits', fontweight='bold')
plt.tight_layout()
plt.savefig('h5_lgn_quality_check.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to: h5_lgn_quality_check.png")

print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)
print("""
If LGN was generated correctly with proper receptive fields:
1. Different digits should have different neuron activity patterns (low correlation)
2. Each sample should have diverse activity levels (many unique values)
3. Spatial structure should be evident (not random noise)

If LGN was generated WITHOUT proper CSV or with wrong parameters:
1. All samples might look similar (high correlation)
2. Activity might be uniform or random
3. No clear spatial structure

Based on the correlation values and diversity metrics above,
you can diagnose if the H5 file was generated correctly.
""")
