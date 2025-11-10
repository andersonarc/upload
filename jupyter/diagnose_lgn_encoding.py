#!/usr/bin/env python
"""
Diagnose LGN encoding to understand why decoded images don't look like MNIST.

This script will:
1. Load the original MNIST images
2. Compare with LGN spike probabilities
3. Check temporal structure
4. Verify encoding is working at all
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

SAMPLE_IDX = int(os.environ.get('SAMPLE_IDX', '10'))

print(f"Diagnosing sample {SAMPLE_IDX}")

# Load spike probabilities
print("\nLoading spike probabilities...")
with h5py.File('spikes-128.h5', 'r') as f:
    spike_probs = f['spike_trains'][SAMPLE_IDX]  # [time, neurons]
    label = f['labels'][SAMPLE_IDX]

    print(f"\nH5 file attributes:")
    for key in f.attrs.keys():
        print(f"  {key}: {f.attrs[key]}")

    pre_delay = f.attrs.get('pre_delay', 50)
    im_slice = f.attrs.get('im_slice', 100)
    post_delay = f.attrs.get('post_delay', 150)

print(f"\nSample {SAMPLE_IDX}: Label = {label}")
print(f"Spike probabilities shape: {spike_probs.shape}")
print(f"Time structure: pre_delay={pre_delay}ms, stimulus={im_slice}ms, post={post_delay}ms")

# Load original MNIST image
print("\nLoading original MNIST image...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# The H5 file was generated with a specific seed and shuffling
# Try to match the sample
# From mnist.py: seed=3000, using training data, shuffled
np.random.seed(3000)
perm = np.random.permutation(len(x_train))
x_train_shuffled = x_train[perm]
y_train_shuffled = y_train[perm]

original_img = x_train_shuffled[SAMPLE_IDX]
original_label = y_train_shuffled[SAMPLE_IDX]

print(f"Original MNIST image: shape={original_img.shape}, label={original_label}")
print(f"Label match: {original_label == label}")

# Analyze spike probabilities
true_probs = spike_probs / 1.3

print(f"\n{'='*80}")
print("SPIKE PROBABILITY ANALYSIS")
print(f"{'='*80}")

print(f"\nOverall statistics:")
print(f"  Mean: {np.mean(true_probs):.6f}")
print(f"  Std: {np.std(true_probs):.6f}")
print(f"  Max: {np.max(true_probs):.6f}")
print(f"  Min: {np.min(true_probs):.6f}")

print(f"\nTemporal breakdown:")
for name, start, end in [
    ("Pre-stimulus (baseline)", 0, pre_delay),
    ("Stimulus period", pre_delay, pre_delay + im_slice),
    ("Post-stimulus", pre_delay + im_slice, spike_probs.shape[0])
]:
    window_probs = true_probs[start:end, :]
    print(f"  {name} [{start}-{end}ms]:")
    print(f"    Mean: {np.mean(window_probs):.6f}")
    print(f"    Max: {np.max(window_probs):.6f}")
    print(f"    Active neurons (>0.01): {np.sum(np.max(window_probs, axis=0) > 0.01)}")

# Check for structure
print(f"\nSpatial structure check:")
# If encoding is working, different neurons should have different activity levels
neuron_means = np.mean(true_probs[pre_delay:pre_delay+im_slice, :], axis=0)
print(f"  Unique activity levels: {len(np.unique(np.round(neuron_means, 6)))}/{len(neuron_means)}")
print(f"  Activity range: [{np.min(neuron_means):.6f}, {np.max(neuron_means):.6f}]")
print(f"  Active neurons (>0.001): {np.sum(neuron_means > 0.001)}")
print(f"  Active neurons (>0.01): {np.sum(neuron_means > 0.01)}")
print(f"  Active neurons (>0.1): {np.sum(neuron_means > 0.1)}")

# Visualization
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# Row 1: Original image and analysis
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(original_img, cmap='gray')
ax1.set_title(f'Original MNIST\\nSample {SAMPLE_IDX}, Label {label}')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
# Resized version (what LGN sees)
img_resized = tf.image.resize_with_pad(
    original_img[..., None] / 255.0,
    120, 240,
    method='lanczos5'
).numpy()[:, :, 0]
ax2.imshow(img_resized, cmap='gray')
ax2.set_title('Resized to 120x240\\n(LGN input)')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
# Normalized version (with intensity scaling)
intensity = 2.0
img_normalized = (img_resized - 0.5) * intensity / 0.5
ax3.imshow(img_normalized, cmap='seismic', vmin=-2, vmax=2)
ax3.set_title(f'Normalized (intensity={intensity})\\nRange: [{img_normalized.min():.2f}, {img_normalized.max():.2f}]')
ax3.axis('off')

ax4 = fig.add_subplot(gs[0, 3])
ax4.hist(img_normalized.flatten(), bins=50, color='black', alpha=0.7)
ax4.set_xlabel('Pixel Intensity')
ax4.set_ylabel('Count')
ax4.set_title('Input Intensity Distribution')
ax4.grid(True, alpha=0.3)

# Row 2: Temporal structure
ax5 = fig.add_subplot(gs[1, 0])
temporal_mean = np.mean(true_probs, axis=1)
ax5.plot(temporal_mean, 'k-', linewidth=1)
ax5.axvline(pre_delay, color='green', linestyle='--', label='Stim Start')
ax5.axvline(pre_delay + im_slice, color='red', linestyle='--', label='Stim End')
ax5.set_xlabel('Time (ms)')
ax5.set_ylabel('Mean Probability (all neurons)')
ax5.set_title('Temporal Profile')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 1])
# Heatmap: time x sample of neurons
sample_neurons = np.linspace(0, spike_probs.shape[1]-1, 100, dtype=int)
im6 = ax6.imshow(true_probs[:, sample_neurons].T, aspect='auto', cmap='hot',
                 interpolation='nearest')
ax6.axvline(pre_delay, color='cyan', linestyle='--', linewidth=1)
ax6.axvline(pre_delay + im_slice, color='cyan', linestyle='--', linewidth=1)
ax6.set_xlabel('Time (ms)')
ax6.set_ylabel('Neuron ID (sampled)')
ax6.set_title('Spike Probability Heatmap\\n(100 random neurons)')
plt.colorbar(im6, ax=ax6, label='Probability')

ax7 = fig.add_subplot(gs[1, 2])
# Distribution during stimulus
stim_probs = true_probs[pre_delay:pre_delay+im_slice, :]
ax7.hist(stim_probs.flatten(), bins=100, color='black', alpha=0.7, log=True)
ax7.set_xlabel('Spike Probability')
ax7.set_ylabel('Count (log scale)')
ax7.set_title('Probability Distribution\\n(Stimulus Period)')
ax7.axvline(np.mean(stim_probs), color='red', linestyle='--',
           label=f'Mean={np.mean(stim_probs):.6f}')
ax7.legend()
ax7.grid(True, alpha=0.3)

ax8 = fig.add_subplot(gs[1, 3])
# Neuron activity during stimulus
neuron_activity = np.mean(true_probs[pre_delay:pre_delay+im_slice, :], axis=0)
downsample = max(1, len(neuron_activity) // 1000)
ax8.plot(neuron_activity[::downsample], 'k-', linewidth=0.5, alpha=0.7)
ax8.set_xlabel(f'Neuron ID (/{downsample})')
ax8.set_ylabel('Mean Activity')
ax8.set_title('Per-Neuron Activity\\n(Stimulus Period)')
ax8.grid(True, alpha=0.3)

# Row 3: Attempted decoding
ax9 = fig.add_subplot(gs[2, 0])
# Reshape neurons to square grid
n_neurons = len(neuron_activity)
grid_size = int(np.sqrt(n_neurons))
grid_activity = np.zeros(grid_size * grid_size)
grid_activity[:n_neurons] = neuron_activity
grid_activity = grid_activity.reshape(grid_size, grid_size)
im9 = ax9.imshow(grid_activity, cmap='hot', aspect='auto')
ax9.set_title(f'Grid Reshape ({grid_size}x{grid_size})\\n(No smoothing)')
plt.colorbar(im9, ax=ax9)

ax10 = fig.add_subplot(gs[2, 1])
# With gaussian smoothing
from scipy.ndimage import gaussian_filter
smoothed = gaussian_filter(grid_activity, sigma=2.0)
im10 = ax10.imshow(smoothed, cmap='gray', aspect='auto')
ax10.set_title('Smoothed (sigma=2)')
plt.colorbar(im10, ax=ax10)

ax11 = fig.add_subplot(gs[2, 2])
# Most active neurons
top_k = 500
top_indices = np.argsort(neuron_activity)[-top_k:]
top_activity = np.zeros_like(neuron_activity)
top_activity[top_indices] = neuron_activity[top_indices]
grid_top = np.zeros(grid_size * grid_size)
grid_top[:n_neurons] = top_activity
grid_top = grid_top.reshape(grid_size, grid_size)
smoothed_top = gaussian_filter(grid_top, sigma=2.0)
im11 = ax11.imshow(smoothed_top, cmap='gray', aspect='auto')
ax11.set_title(f'Top {top_k} Neurons Only\\n(Smoothed)')
plt.colorbar(im11, ax=ax11)

ax12 = fig.add_subplot(gs[2, 3])
# Distribution of top neurons
ax12.hist(neuron_activity[top_indices], bins=50, color='red', alpha=0.7)
ax12.set_xlabel('Activity')
ax12.set_ylabel('Count')
ax12.set_title(f'Top {top_k} Neuron\\nActivity Distribution')
ax12.grid(True, alpha=0.3)

plt.suptitle(f'LGN Encoding Diagnosis - Sample {SAMPLE_IDX} (Label {label})',
             fontsize=16, fontweight='bold')

output_file = f'lgn_diagnosis_sample_{SAMPLE_IDX}.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n{'='*80}")
print(f"Saved diagnosis to: {output_file}")
print(f"{'='*80}")

print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")
print("""
Key things to check:

1. **Temporal Profile**: Does activity increase during stimulus period?
   - If flat: Encoding might not be working
   - If peaked: LGN is responding to stimulus

2. **Activity Distribution**: What's the range of spike probabilities?
   - If all near zero: Weak/broken encoding
   - If diverse: Encoding is working

3. **Spatial Structure**: Do different neurons have different activities?
   - If all similar: No spatial information
   - If varied: Spatial structure present

4. **Decoded Image**: Does it resemble the original digit?
   - If random noise: Spatial organization is lost
   - If blurry digit: Encoding preserves some structure
   - If nothing: Encoding is completely broken

The problem might be:
- LGN neurons aren't spatially organized in the array
- Need to load receptive field positions to decode properly
- Temporal integration is required (not just averaging)
- ON/OFF cells need to be separated
""")
