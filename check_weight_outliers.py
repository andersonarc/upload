#!/usr/bin/env python3
"""Check for extreme weight outliers in checkpoint and H5"""

import tensorflow as tf
import h5py
import numpy as np

print("=" * 80)
print("Checking for Weight Outliers")
print("=" * 80)

# Load from checkpoint
print("\n1. CHECKPOINT WEIGHTS:")
reader = tf.train.load_checkpoint('training_code/ckpt_51978-153')

rec_weights_ckpt = reader.get_tensor('model/layer_with_weights-1/cell/recurrent_weight_values/.ATTRIBUTES/VARIABLE_VALUE')
inp_weights_ckpt = reader.get_tensor('model/layer_with_weights-0/_weights/.ATTRIBUTES/VARIABLE_VALUE')

print(f"\nRecurrent weights (checkpoint):")
print(f"  Mean: {rec_weights_ckpt.mean():.6f}")
print(f"  Std: {rec_weights_ckpt.std():.6f}")
print(f"  Min: {rec_weights_ckpt.min():.6f}")
print(f"  Max: {rec_weights_ckpt.max():.6f}")
print(f"  95th percentile: {np.percentile(np.abs(rec_weights_ckpt), 95):.6f}")
print(f"  99th percentile: {np.percentile(np.abs(rec_weights_ckpt), 99):.6f}")
print(f"  99.9th percentile: {np.percentile(np.abs(rec_weights_ckpt), 99.9):.6f}")

# Count extreme outliers
for threshold in [1.0, 2.0, 5.0, 10.0, 20.0]:
    count = np.sum(np.abs(rec_weights_ckpt) > threshold)
    pct = 100 * count / len(rec_weights_ckpt)
    print(f"  |w| > {threshold:4.1f}: {count:7d} ({pct:.3f}%)")

print(f"\nInput weights (checkpoint):")
print(f"  Mean: {inp_weights_ckpt.mean():.6f}")
print(f"  Std: {inp_weights_ckpt.std():.6f}")
print(f"  Min: {inp_weights_ckpt.min():.6f}")
print(f"  Max: {inp_weights_ckpt.max():.6f}")
print(f"  95th percentile: {np.percentile(inp_weights_ckpt, 95):.6f}")
print(f"  99th percentile: {np.percentile(inp_weights_ckpt, 99):.6f}")
print(f"  99.9th percentile: {np.percentile(inp_weights_ckpt, 99.9):.6f}")

for threshold in [1.0, 2.0, 5.0, 10.0, 20.0]:
    count = np.sum(inp_weights_ckpt > threshold)
    pct = 100 * count / len(inp_weights_ckpt)
    print(f"  w > {threshold:4.1f}: {count:7d} ({pct:.3f}%)")

# Load from H5
print("\n" + "=" * 80)
print("2. H5 WEIGHTS:")
print("=" * 80)

with h5py.File('jupyter/ckpt_51978-153.h5', 'r') as f:
    rec_weights_h5 = np.array(f['recurrent/weights'])
    inp_weights_h5 = np.array(f['input/weights'])

print(f"\nRecurrent weights (H5):")
print(f"  Mean: {rec_weights_h5.mean():.6f}")
print(f"  Std: {rec_weights_h5.std():.6f}")
print(f"  Min: {rec_weights_h5.min():.6f}")
print(f"  Max: {rec_weights_h5.max():.6f}")

print(f"\nInput weights (H5):")
print(f"  Mean: {inp_weights_h5.mean():.6f}")
print(f"  Std: {inp_weights_h5.std():.6f}")
print(f"  Min: {inp_weights_h5.min():.6f}")
print(f"  Max: {inp_weights_h5.max():.6f}")

# Compare
print("\n" + "=" * 80)
print("3. COMPARISON:")
print("=" * 80)

print(f"\nAre weights identical?")
# They might differ slightly due to sign application
rec_mean_diff = abs(rec_weights_h5.mean() - rec_weights_ckpt.mean())
inp_mean_diff = abs(inp_weights_h5.mean() - inp_weights_ckpt.mean())

print(f"  Recurrent mean difference: {rec_mean_diff:.6f}")
print(f"  Input mean difference: {inp_mean_diff:.6f}")

if rec_mean_diff < 0.001 and inp_mean_diff < 0.001:
    print(f"  ✅ Weights match - c2.py copied correctly")
else:
    print(f"  ⚠️  Weights differ - check c2.py conversion")

# Check if the max outlier (29.69) exists
print(f"\n" + "=" * 80)
print("4. OUTLIER ANALYSIS:")
print("=" * 80)

rec_max_idx = np.argmax(np.abs(rec_weights_h5))
inp_max_idx = np.argmax(inp_weights_h5)

print(f"\nMax recurrent weight:")
print(f"  Value: {rec_weights_h5[rec_max_idx]:.6f}")
print(f"  Index: {rec_max_idx}")
print(f"  After scaling (*vsc/1000): {rec_weights_h5[rec_max_idx] * 27.7 / 1000:.6f} nA")

print(f"\nMax input weight:")
print(f"  Value: {inp_weights_h5[inp_max_idx]:.6f}")
print(f"  Index: {inp_max_idx}")
print(f"  After scaling (*vsc/1000): {inp_weights_h5[inp_max_idx] * 27.7 / 1000:.6f} nA")

# Check if this is problematic
if rec_weights_h5[rec_max_idx] * 27.7 / 1000 > 1.0:
    print(f"\n  ⚠️  WARNING: Max recurrent weight is very large!")
    print(f"  This could cause instability")

if inp_weights_h5[inp_max_idx] * 27.7 / 1000 > 1.0:
    print(f"\n  ⚠️  WARNING: Max input weight is very large!")
    print(f"  This could cause instability")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

# Check coefficient of variation
rec_cv = rec_weights_h5.std() / abs(rec_weights_h5.mean())
inp_cv = inp_weights_h5.std() / inp_weights_h5.mean()

print(f"\nCoefficient of variation (std/mean):")
print(f"  Recurrent: {rec_cv:.2f}")
print(f"  Input: {inp_cv:.2f}")

if rec_cv > 10 and inp_cv > 5:
    print(f"\n✅ High variation suggests weights are TRAINED")
    print(f"   - Diverse weight magnitudes")
    print(f"   - Some connections strong, some weak")
else:
    print(f"\n⚠️  Low variation - weights might be UNTRAINED")

# Final verdict
extreme_rec = np.sum(np.abs(rec_weights_h5) > 10)
extreme_inp = np.sum(inp_weights_h5 > 5)

if extreme_rec > 0 or extreme_inp > 0:
    print(f"\n⚠️  EXTREME OUTLIERS DETECTED:")
    print(f"  Recurrent |w| > 10: {extreme_rec}")
    print(f"  Input w > 5: {extreme_inp}")
    print(f"\n  These outliers after scaling:")
    print(f"  Recurrent: up to {rec_weights_h5[rec_max_idx] * 27.7 / 1000:.3f} nA")
    print(f"  Input: up to {inp_weights_h5[inp_max_idx] * 27.7 / 1000:.3f} nA")
    print(f"\n  Typical GLIF3 weights: 0.001-0.1 nA")
    print(f"  Values > 0.5 nA are very strong")
    print(f"  Values > 1.0 nA could cause problems")
else:
    print(f"\n✅ No extreme outliers")

print("\n" + "=" * 80)
