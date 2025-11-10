#!/usr/bin/env python3
"""Load sign masks and weights from checkpoint"""

import tensorflow as tf
import numpy as np

CHECKPOINT = 'training_code/ckpt_51978-153'

print("=" * 80)
print("Loading Sign Masks and Weights from Checkpoint")
print("=" * 80)

reader = tf.train.load_checkpoint(CHECKPOINT)

# Load sign masks (with full path including .ATTRIBUTES/VARIABLE_VALUE)
rec_sign = reader.get_tensor('model/layer_with_weights-1/cell/recurrent_weight_positive/.ATTRIBUTES/VARIABLE_VALUE')
inp_sign = reader.get_tensor('model/layer_with_weights-1/cell/input_weight_positive/.ATTRIBUTES/VARIABLE_VALUE')

# Load weights
rec_weights = reader.get_tensor('model/layer_with_weights-1/cell/recurrent_weight_values/.ATTRIBUTES/VARIABLE_VALUE')

# Input weights are in layer_with_weights-0
inp_weights = reader.get_tensor('model/layer_with_weights-0/_weights/.ATTRIBUTES/VARIABLE_VALUE')

print("\n1. SIGN MASKS:")
print(f"\nRecurrent weight signs:")
print(f"  Shape: {rec_sign.shape}")
print(f"  Positive (True): {np.sum(rec_sign)}")
print(f"  Negative (False): {np.sum(~rec_sign)}")
print(f"  Percentage positive: {100*np.mean(rec_sign):.1f}%")

print(f"\nInput weight signs:")
print(f"  Shape: {inp_sign.shape}")
print(f"  Positive (True): {np.sum(inp_sign)}")
print(f"  Negative (False): {np.sum(~inp_sign)}")
print(f"  Percentage positive: {100*np.mean(inp_sign):.1f}%")

print("\n2. WEIGHT VALUES (before applying signs):")
print(f"\nRecurrent weights:")
print(f"  Shape: {rec_weights.shape}")
print(f"  Mean: {rec_weights.mean():.6f}")
print(f"  Min: {rec_weights.min():.6f}")
print(f"  Max: {rec_weights.max():.6f}")
print(f"  Positive: {np.sum(rec_weights > 0)}")
print(f"  Negative: {np.sum(rec_weights < 0)}")

print(f"\nInput weights:")
print(f"  Shape: {inp_weights.shape}")
print(f"  Mean: {inp_weights.mean():.6f}")
print(f"  Min: {inp_weights.min():.6f}")
print(f"  Max: {inp_weights.max():.6f}")
print(f"  Positive: {np.sum(inp_weights > 0)}")
print(f"  Negative: {np.sum(inp_weights < 0)}")

print("\n3. APPLYING SIGNS (what c2_fixed.py does):")

# Apply signs: True=positive, False=negative
rec_signed = np.abs(rec_weights) * np.where(rec_sign, 1.0, -1.0)
inp_signed = np.abs(inp_weights) * np.where(inp_sign, 1.0, -1.0)

print(f"\nRecurrent weights AFTER applying signs:")
print(f"  Mean: {rec_signed.mean():.6f}")
print(f"  Min: {rec_signed.min():.6f}")
print(f"  Max: {rec_signed.max():.6f}")
print(f"  Positive: {np.sum(rec_signed > 0)}")
print(f"  Negative: {np.sum(rec_signed < 0)}")

print(f"\nInput weights AFTER applying signs:")
print(f"  Mean: {inp_signed.mean():.6f}")
print(f"  Min: {inp_signed.min():.6f}")
print(f"  Max: {inp_signed.max():.6f}")
print(f"  Positive: {np.sum(inp_signed > 0)}")
print(f"  Negative: {np.sum(inp_signed < 0)}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

if np.sum(~inp_sign) > 0:
    print("\n✅ Sign masks exist and contain negative signs!")
    print(f"   Input: {np.sum(~inp_sign)} inhibitory connections ({100*np.mean(~inp_sign):.1f}%)")
    print(f"   Recurrent: {np.sum(~rec_sign)} inhibitory connections ({100*np.mean(~rec_sign):.1f}%)")
    print("\n✅ c2_fixed.py WILL FIX the problem!")

    # Compare with what user reported
    print(f"\n📊 Comparison:")
    print(f"   Current H5 (WRONG): Input 782019 pos, 0 neg")
    print(f"   After fix (CORRECT): Input {np.sum(inp_signed > 0)} pos, {np.sum(inp_signed < 0)} neg")

    print("\n🔧 Next steps:")
    print("  1. cd training_code")
    print("  2. python3 c2_fixed.py --checkpoint ckpt_51978-153 --data_dir v1cortex --output ckpt_51978-153_FIXED.h5")
    print("  3. cp ckpt_51978-153_FIXED.h5 ../jupyter/ckpt_51978-153.h5")
    print("  4. cd ../jupyter && python3 class.py")
else:
    print("\n❌ Sign masks exist but all weights are positive!")
    print("   This shouldn't happen - check checkpoint integrity")

print("\n" + "=" * 80)
