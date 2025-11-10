#!/usr/bin/env python3
"""
Check Weight Signs in Checkpoint and H5

Purpose: Verify if weight signs are preserved correctly
"""

import tensorflow as tf
import h5py
import numpy as np
import sys

print("=" * 80)
print("Weight Sign Diagnostic")
print("=" * 80)

# Check checkpoint
CHECKPOINT = 'ckpt_51978-153'
H5_FILE = 'ckpt_51978-153.h5'

print("\n1. Checking TensorFlow checkpoint...")
try:
    reader = tf.train.load_checkpoint(CHECKPOINT)
    var_map = reader.get_variable_to_shape_map()

    print("\nVariables in checkpoint:")
    for var_name in sorted(var_map.keys()):
        if '.ATTRIBUTES' not in var_name:
            print(f"  {var_name}: {var_map[var_name]}")

    # Check for sign variables
    print("\n2. Looking for sign masks...")
    rec_sign_found = False
    inp_sign_found = False

    for var_name in var_map.keys():
        if 'recurrent_weights_sign' in var_name:
            rec_sign = reader.get_tensor(var_name)
            print(f"\n✓ Found: {var_name}")
            print(f"  Shape: {rec_sign.shape}")
            print(f"  Positive: {np.sum(rec_sign)}")
            print(f"  Negative: {np.sum(~rec_sign)}")
            rec_sign_found = True

        if 'input_weights_sign' in var_name:
            inp_sign = reader.get_tensor(var_name)
            print(f"\n✓ Found: {var_name}")
            print(f"  Shape: {inp_sign.shape}")
            print(f"  Positive: {np.sum(inp_sign)}")
            print(f"  Negative: {np.sum(~inp_sign)}")
            inp_sign_found = True

    if not rec_sign_found:
        print("\n⚠️  WARNING: No recurrent weight sign mask found!")
    if not inp_sign_found:
        print("\n⚠️  WARNING: No input weight sign mask found!")

    # Load actual weights
    print("\n3. Checking weight values in checkpoint...")

    for var_name in var_map.keys():
        if 'sparse_recurrent_weights' in var_name and 'sign' not in var_name:
            rec_weights = reader.get_tensor(var_name)
            print(f"\nRecurrent weights: {var_name}")
            print(f"  Shape: {rec_weights.shape}")
            print(f"  Mean: {rec_weights.mean():.6f}")
            print(f"  Min: {rec_weights.min():.6f}")
            print(f"  Max: {rec_weights.max():.6f}")
            print(f"  Positive values: {np.sum(rec_weights > 0)}")
            print(f"  Negative values: {np.sum(rec_weights < 0)}")
            print(f"  Zero values: {np.sum(rec_weights == 0)}")

            if np.all(rec_weights >= 0):
                print(f"  ⚠️  ALL WEIGHTS ARE NON-NEGATIVE!")
                print(f"  This means signs need to be applied separately")

        if 'sparse_input_weights' in var_name and 'sign' not in var_name:
            inp_weights = reader.get_tensor(var_name)
            print(f"\nInput weights: {var_name}")
            print(f"  Shape: {inp_weights.shape}")
            print(f"  Mean: {inp_weights.mean():.6f}")
            print(f"  Min: {inp_weights.min():.6f}")
            print(f"  Max: {inp_weights.max():.6f}")
            print(f"  Positive values: {np.sum(inp_weights > 0)}")
            print(f"  Negative values: {np.sum(inp_weights < 0)}")
            print(f"  Zero values: {np.sum(inp_weights == 0)}")

            if np.all(inp_weights >= 0):
                print(f"  ⚠️  ALL WEIGHTS ARE NON-NEGATIVE!")
                print(f"  This means signs need to be applied separately")

except Exception as e:
    print(f"ERROR loading checkpoint: {e}")
    import traceback
    traceback.print_exc()

# Check H5 file
print("\n" + "=" * 80)
print("4. Checking H5 file...")
print("=" * 80)

try:
    with h5py.File(H5_FILE, 'r') as f:
        rec_weights = np.array(f['recurrent/weights'])
        inp_weights = np.array(f['input/weights'])

        print("\nRecurrent weights in H5:")
        print(f"  Mean: {rec_weights.mean():.6f}")
        print(f"  Min: {rec_weights.min():.6f}")
        print(f"  Max: {rec_weights.max():.6f}")
        print(f"  Positive: {np.sum(rec_weights > 0)}")
        print(f"  Negative: {np.sum(rec_weights < 0)}")

        print("\nInput weights in H5:")
        print(f"  Mean: {inp_weights.mean():.6f}")
        print(f"  Min: {inp_weights.min():.6f}")
        print(f"  Max: {inp_weights.max():.6f}")
        print(f"  Positive: {np.sum(inp_weights > 0)}")
        print(f"  Negative: {np.sum(inp_weights < 0)}")

        # Check receptor types
        rec_receptors = np.array(f['recurrent/receptor_types'])
        inp_receptors = np.array(f['input/receptor_types'])

        print("\n5. Receptor type analysis:")
        print("\nRecurrent synapses by receptor:")
        for r in range(4):
            mask = rec_receptors == r
            w = rec_weights[mask]
            print(f"  Receptor {r}: {np.sum(mask)} synapses")
            print(f"    Weights: mean={w.mean():.6f}, min={w.min():.6f}, max={w.max():.6f}")
            print(f"    Positive: {np.sum(w > 0)}, Negative: {np.sum(w < 0)}")

        print("\nInput synapses by receptor:")
        for r in range(4):
            mask = inp_receptors == r
            w = inp_weights[mask]
            print(f"  Receptor {r}: {np.sum(mask)} synapses")
            print(f"    Weights: mean={w.mean():.6f}, min={w.min():.6f}, max={w.max():.6f}")
            print(f"    Positive: {np.sum(w > 0)}, Negative: {np.sum(w < 0)}")

except Exception as e:
    print(f"ERROR loading H5: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)

print("\nDale's Law with SignedConstraint:")
print("  - Excitatory neurons (E) can only have positive weights")
print("  - Inhibitory neurons (I) can only have negative weights")
print("  - Signs are enforced by source neuron type, not receptor type")

print("\nReceptor types (target side):")
print("  0: AMPA (excitatory receptor)")
print("  1: GABA_A (inhibitory receptor)")
print("  2: NMDA (excitatory receptor)")
print("  3: GABA_B (inhibitory receptor)")

print("\nExpected behavior:")
print("  - E → E (via AMPA/NMDA): positive weights ✓")
print("  - I → E (via GABA): negative weights ✓")
print("  - E → I (via AMPA/NMDA): positive weights ✓")
print("  - I → I (via GABA): negative weights ✓")

print("\nIf ALL input weights are positive:")
print("  Problem: No inhibitory LGN neurons OR signs not applied")
print("  Solution: Check if sign masks need to be applied to weights")

print("\n" + "=" * 80)
