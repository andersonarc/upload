#!/usr/bin/env python3
"""Check what variables exist in the TensorFlow checkpoint"""

import tensorflow as tf
import sys

CHECKPOINT = 'training_code/ckpt_51978-153'

print("=" * 80)
print("Checking TensorFlow Checkpoint Variables")
print("=" * 80)

try:
    reader = tf.train.load_checkpoint(CHECKPOINT)
    var_map = reader.get_variable_to_shape_map()

    print(f"\nCheckpoint: {CHECKPOINT}")
    print(f"Total variables: {len(var_map)}\n")

    # Look for sign-related variables
    sign_vars = []
    weight_vars = []

    for var_name in sorted(var_map.keys()):
        if '.ATTRIBUTES' not in var_name:
            if 'sign' in var_name.lower():
                sign_vars.append((var_name, var_map[var_name]))
            if 'weight' in var_name.lower():
                weight_vars.append((var_name, var_map[var_name]))

    print("=" * 80)
    print("WEIGHT VARIABLES:")
    print("=" * 80)
    for name, shape in weight_vars:
        print(f"  {name}: {shape}")

    print("\n" + "=" * 80)
    print("SIGN MASK VARIABLES:")
    print("=" * 80)
    if sign_vars:
        for name, shape in sign_vars:
            print(f"  ✓ {name}: {shape}")
            # Try to load and check
            try:
                sign_mask = reader.get_tensor(name)
                print(f"    Positive: {sign_mask.sum()}")
                print(f"    Negative: {(~sign_mask).sum()}")
            except:
                pass
        print("\n✅ Sign masks EXIST in checkpoint!")
        print("   c2_fixed.py should work correctly")
    else:
        print("  ❌ NO sign mask variables found!")
        print("\n⚠️  PROBLEM: Checkpoint doesn't contain sign masks")
        print("   This means either:")
        print("     1. Signs are already baked into the weight values")
        print("     2. Checkpoint was saved incorrectly")
        print("     3. Dale's law wasn't actually used during training")

    # Check if weights already have signs
    print("\n" + "=" * 80)
    print("CHECKING IF WEIGHTS HAVE SIGNS BAKED IN:")
    print("=" * 80)

    for name, shape in weight_vars:
        if 'sparse_recurrent_weights' in name or 'sparse_input_weights' in name:
            weights = reader.get_tensor(name)
            pos = (weights > 0).sum()
            neg = (weights < 0).sum()
            zero = (weights == 0).sum()

            print(f"\n{name}:")
            print(f"  Shape: {weights.shape}")
            print(f"  Positive: {pos}")
            print(f"  Negative: {neg}")
            print(f"  Zero: {zero}")

            if neg > 0:
                print(f"  ✅ Weights have MIXED signs - signs are baked in!")
            else:
                print(f"  ⚠️  Weights are ALL non-negative - need sign masks!")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)

    if sign_vars:
        print("✅ Sign masks exist - use c2_fixed.py")
    else:
        # Check if weights have signs
        rec_weights = None
        for name, shape in weight_vars:
            if 'sparse_recurrent_weights' in name:
                rec_weights = reader.get_tensor(name)
                break

        if rec_weights is not None and (rec_weights < 0).sum() > 0:
            print("✅ Weights have signs baked in - original c2.py should work!")
            print("   But check why your H5 has all positive weights...")
        else:
            print("❌ NO signs found anywhere - checkpoint may be broken!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
