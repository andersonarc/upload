#!/usr/bin/env python3
"""Check if H5 contains trained or untrained weights"""

import h5py
import numpy as np
import sys

H5_FILE = 'jupyter/ckpt_51978-153.h5'

print("=" * 80)
print("Checking if H5 weights are TRAINED vs UNTRAINED")
print("=" * 80)

try:
    with h5py.File(H5_FILE, 'r') as f:
        rec_w = np.array(f['recurrent/weights'])
        inp_w = np.array(f['input/weights'])

        print("\nWeight Statistics:")
        print(f"\nRecurrent weights:")
        print(f"  Count: {len(rec_w)}")
        print(f"  Mean: {rec_w.mean():.6f}")
        print(f"  Std:  {rec_w.std():.6f}")
        print(f"  Min:  {rec_w.min():.6f}")
        print(f"  Max:  {rec_w.max():.6f}")
        print(f"  Median: {np.median(rec_w):.6f}")

        print(f"\nInput weights:")
        print(f"  Count: {len(inp_w)}")
        print(f"  Mean: {inp_w.mean():.6f}")
        print(f"  Std:  {inp_w.std():.6f}")
        print(f"  Min:  {inp_w.min():.6f}")
        print(f"  Max:  {inp_w.max():.6f}")
        print(f"  Median: {np.median(inp_w):.6f}")

        # Check weight distribution
        print("\n" + "=" * 80)
        print("Weight Distribution Analysis:")
        print("=" * 80)

        # Trained networks typically have:
        # - Diverse weight magnitudes
        # - Many small weights (pruned)
        # - Some large weights (important connections)

        # Untrained networks have:
        # - Uniform/similar magnitudes
        # - Gaussian-like distribution
        # - No pruning pattern

        print("\nRecurrent weight magnitude distribution:")
        rec_abs = np.abs(rec_w)
        for threshold in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
            pct = 100 * np.mean(rec_abs > threshold)
            print(f"  > {threshold:4.2f}: {pct:5.1f}%")

        print("\nInput weight distribution:")
        for threshold in [0.01, 0.1, 0.5, 1.0, 5.0]:
            pct = 100 * np.mean(inp_w > threshold)
            print(f"  > {threshold:4.2f}: {pct:5.1f}%")

        # Check if weights look structured vs random
        print("\n" + "=" * 80)
        print("DIAGNOSIS:")
        print("=" * 80)

        # Heuristics:
        # - If most weights are very similar → likely UNTRAINED
        # - If wide distribution with many zeros/small values → likely TRAINED
        # - If std/mean ratio is low → likely UNTRAINED

        rec_cv = rec_w.std() / np.abs(rec_w.mean()) if rec_w.mean() != 0 else float('inf')
        inp_cv = inp_w.std() / inp_w.mean() if inp_w.mean() != 0 else float('inf')

        print(f"\nCoefficient of variation (std/mean):")
        print(f"  Recurrent: {rec_cv:.2f}")
        print(f"  Input: {inp_cv:.2f}")
        print(f"  (Higher = more diverse = likely trained)")

        # Check for very large outliers (29.69 reported earlier)
        rec_outliers = np.sum(np.abs(rec_w) > 10)
        inp_outliers = np.sum(inp_w > 5)

        print(f"\nExtreme values:")
        print(f"  Recurrent |w| > 10: {rec_outliers} ({100*rec_outliers/len(rec_w):.3f}%)")
        print(f"  Input w > 5: {inp_outliers} ({100*inp_outliers/len(inp_w):.3f}%)")

        if rec_outliers > 0 or inp_outliers > 0:
            print(f"\n  ⚠️  Extreme weight values detected!")
            print(f"  These could cause instability even with trained weights")
            print(f"  Max recurrent: {rec_w.max():.4f} (after scaling: ~{rec_w.max() * 24 / 1000:.4f} nA)")
            print(f"  Max input: {inp_w.max():.4f} (after scaling: ~{inp_w.max() * 24 / 1000:.4f} nA)")

        # Final assessment
        print("\n" + "=" * 80)
        print("CONCLUSION:")
        print("=" * 80)

        if rec_cv < 5 and inp_cv < 5:
            print("\n❌ Weights appear UNTRAINED")
            print("   - Low coefficient of variation")
            print("   - Similar magnitudes across connections")
            print("\n🔧 Solution: Verify c2.py loaded checkpoint correctly")
        elif rec_outliers > len(rec_w) * 0.001:  # > 0.1% are extreme
            print("\n⚠️  Weights may be trained but have EXTREME values")
            print("   - Very large weight outliers detected")
            print("   - Could cause network instability")
            print("\n🔧 Solution: Clip extreme weights or check if scaling is correct")
        else:
            print("\n✅ Weights appear TRAINED")
            print("   - Diverse weight distribution")
            print("   - Reasonable value ranges")
            print("\n⚠️  But network still produces garbage output...")
            print("   Problem must be elsewhere:")
            print("   - Neuron parameters wrong?")
            print("   - Connectivity topology wrong?")
            print("   - Delays wrong?")
            print("   - Input/output encoding wrong?")

except FileNotFoundError:
    print(f"ERROR: {H5_FILE} not found")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
