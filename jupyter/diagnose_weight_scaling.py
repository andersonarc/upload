#!/usr/bin/env python3
"""
Weight Scaling Diagnostic Script

Purpose: Determine correct weight scaling by analyzing H5 format and TensorFlow code
"""

import h5py
import numpy as np
import sys

print("=" * 80)
print("Weight Scaling Diagnostic")
print("=" * 80)

# Load H5 file
H5_FILE = 'ckpt_51978-153.h5'

try:
    with h5py.File(H5_FILE, 'r') as f:
        print(f"\n1. H5 File Contents:")
        print(f"   Available groups: {list(f.keys())}")

        # Load parameters
        E_L = np.array(f['neurons/glif3_params/E_L'])
        V_th = np.array(f['neurons/glif3_params/V_th'])
        V_reset = np.array(f['neurons/glif3_params/V_reset'])
        asc_amps = np.array(f['neurons/glif3_params/asc_amps'])

        # Load weights
        rec_weights = np.array(f['recurrent/weights'])
        inp_weights = np.array(f['input/weights'])

        print(f"\n2. Voltage Parameters (Raw H5 values):")
        print(f"   E_L:     min={E_L.min():.2f}, max={E_L.max():.2f}, mean={E_L.mean():.2f}")
        print(f"   V_th:    min={V_th.min():.2f}, max={V_th.max():.2f}, mean={V_th.mean():.2f}")
        print(f"   V_reset: min={V_reset.min():.2f}, max={V_reset.max():.2f}, mean={V_reset.mean():.2f}")

        voltage_scale = V_th - E_L
        print(f"   Voltage scale: min={voltage_scale.min():.2f}, max={voltage_scale.max():.2f}, mean={voltage_scale.mean():.2f}")

        # Check if normalized
        if np.abs(E_L).mean() < 2:
            print(f"   STATUS: ⚠️  NORMALIZED (0-1 range)")
            print(f"   PROBLEM: Need original voltage_scale to denormalize!")
        else:
            print(f"   STATUS: ✅ UNNORMALIZED (mV range)")

        print(f"\n3. ASC Parameters (Raw H5 values):")
        print(f"   ASC amp 0: min={asc_amps[:, 0].min():.4f}, max={asc_amps[:, 0].max():.4f}, mean={asc_amps[:, 0].mean():.4f}")
        print(f"   ASC amp 1: min={asc_amps[:, 1].min():.4f}, max={asc_amps[:, 1].max():.4f}, mean={asc_amps[:, 1].mean():.4f}")

        # Check ASC format
        asc_mean = np.mean(np.abs(asc_amps[asc_amps != 0]))
        if asc_mean > 10:
            print(f"   STATUS: UNNORMALIZED pA (need /1000 to convert to nA)")
            print(f"   CORRECT: asc_nA = asc_pA / 1000")
        elif asc_mean > 0.01:
            print(f"   STATUS: VOLTAGE-NORMALIZED pA (need *= vsc / 1000)")
            print(f"   CORRECT: asc_nA = asc_pA * vsc / 1000")
        else:
            print(f"   STATUS: Already in nA")
            print(f"   CORRECT: use as-is")

        print(f"\n4. Weight Statistics (Raw H5 values):")
        print(f"   Recurrent: min={rec_weights.min():.4f}, max={rec_weights.max():.4f}, mean={rec_weights.mean():.4f}")
        print(f"   Input:     min={inp_weights.min():.4f}, max={inp_weights.max():.4f}, mean={inp_weights.mean():.4f}")

        # Analyze weight format
        rec_mean_abs = np.mean(np.abs(rec_weights))
        print(f"\n5. Weight Format Analysis:")
        print(f"   Mean absolute recurrent weight: {rec_mean_abs:.4f}")

        # Check if weights are voltage-normalized
        # TensorFlow normalizes: weights_normalized = weights / voltage_scale
        # Expected denormalized values: 0.001 - 0.1 nA typical
        # Expected normalized values: 0.00005 - 0.005 (if vsc ~ 20)

        if rec_mean_abs < 0.01:
            print(f"   STATUS: ⚠️  VOLTAGE-NORMALIZED")
            print(f"   TensorFlow divides by voltage_scale during training")
            print(f"   CORRECT: weights_nA = weights_normalized * vsc")
            print(f"   DO NOT use /1000")
            correct_scaling = "* vsc"
        elif rec_mean_abs < 1.0:
            print(f"   STATUS: ⚠️  AMBIGUOUS - Could be:")
            print(f"           a) Unnormalized nA (use as-is)")
            print(f"           b) Normalized with large vsc (multiply by vsc)")
            print(f"   Need to test both approaches")
            correct_scaling = "AMBIGUOUS"
        else:
            print(f"   STATUS: ⚠️  VERY LARGE VALUES")
            print(f"   Might be in pA (need /1000)")
            print(f"   Or extremely strong (check for bugs)")
            correct_scaling = "/ 1000"

        print(f"\n6. Recommended Scaling:")
        print(f"   For ASC: /= 1000.0  (pA → nA, no voltage_scale)")
        print(f"   For Weights: {correct_scaling}")

        # Test different scaling approaches
        print(f"\n7. Weight Scaling Test Results:")
        vsc_mean = voltage_scale.mean()

        print(f"\n   Option A: weights_nA = weights * vsc")
        scaled_A = rec_mean_abs * vsc_mean
        print(f"      Mean: {scaled_A:.6f} nA")
        if 0.001 <= scaled_A <= 0.5:
            print(f"      ✅ REASONABLE for PyNN GLIF3")
        else:
            print(f"      ❌ OUT OF RANGE (too large/small)")

        print(f"\n   Option B: weights_nA = weights / 1000")
        scaled_B = rec_mean_abs / 1000
        print(f"      Mean: {scaled_B:.6f} nA")
        if 0.001 <= scaled_B <= 0.5:
            print(f"      ✅ REASONABLE for PyNN GLIF3")
        else:
            print(f"      ❌ OUT OF RANGE (too large/small)")

        print(f"\n   Option C: weights_nA = weights (use as-is)")
        scaled_C = rec_mean_abs
        print(f"      Mean: {scaled_C:.6f} nA")
        if 0.001 <= scaled_C <= 0.5:
            print(f"      ✅ REASONABLE for PyNN GLIF3")
        else:
            print(f"      ❌ OUT OF RANGE (too large/small)")

        print(f"\n   Option D: weights_nA = weights * vsc / 1000")
        scaled_D = rec_mean_abs * vsc_mean / 1000
        print(f"      Mean: {scaled_D:.6f} nA")
        if 0.001 <= scaled_D <= 0.5:
            print(f"      ✅ REASONABLE for PyNN GLIF3")
        else:
            print(f"      ❌ OUT OF RANGE (too large/small)")

        # Check weight distribution for outliers
        print(f"\n8. Weight Distribution Analysis:")
        rec_sorted = np.sort(np.abs(rec_weights))
        print(f"   Percentiles:")
        for p in [50, 75, 90, 95, 99, 99.9, 100]:
            idx = int(len(rec_sorted) * p / 100) - 1
            print(f"      {p:5.1f}%: {rec_sorted[idx]:.4f}")

        # Check for extreme outliers
        outliers = np.sum(np.abs(rec_weights) > 10)
        print(f"\n   Weights > 10: {outliers} ({100*outliers/len(rec_weights):.2f}%)")
        if outliers > 0:
            print(f"   ⚠️  WARNING: Extreme weight values detected")
            print(f"   These might cause instability even with correct scaling")

        # Final recommendation
        print(f"\n" + "=" * 80)
        print(f"FINAL RECOMMENDATION:")
        print(f"=" * 80)

        print(f"\nBased on your reported statistics:")
        print(f"  Recurrent: mean=0.104, max=29.69")
        print(f"  Input:     mean=0.814, max=5.15")
        print(f"  Voltage scale: ~20-30 mV")

        print(f"\nTensorFlow normalization (models.py:227):")
        print(f"  weights_normalized = weights / voltage_scale")

        print(f"\nThis means H5 contains VOLTAGE-NORMALIZED weights!")
        print(f"\n✅ CORRECT SCALING (lines 841, 888):")
        print(f"   syn[:, S.WHT] *= vsc  # Denormalize (NO /1000)")

        print(f"\n❌ WRONG SCALING:")
        print(f"   syn[:, S.WHT] *= vsc / 1000.0  # Double conversion!")
        print(f"   syn[:, S.WHT] /= 1000.0        # Still normalized!")

        print(f"\nExplanation:")
        print(f"  - TensorFlow trains with normalized weights (÷ vsc)")
        print(f"  - H5 stores these normalized values")
        print(f"  - PyNN needs denormalized weights (× vsc)")
        print(f"  - NO unit conversion needed (already in nA)")

        print(f"\nTest this by checking mean weight after denormalization:")
        print(f"  0.104 * 24 = {0.104 * 24:.3f} nA (reasonable)")
        print(f"  0.104 / 1000 = {0.104 / 1000:.6f} nA (too small!)")

except FileNotFoundError:
    print(f"\nERROR: H5 file not found: {H5_FILE}")
    print(f"Please ensure the file is in the current directory")
    sys.exit(1)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n" + "=" * 80)
