#!/usr/bin/env python
"""Verify ASC normalization status in H5 file"""
import h5py
import numpy as np
import sys

H5_FILE = 'ckpt_51978-153.h5'

try:
    with h5py.File(H5_FILE, 'r') as f:
        asc_amps = np.array(f['neurons/glif3_params/asc_amps'])
        v_th = np.array(f['neurons/glif3_params/V_th'])
        e_l = np.array(f['neurons/glif3_params/E_L'])

        voltage_scale = v_th - e_l

        print("="*80)
        print("ASC NORMALIZATION VERIFICATION")
        print("="*80)

        print("\nSample values (first 5 neuron types):")
        print(f"asc_amps[:5, 0]: {asc_amps[:5, 0]}")
        print(f"asc_amps[:5, 1]: {asc_amps[:5, 1]}")
        print(f"voltage_scale[:5]: {voltage_scale[:5]}")
        print(f"asc_amps[:5, 0] / voltage_scale[:5]: {asc_amps[:5, 0] / voltage_scale[:5]}")

        # Statistics
        print(f"\nasc_amps magnitude:")
        print(f"  Mean: {np.mean(np.abs(asc_amps)):.4f}")
        print(f"  Std: {np.std(asc_amps):.4f}")
        print(f"  Range: [{np.min(asc_amps):.4f}, {np.max(asc_amps):.4f}]")

        print(f"\nvoltage_scale magnitude:")
        print(f"  Mean: {np.mean(voltage_scale):.4f}")
        print(f"  Std: {np.std(voltage_scale):.4f}")
        print(f"  Range: [{np.min(voltage_scale):.4f}, {np.max(voltage_scale):.4f}]")

        # Diagnosis
        nonzero_asc = asc_amps[asc_amps != 0]
        if len(nonzero_asc) > 0:
            mean_abs_asc = np.mean(np.abs(nonzero_asc))
        else:
            mean_abs_asc = 0
            print("\n⚠️  WARNING: All ASC values are zero!")

        mean_vscale = np.mean(voltage_scale)

        print(f"\n{'='*80}")
        print(f"DIAGNOSIS:")
        print(f"{'='*80}")

        if mean_abs_asc > 10:
            print(f"\n✓ asc_amps values are ~{mean_abs_asc:.1f} (>> 1)")
            print(f"  → UNNORMALIZED (in pA)")
            print(f"  → H5 file contains PHYSICAL units")
            print(f"\n🔴 BUG CONFIRMED:")
            print(f"  class.py multiplies by voltage_scale when it should NOT")
            print(f"  Result: ASC values are {mean_vscale:.0f}x too large!")
            print(f"\n  Fix in class.py lines 123-124:")
            print(f"    Change: *= voltage_scale / 1000")
            print(f"    To:     /= 1000")

        elif mean_abs_asc < 5 and mean_abs_asc > 0:
            print(f"\n✓ asc_amps values are ~{mean_abs_asc:.2f} (<< 10)")
            print(f"  → NORMALIZED (dimensionless)")
            print(f"  → H5 file contains normalized values")
            print(f"\n✅ NO BUG:")
            print(f"  class.py is CORRECT")
            print(f"  ASC normalization is working as intended")
            print(f"\n  ASC is NOT the root cause of failure")

        else:
            print(f"\n⚠️  Ambiguous: asc_amps ~ {mean_abs_asc:.2f}")
            print(f"  → Unclear if normalized or not")
            print(f"  → Need expert judgment")

        # Additional check: ratio test
        print(f"\n{'='*80}")
        print(f"RATIO TEST:")
        print(f"{'='*80}")

        if len(nonzero_asc) > 0:
            # If normalized, asc_amps should be approximately asc_amps_physical / voltage_scale
            # So asc_amps * voltage_scale should give back physical values
            sample_indices = np.where(asc_amps[:, 0] != 0)[0][:5]
            if len(sample_indices) > 0:
                print("\nIf normalized, asc * voltage_scale should give ~10-100 (pA range)")
                print("If unnormalized, asc / voltage_scale should give ~0.5-5 (normalized range)")
                print()
                for idx in sample_indices:
                    asc_val = asc_amps[idx, 0]
                    vs = voltage_scale[idx]
                    print(f"  Type {idx}: asc={asc_val:8.4f}, vs={vs:6.2f}, "
                          f"asc*vs={asc_val*vs:8.4f}, asc/vs={asc_val/vs:8.4f}")

        print(f"\n{'='*80}")

except FileNotFoundError:
    print(f"ERROR: H5 file not found: {H5_FILE}")
    print("File needs to be downloaded or generated")
    sys.exit(1)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
