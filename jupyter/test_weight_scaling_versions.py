#!/usr/bin/env python3
"""
Test Different Weight Scaling Approaches

This script shows exactly what each scaling option does to the weights.
"""

import numpy as np

print("=" * 80)
print("Weight Scaling Test - All Versions")
print("=" * 80)

# Your reported statistics
rec_mean = 0.104
rec_max = 29.69
inp_mean = 0.814
inp_max = 5.15
vsc_mean = 24.0  # Approximate from GLIF3 params

print(f"\nInput values (from H5):")
print(f"  Recurrent weight mean: {rec_mean}")
print(f"  Recurrent weight max:  {rec_max}")
print(f"  Input weight mean:     {inp_mean}")
print(f"  Voltage scale (mean):  {vsc_mean} mV")

print(f"\n" + "=" * 80)
print(f"SCALING OPTIONS:")
print(f"=" * 80)

print(f"\nOption A: weights * vsc / 1000  (ORIGINAL BUGGY CODE)")
print(f"  Lines 841, 888: syn[:, S.WHT] *= vsc / 1000.0")
print(f"  Recurrent mean: {rec_mean} * {vsc_mean} / 1000 = {rec_mean * vsc_mean / 1000:.6f} nA")
print(f"  Recurrent max:  {rec_max} * {vsc_mean} / 1000 = {rec_max * vsc_mean / 1000:.6f} nA")
print(f"  Input mean:     {inp_mean} * {vsc_mean} / 1000 = {inp_mean * vsc_mean / 1000:.6f} nA")
if rec_mean * vsc_mean / 1000 < 0.001:
    print(f"  ❌ TOO SMALL - neurons won't fire properly")
else:
    print(f"  ✅ Reasonable range")

print(f"\nOption B: weights / 1000  (DIVIDE BY 1000 ONLY)")
print(f"  Lines 841, 888: syn[:, S.WHT] /= 1000.0")
print(f"  Recurrent mean: {rec_mean} / 1000 = {rec_mean / 1000:.6f} nA")
print(f"  Recurrent max:  {rec_max} / 1000 = {rec_max / 1000:.6f} nA")
print(f"  Input mean:     {inp_mean} / 1000 = {inp_mean / 1000:.6f} nA")
if rec_mean / 1000 < 0.001:
    print(f"  ❌ TOO SMALL - weights still normalized by vsc")
else:
    print(f"  ✅ Reasonable range")

print(f"\nOption C: No scaling  (USE H5 VALUES AS-IS)")
print(f"  Lines 841, 888: # No multiplication or division")
print(f"  Recurrent mean: {rec_mean} nA")
print(f"  Recurrent max:  {rec_max} nA")
print(f"  Input mean:     {inp_mean} nA")
print(f"  ⚠️  If weights are normalized, this is WRONG")
print(f"  ⚠️  If weights are in pA, this is WRONG")

print(f"\nOption D: weights * vsc  (DENORMALIZE ONLY) ⭐ RECOMMENDED")
print(f"  Lines 841, 888: syn[:, S.WHT] *= vsc")
print(f"  Recurrent mean: {rec_mean} * {vsc_mean} = {rec_mean * vsc_mean:.6f} nA")
print(f"  Recurrent max:  {rec_max} * {vsc_mean} = {rec_max * vsc_mean:.6f} nA")
print(f"  Input mean:     {inp_mean} * {vsc_mean} = {inp_mean * vsc_mean:.6f} nA")
if 0.01 <= rec_mean * vsc_mean <= 10:
    print(f"  ✅ GOOD - matches typical PyNN GLIF3 weights")
else:
    print(f"  ⚠️  Check if reasonable")

print(f"\n" + "=" * 80)
print(f"ANALYSIS:")
print(f"=" * 80)

print(f"\nFrom TensorFlow training code (models.py:227):")
print(f"  weights_stored = weights_original / voltage_scale")
print(f"  Therefore: weights_original = weights_stored * voltage_scale")

print(f"\nH5 file contains VOLTAGE-NORMALIZED weights")
print(f"  ✅ TensorFlow divides by vsc during training")
print(f"  ✅ PyNN needs to multiply by vsc to denormalize")

print(f"\n⭐ RECOMMENDATION: Use Option D")
print(f"  Change lines 841, 888 FROM:")
print(f"    syn[:, S.WHT] /= 1000.0")
print(f"  Change TO:")
print(f"    syn[:, S.WHT] *= vsc")

print(f"\nExpected result after fix:")
print(f"  Mean recurrent weight: ~{rec_mean * vsc_mean:.3f} nA")
print(f"  Mean input weight:     ~{inp_mean * vsc_mean:.3f} nA")
print(f"  Max recurrent weight:  ~{rec_max * vsc_mean:.3f} nA (strong but not impossible)")

print(f"\n" + "=" * 80)
print(f"COMPARISON WITH YOUR TEST RESULTS:")
print(f"=" * 80)

print(f"\nYou reported:")
print(f"  'without /1000 network is silent'")
print(f"  'with /1000 output is garbage'")

print(f"\nMy interpretation:")
print(f"  'without /1000' = Option C (no scaling)")
print(f"    → Silent because weights still normalized (~0.1 instead of ~2.5)")
print(f"  'with /1000' = Option B (divide by 1000)")
print(f"    → Garbage because weights now 24x too small (~0.0001 instead of ~2.5)")

print(f"\nWhat to test next:")
print(f"  Option D: syn[:, S.WHT] *= vsc (just denormalize)")
print(f"    → Should give reasonable network activity")

print(f"\n" + "=" * 80)
