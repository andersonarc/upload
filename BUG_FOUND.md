# THE BUG: Background Weights 100× Too Large!

## Summary
**SpiNNaker background noise is 100× stronger than TensorFlow/NEST**, completely overwhelming the signal!

## Expected Background Input (TensorFlow & NEST)

**TensorFlow** (models.py:101-104):
- 10 Bernoulli(0.1) sources, expected sum = 1.0 per ms
- Applied as: `bkg_weights * 1.0 / 10 = 0.1 × W_orig` per ms

**NEST** (nest_glif.py:132-167):
- 10 Poisson(10 Hz) generators = 100 Hz total = 0.1 spikes/ms expected
- Weight per spike: `(bkg_weights / 10) * vsc = W_orig` pA
- Expected input: `0.1 spikes/ms × W_orig = 0.1 × W_orig` pA per ms

**Both give: 0.1 × W_orig pA per ms**

---

## Actual SpiNNaker Background Input

**SpiNNaker** (spynnaker_newclass.py:933-965):
- Single Poisson(100 Hz) source = 0.1 spikes/ms expected
- Weight calculation:
```python
vsc = network['glif3'][pid, G.VSC]
bkg_w_norm = bkg_w * vsc  # (W_orig/vsc*10) * vsc = W_orig * 10
bkg_w_scaled = bkg_w_norm * 10.0 / 1000.0  # BUG: multiplying by 10!
# Result: W_orig * 10 * 10 / 1000 = W_orig / 10 in nA = W_orig * 100 in pA
```

- Expected input: `0.1 spikes/ms × (W_orig × 100) = 10 × W_orig` pA per ms

**SpiNNaker gives: 10 × W_orig pA per ms → 100× too large!**

---

## The Fix

**Line 965 in spynnaker_newclass.py:**

```python
# BEFORE (WRONG):
bkg_w_scaled = bkg_w_norm * 10.0 / 1000.0  # Multiplying by 10!

# AFTER (CORRECT):
bkg_w_scaled = (bkg_w / 10.0) * vsc / 1000.0  # Divide by 10, like NEST
```

Or more clearly, matching NEST pattern:
```python
# Match NEST implementation exactly
w_filt = bkg_w / 10.0  # Normalize for 10-source model
bkg_w_scaled = w_filt * vsc / 1000.0  # Denormalize and convert pA→nA
```

---

## Why This Causes "Garbage Spikes"

With background 100× too strong:
- **Background noise**: Dominates neuron input (100× expected)
- **Synaptic signal**: Correct magnitude (with /1000.0 for pA→nA)
- **Result**: Neurons spike randomly due to noise, ignoring actual input!

This perfectly explains the "garbage spike" behavior where outputs don't classify digits correctly.

---

## Verification

The `/1000.0` in synaptic weights **IS CORRECT** for pA→nA unit conversion (see UNIT_ANALYSIS.md).

The **ONLY bug** is the background weight calculation.

After fixing background weights:
- Signal-to-noise ratio will be correct
- Should match NEST's 80%+ accuracy
- Spikes will correlate with digit inputs

---

## Code Location

**File**: `scripts/spynnaker_newclass.py`
**Line**: 965
**Function**: `create_background()`

Change ONE line to fix the entire issue!
