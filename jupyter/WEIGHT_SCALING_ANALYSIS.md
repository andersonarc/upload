# Weight Scaling Analysis - Critical Findings

**Date**: 2025-11-10

---

## The Confusion

Looking at the scaling options:

### Option A: `weights * vsc / 1000` (ORIGINAL CODE)
- Mean: 0.002496 nA
- Max: 0.712560 nA
- **Status: Actually REASONABLE range for PyNN GLIF3!**

### Option B: `weights / 1000` (What you tested "with /1000")
- Mean: 0.000104 nA
- Max: 0.029690 nA
- **Status: TOO SMALL - barely any effect**

### Option C: No scaling (What you tested "without /1000")
- Mean: 0.104 nA
- Max: 29.69 nA
- **Status: Max value WAY too large (712 nA after denormalization)**

### Option D: `weights * vsc` (Pure denormalization)
- Mean: 2.496 nA
- Max: **712.560 nA** ⚠️
- **Status: Max value EXTREMELY large - would cause instability!**

---

## Critical Insight

The **ORIGINAL CODE** (Option A: `* vsc / 1000`) might actually be **CORRECT**!

### Reasoning:

1. **TensorFlow normalizes**: `weights_stored = weights / voltage_scale`
2. **Weights are stored in picoamperes** (pA) - common in neuroscience
3. **PyNN needs nanoamperes** (nA)
4. **Conversion**: `weights_nA = weights_pA * vsc / 1000`

This matches Option A perfectly!

### The /1000 does TWO things:
- **Denormalize**: multiply by vsc
- **Convert units**: divide by 1000 (pA → nA)

---

## Then Why "Garbage Output"?

If the original code (Option A) was correct for weight scaling, why does the simulation fail?

### Possible explanations:

1. **ASC bug was the critical issue** (now fixed)
   - Test with BOTH fixes together:
     - ASC: `/= 1000` (no vsc)
     - Weights: `*= vsc / 1000` (ORIGINAL code)

2. **Weight distribution issues**
   - Max weight of 29.69 in H5 is an OUTLIER
   - Even with correct scaling: 29.69 * 24 / 1000 = 0.71 nA (very strong!)
   - Might need weight clipping or outlier removal

3. **Different bug entirely**
   - Receptor types (inhibitory vs excitatory)
   - Delays
   - Neuron parameters
   - Input encoding

---

## What You Should Test

### Test 1: ASC fix + ORIGINAL weight scaling
```python
# Lines 123-124 (ASC - FIXED):
network['glif3'][:, G.AA0] /= 1000.0  # NEW
network['glif3'][:, G.AA1] /= 1000.0  # NEW

# Lines 841, 888 (Weights - REVERT TO ORIGINAL):
syn[:, S.WHT] *= vsc / 1000.0  # ORIGINAL CODE
```

**Hypothesis**: This might actually work!

### Test 2: Check for weight outliers
```python
# After loading weights, before scaling
import numpy as np
rec_weights = network['recurrent'][:, 2]
print(f"Weights > 10: {np.sum(np.abs(rec_weights) > 10)}")
print(f"Max weight: {np.max(np.abs(rec_weights))}")

# Optional: Clip extreme values
rec_weights = np.clip(rec_weights, -10, 10)
```

### Test 3: Try Option D if Test 1 fails
```python
# Lines 841, 888:
syn[:, S.WHT] *= vsc  # No /1000
```

But be aware: this gives mean ~2.5 nA, max ~712 nA (very strong!)

---

## Key Question

**When you tested "with /1000", did you:**

A) Keep `vsc` and just change from `*=` to `/=`?
   ```python
   # FROM: syn[:, S.WHT] *= vsc / 1000.0
   # TO:   syn[:, S.WHT] /= 1000.0
   ```
   This is **Option B** (wrong)

B) Remove `vsc` multiplication entirely?
   ```python
   # FROM: syn[:, S.WHT] *= vsc / 1000.0
   # TO:   syn[:, S.WHT] /= 1000.0
   ```
   Same as A

C) Change the operation but keep vsc?
   ```python
   # FROM: syn[:, S.WHT] *= vsc / 1000.0
   # TO:   syn[:, S.WHT] = syn[:, S.WHT] * vsc / 1000.0
   ```
   This would be same as original (Option A)

**Please clarify what exact code you tested!**

---

## Units Clarification

### TensorFlow Training:
- Original weights: some unit (let's call it X)
- Normalized: X / voltage_scale (mV) = X/mV
- Stored in H5: X/mV units

### PyNN Needs:
- Current-based synapses: nanoamperes (nA)

### If X = picoamperes (pA):
- H5 stores: pA/mV
- Denormalize: pA/mV * mV = pA
- Convert: pA / 1000 = nA
- **CORRECT: `*= vsc / 1000`** ✅

### If X = nanoamperes (nA):
- H5 stores: nA/mV
- Denormalize: nA/mV * mV = nA
- Convert: none needed
- **CORRECT: `*= vsc`** ✅

---

## My Recommendation

### Step 1: Test with ASC fix ONLY
Keep original weight scaling:
```python
# ASC (fixed):
network['glif3'][:, G.AA0] /= 1000.0
network['glif3'][:, G.AA1] /= 1000.0

# Weights (ORIGINAL):
syn[:, S.WHT] *= vsc / 1000.0
```

If this works → ASC was the only bug!

### Step 2: If still garbage, check weight statistics
```python
# After scaling, before creating projections
print(f"Scaled weight mean: {np.mean(np.abs(syn[:, S.WHT]))}")
print(f"Scaled weight max: {np.max(np.abs(syn[:, S.WHT]))}")
```

Expected: mean ~0.002-0.02 nA, max < 1.0 nA

### Step 3: If weights look wrong, try alternative scaling
```python
syn[:, S.WHT] *= vsc  # Option D
```

But monitor for instability (extreme weight values).

---

## Bottom Line

I now suspect the **ORIGINAL CODE was actually correct** for weights!

The ASC bug was the critical issue (confirmed by you). Please test with:
- ASC: `/= 1000` (FIXED)
- Weights: `*= vsc / 1000` (ORIGINAL)

This combination might be the solution!
