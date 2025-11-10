# CRITICAL FINDING: Weight Scaling Analysis

**Date**: 2025-11-10
**Status**: URGENT - Test This Immediately

---

## TL;DR

**The ORIGINAL weight scaling code was probably CORRECT!**

Test with:
```python
# ASC (FIXED):
network['glif3'][:, G.AA0] /= 1000.0
network['glif3'][:, G.AA1] /= 1000.0

# Weights (REVERT TO ORIGINAL):
syn[:, S.WHT] *= vsc / 1000.0  # NOT /= 1000 !!
```

---

## What I Realized

Looking at your weight statistics and analyzing all scaling options:

### Your Data:
- Recurrent mean: 0.104, max: 29.69
- Input mean: 0.814, max: 5.15
- Voltage scale: ~24 mV

### Scaling Analysis:

| Option | Code | Mean (nA) | Max (nA) | Status |
|--------|------|-----------|----------|--------|
| **A (Original)** | `*= vsc / 1000` | 0.0025 | 0.71 | ✅ **REASONABLE** |
| B (Your test?) | `/= 1000` | 0.0001 | 0.030 | ❌ TOO SMALL |
| C (No scale) | use as-is | 0.104 | 29.69 | ⚠️ MAX TOO LARGE |
| D (Denorm only) | `*= vsc` | 2.5 | 712 | ❌ MAX EXTREME |

**Option A (original) is the only one with reasonable values!**

---

## Why The Original Code Is Correct

### TensorFlow Training (models.py:227):
```python
weights = weights / voltage_scale  # Normalize
```

### Units in H5:
- TensorFlow stores: **pA / mV** (picoamperes per millivolt)
- PyNN needs: **nA** (nanoamperes)

### Conversion:
```python
weights_nA = weights_H5 * vsc / 1000
             \_____________/   \___/
                    |            |
             denormalize    pA → nA
             (× mV)         (÷ 1000)
```

This is EXACTLY what the original code does!

---

## Why You Saw "Garbage Output"

The **ASC bug** was the critical issue, NOT weight scaling!

### Before Your Fixes:
- ❌ ASC: `*= vsc / 1000` (20x too large)
- ✅ Weights: `*= vsc / 1000` (CORRECT)
- **Result**: Neuron dynamics broken → garbage output

### What You Tested (I think):
- ✅ ASC: `/= 1000` (FIXED)
- ❌ Weights: `/= 1000` (now 24x too small)
- **Result**: Weak connections → might explain "garbage"

### What You Should Test:
- ✅ ASC: `/= 1000` (FIXED)
- ✅ Weights: `*= vsc / 1000` (ORIGINAL)
- **Result**: Both correct → should work!

---

## Exact Code Changes

### Current class.py Lines 123-124 (KEEP THIS):
```python
network['glif3'][:, G.AA0] /= 1000.0  # CORRECT
network['glif3'][:, G.AA1] /= 1000.0  # CORRECT
```

### Current class.py Lines 841, 888 (CHANGE BACK):
```python
# CHANGE FROM:
syn[:, S.WHT] /= 1000.0

# CHANGE TO (ORIGINAL CODE):
syn[:, S.WHT] *= vsc / 1000.0
```

---

## Test Protocol

### Step 1: Revert weight scaling to original
```bash
cd /home/user/upload/jupyter
# Edit class.py lines 841 and 888
# FROM: syn[:, S.WHT] /= 1000.0
# TO:   syn[:, S.WHT] *= vsc / 1000.0
```

### Step 2: Verify ASC fix is still in place
```bash
# Lines 123-124 should be:
network['glif3'][:, G.AA0] /= 1000.0
network['glif3'][:, G.AA1] /= 1000.0
```

### Step 3: Run simulation
```bash
python3 class.py
```

### Step 4: Check results
Expected:
- Network should be ACTIVE (not silent)
- Output should be MEANINGFUL (not garbage)
- Accuracy should improve dramatically

---

## If This Still Gives Garbage

Then check:

### 1. Weight outliers
That max weight of 29.69 is concerning:
```python
# After loading H5, before scaling
rec_weights = network['recurrent'][:, 2]
outliers = np.abs(rec_weights) > 10
print(f"Outliers: {np.sum(outliers)} ({100*np.sum(outliers)/len(rec_weights):.2f}%)")
if np.sum(outliers) > 0:
    network['recurrent'][:, 2] = np.clip(rec_weights, -10, 10)
```

### 2. Run diagnostic script
```bash
python3 diagnose_weight_scaling.py
```

This will analyze your H5 file and confirm scaling approach.

### 3. Check receptor types
Verify inhibitory weights are negative:
```python
# After weight scaling
for receptor in [0, 1, 2, 3]:
    mask = network['recurrent'][:, 3] == receptor
    weights = network['recurrent'][mask, 2]
    print(f"Receptor {receptor}: mean={weights.mean():.4f}, range=[{weights.min():.4f}, {weights.max():.4f}]")
```

Expected:
- Receptors 0, 2 (AMPA, NMDA): positive or mixed
- Receptors 1, 3 (GABA_A, GABA_B): negative or mixed

---

## Why I'm Confident

### Evidence:
1. ✅ Original code gives reasonable weight magnitudes (0.001-0.7 nA)
2. ✅ TensorFlow code shows clear normalization by vsc
3. ✅ Typical neuroscience practice: store in pA, use in nA
4. ✅ The /1000 serves dual purpose: denormalize AND convert units

### What was wrong:
- ❌ ASC had SAME pattern but H5 stores UNNORMALIZED pA
- ❌ This broke neuron dynamics completely
- ❌ Weight scaling was CORRECT but hidden by ASC bug

---

## Bottom Line

**TEST THIS IMMEDIATELY:**

```python
# jupyter/class.py

# Lines 123-124 (ASC - KEEP FIXED):
network['glif3'][:, G.AA0] /= 1000.0
network['glif3'][:, G.AA1] /= 1000.0

# Lines 841, 888 (Weights - REVERT TO ORIGINAL):
vsc = network['glif3'][int(tgt_key[0]), G.VSC]
syn[:, S.WHT] *= vsc / 1000.0  # ORIGINAL CODE
```

If this works → Problem solved!
If this fails → Run diagnostic scripts and report findings

---

**Let me know the results!**
