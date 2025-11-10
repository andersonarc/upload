# SECOND PASS: ASC Scaling Verification

**Status**: Attempting to DISPROVE initial finding
**User Claim**: "ASC values are likely stored normalized"
**My Initial Claim**: "ASC values stored unnormalized → class.py bug"

---

## Critical Question

**What format are asc_amps values stored in the H5 file?**
- Option A: UNNORMALIZED (pA) → class.py has bug
- Option B: NORMALIZED (dimensionless) → class.py is correct

---

## Data Flow Analysis

### Step 1: Source Data (Allen Institute Pickle)

**File**: `v1cortex/network_dat.pkl`

**Format**: UNNORMALIZED (pA)
- Raw experimental measurements of after-spike current amplitudes
- Typical values: 10-100 pA
- **Evidence**: Allen Institute GLIF models use physical units

### Step 2: load_sparse.py Loading

**Code** (lines 91-96):
```python
for i, node_type in enumerate(d['nodes']):
    tf_ids = bmtk_id_to_tf_id[np.array(node_type['ids'])]
    tf_ids = tf_ids[tf_ids >= 0]
    node_type_ids[tf_ids] = i
    for k, v in node_params.items():
        v[i] = node_type['params'][k]  # ← Loads asc_amps directly
```

**Result**: `network['node_params']['asc_amps']` contains **UNNORMALIZED (pA)** values

### Step 3: TensorFlow Training (models.py:154)

**Code**:
```python
self._params = network['node_params']  # Gets unnormalized from load_sparse
voltage_scale = self._params['V_th'] - self._params['E_L']
self._params['asc_amps'] = self._params['asc_amps'] / voltage_scale[..., None]
```

**Operation**: NORMALIZES asc_amps by dividing by voltage_scale
**Result**: TensorFlow uses **NORMALIZED (dimensionless)** values internally

**Important**: This only affects TensorFlow's internal copy of `self._params`, not the original `network['node_params']` dict from load_sparse.

### Step 4: c2.py Checkpoint Loading (lines 11-99)

**Code**:
```python
def load_tf_checkpoint(checkpoint_path, network):
    # ... (lines 23-29) Creates NEW network via load_billeh
    input_population, network, bkg_weights = load_sparse.load_billeh(...)

    # ... (lines 32-41) Builds TensorFlow model (which normalizes asc_amps internally)
    model = classification_tools.create_model(network, ...)

    # ... (lines 64-97) Extracts ONLY trainable weights (recurrent, input, readout)
    # DOES NOT extract node_params or asc_amps!
```

**Result**: Returns dict with ONLY weights, NO node_params

### Step 5: c2.py Conversion (lines 101-189)

**Code**:
```python
def convert_to_pynn_format(checkpoint_path, data_dir, output_h5, ...):
    # Line 103-109: Load network from scratch
    input_population, network, bkg_weights = load_sparse.load_billeh(...)

    # Line 113: Load trained weights
    model_vars = load_tf_checkpoint(checkpoint_path, network)

    # Line 127: Get node_params from THIS network (not from checkpoint)
    node_params = network['node_params']

    # Lines 135-152: Use weights from checkpoint OR fallback to initial
    # (Only affects weights, not node_params)

    # Lines 187-189: Store node_params AS-IS
    params_grp = neuron_grp.create_group('glif3_params')
    for key, val in node_params.items():
        params_grp.create_dataset(key, data=val)
```

**Critical Analysis**:
- Line 103-109: `load_billeh()` returns `network['node_params']` with **UNNORMALIZED asc_amps**
- Line 113: `load_tf_checkpoint()` returns ONLY weights, does NOT return node_params
- Line 127: Uses node_params from the load_billeh() network (unnormalized)
- Lines 188-189: Stores node_params directly WITHOUT any normalization

**Result**: H5 file contains **UNNORMALIZED (pA)** asc_amps

**Confidence**: 99% - Code clearly shows no normalization

### Step 6: class.py Loading (lines 100-124)

**Code**:
```python
# Lines 100-117: Load from H5
network['glif3'] = np.stack([
    file['neurons/glif3_params/C_m'],
    file['neurons/glif3_params/E_L'],
    file['neurons/glif3_params/V_reset'],
    file['neurons/glif3_params/V_th'],
    file['neurons/glif3_params/asc_amps'][:, 0],  # ← Loads as-is
    file['neurons/glif3_params/asc_amps'][:, 1],  # ← Loads as-is
    ...
], axis=1)

# Lines 120-124: Scale parameters
network['glif3'][:, G.CM]  /= 1000.0  # pF -> nF
network['glif3'][:, G.G]   /= 1000.0  # nS -> uS
network['glif3'][:, G.VSC] = network['glif3'][:, G.THR] - network['glif3'][:, G.EL]
network['glif3'][:, G.AA0] *= network['glif3'][:, G.VSC] / 1000.0
network['glif3'][:, G.AA1] *= network['glif3'][:, G.VSC] / 1000.0
```

**Operation on ASC**:
```
asc_amps *= voltage_scale / 1000
```

**Two Scenarios**:

**Scenario A: H5 has UNNORMALIZED (pA)**
- Input: pA
- Operation: pA * voltage_scale / 1000
- Output: pA * voltage_scale / 1000
- **For voltage_scale = 20 mV**: pA * 20 / 1000 = pA * 0.02
- **Should be**: pA / 1000 = nA
- **Result**: Values are **20x too large** (0.02 * pA instead of 0.001 * pA)

**Scenario B: H5 has NORMALIZED (dimensionless)**
- Input: pA / voltage_scale (dimensionless)
- Operation: (pA / voltage_scale) * voltage_scale / 1000
- Output: pA / 1000 = nA
- **Result**: **CORRECT**

### Step 7: PyNN GLIF3 Expected Format

**File**: `glif3/glif3_curr.py` lines 34-37

**Documentation**:
```python
asc_amp_0 : float
    Fast after-spike current amplitude (nA). Default: 0.0
asc_amp_1 : float
    Slow after-spike current amplitude (nA). Default: 0.0
```

**Expected**: nA (nanoamps)

---

## Proof: What Format Does H5 Have?

### Code Evidence

Based on c2.py code analysis: **UNNORMALIZED (pA)**
- c2.py line 127 gets node_params from load_billeh (unnormalized)
- c2.py lines 188-189 store without modification
- No normalization code exists between loading and storing

**Confidence**: 99% based on code

### How to VERIFY Empirically

**Method 1: Inspect H5 file values**
```python
import h5py
import numpy as np

with h5py.File('ckpt_51978-153.h5', 'r') as f:
    asc_amps = np.array(f['neurons/glif3_params/asc_amps'])
    v_th = np.array(f['neurons/glif3_params/V_th'])
    e_l = np.array(f['neurons/glif3_params/E_L'])

    voltage_scale = v_th - e_l

    print(f"asc_amps[0]: {asc_amps[0]}")
    print(f"voltage_scale[0]: {voltage_scale[0]}")
    print(f"asc_amps[0] / voltage_scale[0]: {asc_amps[0] / voltage_scale[0]}")

    # If asc_amps values are ~10-100: UNNORMALIZED (pA)
    # If asc_amps values are ~0.5-5: NORMALIZED (dimensionless)
    # If voltage_scale is ~15-25: Typical
```

**Expected if UNNORMALIZED**: asc_amps ~ 10-100, voltage_scale ~ 15-25
**Expected if NORMALIZED**: asc_amps ~ 0.5-5, voltage_scale ~ 15-25

**Method 2: Check TensorFlow training logs**

If c2.py was run with logs enabled, check for the output from line 137 or 151:
```
Using TRAINED recurrent weights: mean=..., std=...
```

But node_params are NOT trained, so they wouldn't appear in logs.

**Method 3: Compare with original pickle**
```python
import pickle
with open('v1cortex/network_dat.pkl', 'rb') as f:
    d = pickle.load(f)
    original_asc = d['nodes'][0]['params']['asc_amps']
    print(f"Original asc_amps: {original_asc}")

import h5py
with h5py.File('ckpt_51978-153.h5', 'r') as f:
    h5_asc = f['neurons/glif3_params/asc_amps'][0]
    print(f"H5 asc_amps: {h5_asc}")

# If they match: UNNORMALIZED
# If h5 is ~20x smaller: NORMALIZED
```

---

## User's Claim: "Likely stored normalized"

**Possible explanations**:

1. **User is using different c2.py version** - Code was modified to normalize before storing
2. **Manual H5 editing** - User manually fixed H5 file after generation
3. **Memory of previous fix** - User remembers fixing this before and assumes current file is fixed
4. **Misunderstanding data flow** - User thinks TensorFlow's normalization affects stored values

**Response**: Need to CHECK ACTUAL H5 FILE to know for certain.

---

## Attempting to Disprove My Finding

**Claim to disprove**: "class.py has bug where it multiplies by voltage_scale instead of dividing"

**Counter-evidence needed**:
1. Proof that H5 file contains normalized values, OR
2. Proof that class.py formula is correct for unnormalized inputs

**Checking (1) - H5 contains normalized values?**
- Code analysis says NO (c2.py doesn't normalize)
- Can only be definitively proven by inspecting actual H5 file

**Checking (2) - class.py formula correct for unnormalized?**
- Formula: `asc *= voltage_scale / 1000`
- Input: pA (unnormalized)
- Output: pA * voltage_scale / 1000
- Expected: nA = pA / 1000
- **Mismatch**: Factor of `voltage_scale` error
- **Cannot be correct for unnormalized input**

---

## Conclusions

### Based on Code Analysis

**Finding**: c2.py stores UNNORMALIZED (pA) values
**Confidence**: 99%
**Evidence**: Clear code path from load_sparse → c2.py storage with no normalization

**Finding**: class.py expects NORMALIZED (dimensionless) values
**Confidence**: 90%
**Evidence**: Formula `asc *= voltage_scale / 1000` only makes sense if input is normalized

**Finding**: class.py multiplies by voltage_scale when it should divide (IF values are normalized)
**Confidence**: 95%
**Evidence**: Mathematical analysis of formula

### If Values Are Unnormalized (Code suggests this)

**Bug location**: c2.py should normalize before storing
**OR**: class.py should divide by voltage_scale, not multiply
**Result**: ASC amplitudes are voltage_scale (~20x) too large
**Impact**: CRITICAL - breaks network dynamics completely

### If Values Are Normalized (User claims this)

**Bug location**: None - everything is correct
**Result**: ASC amplitudes are correct
**Impact**: None - ASC is not the problem

---

## Required Action: Empirical Verification

**Cannot definitively resolve without inspecting actual H5 file**

### Test Script

```python
#!/usr/bin/env python
"""Verify ASC normalization status in H5 file"""
import h5py
import numpy as np

with h5py.File('jupyter/ckpt_51978-153.h5', 'r') as f:
    asc_amps = np.array(f['neurons/glif3_params/asc_amps'])
    v_th = np.array(f['neurons/glif3_params/V_th'])
    e_l = np.array(f['neurons/glif3_params/E_L'])

    voltage_scale = v_th - e_l

    print("Sample values (first 5 neuron types):")
    print(f"asc_amps[0]: {asc_amps[:5, 0]}")
    print(f"asc_amps[1]: {asc_amps[:5, 1]}")
    print(f"voltage_scale: {voltage_scale[:5]}")
    print(f"asc_amps[0] / voltage_scale: {asc_amps[:5, 0] / voltage_scale[:5]}")

    # Statistics
    print(f"\nasc_amps magnitude:")
    print(f"  Mean: {np.mean(np.abs(asc_amps)):.2f}")
    print(f"  Range: [{np.min(asc_amps):.2f}, {np.max(asc_amps):.2f}]")

    print(f"\nvoltage_scale magnitude:")
    print(f"  Mean: {np.mean(voltage_scale):.2f}")
    print(f"  Range: [{np.min(voltage_scale):.2f}, {np.max(voltage_scale):.2f}]")

    # Diagnosis
    mean_abs_asc = np.mean(np.abs(asc_amps[asc_amps != 0]))
    mean_vscale = np.mean(voltage_scale)

    print(f"\n{'='*60}")
    print(f"DIAGNOSIS:")
    print(f"{'='*60}")

    if mean_abs_asc > 10:
        print(f"✓ asc_amps values are ~{mean_abs_asc:.1f} (>> 1)")
        print(f"  → UNNORMALIZED (in pA)")
        print(f"  → class.py has BUG (multiplies by voltage_scale)")
        print(f"  → ASC values are {mean_vscale:.0f}x too large!")
    elif mean_abs_asc < 5:
        print(f"✓ asc_amps values are ~{mean_abs_asc:.2f} (<< 10)")
        print(f"  → NORMALIZED (dimensionless)")
        print(f"  → class.py is CORRECT")
        print(f"  → ASC values are OK")
    else:
        print(f"⚠️  Ambiguous: asc_amps ~ {mean_abs_asc:.2f}")
        print(f"  → Need expert judgment")
```

**Run this script to determine ground truth**

---

## Final Answer

**Based on code analysis**: Bug exists, ASC is ~20x too large

**Based on user claim**: No bug, ASC is correct

**Resolution**: **NEED TO RUN VERIFICATION SCRIPT** on actual H5 file

**I cannot definitively disprove my finding without empirical data.**

---

## Revised Recommendations

### Option A: If Verification Shows UNNORMALIZED

**Fix class.py lines 123-124**:
```python
# Change FROM:
network['glif3'][:, G.AA0] *= network['glif3'][:, G.VSC] / 1000.0
network['glif3'][:, G.AA1] *= network['glif3'][:, G.VSC] / 1000.0

# Change TO:
network['glif3'][:, G.AA0] /= 1000.0  # pA -> nA (no voltage_scale)
network['glif3'][:, G.AA1] /= 1000.0  # pA -> nA (no voltage_scale)
```

**OR fix c2.py to normalize before storing**:
```python
# After line 127:
voltage_scale = node_params['V_th'] - node_params['E_L']
node_params['asc_amps'] = node_params['asc_amps'] / voltage_scale[:, None]
# Then class.py formula becomes correct
```

### Option B: If Verification Shows NORMALIZED

**No fix needed** - class.py is correct

**Then investigate other bug sources**:
- GLIF3 implementation details
- Weight normalization/denormalization
- Spike timing issues
- Numerical precision issues
