# Phase 6: ASC / Input / Output Analysis

**Date**: 2025-11-10
**Status**: ⚠️ CRITICAL BUG FOUND in ASC scaling

---

## 🚨 CRITICAL BUG #1: ASC Scaling Mismatch

### Data Flow Analysis

**Step 1: load_sparse.py loads node_params from network_dat.pkl**
- File: `/home/user/upload/training_code/load_sparse.py` lines 78-89
- ASC values: **UNNORMALIZED (in pA)**
- Source: Allen Institute GLIF parameters

**Step 2: TensorFlow normalizes for training (models.py:154)**
```python
voltage_scale = self._params['V_th'] - self._params['E_L']
self._params['asc_amps'] = self._params['asc_amps'] / voltage_scale[..., None]
```
- TensorFlow divides by voltage_scale
- Internal format: **NORMALIZED** (dimensionless)

**Step 3: c2.py stores to H5 file (c2.py:127, 188-189)**
```python
node_params = network['node_params']  # From load_sparse.py
...
for key, val in node_params.items():
    params_grp.create_dataset(key, data=val)
```
- Stores node_params **WITHOUT normalization**
- H5 file format: **UNNORMALIZED (in pA)**

**Step 4: class.py loads and scales (class.py:105-106, 123-124)**
```python
# Load from H5
file['neurons/glif3_params/asc_amps'][:, 0]  # Gets UNNORMALIZED pA values
file['neurons/glif3_params/asc_amps'][:, 1]

# Scale (WRONG!)
network['glif3'][:, G.VSC] = network['glif3'][:, G.THR] - network['glif3'][:, G.EL] # voltage scale
network['glif3'][:, G.AA0] *= network['glif3'][:, G.VSC] / 1000.0 # pA -> nA
network['glif3'][:, G.AA1] *= network['glif3'][:, G.VSC] / 1000.0 # pA -> nA
```

### The Bug

**What class.py does**:
```
ASC_nA = ASC_pA * voltage_scale / 1000
```

**What it should do**:
```
ASC_nA = ASC_pA / 1000
```

### Impact

For typical voltage_scale values (~15-25 mV):
- **ASC amplitudes are 15-25x too large**
- After-spike currents massively overshoots
- Neurons spike too frequently after initial spike
- Could cause runaway activity or silence (depending on sign)

**Severity**: 🔴 CRITICAL - This bug alone could explain complete SpiNNaker failure

### Evidence

**File**: `/home/user/upload/training_code/models.py:154`
```python
self._params['asc_amps'] = self._params['asc_amps'] / voltage_scale[..., None]
```
Proves TensorFlow expects UNNORMALIZED input and normalizes it.

**File**: `/home/user/upload/training_code/c2.py:188-189`
```python
for key, val in node_params.items():
    params_grp.create_dataset(key, data=val)
```
Proves c2.py stores UNNORMALIZED values.

**File**: `/home/user/upload/jupyter/class.py:123-124`
```python
network['glif3'][:, G.AA0] *= network['glif3'][:, G.VSC] / 1000.0
network['glif3'][:, G.AA1] *= network['glif3'][:, G.VSC] / 1000.0
```
Proves class.py incorrectly multiplies by voltage_scale.

### Fix

**In class.py lines 123-124, change**:
```python
# WRONG:
network['glif3'][:, G.AA0] *= network['glif3'][:, G.VSC] / 1000.0
network['glif3'][:, G.AA1] *= network['glif3'][:, G.VSC] / 1000.0

# CORRECT:
network['glif3'][:, G.AA0] /= 1000.0  # pA -> nA
network['glif3'][:, G.AA1] /= 1000.0  # pA -> nA
```

**Alternative**: Have c2.py normalize before storing (match TensorFlow format):
```python
# In c2.py after line 127:
voltage_scale = node_params['V_th'] - node_params['E_L']
node_params['asc_amps'] = node_params['asc_amps'] / voltage_scale[:, None]
# Then class.py can use: *= VSC / 1000
```

---

## ✅ Input Encoding Verification

### Code Analysis (class.py:263-279)

**Poisson Sampling with 1.3 removal**:
```python
def create_spike_times(spike_trains, timestep=1.0, scale=1.0):
    for i in range(lgn_size):
        times = []
        for t in range(spike_trains.shape[0]):
            if np.clip((spike_trains[t, i] / 1.3) * scale, 0.0, 1.0) > np.random.rand():
                times.append(float(t * timestep))
        spike_times.append(times)
    return spike_times
```

**Analysis**:
1. **Line 274**: Divides by 1.3 (removes training-time scaling)
2. **Line 274**: Multiplies by `scale` (currently 1.0, so no effect)
3. **Line 274**: Clips to [0, 1] (ensures valid probability)
4. **Line 274**: Poisson sampling via `> np.random.rand()`
5. **Line 275**: Converts to spike time in milliseconds

**Status**: ✅ **CORRECT** - Matches TensorFlow preprocessing

### Verification from LGN Diagnostic

Previous analysis (`jupyter/analyze_working_vs_failed_lgn.py`) showed:
- LGN encoding produces recognizable digits
- No systematic differences between "working" and "failed" samples
- Input spike statistics are reasonable

**Conclusion**: Input encoding is NOT the bug

---

## ✅ Output Decoding Verification

### Code Analysis (class.py:1000-1009)

**Vote Counting Logic**:
```python
votes = np.zeros(10)
for i in range(0, 10):
    start = i*30
    end = (1+i)*30
    keys = network['output'][start:end]  # 30 neurons per class
    for key in keys:
        spiketrain = gid2train[key]
        mask = (spiketrain > 50) & (spiketrain < 100)  # Response window
        count = mask.sum()
        votes[i] += count
```

**Analysis**:
1. **Line 1000**: Creates 10 vote counters (one per class)
2. **Line 1001**: Iterates over classes 0-9
3. **Line 1002-1003**: Gets 30 neurons for each class
4. **Line 1004-1008**: Counts spikes in [50, 100] ms window
5. Winner: `votes.argmax()`

**Readout Ordering** (previously analyzed):
- network['output'] contains neurons from `localized_readout_neuron_ids_{5-14}`
- TensorFlow ALSO uses indices 5-14 for classes 0-9
- **Mapping is CORRECT** (see PP_ROUTING_CORRECTION.md)

**Response Window**: [50, 100] ms
- Standard for MNIST classification
- Allows initial transients to settle
- Captures steady-state activity

**Status**: ✅ **CORRECT** - No bugs found

---

## ✅ Weight Scaling Verification

### Recurrent Weights (class.py:130)

```python
network['recurrent'] = np.stack((
    file['recurrent/sources'],
    file['recurrent/targets'],
    file['recurrent/weights'],  # Loaded as-is from H5
    ...
), axis=1)
```

**No scaling applied** - Assumes H5 has weights in nA.

### Input Weights (class.py:140)

```python
network['input'] = np.stack((
    file['input/sources'],
    file['input/targets'],
    file['input/weights'],  # Loaded as-is from H5
    ...
), axis=1)
```

**No scaling applied** - Assumes H5 has weights in nA.

### Verification Needed

Check c2.py to ensure weights are stored in nA, not pA.

**c2.py weight handling** (lines 135-152):
```python
if 'recurrent_weights' in model_vars:
    rec_weights = model_vars['recurrent_weights']  # From TensorFlow
else:
    rec_weights = network['synapses']['weights']  # From load_sparse

if 'input_weights' in model_vars:
    inp_weights = model_vars['input_weights']  # From TensorFlow
else:
    inp_weights = input_population['weights']  # From load_sparse
```

Weights come from either:
1. TensorFlow checkpoint (via load_tf_checkpoint)
2. Initial weights from load_sparse

**TensorFlow weights format** (models.py):
- Normalized by voltage_scale during training
- Need to check if c2.py denormalizes before storing

**Initial weights format** (load_sparse.py):
- Loaded from network_dat.pkl
- Unknown if normalized or not

### Conclusion on Weights

⚠️ **UNCERTAIN** - Need to verify:
1. Are TensorFlow checkpoint weights normalized?
2. Does c2.py denormalize before storing?
3. What format does class.py expect?

**But**: If weights were totally wrong, we wouldn't see ANY structure in outputs (we see some correlation with inputs, just wrong answers).

---

## Summary

| Component | Status | Severity |
|-----------|--------|----------|
| **ASC Scaling** | 🔴 **CRITICAL BUG** | Could cause 15-25x too much post-spike current |
| Input Encoding | ✅ Correct | N/A |
| Output Decoding | ✅ Correct | N/A |
| Weight Scaling | ⚠️ Needs verification | Medium |
| Readout Ordering | ✅ Correct (corrected from Phase 5 error) | N/A |

---

## Recommended Actions

### Immediate Priority: Fix ASC Bug

**Test 1**: Run SpiNNaker with corrected ASC scaling
```python
# class.py lines 123-124
network['glif3'][:, G.AA0] /= 1000.0  # NOT *= VSC / 1000.0
network['glif3'][:, G.AA1] /= 1000.0  # NOT *= VSC / 1000.0
```

**Expected outcome**: If ASC bug is root cause, accuracy should improve dramatically (possibly from 0% to 60-80%).

### Secondary: Verify Weight Scaling

Create diagnostic to compare:
1. TensorFlow checkpoint weights (raw format)
2. H5 file weights (stored format)
3. Expected denormalized format

---

## Confidence Assessment

**ASC Bug**: 95% confident this is real
- Clear mismatch between TensorFlow normalization and class.py denormalization
- Mathematical error is obvious (multiplying instead of dividing)
- Magnitude of error (15-25x) is sufficient to break network

**Impact**: 70% confident this is THE root cause
- ASC currents control post-spike dynamics
- 20x error would cause completely wrong firing patterns
- Could explain both "garbage" and "silence" outputs depending on ASC sign

**Caveat**: Other bugs may also exist (weights, GLIF3 implementation, etc.)
