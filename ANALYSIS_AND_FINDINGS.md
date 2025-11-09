# Mouse V1 Cortex SpiNNaker Simulation - Debug Analysis

## Executive Summary

The network is failing to produce meaningful spike outputs on SpiNNaker despite being trained to ~0.8 accuracy in TensorFlow. Analysis reveals multiple critical bugs and dimensional mismatches between the training (TensorFlow) and inference (PyNN/SpiNNaker) pipelines.

## Architecture Overview

```
MNIST → LGN Encoding → Input Spikes → V1 (51,978 GLIF3 neurons) → Readout (300 neurons, 30/class)
         (17,400 neurons)            (786K synapses)              (14.4M recurrent synapses)
```

## Critical Issues Identified

### Issue 1: **WEIGHT SCALING BUG** ⚠️ CRITICAL

**Location:** `jupyter/spinnaker.ipynb`, functions `create_V1()` and `create_LGN()`

**Problem:**
```python
# Current code (WRONG)
vsc = network['glif3'][int(tgt_key[0]), G.VSC]
syn[:, S.WHT] *= vsc / 1000.0
```

**Analysis:**
- In TensorFlow (`models.py:227, 235`): weights are **divided** by `voltage_scale` for normalization
- In H5 file: weights are stored in normalized form (already divided by voltage_scale)
- In PyNN: weights are being **multiplied** by voltage_scale, which **REVERSES** the TensorFlow normalization
- Then divided by 1000, resulting in net effect: `weights / 1000`

**Impact:**
- This is mathematically backwards from TensorFlow
- The `/1000` conversion appears to be a units fix (pA → nA), but the multiplication by voltage_scale undoes the TensorFlow normalization incorrectly

**Expected behavior:**
In TensorFlow, the weight normalization ensures that weights are dimensionless and voltage-independent. When converting to PyNN, we need to account for:
1. PyNN uses absolute voltage units (mV) not normalized voltages
2. PyNN weights should represent synaptic current (nA)

However, the current implementation may be introducing incorrect scaling.

**Recommendation:**
- Verify what units the H5 weights are actually in
- Check if weights should be multiplied or divided by voltage_scale
- Test with weights both with and without the voltage_scale multiplication
- Consider that different weight scaling might be needed for SpiNNaker's fixed-point arithmetic

---

### Issue 2: **INPUT MODE MISMATCH** ⚠️ CRITICAL

**Location:** Training vs Inference spike generation

**Problem:**

Training command:
```bash
python multi_training.py --nocurrent_input
```

This sets `current_input=False`, which means (`stim_dataset.py:344-345`):
```python
else:
    _z = tf.cast(tf.random.uniform(tf.shape(_p)) < _p, tf.float32)  # BINARY SPIKES (0 or 1)
```

But `mnist.py:60-61` (used for generating test spikes) ALWAYS uses:
```python
_p = 1 - tf.exp(-firing_rates / 1000.)
_z = _p * 1.3  # CONTINUOUS VALUES (probabilities * 1.3)
```

**Analysis:**
- **Training:** Network was trained with **binary spikes** (0 or 1) sampled from firing rate probabilities
- **Inference (mnist.py):** Generates **continuous probability values** (* 1.3)
- **Inference (notebook):** Correctly divides by 1.3 and samples to get binary spikes

```python
# jupyter/spinnaker.ipynb create_spike_times()
if np.clip((spike_trains[t, i] / 1.3) * scale, 0.0, 1.0) > np.random.rand():
```

**Status:** The notebook appears to correctly handle this by dividing by 1.3 and sampling. However, verify that the statistical properties match training.

**Recommendation:**
- The notebook correction looks correct
- But verify that the spike statistics (rate, temporal patterns) match what the network saw during training
- Consider using the exact same spike generation code as training

---

### Issue 3: **SYNAPSE TYPE / RECEPTOR MAPPING**

**Location:** GLIF3 C implementation vs PyNN projections

**Finding:**
- C code (`synapse_types_glif3_impl.h:38-39`):
  ```c
  #define NUM_EXCITATORY_RECEPTORS 4
  #define NUM_INHIBITORY_RECEPTORS 0
  ```
- **ALL 4 receptor types are excitatory**
- Inhibition is achieved through **negative weights** in excitatory receptors

**Implication:**
- The notebook correctly uses: `receptor_type = f'synapse_{int(syn[0, S.RTY])}'`
- With receptor types 0, 1, 2, 3
- Negative weights ARE present in the data (verified: 3.17M negative of 14.4M recurrent)

**Status:** ✓ This appears correct

---

### Issue 4: **RESPONSE WINDOW UNCERTAINTY**

**Location:** Classification readout timing

**Problem:**
- Notebook uses response window `[50-100 ms]` for vote counting (or `[50-200 ms]`)
- Training parameters: `--pre_delay=50 --im_slice=100 --post_delay=450`
- H5 file specifies: `response_window = [50, 100]`

**Analysis:**
- Image presentation: 50-150 ms
- Current response window: 50-100 ms (first 50ms of image)
- This might be too early - network may need time to integrate

**Recommendation:**
- Test with response window `[100-200 ms]` (during image presentation)
- Or `[50-250 ms]` (image + some decay time)

---

### Issue 5: **PARAMETER UNIT CONVERSIONS**

**Location:** `jupyter/spinnaker.ipynb` network loading

**Current conversions:**
```python
network['glif3'][:, G.CM]  /= 1000.0 # pF -> nF ✓
network['glif3'][:, G.G]   /= 1000.0 # nS -> uS ✓
network['glif3'][:, G.AA0] /= 1000.0 # pA -> nA ✓
network['glif3'][:, G.AA1] /= 1000.0 # pA -> nA ✓
```

**Missing conversion?**
- Synaptic weights: Currently `* vsc / 1000.0`
- May need different scaling

**Verify units:**
- C code expects: currents in nA, voltages in mV, conductances in uS
- Weights should produce nA when a spike arrives

---

### Issue 6: **GLIF3 vs IF_curr_exp CONVERSION**

**Location:** Commented code in notebook trying IF_curr_exp

**Finding:**
There's commented code that converts GLIF3 to IF_curr_exp:
```python
def glif32ice(network):
    glif3 = network['glif3']
    network['ice'] = np.stack([
        glif3[:, G.CM],                 # C_m
        glif3[:, G.CM] / glif3[:, G.G], # tau_m
        glif3[:, G.EL],                 # V_rest
        glif3[:, G.RST],                # V_reset
        glif3[:, G.THR],                # V_thresh
        glif3[:, G.RFR],                # tau_refrac
        glif3[:, G.TA0],                # tau_syn_E
        glif3[:, G.TA2],                # tau_syn_I
        ...
```

**Issue:** This maps tau_syn0 → tau_syn_E and tau_syn2 → tau_syn_I, but GLIF3 has 4 independent receptors, not just E/I

---

### Issue 7: **INSUFFICIENT V1 ACTIVITY**

**Observed:**
```
LGN 0: 8 neurons, 42 total spikes  ✓ LGN is spiking
LGN 1: 8 neurons, 17 total spikes
...
Class 0-9: 0 total spikes (all time)  ✗ V1 is NOT spiking
```

**Possible causes:**
1. **Weights too small** (Issue #1)
2. **Wrong weight signs**
3. **Thresholds too high / resting potentials wrong**
4. **Synaptic conductances not working**
5. **Spikes not arriving from LGN**

---

## Recommended Debug Steps

### Priority 1: Fix Weight Scaling

Test three scenarios:
```python
# Scenario A: No voltage_scale multiplication
syn[:, S.WHT] /= 1000.0

# Scenario B: Current approach
syn[:, S.WHT] *= vsc / 1000.0

# Scenario C: Divide by voltage_scale (match TF)
syn[:, S.WHT] /= (vsc * 1000.0)
```

### Priority 2: Add Detailed Logging

Add logging to verify:
1. LGN spikes are arriving at V1 neurons
2. V1 neuron voltages are changing
3. Synaptic currents are non-zero
4. Weight values are reasonable

```python
# Log first V1 population voltage
if V1[key] is not None:
    v = V1[key].get_data('v').segments[0].analogsignals[0]
    print(f"V1[{i}] voltage range: {v.min():.2f} to {v.max():.2f} mV")
    print(f"V1[{i}] threshold: {network['glif3'][pid, G.THR]:.2f} mV")
```

### Priority 3: Simplify for Testing

Create a minimal test case:
- Single LGN neuron → Single V1 neuron
- Known spike times
- Verify V1 responds

### Priority 4: Compare TensorFlow Inference

Run the same input through TensorFlow model and compare:
- Spike counts per population
- Voltage traces
- Response timing

---

## Unit Reference

| Parameter | TensorFlow | H5 File | PyNN/SpiNNaker | C Code |
|-----------|-----------|---------|----------------|--------|
| Voltage | Normalized | mV | mV | mV |
| Current | Normalized | ? | nA | nA |
| Capacitance | pF | pF | nF (÷1000) | nF |
| Conductance | nS | nS | uS (÷1000) | uS |
| Weight | Normalized | Normalized? | nA? | nA |

---

## Files Requiring Changes

1. `jupyter/spinnaker.ipynb` - weight scaling in create_V1() and create_LGN()
2. `training_code/mnist.py` - spike generation mode flag (for consistency)
3. Potentially: new logging/debug version of notebook

---

## Questions to Answer

1. **What are the actual units of weights in the H5 file?**
   - Are they normalized (dimensionless)?
   - Are they in pA, nA, or something else?

2. **What should the weight conversion be?**
   - Multiplication or division by voltage_scale?
   - What about the /1000 factor?

3. **Are the trained weights actually being loaded?**
   - Check c2.py output logs
   - Verify checkpoint loading succeeded

4. **Is the GLIF3 C implementation correct?**
   - Test with simple inputs
   - Verify voltage dynamics

