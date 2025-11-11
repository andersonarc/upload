# Implementation Comparison: TensorFlow Training vs NEST vs SpiNNaker

## Date: 2025-11-11

## Overview

This document compares three implementations of the mouse V1 cortex GLIF3 model:
1. **TensorFlow** (ground truth - used for training)
2. **NEST Simulator** (nest_glif.py - current implementation)
3. **SpiNNaker** (newclass.py - existing implementation)

## CRITICAL FINDINGS

### 1. **SYNAPSE TYPE MISMATCH** ⚠️

**TensorFlow (Training Code)**:
- Uses **ALPHA (double-exponential) SYNAPSES**
- Two state variables per synapse: `psc_rise` and `psc`
- models.py:318-319:
  ```python
  new_psc_rise = self.syn_decay * psc_rise + rec_inputs * self.psc_initial
  new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
  ```
- models.py:174: `self._psc_initial = np.e / np.array(self._params['tau_syn'])`

**NEST glif_psc**:
- Uses **SIMPLE EXPONENTIAL PSCs**
- Single state variable per synapse
- Standard exponential decay: `I_syn(t+dt) = I_syn(t) * exp(-dt/tau) + weight * spike`

**SpiNNaker (Current)**:
- Uses **SIMPLE EXPONENTIAL SYNAPSES** (after revert in commit 909a17b)
- glif3/synapse_types_glif3_impl.h line 66-70:
  ```c
  static inline void synapse_types_shape_input(synapse_types_t *parameters) {
      exp_shaping(&parameters->syn_0);
      exp_shaping(&parameters->syn_1);
      // ...
  }
  ```

**SpiNNaker (Old - commit cac849a)**:
- Previously implemented **ALPHA SYNAPSES** (matching TensorFlow!)
- Two state variables: `syn_X_rise` and `syn_X`
- Was reverted to simple exponential (commit 909a17b: "Revert to simple exponential synapses")

**CONCLUSION**: TensorFlow training uses alpha synapses, but both NEST and SpiNNaker (current) use simple exponential. This is a **fundamental mismatch** that could explain discrepancies.

---

### 2. **WEIGHT UNITS: nA vs pA** ⚠️

**TensorFlow**:
- Stores weights NORMALIZED (divided by voltage_scale)
- Units: dimensionless (normalized by mV)

**NEST**:
- Weights in **pA** (picoamps)
- Denormalization: `w_nest = w_tf * voltage_scale` (pA)

**SpiNNaker**:
- Weights in **nA** (nanoamps)
- newclass.py:848: `syn[:, S.WHT] *= vsc / 1000.0` (converts pA → nA)
- newclass.py:895: `syn[:, S.WHT] *= vsc / 1000.0` (LGN weights also /1000)

**CONCLUSION**: SpiNNaker divides by 1000 (pA→nA conversion), NEST does not. This is correct - NEST uses pA, SpiNNaker uses nA. My NEST implementation correctly uses pA (no /1000).

---

### 3. **BACKGROUND WEIGHTS IMPLEMENTATION**

#### TensorFlow (Ground Truth)
- models.py:95-98:
  ```python
  rest_of_brain = tf.reduce_sum(tf.cast(
      tf.random.uniform((shp[0], shp[1], 10)) < .1, self._compute_dtype), -1)
  noise_input = self._bkg_weights[None, None] * rest_of_brain[..., None] / 10.
  ```
- **10 independent Bernoulli(0.1) sources**, sum = 0-10
- Divided by 10 → mean 0.1 per source
- Storage (models.py:271): `self.bkg_weights = tf.Variable(bkg_weights * 10., ...)`

#### NEST (My Implementation) ✅
- nest_glif.py:131-170
- **10 independent Poisson(10 Hz) generators**
- Each with weight `(bkg_weights / 10) * vsc`
- Matches TensorFlow statistics (mean AND variance)

#### SpiNNaker (newclass.py) ❌
- newclass.py:911-979: `create_background(..., rate=100.0)`
- **SINGLE Poisson(100 Hz) source** per neuron
- Weights: `bkg_w_norm * 10.0 / 1000.0` (line 953)
- **WRONG**: Single source has DIFFERENT variance than sum of 10 sources!

**CONCLUSION**: My NEST implementation is correct (10 sources). SpiNNaker implementation uses single source, which has wrong variance.

---

### 4. **GLIF3 NEURON MODEL**

All three implementations use equivalent GLIF3 dynamics:

**Membrane voltage**:
```
dV/dt = (1/C_m) * [I_total - g*(V - E_L)]
```

**After-spike currents**:
```
dI_asc_j/dt = -k_j * I_asc_j
On spike: I_asc_j → I_asc_j * exp(-k_j * t_ref) + asc_amp_j
```

**Key differences**:
- NEST: Uses `asc_decay` parameter (= k)
- SpiNNaker C code (glif3_neuron_impl.h:152-153): Explicit exponential decay
- TensorFlow (models.py:325-326): `exp(-dt * k) * asc + spike * asc_amps`

**CONCLUSION**: Neuron models are equivalent across implementations.

---

## SUMMARY TABLE

| Feature                | TensorFlow (Training)      | NEST (My Impl)            | SpiNNaker (newclass.py)   |
|------------------------|----------------------------|---------------------------|---------------------------|
| **Synapse Type**       | Alpha (double-exp) ⚠️      | Simple exponential ⚠️     | Simple exponential ⚠️     |
| **Weight Units**       | Normalized                 | pA ✅                     | nA ✅                     |
| **Background Model**   | 10 Bernoulli(0.1) ✅       | 10 Poisson(10Hz) ✅       | 1 Poisson(100Hz) ❌       |
| **Background Weights** | `*10` storage, `/10` apply | `/10` per source ✅       | `*10 /1000` single ❌     |
| **GLIF3 Dynamics**     | Correct ✅                 | Correct ✅                | Correct ✅                |
| **Spike Input**        | Stochastic sampling ✅     | Stochastic sampling ✅    | Stochastic sampling ✅    |
| **Delays**             | Sparse (rounded to dt) ✅  | Per-synapse ✅            | Per-synapse ✅            |

---

## RECOMMENDATIONS

### Immediate Actions

1. **Test NEST performance**: Determine if simple exponential synapses are "close enough" to alpha synapses
   - If accuracy is good (>80%): Alpha synapses may not be critical
   - If accuracy is poor: Need to implement alpha synapses in NEST

2. **Fix SpiNNaker background model**:
   - Change from 1×Poisson(100Hz) to 10×Poisson(10Hz)
   - Update weight calculation to match NEST implementation

3. **Consider alpha synapse implementation**:
   - Option A: Use NEST's built-in alpha synapse models
   - Option B: Implement custom alpha PSC in glif_psc model
   - Option C: Accept reduced accuracy with simple exponential

### Long-term Investigations

1. **Quantify synapse type impact**:
   - Run TensorFlow inference with simple exponential synapses
   - Compare accuracy: alpha vs simple exponential

2. **Validate weight units thoroughly**:
   - Verify pA vs nA conversion is correct
   - Check that all weight paths are consistent

3. **Compare with TensorFlow inference**:
   - Run TensorFlow model inference on same test samples
   - Establish ground truth accuracy for comparison

---

## FILES ANALYZED

### TensorFlow Training Code
- `/home/user/upload/training_code/models.py` (lines 173-174, 279-280, 318-319)
- `/home/user/upload/training_code/mnist.py` (spike generation)

### NEST Implementation
- `/home/user/upload/new_things/nest_glif.py` (main implementation)
- NEST glif_psc model (built-in)

### SpiNNaker Implementation
- `/home/user/upload/new_things/newclass.py` (main Python interface)
- `/home/user/upload/glif3/glif3_neuron_impl.h` (C neuron model)
- `/home/user/upload/glif3/synapse_types_glif3_impl.h` (C synapse model)
- Git commit cac849a (old alpha synapse implementation)
- Git commit 909a17b (revert to simple exponential)

---

## NEXT STEPS

1. ✅ Complete test on 10 diverse samples from mnist24.h5
2. ⏳ Analyze test results and determine accuracy
3. ⏳ Cross-check SpiNNaker implementation (if NEST works)
4. ⏳ Create visualizations of network activity
5. ⏳ Consider alpha synapse implementation if needed

---

## NOTES

- **Alpha vs Exponential**: The synapse type mismatch is likely the MOST CRITICAL difference
- **Background model**: NEST implementation is correct, SpiNNaker needs fixing
- **Weight units**: Both implementations handle units correctly (pA vs nA)
- **TensorFlow ground truth**: Should be used as reference for all comparisons
