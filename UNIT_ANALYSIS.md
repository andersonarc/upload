# Careful Unit Analysis: NEST vs SpiNNaker GLIF3

## NEST Implementation (glif_psc)

### Parameters (from h5 file)
- C_m: **pF** (picofarads)
- g: **nS** (nanosiemens)
- E_L, V_reset, V_th: **mV** (millivolts)
- asc_amps: **pA** (picoamperes)
- tau_syn: **ms** (milliseconds)

### Synaptic Current Computation
```cpp
PSCInitialValues_[i] = 1.0 * e / tau_syn_[i]  // dimensionless / ms
y1_[i] += PSCInitialValues_[i] * weight * spike
I_syn_ = Σ y2_[i]  // pA
```

**Critical**: NEST expects all currents in **pA**

### Voltage Update
```cpp
P30_ = (1 - P33_) * Tau_ / C_m_  // (1 - dimensionless) * ms / pF = ms/pF
U_ = v_old * P33_ + (I_syn + ASCurrents_sum_) * P30_
```

Dimensional check:
- `I * P30 = pA * (ms/pF) = pA*ms/pF = (10^-12 A) * (10^-3 s) / (10^-12 F)`
- `= A*s/F = coulombs / F = volts`

✅ **Units check out with currents in pA**

---

## SpiNNaker GLIF3 Implementation

### Parameter Conversions (spynnaker_newclass.py:120-124)
```python
network['glif3'][:, G.CM]  /= 1000.0  # pF -> nF
network['glif3'][:, G.G]   /= 1000.0  # nS -> uS
network['glif3'][:, G.AA0] /= 1000.0  # pA -> nA
network['glif3'][:, G.AA1] /= 1000.0  # pA -> nA
```

### C Code Parameters
- C_m: **nF** (nanofarads)
- g: **uS** (microsiemens)
- E_L, V_reset, V_th: **mV** (millivolts) - unchanged
- asc_amps: **nA** (nanoamperes)

### Voltage Update (glif3_neuron_impl.h:228-230)
```c
current_factor = (1 - v_decay) / g  // 1 / uS
g_times_EL = g * E_L                // uS * mV
V = V * v_decay + current_factor * (I_total + g_times_EL)
```

Dimensional check for `g * E_L`:
- `uS * mV = (10^-6 S) * (10^-3 V) = 10^-9 A = nA` ✅

So `I_total` must be in **nA** for dimensional consistency!

Dimensional check for voltage update:
- `current_factor * I_total = (1/uS) * nA = (1/(10^-6 S)) * (10^-9 A)`
- `= (10^6 / S) * (10^-9 A) = 10^-3 * A/S = 10^-3 * A/(A/V) = 10^-3 V = mV` ✅

**Units check out with currents in nA**

---

## Weight Scaling Analysis

### TensorFlow Training
Weights stored in h5 file are **dimensionless** (normalized by voltage_scale).

### NEST Inference (nest_glif.py:188-189)
```python
w_scaled = w_arr * vsc[tgt_arr]  # dimensionless * mV
```

For dimensional consistency with NEST expecting **pA**:
- If weights have implicit units of **pA/mV**, then:
- `w_scaled = (pA/mV) * mV = pA` ✅

### SpiNNaker Inference (spynnaker_newclass.py:856)
```python
syn[:, S.WHT] *= vsc / 1000.0  # dimensionless * mV / 1000
```

For dimensional consistency with SpiNNaker expecting **nA**:
- If weights have implicit units of **pA/mV**, then:
- `w_scaled = (pA/mV) * mV / 1000 = pA / 1000 = nA` ✅

---

## Conclusion on `/1000.0`

**The `/1000.0` factor IS CORRECT for pA → nA conversion!**

NEST uses pA → weights scaled by `vsc`
SpiNNaker uses nA → weights scaled by `vsc / 1000`

This is **not the bug**. The issue must be elsewhere.

---

## So What IS the Bug?

Since unit conversion is correct, possible issues:

1. **Synapse dynamics** - Alpha function implementation differs
2. **PSC normalization** - `e/tau` vs peak-at-tau
3. **Timing** - How/when synapses update vs neuron updates
4. **Background weights** - Still suspicious (has *10 and /1000)
5. **Initial conditions** - Different starting states
6. **Receptor type mapping** - Wrong synapse indices

Need to investigate these systematically.
