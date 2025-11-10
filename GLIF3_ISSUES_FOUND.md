# GLIF3 Implementation Issues Found

## Summary
Network produces incorrect output on SpiNNaker. Two manifestations observed:
- **Before recent changes (f65861a)**: Garbage output (non-zero but incorrect)
- **After recent changes (443e46a)**: Complete silence (zero spikes)

## Issues Identified

### 1. Alpha Synapse psc Timing Bug
**Status**: Fixed in commit 87b53b5 (but may have introduced new issues)

**Problem**: The psc calculation used psc_rise values AFTER spikes were added, but TensorFlow uses values from BEFORE spikes are added.

**TensorFlow** (line 318-319):
```python
new_psc_rise = syn_decay * psc_rise + rec_inputs * psc_initial
new_psc = psc * syn_decay + dt * syn_decay * psc_rise  # Uses OLD psc_rise
```

**Original SpiNNaker code**:
```c
exp_shaping(&p->syn_0_rise);  // Decays psc_rise
p->syn_0.synaptic_input_value = decay(...) + decay(psc_rise);  // Uses decayed value
```

**Execution order**:
1. neuron_transfer() adds spikes to psc_rise
2. shape_input() reads psc_rise (already has current spikes added) ❌

**Fix Applied**: Added psc_rise_prev fields to store previous timestep values.

### 2. Sub-Timestep Handling Bugs
**Status**: Fixed in commits 412fa17 and 443e46a

**Bug 2a**: psc_rise_prev updated every sub-step instead of once per full timestep
**Bug 2b**: Used `dt` instead of `ts` in psc calculation, multiplying contribution by N

### 3. Spike Input Scaling Bug
**Status**: Attempted fix in commit 412fa17, but may have type errors

**Problem**: TensorFlow uses `psc_initial = e/tau`, and spikes are added AFTER decay. But in SpiNNaker, spikes are added BEFORE decay, causing them to be weakened by one timestep's worth of decay.

**Attempted Fix** (line 57-60):
```c
REAL e_approx = expulr(ONE);
REAL dt_over_tau = kdivk(dt, params->tau);
REAL compensation = expulr(dt_over_tau);  // exp(dt/tau)
decay_t init = kdivk(e_approx, params->tau) * compensation;  // ← TYPE ERROR?
```

**Potential Issue**: Direct multiplication `REAL * decay_t` may not work correctly in fixed-point arithmetic. Standard SpiNNaker code uses `decay_s1615_to_u032()` for this operation.

### 4. Struct Format Changes
**Working version (f65861a)**:
- 9 fields: (tau, init) × 4 + timestep
- Used standard `exp_params_t`

**Current version (443e46a)**:
- 13 fields: (tau, init_rise, init_main) × 4 + timestep
- Custom `double_exp_params_t`
- Added 6 runtime state fields (dt, 4× psc_rise_prev, 2× counters)

This allows proper state persistence for psc_rise, which was missing before.

## Recommendations

1. **Fix type error on line 60**: Use `decay_s1615_to_u032()` instead of direct multiplication
2. **Verify neuron dynamics**: Ensure voltage integration, ASC updates, and refractory handling are correct
3. **Test systematically**: Start with simplest case (single neuron, single spike) and verify each component

## Files Modified
- `glif3/synapse_types_glif3_impl.h`: Synapse dynamics
- `glif3/glif3_synapse_type.py`: Python struct definition
- `glif3/glif3_neuron_impl.h`: Neuron dynamics (separate fixes)
