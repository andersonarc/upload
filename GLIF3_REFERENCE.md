# GLIF3 Implementation Reference

## 1. TensorFlow Reference Code (Ground Truth)

From `training_code/models.py` lines 318-341:

```python
# Line 318-319: Alpha synapse dynamics
new_psc_rise = syn_decay * psc_rise + rec_inputs * psc_initial
new_psc = psc * syn_decay + self._dt * self.syn_decay * psc_rise  # Uses OLD psc_rise

# Line 321: Refractory period
new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)

# Line 325-326: After-spike currents
new_asc_1 = tf.exp(-self._dt * k[:, 0]) * asc_1 + prev_z * asc_amps[:, 0]
new_asc_2 = tf.exp(-self._dt * k[:, 1]) * asc_2 + prev_z * asc_amps[:, 1]

# Line 328-334: Voltage update
reset_current = prev_z * (self.v_reset - self.v_th)
input_current = tf.reduce_sum(psc, -1)
decayed_v = self.decay * v
gathered_g = self.param_g * self.e_l
c1 = input_current + asc_1 + asc_2 + gathered_g
new_v = decayed_v + self.current_factor * c1 + reset_current

# Line 339-341: Spike generation and refractory masking
new_z = spike_gauss(v_sc, ...)
new_z = tf.where(new_r > 0., tf.zeros_like(new_z), new_z)  # No spikes during refractory
```

### Key Initialization Values (line 157-174):

```python
self._dt = dt  # Full simulation timestep (1.0 ms)

# Line 171-172: Voltage decay
tau = self._params['C_m'] / self._params['g']
self._decay = tf.exp(-self._dt / tau)
self._current_factor = 1 / self._params['C_m'] * (1 - self._decay) * tau

# Line 174: Synapse initialization (DIFFERENT from standard SpiNNaker!)
self._psc_initial = np.e / np.array(self._params['tau_syn'])  # e/tau, NOT tau*(1-exp(-dt/tau))
```

## 2. GLIF3 Continuous-Time Equations

From Allen Institute GLIF model specification:

### Voltage dynamics:
```
dV/dt = (1/C_m) * [I_syn + I_asc_1 + I_asc_2 - g*(V - E_L)]
```

### After-spike currents:
```
dI_asc_j/dt = -k_j * I_asc_j,  j=1,2
```

### Alpha synapses (double exponential):
```
d(psc_rise)/dt = -psc_rise/tau_syn + spike_inputs(t) * psc_initial
d(psc)/dt = -psc/tau_syn + psc_rise
```

### Spike and reset:
```
If V >= V_th:
  - Spike occurs
  - V -> V (voltage NOT reset to V_reset!)
  - At next timestep: V += (V_reset - V_th)  [soft reset via reset_current]
  - I_asc_j += asc_amp_j  [at next timestep]
  - Refractory timer starts
```

## 3. SpiNNaker Sub-Timestep Integration

### Purpose of sub-timesteps:
Sub-timesteps provide more accurate numerical integration of ODEs. With `n_steps_per_timestep=2`:
- Full timestep: 1.0 ms
- Sub-timestep duration: 0.5 ms
- State updates happen twice per full timestep

### Execution order per full timestep:
```
1. neuron_transfer() - called ONCE per full timestep
   - Adds spikes from ring buffer to synapse state
   - Runs BEFORE sub-step loop

2. for i_step in n_steps_per_timestep down to 1:
   - Read synapse state (psc values)
   - Update neuron voltage
   - Check for spike
   - Update synapse state (decay)
```

### Decay computation with sub-steps:
For sub-steps, decay is computed for the SUB-TIMESTEP duration:
```c
ts = time_step_ms / n_steps_per_timestep  // e.g., 1.0/2 = 0.5 ms
decay = exp(-ts / tau)                     // e.g., exp(-0.5/5) = 0.905
```

Total decay over full timestep: `decay^n = 0.905^2 = 0.819 ≈ exp(-1/5) = 0.8187` ✓

## 4. Critical Timing Issues

### Issue 1: Spike timing in synapses

**TensorFlow semantics:**
```
Timestep T:
  new_psc = decay*psc + dt*decay*OLD_psc_rise  // psc_rise from BEFORE inputs added
  new_psc_rise = decay*psc_rise + inputs       // inputs added AFTER psc calculated
```

**SpiNNaker with sub-steps:**
```
Timestep T:
  neuron_transfer: psc_rise += inputs  // Happens BEFORE sub-step loop!
  Sub-step 1: psc uses psc_rise (ALREADY has inputs!) ❌
```

**Solution:**
Save `psc_rise_prev` at END of previous full timestep, use that value in current timestep.

### Issue 2: Spike reset timing in neurons

**TensorFlow semantics:**
```
Timestep T:
  Spike occurs, set prev_z=1
Timestep T+1:
  reset_current = prev_z * (v_reset - v_th)  // Uses spike from PREVIOUS timestep
  new_asc = ... + prev_z * asc_amp           // Uses spike from PREVIOUS timestep
```

**SpiNNaker with sub-steps:**
```
Timestep T, Sub-step 1:
  Spike occurs, set spiked_last_step=1
Timestep T, Sub-step 2:
  Apply reset_current (spiked_last_step=1)  ❌ // Should wait until next FULL timestep!
```

**Solution:**
Only apply reset/ASC updates on FIRST sub-step of NEXT full timestep, not same timestep.

## 5. Correct Implementation Strategy

### Synapse (synapse_types_shape_input):
1. Track sub-steps with counter (counts down from n to 1)
2. On LAST sub-step (counter==1): save psc_rise_prev = psc_rise
3. Calculate: `psc = decay(psc) + decay(psc_rise_prev)`  // Use OLD value
4. Then decay: `psc_rise = decay(psc_rise)`

### Neuron (neuron_model_state_update):
1. Track sub-steps with counter (counts down from n to 1)
2. Detect first sub-step: `is_first = (counter == n)` BEFORE decrementing
3. Only apply reset/ASC if `spiked_last_step && is_first`
4. This ensures reset happens in NEXT full timestep, not current one

## 6. Open Questions

1. Should psc contribution include `dt` factor?
   - TensorFlow line 319: `dt * decay * psc_rise`
   - With sub-steps, adding `dt` per sub-step causes N× multiplication
   - Current solution: NO dt factor (matches working commit f65861a)

2. Exact spike input normalization:
   - TensorFlow uses: `e / tau`
   - Standard SpiNNaker uses: `tau * (1 - exp(-dt/tau))`
   - Current: Using TensorFlow's `e / tau` (commit eebac81)
