# NEST GLIF3 Inference Progress

## Current Status
- **Accuracy**: 40% (2/5 correct on mnist.h5)
- **Implementation**: GLIF3 with proper spike sampling and weight scaling
- **Issues**: Very few output spikes in target 50-100ms window (0-4 spikes vs 5-27 in 50-200ms)

## Recent Fixes
1. ✅ Fixed spike probability sampling (was treating all probabilities as deterministic spikes)
2. ✅ Fixed weight unit conversion (NEST uses pA, PyNN SpiNNaker uses nA)
3. ✅ Weight scaling: `w_scaled = w_arr * vsc[tgt_arr]` (NOT /1000)
4. ✅ Found critical unit mismatch between NEST and SpiNNaker GLIF models

## Key Discoveries
- Spike trains contain PROBABILITIES (0-0.135), not binary spikes
- Must sample using: `if np.clip(spike_trains[t,i]/1.3, 0, 1) > np.random.rand()`
- Sampling reduces spikes from 8.6M to 74K (0.9%)
- Background weights NOW AVAILABLE in updated checkpoint

## New Checkpoint Info
- File: `ckpt_51978-153_NEW.h5`
- Background weights: shape (51978, 4), range [0, 2.7], mean 1.02
- 51978 non-zero values (1 per neuron, appears to be in first receptor type column)

## Next Steps (from user guidance)
1. ✅ Update repo and desplit new checkpoint with background weights
2. ✅ Desplit mnist24.h5 (24 samples for testing)
3. ⏳ Examine training_code (multi_training.py, models.py) for proper implementation
4. ⏳ Implement background weights like TensorFlow reference
5. ⏳ Compare spike generation with c2.py/mnist.py
6. ⏳ Add diagnostic output (voltages, spike stats)
7. ⏳ Consider normalized voltage implementation if needed
8. ⏳ Achieve 80% accuracy on mnist24.h5

## Current Implementation Details

### Spike Sampling (nest_glif.py lines 60-68)
```python
for neuron_idx in range(sample_spikes.shape[1]):
    times = []
    for t_idx in range(sample_spikes.shape[0]):
        prob = np.clip((sample_spikes[t_idx, neuron_idx] / 1.3), 0.0, 1.0)
        if prob > np.random.rand():
            times.append(float(t_idx + 1.0))
    if len(times) > 0:
        spike_times[neuron_idx] = times
```

### Weight Scaling (nest_glif.py lines 137-141)
```python
# Calculate voltage scale for each target neuron
vsc = np.array([glif_params['V_th'][neurons[i]] - glif_params['E_L'][neurons[i]] for i in range(len(neurons))])

# Scale weights by voltage scale (NOT /1000 - NEST uses pA, SpiNNaker uses nA)
w_scaled = w_arr * vsc[tgt_arr]
```

### GLIF3 Parameters (nest_glif.py lines 96-110)
```python
params = {
    'C_m': C_m,
    'E_L': E_L,
    'V_reset': V_reset,
    'V_th': V_th,
    'V_m': E_L,  # Initial voltage = leak potential
    'g': g,
    't_ref': t_ref,
    'tau_syn': tau_syn,  # 4 receptor types
    'asc_amps': asc_amps,  # 2 after-spike currents
    'asc_decay': asc_decay,  # k parameters
    'after_spike_currents': True,  # Enable GLIF3 ASC
    'spike_dependent_threshold': False,
    'adapting_threshold': False
}
```

## Unit Mismatches Discovered

| Parameter | NEST (glif_psc) | PyNN SpiNNaker | Conversion |
|-----------|-----------------|----------------|------------|
| C_m | pF | nF | /1000 in PyNN |
| g | nS | uS | /1000 in PyNN |
| asc_amps | pA | nA | /1000 in PyNN |
| **weights** | **pA** | **nA** | **vsc/1000 in PyNN → vsc in NEST** |

## Files
- Main implementation: `new_things/nest_glif.py`
- Test script: `new_things/run_all_glif.sh`
- Old checkpoint: `new_things/ckpt_51978-153.h5` (290MB, no background weights)
- **New checkpoint**: `new_things/ckpt_51978-153_NEW.h5` (290MB, WITH background weights)
- Old samples: `new_things/mnist.h5` (5 samples)
- **New samples**: `new_things/mnist24.h5` (24 samples)

## Sample Results (nest_glif.py on mnist.h5)
```
Sample 0 (label 6): Predicted 5 (50-200ms) / 4 (50-100ms) - WRONG
Sample 1 (label 4): Predicted 2 (50-200ms) / 4 (50-100ms) - CORRECT in target window
Sample 2 (label 4): Predicted 2 (50-200ms) / 0 (50-100ms) - WRONG
Sample 3 (label 4): Predicted 4 (50-200ms) / 4 (50-100ms) - CORRECT
Sample 4 (label 9): Predicted 4 (50-200ms) / 4 (50-100ms) - WRONG

Accuracy: 40% (2/5 in 50-200ms window)
```

## Observations
- 50-200ms window: 5-27 spikes per class (good activity)
- 50-100ms window: 0-4 spikes per class (too few!)
- Network seems biased toward class 4 in target window
- Need background weights to provide baseline activity

## Reference: TensorFlow Training Code
User says TensorFlow (training_code/) is the ground truth reference, not SpiNNaker.
- `multi_training.py` - training script
- `models.py` - model definitions
- `c2.py` - checkpoint conversion
- `mnist.py` - spike generation

Background weights implementation in models.py:269-271:
```python
bkg_weights = bkg_weights / np.repeat(voltage_scale[self._node_type_ids], self._n_receptors)
self.bkg_weights = tf.Variable(bkg_weights * 10., name='rest_of_brain_weights', trainable=train_bkg)
```
