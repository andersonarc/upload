# NEST GLIF3 Implementation Status

## Date: 2025-11-11

## Implementation Details - VERIFIED CORRECT

### 1. Background Model ✅
- **TensorFlow**: `sum(10 Bernoulli(0.1)) / 10 * bkg_weights`
- **NEST**: 10 independent Poisson(10Hz) generators, each with `bkg_weights/10 * vsc`
- **Result**: Matches TensorFlow statistics (mean and variance)

### 2. Weight Normalization ✅
- **TensorFlow**: Normalizes all weights by dividing by `voltage_scale` (V_th - E_L)
- **Saved in h5**: Normalized weights (after `/vsc`) with `*10` factor for background
- **NEST**: Denormalizes by multiplying by `vsc` (since NEST uses actual mV, not normalized)
  - Input weights: `w_h5 * vsc`
  - Recurrent weights: `w_h5 * vsc`
  - Background weights: `(w_h5 / 10) * vsc` (the `/10` accounts for TensorFlow's `*10` storage)

### 3. Input Representation ✅
- **Mode**: `current_input=False` → Spike-based input (NOT continuous currents)
- **TensorFlow**: Uses discrete spike events sampled from probabilities
- **NEST**: Stochastic sampling: `if clip(spike_trains[t,i]/1.3, 0,1) > rand()`
- **Result**: Both use event-based coding with variance

### 4. Timing Parameters ✅
- **Stimulus**: 50-150ms (100ms duration)
- **Target response window**: 50-100ms (first half of stimulus)
- **Pre-delay**: 50ms blank
- **Post-delay**: 450ms blank
- **Total sequence**: 600ms

## Current Performance

### mnist.h5 (5 samples):
- **50-200ms**: 80% (4/5 correct)
- **50-150ms**: Variable
- **50-100ms**: 20% (1/5 correct) ❌ TARGET WINDOW

### Observed Issues:
1. **Low spike counts in 50-100ms**: Only 30-60 total votes across all 10 classes
2. **Nearly uniform distribution**: Margin between 1st and 2nd place often <10 votes
3. **Network responds slowly**: Signal emerges in 50-200ms but not 50-100ms

## Open Questions / Next Steps

1. **Test on diverse samples** (mnist24.h5): Running now
2. **Analyze error patterns**: Are mistakes sensible (confusing similar digits)?
3. **Check for biases**: Any hyperactive output classes?
4. **Compare with TensorFlow**: Need to verify TensorFlow achieves >80% on same samples
5. **Investigate slow response**: Why does network need 100-200ms to respond clearly?

## Files
- Main implementation: `new_things/nest_glif.py`
- Checkpoint: `new_things/ckpt_51978-153_NEW.h5` (with background weights)
- Test data: `new_things/mnist24.h5` (24 samples, diverse classes)
- Training reference: `training_code/` (TensorFlow ground truth)
