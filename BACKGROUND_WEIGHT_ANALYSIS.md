# Background Weight Analysis

## TensorFlow Training (models.py)

### Storage (line 275-277)
```python
bkg_weights = bkg_weights / np.repeat(voltage_scale[node_type_ids], n_receptors)  # Normalize
self.bkg_weights = tf.Variable(bkg_weights * 10., ...)  # Scale by 10
```

**Stored in h5**: `bkg_stored = (W_original / voltage_scale) * 10`

### Inference (line 101-104)
```python
rest_of_brain = tf.reduce_sum(tf.random.uniform((batch, time, 10)) < .1, dtype), -1)
noise_input = bkg_weights[None, None] * rest_of_brain[..., None] / 10.
```

- `rest_of_brain` = sum of 10 Bernoulli(0.1) per timestep
- Expected value = 10 × 0.1 = **1.0** per timestep
- `noise_input` expected = `bkg_stored * 1.0 / 10 = [(W_orig / vsc) * 10] / 10 = W_orig / vsc`

After denormalization in neuron model, this becomes **W_orig** in pA.

---

## NEST Implementation (nest_glif.py)

### Setup (line 132-134)
```python
bkg_generators = nest.Create('poisson_generator', 10)  # 10 separate generators
for gen in bkg_generators:
    gen.set({'rate': 10.0, ...})  # Each at 10 Hz
```

### Weights (line 154-155)
```python
w_filt = weights[mask] / 10.0  # bkg_stored / 10 = (W_orig / vsc) * 10 / 10 = W_orig / vsc
w_scaled = w_filt * vsc  # W_orig in pA
```

### Connection (line 160-167)
```python
# Connect EACH of the 10 generators to each neuron with weight w_scaled
for gen in bkg_generators:
    nest.Connect(..., syn_spec={'weight': w_scaled})
```

**Analysis:**
- Each generator: 10 Hz = 0.01 spikes/ms expected
- 10 generators = 0.1 spikes/ms total expected
- Weight per spike: **W_orig** pA
- Expected input per ms: 0.1 × W_orig = **0.1 × W_orig** pA per ms

Wait, that doesn't match TensorFlow...

Actually, let me reconsider the timestep interpretation:
- TensorFlow: 1 ms timesteps, expects ~1 spike per timestep
- NEST: continuous time, 10 generators × 10 Hz = 100 Hz total = 0.1 spikes per ms

Oh! The units are different!
- TensorFlow: **per timestep** (1 ms)
- NEST: **per millisecond** (continuous)

So NEST at 100 Hz total gives 0.1 spikes per ms on average.
TensorFlow expects 1.0 spikes per timestep (= per ms).

There's a **10× mismatch**!

### Correction for NEST
If TensorFlow expects 1.0 spike/ms and NEST provides 0.1 spike/ms, then NEST weights should be 10× larger to compensate!

But the code has: `w_filt = weights / 10.0`, which makes weights **smaller**!

This seems wrong... unless the 10 generators are meant to each fire at higher rates?

Let me check: 10 generators × 10 Hz = 100 Hz total = 0.1 expected spikes per ms

To match TensorFlow's 1.0 spike per ms, we'd need 1000 Hz total, or 100 Hz per generator!

**But NEST uses 10 Hz per generator, not 100 Hz!**

This is confusing. Let me check if my interpretation of TensorFlow is correct...

---

## TensorFlow Re-analysis

Line 101-102:
```python
rest_of_brain = tf.reduce_sum(
    tf.random.uniform((shp[0], shp[1], 10)) < .1, dtype), -1)
```

- `tf.random.uniform(...) < .1`: Bernoulli(p=0.1), gives 0 or 1
- Shape: (batch, time, 10) → after reduce_sum → (batch, time)
- Each of 10 values has p=0.1 of being 1
- Expected sum = 10 × 0.1 = **1.0**

So yes, expected value is 1.0 per timestep.

Line 103-104:
```python
noise_input = bkg_weights[None, None] * rest_of_brain[..., None] / 10.
```

Expected: `bkg_weights * 1.0 / 10`

Where `bkg_weights = (W_orig / vsc) * 10`, so:
Expected = `(W_orig / vsc) * 10 * 1.0 / 10 = W_orig / vsc`

This is **normalized current** which gets added to other normalized currents.

---

## Back to NEST

Wait, I need to think about what "spike rate" means in discrete time.

If Poisson generator has rate 10 Hz:
- In 1 ms interval, expected spikes = 10 Hz × 0.001 s = 0.01
- For 10 generators: expected total = 0.1 spikes per ms

But TensorFlow samples 10 Bernoulli(0.1) which gives expected 1.0 per ms!

**So NEST is generating 10× fewer spikes than TensorFlow expects!**

To fix this, either:
1. Increase rate to 100 Hz per generator (10 × 100 Hz = 1000 Hz total = 1.0 spike/ms)
2. OR increase weight by 10×

Looking at nest_glif.py line 154, it divides by 10: `w_filt = weights / 10.0`

Maybe this is wrong? Should it NOT divide?

Let me check what the actual effective input is:
- Rate: 10 Hz per generator × 10 generators = 100 Hz total
- Weight: W_orig pA per spike (after `/10` and `*vsc`)
- Expected input: 100 Hz × W_orig = 100 spikes/s × W_orig = 0.1 spikes/ms × W_orig

In TensorFlow:
- Expected spikes: 1.0 per ms
- Weight: W_orig / vsc (normalized), becomes W_orig after denorm
- Expected input: 1.0 spikes/ms × (W_orig / 10) = 0.1 × W_orig per ms

Wait, TensorFlow divides by 10 at line 104! So both give 0.1 × W_orig per ms!

Hmm, they actually DO match!

But then... why does TensorFlow sample 10 Bernoulli(0.1) and divide by 10, instead of just sampling 1 Bernoulli(0.1)?

I think it's for variance reduction - averaging 10 sources reduces variance.

OK so NEST and TensorFlow both give expected input of **0.1 × W_orig** per ms.

Now what about SpiNNaker?

---

## SpiNNaker Implementation

### Setup (line 933-936)
```python
BKG_pop = sim.Population(size, sim.SpikeSourcePoisson,
    cellparams={'rate': 100.0, 'start': 0.0, 'duration': 1000.0})
```

**Single** Poisson source at **100 Hz** per neuron (not 10 sources at 10 Hz each!)

### Weights (line 960-965)
```python
vsc = network['glif3'][pid, G.VSC]
bkg_w_norm = bkg_w * vsc  # (W_orig/vsc*10) * vsc = W_orig * 10
bkg_w_scaled = bkg_w_norm * 10.0 / 1000.0  # W_orig * 10 * 10 / 1000 = W_orig / 10 in nA
```

Wait, let me recalculate:
- `bkg_w` from h5 = `(W_orig / vsc) * 10`
- `bkg_w_norm = bkg_w * vsc = W_orig * 10`
- `bkg_w_scaled = (W_orig * 10) * 10 / 1000 = W_orig * 100 / 1000 = W_orig / 10` in nA

Converting to pA: `W_orig / 10` nA = `W_orig * 100` pA

**Expected input:**
- Rate: 100 Hz = 0.1 spikes/ms
- Weight: W_orig * 100 pA per spike
- Expected input: 0.1 × (W_orig × 100) = **10 × W_orig** pA per ms

**This is 100× larger than TensorFlow/NEST!**

---

## The Bug!

**TensorFlow/NEST**: 0.1 × W_orig pA per ms
**SpiNNaker**: 10 × W_orig pA per ms

**SpiNNaker background is 100× too strong!**

### Root Cause
```python
bkg_w_scaled = bkg_w_norm * 10.0 / 1000.0
```

Should be:
```python
bkg_w_scaled = bkg_w_norm / 10.0 / 1000.0  # Divide by 10, not multiply!
# OR equivalently:
bkg_w_scaled = bkg_w / 10.0  # Like NEST, then let denorm happen naturally
```

Actually, matching NEST pattern:
```python
bkg_w_scaled = (bkg_w / 10.0) * vsc / 1000.0  # /10 for 10-source normalization, /1000 for pA->nA
```
