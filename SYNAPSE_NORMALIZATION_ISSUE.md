# Potential Issue: PSC Normalization Mismatch

## The Discrepancy

### NEST glif_psc
From WebFetch earlier:
> "alpha function is normalized such that an event of weight 1.0 results in a **peak current of 1 pA at t = τ_syn**"

This normalization gives: `init = τ * (1 - exp(-dt/τ))`

For dt=1ms, τ=5ms: `init = 5 * (1 - exp(-0.2)) ≈ 5 * 0.181 = 0.905`

### TensorFlow Training
From models.py line 174:
```python
self._psc_initial = np.e / np.array(self._params['tau_syn'])
```

This gives: `init = e / τ`

For τ=5ms: `init = 2.718 / 5 = 0.544`

### SpiNNaker GLIF3 C Code
From glif3/synapse_types_glif3_impl.h lines 78-82:
```c
// TensorFlow line 174: psc_initial = e / tau
REAL e_approx = expulr(ONE);  // e ≈ 2.718
decay_t init = kdivk(e_approx, params->tau);
```

Uses `e/tau` - **matches TensorFlow training**.

---

## The Problem

**If weights were trained with TensorFlow's `e/tau` normalization, but NEST inference uses peak-at-tau normalization, there's a mismatch!**

NEST effective weight strength: `0.905 / 0.544 ≈ 1.66×` stronger than training!

But wait - NEST achieves 80% accuracy, so either:
1. The normalization doesn't matter as much as I think
2. OR NEST actually uses `e/tau` despite the documentation
3. OR the trained weights somehow compensate

Let me check NEST source code to see what it actually uses...

---

## Investigation Needed

Need to verify NEST's actual PSC normalization in the source code, not just documentation.

If NEST really uses peak-at-tau and still works, then SpiNNaker should too, and the normalization isn't the issue.

If NEST uses `e/tau` (matching docs from Allen Institute), then SpiNNaker is correct.

---

## Alternative Hypothesis

Maybe the issue isn't normalization at all, but something about:
- **Timing**: When synapses update vs when neuron reads them
- **Receptor type mapping**: Synapses going to wrong receptors
- **Delays**: How delays are handled
- **Initial conditions**: Starting from wrong state

Need to systematically test these.
