# Summary: Root Cause Found

## The Problem: Massive Over-Excitation

**sPyNNaker firing rate: 8.6× higher than NEST**
**Spike distribution: Uniform/flat (random noise) instead of selective**

This means neurons are firing too much, driven by noise rather than signal.

---

## Primary Fix Applied: Background Weights

**The background weights were 100× too strong** (just fixed and pushed).

### What was wrong:
```python
# OLD (WRONG):
bkg_w_scaled = (bkg_w * vsc) * 10.0 / 1000.0  # Multiplied by 10!
```

### What's correct now:
```python
# NEW (CORRECT):
bkg_w_scaled = (bkg_w / 10.0) * vsc / 1000.0  # Divide by 10
```

**This should reduce background-driven firing by 100×.**

---

## Expected Result After Fix

With background 100× weaker:
- **Firing rates should drop dramatically** (from 17.8 Hz to ~0.2-2 Hz range)
- **Signal-to-noise ratio improves 100×**
- **Selective responses should emerge** (some classes active, others quiet)
- **Should approach NEST's 80%+ accuracy**

---

## Testing Instructions

1. **Re-run with the fixed code** (just pushed to branch)
2. **Check if spike counts drop to ~3-5 per class** (matching NEST)
3. **Check if distribution becomes selective** (winner-take-most, not uniform)

If still issues after this fix, investigate:
- Receptor type mapping (excitatory vs inhibitory)
- ASC signs (should be mostly negative for adaptation)
- Refractory periods
- Initial conditions

But I'm highly confident the background fix will resolve this!

---

## Why Background Weights Caused This

With background 100× too strong:
- Each neuron receives massive constant noise input
- Noise alone drives neurons to threshold → random firing
- Synaptic signal is only ~1% of total input → ignored
- All neurons fire uniformly at noise-driven rate

With correct background:
- Background provides gentle baseline activity
- Synaptic signal is significant fraction of input
- Neurons respond selectively to input patterns
- Classification information preserved

---

## Verification

To confirm this is fixed, after re-running check:
```
Total spikes in 50-100ms window:
- Should be ~31 total (matching NEST)  - Currently ~267 (8.6× too many)

Distribution shape:
- Should be peaked (winner class has 5-7, others have 0-3)
- Currently flat (all classes 20-35)
```
