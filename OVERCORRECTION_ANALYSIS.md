# Quick Analysis: Why 0 Spikes?

## Voltage Analysis
- Max voltage: -38.9 mV
- Typical threshold: -40 to -50 mV
- **Neurons are CLOSE to threshold but not reaching it!**

This means input is reaching neurons but is ~1-10 mV too weak.

## Possible Issues

### 1. Background Rate Bug
My code creates 10 sources each at `rate` Hz, but `rate=100.0` is passed.
- Current: 10 × 100 Hz = 1000 Hz total (10× too high)
- Should: 10 × 10 Hz = 100 Hz total

But if rate is 10× too high, why are neurons TOO QUIET? Contradiction!

### 2. Synaptic Weights Possibly Too Weak?
User confirmed `/1000.0` is correct for pA→nA conversion.
But what if there's another scaling issue?

Let me check if LGN spikes are actually arriving...

### 3. PSC Normalization Issue
SpiNNaker uses `e/tau ≈ 0.544` for τ=5ms
NEST uses peak-at-tau `≈ 0.905` for τ=5ms
**Ratio**: 0.905/0.544 = 1.66× stronger in NEST

If SpiNNaker synapses are 40% weaker, this could explain the voltage gap!

## Need to Check
1. Are LGN spikes being generated?
2. Are they reaching V1?
3. What's the actual PSC normalization in NEST source code?
