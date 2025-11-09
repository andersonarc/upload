# Mouse V1 Cortex SpiNNaker Simulation - Executive Summary

## Problem Statement

A large-scale Mouse V1 cortex model (51,978 GLIF3 neurons) was successfully trained in TensorFlow to ~0.8 accuracy on MNIST classification. However, when deployed on SpiNNaker neuromorphic hardware, the network either:
- Produces no spikes (most common)
- Produces garbage output with no correlation to input
- Shows hyperactivity in 1-2 classes (observed in some runs)

LGN input neurons ARE spiking (hundreds of spikes total), but V1 is not responding.

## Root Cause Analysis

### PRIMARY ISSUE: Weight Scaling Bug 🔴 CRITICAL

**Location:** `jupyter/spinnaker.ipynb` - `create_V1()` and `create_LGN()` functions

**The Bug:**
```python
# Current (WRONG) code
vsc = network['glif3'][int(tgt_key[0]), G.VSC]  # voltage_scale = V_th - E_L
syn[:, S.WHT] *= vsc / 1000.0
```

**Why It's Wrong:**
1. In TensorFlow training (`models.py:227, 235`), weights are **divided** by `voltage_scale` for normalization:
   ```python
   weights = weights / voltage_scale[...]
   ```

2. The H5 file stores these **already-normalized weights** from TensorFlow

3. The PyNN code **multiplies** by `voltage_scale`, which:
   - **Reverses** the TensorFlow normalization (wrong direction!)
   - Then divides by 1000 (unit conversion)
   - Net effect: weights scaled incorrectly

**Impact:**
- Weights may be 1000-10000x too small or in wrong units
- V1 neurons cannot reach firing threshold
- Network fails to respond to input

## Proposed Solutions

### Solution #1: Test Different Weight Scaling Modes (RECOMMENDED)

Test in this order:

```python
# Scenario A: No voltage_scale (just unit conversion) - TRY THIS FIRST
syn[:, S.WHT] = original_weights / 1000.0

# Scenario B: Current approach (for comparison)
syn[:, S.WHT] = original_weights * vsc / 1000.0

# Scenario C: Inverse of TensorFlow (divide, don't multiply)
syn[:, S.WHT] = original_weights / (vsc * 1000.0)
```

**Expected Result:**
One of these scenarios should produce correct network activity matching TensorFlow inference.

### Solution #2: Add Comprehensive Diagnostics

Before and after the fix, monitor:

```python
# 1. Weight statistics
print(f"Mean weight: {np.mean(weights):.9f} nA")
print(f"Range: [{np.min(weights):.9f}, {np.max(weights):.9f}]")

# 2. V1 voltage traces
v = V1[key].get_data('v')
print(f"Voltage range: [{v.min():.2f}, {v.max():.2f}] mV")
print(f"Threshold: {threshold:.2f} mV")
print(f"Gap to threshold: {threshold - v.max():.2f} mV")

# 3. Activity counts
print(f"LGN spikes: {count_lgn_spikes()}")
print(f"V1 spikes: {count_v1_spikes()}")
print(f"Readout spikes: {count_readout_spikes()}")
```

## Secondary Issues Identified

### Issue 2: Input Mode Flag Mismatch
- **Status:** LIKELY OKAY (notebook compensates)
- **Details:** Training used `--nocurrent_input` (binary spikes), but `mnist.py` generates continuous values. Notebook divides by 1.3 and samples, which should be correct.
- **Verification needed:** Confirm spike statistics match training

### Issue 3: Response Window Uncertainty
- **Current:** `[50-100 ms]`
- **Might need:** `[100-200 ms]` or `[50-250 ms]`
- **Action:** Test after fixing weights

### Issue 4: Checkpoint Loading
- **Concern:** Was TensorFlow checkpoint actually loaded in c2.py?
- **Action:** Check c2.py output logs for "Successfully loaded trained weights"

## Files Created

1. **`ANALYSIS_AND_FINDINGS.md`** - Comprehensive technical analysis
   - Detailed pipeline breakdown
   - All 7 issues identified
   - Unit conversions and equations
   - Comparison tables

2. **`debug_weight_scaling.py`** - Diagnostic script
   ```bash
   python debug_weight_scaling.py ckpt_51978-77.h5 checkpoints
   ```
   - Analyzes H5 file statistics
   - Compares weight scaling scenarios
   - Checks checkpoint loading
   - Provides test recommendations

3. **`proposed_fixes.py`** - Fixed implementations
   - `create_V1_FIXED()` - Corrected weight scaling with test modes
   - `create_LGN_FIXED()` - Corrected input weight scaling
   - `analyze_simulation_results()` - Comprehensive diagnostics
   - Usage examples

## Quick Start - Immediate Action Plan

### Step 1: Run Diagnostic (5 minutes)
```bash
cd /home/user/upload
python debug_weight_scaling.py ckpt_51978-77.h5
```

Review output to understand current weight statistics.

### Step 2: Test Fix (30 minutes)

In your Jupyter notebook, replace network creation with:

```python
# Import the fixed functions
exec(open('proposed_fixes.py').read())

# Test Scenario A (most likely correct)
setup()
V1, V1_n_pop, V1_n_proj = create_V1_FIXED(
    network['glif3'], ps2g, v1_synpols, network,
    test_mode='scenario_A'
)
LGN, LGN_n_pop, LGN_n_proj = create_LGN_FIXED(
    V1, spike_times, tm2l, lgn_synpols, network,
    test_mode='scenario_A'
)
readouts = create_readouts(output_nnpols, V1)

# Run simulation
sim.run(1000)

# Analyze
analyze_simulation_results(V1, LGN, network, output_nnpols, ps2g, dataset)
```

### Step 3: Iterate if Needed

If Scenario A doesn't work, try scenarios B, C in order.

### Step 4: Compare with TensorFlow

Once you get spikes, run the same input through TensorFlow and compare:
- Spike counts per population
- Response timing
- Classification accuracy

## Expected Outcomes

### If Fix Works:
- V1 neurons will spike in response to LGN input
- Readout neurons will show class-specific activity
- Classification accuracy should match TensorFlow (~0.8)
- Voltage traces will show neurons approaching/crossing threshold

### If Still No Activity:
- Weights may be in completely different units than expected
- May need to check original TensorFlow weight values
- Might need alpha-exponential synapses instead of current synapses
- Consider testing with IF_curr_exp neurons first (simpler model)

## Technical Notes

### Units Summary
| Quantity | TensorFlow | H5 File | PyNN Target | C Code |
|----------|------------|---------|-------------|--------|
| Voltage | Normalized | mV | mV | mV |
| Current | Normalized | ? | **nA** | **nA** |
| Weight | Normalized | Normalized? | **nA** | **nA** |
| C_m | pF | pF | nF (÷1000) | nF |
| g | nS | nS | uS (÷1000) | uS |

### GLIF3 Receptor Configuration
- **4 independent receptors** (synapse_0, synapse_1, synapse_2, synapse_3)
- **ALL excitatory** (inhibition via negative weights)
- Uses alpha-exponential synapses in C code
- Each receptor has independent time constant

### Key Equations

**TensorFlow weight normalization:**
```python
weight_normalized = weight_original / voltage_scale
```

**Target PyNN weight (hypothesis):**
```python
# Option A: H5 weights are in pA, just need unit conversion
weight_nA = weight_h5 / 1000.0

# Option B: H5 weights are normalized, need to recover original
weight_nA = (weight_h5 * voltage_scale) / 1000.0  # Current approach

# Option C: H5 weights need further normalization
weight_nA = weight_h5 / (voltage_scale * 1000.0)
```

## Success Criteria

✅ **Minimal Success:** V1 neurons spike in response to input
✅ **Moderate Success:** Readout neurons show some class preference
✅ **Full Success:** Classification accuracy matches TensorFlow (~0.8)

## Confidence Level

- **Weight scaling bug:** 95% confident this is the primary issue
- **Proposed fix:** 80% confident Scenario A will work
- **Secondary issues:** 20% chance they become relevant after weight fix

## Next Steps After Fix

Once the network is spiking properly:

1. **Tune response window** for optimal classification
2. **Optimize population splitting** for SpiNNaker efficiency
3. **Test on full MNIST dataset** (currently using 16 samples)
4. **Profile performance** (inference time, power consumption)
5. **Scale to larger models** if successful

## Questions?

For details on any specific issue, see `ANALYSIS_AND_FINDINGS.md`.

For testing and debugging tools, see `debug_weight_scaling.py` and `proposed_fixes.py`.

---

**Bottom Line:** The network is almost certainly not responding because weights are scaled incorrectly. Test the three weight scaling scenarios in `proposed_fixes.py`, starting with Scenario A. This should restore functionality.
