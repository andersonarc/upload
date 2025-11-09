# SpiNNaker Inference Debugging - Quick Reference

## 📋 What's in This Branch

This branch contains a comprehensive analysis and proposed fixes for the SpiNNaker inference bug where the Mouse V1 cortex model fails to produce meaningful outputs.

## 🔥 The Problem

- **Training:** Network achieves ~0.8 accuracy on MNIST in TensorFlow
- **Inference:** Network produces zero or garbage spikes on SpiNNaker
- **Symptom:** LGN inputs are active (hundreds of spikes) but V1 doesn't respond

## 🎯 The Root Cause

**Weight Scaling Bug** in `jupyter/spinnaker.ipynb`

```python
# WRONG (current code)
syn[:, S.WHT] *= vsc / 1000.0

# LIKELY CORRECT
syn[:, S.WHT] /= 1000.0  # Scenario A - test this first
```

The code multiplies weights by `voltage_scale` when TensorFlow **divides** by it, causing weights to be incorrect by 1000-10000x.

## 📁 Files to Read (in order)

### 1. **EXECUTIVE_SUMMARY.md** ← START HERE
- Quick overview of the problem
- Immediate action plan
- Expected outcomes
- **Read this first!**

### 2. **minimal_test.py** ← TEST THIS FIRST
```bash
python minimal_test.py
```
- Tests GLIF3 with 1 neuron
- Finds correct weight scaling
- **Run this before full simulation**

### 3. **proposed_fixes.py** ← USE THIS
- Drop-in replacement functions for notebook
- 5 weight scaling scenarios to test
- Comprehensive diagnostics
- **Copy these into your notebook**

### 4. **debug_weight_scaling.py** ← ANALYZE WITH THIS
```bash
python debug_weight_scaling.py ckpt_51978-77.h5
```
- Analyzes H5 file statistics
- Compares weight scenarios
- Checks checkpoint loading

### 5. **ANALYSIS_AND_FINDINGS.md** ← DETAILS
- Full technical analysis
- All 7 issues identified
- Pipeline breakdown
- **For deep understanding**

## ⚡ Quick Start

### Option 1: Minimal Test (Recommended First)
```bash
# Test if GLIF3 works and find correct weight scale
python minimal_test.py
# Look for working weight range in output
```

### Option 2: Analyze Current Data
```bash
# Understand current weight statistics
python debug_weight_scaling.py ckpt_51978-77.h5
```

### Option 3: Fix Full Simulation
In your Jupyter notebook:
```python
# Load the fixes
exec(open('proposed_fixes.py').read())

# Replace network creation with:
setup()
V1, V1_n_pop, V1_n_proj = create_V1_FIXED(
    network['glif3'], ps2g, v1_synpols, network,
    test_mode='scenario_A'  # Start with A, then try B, C
)
LGN, LGN_n_pop, LGN_n_proj = create_LGN_FIXED(
    V1, spike_times, tm2l, lgn_synpols, network,
    test_mode='scenario_A'
)
readouts = create_readouts(output_nnpols, V1)

# Run and analyze
sim.run(1000)
analyze_simulation_results(V1, LGN, network, output_nnpols, ps2g, dataset)
```

## 🧪 Test Scenarios

Test these weight scaling approaches in order:

- **Scenario A** (÷1000 only) - **Most likely correct**
- **Scenario B** (*vsc/1000) - Current approach (for comparison)
- **Scenario C** (÷(vsc*1000)) - Inverse of TensorFlow
- **Scenario D** (÷vsc) - Match TensorFlow exactly
- **Scenario E** (*vsc/1000) - TensorFlow + unit conversion

## ✅ Success Criteria

After applying the fix, you should see:

```
✓ LGN is active (N spikes)
✓ V1 is active (N spikes)
✓ Max voltage reached: -45.2 mV (above threshold -53.4 mV)
✓ Readout spikes in response window
✓ Classification: predicted=6, actual=6
```

## 🔍 If It Doesn't Work

1. **Check weight statistics:** Are they in reasonable range (0.01-10 nA)?
2. **Check voltage traces:** Are neurons reaching threshold?
3. **Verify checkpoint loaded:** Did c2.py successfully load trained weights?
4. **Try different response window:** Maybe [100-200ms] instead of [50-100ms]
5. **Compare with TensorFlow:** Run same input through TF and compare outputs

## 📊 Key Statistics to Monitor

From `analyze_simulation_results()`:
```
LGN spikes: [should be > 100]
V1 spikes: [should be > 0 after fix]
Voltage range: [should approach threshold]
Gap to threshold: [should be < 1 mV for active neurons]
```

## 🐛 Other Issues Identified

Secondary issues (less critical):
1. Input mode mismatch (likely compensated by notebook)
2. Response window timing
3. Checkpoint loading verification needed

See ANALYSIS_AND_FINDINGS.md for details on all 7 issues.

## 💡 Why This Happened

### TensorFlow (training)
```python
# Normalizes weights by voltage range
weights = weights / voltage_scale
```

### H5 File
```
# Stores normalized weights
weights = [already normalized values]
```

### PyNN (inference) - WRONG
```python
# Multiplies instead of dividing (backwards!)
weights = weights * voltage_scale / 1000
# Result: weights 1000x-10000x wrong magnitude
```

### PyNN (inference) - CORRECT
```python
# Just unit conversion, no voltage_scale
weights = weights / 1000  # pA → nA
```

## 📞 Getting Help

If you run the proposed fixes and:
- **It works:** Great! Document which scenario worked
- **Minimal test works but full network doesn't:** Check connectivity/population setup
- **Nothing works:** Double-check that GLIF3 model is compiled correctly

## 🎓 Learning Resources

Key files to understand the pipeline:
- `training_code/multi_training.py` - TensorFlow training
- `training_code/models.py` - GLIF3 implementation in TF
- `training_code/c2.py` - Weight conversion to H5
- `jupyter/spinnaker.ipynb` - PyNN inference
- `glif3/glif3_neuron_impl.h` - GLIF3 C implementation

## 📈 Next Steps After Fix

Once network is spiking:
1. ✅ Verify classification accuracy matches TensorFlow
2. ✅ Tune response window for optimal performance
3. ✅ Test on full MNIST dataset
4. ✅ Profile SpiNNaker performance
5. ✅ Consider visualization (side goal from original request)

---

**Bottom line:** Run `minimal_test.py` first, then apply Scenario A in `proposed_fixes.py`. This should fix the issue. See EXECUTIVE_SUMMARY.md for more details.
