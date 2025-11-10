# H5 Weight Conversion Analysis - c2.py

**Status**: Code analysis only (H5 file download failed with 403)
**Date**: 2025-11-10
**Phase**: Review Task #1 - H5 weights validation

---

## Executive Summary

**User Hypothesis**: H5 weights may be untrained or corrupted (previous incident with this)

**Analysis Method**: Deep code review of c2.py conversion logic

**Key Finding**: **CRITICAL FAILURE MODE IDENTIFIED** - c2.py has a silent failure path that uses untrained weights if checkpoint loading fails

---

## c2.py Structure Analysis

### Function 1: `load_tf_checkpoint()` (lines 11-99)

**Purpose**: Load trained weights from TensorFlow checkpoint

**Critical Code Paths**:

```python
# Lines 49-58: Checkpoint restoration
if tf.train.latest_checkpoint(checkpoint_dir):
    restored = checkpoint.restore(checkpoint_path)
    if restored:
        print(f"Successfully restored checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Could not restore checkpoint from {checkpoint_path}")
        return {}  # ← RETURNS EMPTY DICT ON FAILURE
else:
    print(f"Warning: No checkpoint found at {checkpoint_path}")
    return {}  # ← RETURNS EMPTY DICT ON FAILURE
```

**⚠️ ISSUE**: Returns empty dict `{}` on failure instead of raising exception

---

### Function 2: `convert_to_pynn_format()` (lines 101-223)

**Purpose**: Main conversion function

**Critical Section (lines 112-124)**:

```python
model_vars = load_tf_checkpoint(checkpoint_path, network)

# Verify weights were loaded
if not model_vars:
    print("WARNING: No trained weights loaded - using untrained weights!")  # ← WARNING ONLY
else:
    print("Successfully loaded trained weights:")
    for key, value in model_vars.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
```

**⚠️ CRITICAL ISSUE**:
- If `model_vars` is empty (checkpoint failed to load)
- Only prints a WARNING
- **Continues execution with untrained weights**
- No exception raised, no abort

**Recurrent Weights Fallback (lines 135-140)**:

```python
if 'recurrent_weights' in model_vars:
    rec_weights = model_vars['recurrent_weights']
    print(f"Using TRAINED recurrent weights: mean={np.mean(rec_weights):.6f}, std={np.std(rec_weights):.6f}")
else:
    rec_weights = network['synapses']['weights']  # ← USES UNTRAINED WEIGHTS
    print(f"Using UNTRAINED recurrent weights: mean={np.mean(rec_weights):.6f}, std={np.std(rec_weights):.6f}")
```

**Input Weights Fallback (lines 147-152)**:

```python
if 'input_weights' in model_vars:
    inp_weights = model_vars['input_weights']
    print(f"Using TRAINED input weights: mean={np.mean(inp_weights):.6f}, std={np.std(inp_weights):.6f}")
else:
    inp_weights = input_population['weights']  # ← USES UNTRAINED WEIGHTS
    print(f"Using UNTRAINED input weights: mean={np.mean(inp_weights):.6f}, std={np.std(inp_weights):.6f}")
```

**Final Summary (lines 218-222)**:

```python
if model_vars:
    print("✓ Trained weights successfully applied")
else:
    print("✗ Using untrained weights - check checkpoint path")  # ← SILENT FAILURE
```

---

## Failure Scenarios

### Scenario 1: Checkpoint File Not Found

**Trigger**: `checkpoint_path` points to non-existent file

**Result**:
1. Line 57: `tf.train.latest_checkpoint(checkpoint_dir)` returns `None`
2. Line 58: Prints "Warning: No checkpoint found"
3. Line 59: Returns `{}`
4. Line 116: `if not model_vars:` is `True`
5. Line 117: Prints "WARNING: No trained weights loaded"
6. **Continues with untrained weights from `network['synapses']['weights']`**

### Scenario 2: Checkpoint Restoration Fails

**Trigger**: Checkpoint exists but restoration fails (corrupted, version mismatch, etc.)

**Result**:
1. Line 50: `checkpoint.restore()` fails
2. Line 52: `if restored:` is `False`
3. Line 54: Prints "Warning: Could not restore"
4. Line 55: Returns `{}`
5. **Same fallback to untrained weights**

### Scenario 3: Variable Name Mismatch

**Trigger**: Checkpoint loads but variable names don't match expected patterns

**Result**:
1. Lines 84-97: Variable extraction by name matching
2. If no variables match patterns like 'sparse_recurrent_weights'
3. `model_vars` dict remains partially or completely empty
4. **Partial fallback to untrained weights for missing components**

---

## Expected Weight Characteristics

### Trained Weights Should Have:

1. **Non-zero mean** (learned biases toward excitation/inhibition)
2. **Broad distribution** (diverse synaptic strengths)
3. **Sparse but structured patterns** (pruned weak connections)
4. **Readout weights present** (only exist after training)
5. **Asymmetric positive/negative balance** (learned E/I balance)

### Untrained Weights Would Have:

1. **Near-zero mean** (random initialization centered at 0)
2. **Narrow, symmetric distribution** (gaussian or uniform initialization)
3. **No sparsity** (all random weights present)
4. **No readout weights** (not created during initialization)
5. **Balanced positive/negative** (~50/50 split)

---

## Verification Questions

### Q1: Was checkpoint path correct when c2.py was run?
**Check**: User's conversion command
**Evidence needed**: Original command line used to run c2.py

### Q2: Did checkpoint restoration succeed?
**Check**: Console output from c2.py run
**Look for**:
- "Successfully restored checkpoint" (SUCCESS)
- "Warning: Could not restore checkpoint" (FAILURE)
- "WARNING: No trained weights loaded" (FAILURE)

### Q3: Were trained weights actually used?
**Check**: Weight statistics printed by c2.py
**Look for**:
- "Using TRAINED recurrent weights" (SUCCESS)
- "Using UNTRAINED recurrent weights" (FAILURE)

### Q4: Do weight statistics match expectations?
**Check**: class.py output (lines 408-424 print weight stats)
**Expected for trained**:
- Mean: NOT close to 0
- Range: Broad (multiple orders of magnitude)
- Distribution: Non-uniform

---

## Testing Recommendations

### Test 1: Re-run c2.py with correct checkpoint

```bash
cd training_code
python c2.py \
  --checkpoint /path/to/ckpt_51978-153 \
  --data_dir v1cortex \
  --output ckpt_51978-153_VERIFIED.h5
```

**Check output for**:
- "Successfully restored checkpoint"
- "Using TRAINED recurrent weights"
- "✓ Trained weights successfully applied"

### Test 2: Compare weight statistics

**If H5 file accessible**:
```python
import h5py
import numpy as np

with h5py.File('ckpt_51978-153.h5', 'r') as f:
    rec_w = f['recurrent/weights'][:]
    inp_w = f['input/weights'][:]

    print(f"Recurrent: mean={np.mean(rec_w):.6f}, std={np.std(rec_w):.6f}")
    print(f"Input: mean={np.mean(inp_w):.6f}, std={np.std(inp_w):.6f}")
    print(f"Has readout: {'readout_weights' in f['readout']}")
```

### Test 3: Create untrained baseline for comparison

```bash
# Use c2.py with intentionally wrong checkpoint path
python c2.py \
  --checkpoint /nonexistent/path \
  --data_dir v1cortex \
  --output UNTRAINED_BASELINE.h5
```

Compare statistics: `UNTRAINED_BASELINE.h5` vs `ckpt_51978-153.h5`

---

## class.py Weight Loading (lines 126-143)

```python
network['recurrent'] = np.stack((
    file['recurrent/sources'],
    file['recurrent/targets'],
    file['recurrent/weights'],     # ← Loads weights as-is
    file['recurrent/receptor_types'],
    np.arange(len(file['recurrent/weights'])),
    file['recurrent/delays']
), axis=1)

network['input'] = np.stack((
    file['input/sources'],
    file['input/targets'],
    file['input/weights'],         # ← Loads weights as-is
    file['input/receptor_types'],
    np.arange(len(file['input/weights']))
), axis=1)
```

**No validation or transformation** - class.py trusts H5 file completely

**Weight Statistics Printed (lines 408-414)**:
```python
print(np.mean(network['recurrent'][:, S.WHT]))  # Recurrent mean
print(np.min(network['recurrent'][:, S.WHT]))   # Recurrent min
print(np.max(network['recurrent'][:, S.WHT]))   # Recurrent max
print(np.mean(network['input'][:, S.WHT]))      # Input mean
print(np.min(network['input'][:, S.WHT]))       # Input min
print(np.max(network['input'][:, S.WHT]))       # Input max
```

**These values would definitively show if weights are trained or untrained**

---

## Hypothesis Status

| Indicator | Trained | Untrained | Status |
|-----------|---------|-----------|--------|
| Readout weights exist | ✓ | ✗ | ❓ PENDING |
| Mean near zero | ✗ | ✓ | ❓ PENDING |
| Broad distribution | ✓ | ✗ | ❓ PENDING |
| Non-uniform histogram | ✓ | ✗ | ❓ PENDING |
| Sparse connections | ✓ | ✗ | ❓ PENDING |

**Cannot determine without**:
1. Actual H5 file access, OR
2. Console output from c2.py run, OR
3. Console output from class.py run (weight statistics)

---

## CRITICAL RECOMMENDATIONS

### Immediate Actions:

1. **Check c2.py conversion logs** - Did checkpoint load succeed?
2. **Check class.py execution logs** - What are the actual weight statistics?
3. **If statistics unavailable**: Re-run c2.py with verified checkpoint path
4. **Create comparison baseline**: Generate UNTRAINED H5 to compare against

### Long-term Fix for c2.py:

```python
# Replace lines 116-124 with:
if not model_vars:
    raise RuntimeError(
        "FATAL: No trained weights loaded from checkpoint!\n"
        "This would result in using untrained weights.\n"
        f"Checkpoint path: {checkpoint_path}\n"
        "Aborting to prevent silent failure."
    )
```

**Rationale**: Fail loudly instead of silently using untrained weights

---

## Conclusion

**Code Review Verdict**: **CRITICAL VULNERABILITY FOUND**

c2.py has a silent failure mode that will use untrained weights if:
- Checkpoint path is wrong
- Checkpoint file is corrupted
- TensorFlow version incompatibility
- Variable naming mismatch

This matches user's hypothesis exactly - there WAS a previous incident with untrained weights, and the code structure makes it easy to repeat.

**Status**: ✗ CANNOT CONFIRM if current H5 is trained/untrained without:
- Actual H5 file examination
- c2.py conversion logs
- class.py execution logs with weight statistics

**Next Steps**: Proceed to Phase 5 (Population-Projection routing analysis)
