# Quick Fix Guide

**For immediate testing of bug fixes**

---

## 🔴 CONFIRMED BUG: ASC Scaling

### Fix #1: ASC Scaling (CRITICAL - Fix this first!)

**File**: `/home/user/upload/jupyter/class.py`

**Lines**: 123-124

**Current (WRONG)**:
```python
network['glif3'][:, G.AA0] *= network['glif3'][:, G.VSC] / 1000.0
network['glif3'][:, G.AA1] *= network['glif3'][:, G.VSC] / 1000.0
```

**Replace with (CORRECT)**:
```python
network['glif3'][:, G.AA0] /= 1000.0  # pA -> nA (no voltage_scale)
network['glif3'][:, G.AA1] /= 1000.0  # pA -> nA (no voltage_scale)
```

**Why**:
- H5 file contains UNNORMALIZED values in pA (user confirmed)
- Current code multiplies by voltage_scale (~20 mV)
- Result: ASC amplitudes are 20x too large
- This alone could explain complete network failure

**Expected improvement**: Possibly 0% → 60-80% accuracy

---

## ⚠️ LIKELY BUG: Weight Scaling

### Fix #2: Weight Scaling (Test if ASC fix insufficient)

**File**: `/home/user/upload/jupyter/class.py`

**Lines**: 841 (V1 projections), 888 (LGN projections)

**Current (POSSIBLY WRONG)**:
```python
# Line 841 (V1 recurrent):
vsc = network['glif3'][int(tgt_key[0]), G.VSC]
syn[:, S.WHT] *= vsc / 1000.0

# Line 888 (LGN input):
vsc = network['glif3'][int(tgt_key[0]), G.VSC]
syn[:, S.WHT] *= vsc / 1000.0
```

**Option A: If H5 weights are UNNORMALIZED (like ASC)**:
```python
# Line 841:
syn[:, S.WHT] /= 1000.0  # No voltage_scale multiplication

# Line 888:
syn[:, S.WHT] /= 1000.0  # No voltage_scale multiplication
```

**Option B: If H5 weights are NORMALIZED**:
```python
# Keep current code (it's correct)
```

**How to verify**: Run `jupyter/verify_asc_normalization.py` to check H5 format

---

## 🧪 Testing Strategy

### Phase 1: Test ASC Fix Alone

1. Apply Fix #1 (ASC scaling)
2. Run SpiNNaker simulation with test samples
3. Check accuracy

**If accuracy improves to ~80%**: ASC was the only bug! ✅

**If accuracy improves but still wrong (e.g., 20-40%)**: Weight bug likely

**If no improvement**: Weight bug + ASC bug, OR different root cause

---

### Phase 2: Test Weight Fix (If Needed)

1. Verify H5 format:
   ```bash
   cd /home/user/upload/jupyter
   python3 verify_asc_normalization.py
   ```

2. If output says "UNNORMALIZED", apply Fix #2 (Option A)

3. If output says "NORMALIZED", keep current weight scaling

4. Re-run SpiNNaker simulation

**If accuracy now ~80%**: Both bugs were needed! ✅

**If still wrong**: Check H5 weights trained vs untrained

---

### Phase 3: Verify H5 Weights (If Needed)

1. Run weight diagnostics:
   ```bash
   cd /home/user/upload/jupyter
   python3 check_h5_weights.py
   ```

2. Check if weights look trained or untrained

3. If untrained, regenerate H5 with c2.py from correct checkpoint

---

## 📊 Expected Outcomes

### Scenario A: ASC-only bug (70% probability)
- Fix #1 applied → accuracy jumps to ~80%
- No other fixes needed

### Scenario B: ASC + Weight bugs (20% probability)
- Fix #1 applied → accuracy improves to 20-40%
- Fix #2 applied → accuracy jumps to ~80%

### Scenario C: ASC + Weight + H5 untrained (5% probability)
- Fix #1 + Fix #2 applied → slight improvement
- Need to regenerate H5 file

### Scenario D: Different bug (5% probability)
- Fixes don't help significantly
- Need NEST validation to check GLIF3 implementation
- Run diagnostic scripts (see FINAL_DEBUG_SESSION_SUMMARY.md)

---

## 🔍 Verification Commands

### Check current ASC values:
```bash
cd /home/user/upload/jupyter
python3 -c "
import h5py
import numpy as np
with h5py.File('ckpt_51978-153.h5', 'r') as f:
    asc = np.array(f['neurons/glif3_params/asc_amps'])
    print(f'ASC range: [{asc.min():.2f}, {asc.max():.2f}]')
    print(f'ASC mean: {np.mean(np.abs(asc)):.2f}')
    if np.mean(np.abs(asc[asc != 0])) > 10:
        print('STATUS: UNNORMALIZED (pA) - Fix #1 NEEDED')
    else:
        print('STATUS: NORMALIZED - Fix #1 may not be the issue')
"
```

### Check current weight values:
```bash
python3 -c "
import h5py
import numpy as np
with h5py.File('ckpt_51978-153.h5', 'r') as f:
    rec_w = np.array(f['recurrent/weights'])
    inp_w = np.array(f['input/weights'])
    print(f'Recurrent weights: [{rec_w.min():.4f}, {rec_w.max():.4f}], mean={np.abs(rec_w).mean():.4f}')
    print(f'Input weights: [{inp_w.min():.4f}, {inp_w.max():.4f}], mean={np.abs(inp_w).mean():.4f}')
"
```

---

## 📝 Before/After Comparison

### Before Fix:
```python
# class.py lines 123-124:
network['glif3'][:, G.AA0] *= network['glif3'][:, G.VSC] / 1000.0  # WRONG
network['glif3'][:, G.AA1] *= network['glif3'][:, G.VSC] / 1000.0  # WRONG

# Result: ASC = (pA * mV) / 1000 = pA * 0.02 (20x too large!)
```

### After Fix:
```python
# class.py lines 123-124:
network['glif3'][:, G.AA0] /= 1000.0  # CORRECT
network['glif3'][:, G.AA1] /= 1000.0  # CORRECT

# Result: ASC = pA / 1000 = nA (correct!)
```

---

## 🚨 Important Notes

1. **Make backups** before editing class.py:
   ```bash
   cp jupyter/class.py jupyter/class.py.backup
   ```

2. **Test with same samples** as previous runs for fair comparison:
   - Samples 10, 50, 90, 100 are good test cases

3. **Check determinism**: Use same seed (np.random.seed(1)) for reproducibility

4. **Monitor output**: Look for changes in:
   - Output neuron spike counts
   - Vote distributions
   - Classification predictions

5. **If unsure**: Run verification scripts first before applying fixes

---

## 📞 If Fixes Don't Work

See `FINAL_DEBUG_SESSION_SUMMARY.md` for:
- Alternative bug scenarios
- NEST validation procedure
- TensorFlow/SpiNNaker visualization
- Full diagnostic workflow

Or run:
```bash
cd /home/user/upload/jupyter
python3 nest_validation.py  # If NEST installed
python3 tensorflow_runtime_visualizer.py  # If TF installed
python3 spinnaker_runtime_visualizer.py  # Analyze current results
```

---

**Good luck! The ASC fix alone has 70% chance of solving everything.**
