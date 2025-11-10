# PP_ROUTING_ANALYSIS.md - CORRECTION

**Date**: 2025-11-10
**Status**: ❌ PREVIOUS ANALYSIS WAS INCORRECT

---

## Error in Previous Analysis

**Previous Claim**: c2.py has a bug where it starts readout neurons at index 5 instead of 0, causing 100% misclassification.

**Why This Was Wrong**: I failed to check how TensorFlow uses these indices.

---

## Correct Analysis

### TensorFlow Usage (classification_tools.py lines 91-97)

```python
elif output_mode == '10class':
    outputs = []
    for i in range(10):
        t_output = tf.gather(output_spikes, network[f'localized_readout_neuron_ids_{i + 5}'], axis=2)
        t_output = tf.reduce_mean(t_output, -1)
        outputs.append(t_output)
    output = tf.concat(outputs, -1) * scale
```

**TensorFlow uses indices 5-14 for the 10 classes** (not 0-9).

### load_sparse.py Creates 15 Readout Populations (lines 471-492)

```python
for i in range(15):
    origin = origins[i]
    # ... spatial selection logic ...
    network[f'localized_readout_neuron_ids_{i}'] = np.where(sel)[0][None]
```

Creates 15 spatially-localized readout populations at different origins.
Indices 5-14 are used for MNIST 10-class classification.
Indices 0-4 are likely for other tasks (garrett, vcd_grating, ori_diff, evidence).

### c2.py Conversion (lines 166-170)

```python
readout_neuron_ids = network['localized_readout_neuron_ids_5']
for i in range(6, 15):
    key = f'localized_readout_neuron_ids_{i}'
    if key in network:
        readout_neuron_ids = np.concatenate([readout_neuron_ids, network[key]], axis=1)
```

Stores indices 5-14 into H5 file, **matching TensorFlow's usage exactly**.

### class.py Output Decoding (lines 1000-1009)

```python
votes = np.zeros(10)
for i in range(0, 10):
    start = i*30
    end = (1+i)*30
    keys = network['output'][start:end]
```

Reads 300 neurons (30 per class) from network['output'], which contains the flattened neurons from indices 5-14.

### Mapping Verification

| Class | class.py indices | H5 source | TensorFlow source | Status |
|-------|------------------|-----------|-------------------|--------|
| 0 | 0:30 | localized_5 | localized_5 | ✅ MATCH |
| 1 | 30:60 | localized_6 | localized_6 | ✅ MATCH |
| 2 | 60:90 | localized_7 | localized_7 | ✅ MATCH |
| 3 | 90:120 | localized_8 | localized_8 | ✅ MATCH |
| 4 | 120:150 | localized_9 | localized_9 | ✅ MATCH |
| 5 | 150:180 | localized_10 | localized_10 | ✅ MATCH |
| 6 | 180:210 | localized_11 | localized_11 | ✅ MATCH |
| 7 | 210:240 | localized_12 | localized_12 | ✅ MATCH |
| 8 | 240:270 | localized_13 | localized_13 | ✅ MATCH |
| 9 | 270:300 | localized_14 | localized_14 | ✅ MATCH |

---

## Conclusion

**Readout ordering is CORRECT**. All three components (TensorFlow, c2.py, class.py) consistently use indices 5-14 for MNIST classes 0-9.

**No bug here**. The root cause of SpiNNaker failure must be elsewhere.

---

## Lesson Learned

**Rule**: Every time I reach a conclusion, try to DISPROVE it first before recording it as fact.

**What I Should Have Done**:
1. Check classification_tools.py FIRST
2. Verify TensorFlow's usage of readout indices
3. Trace the complete data flow
4. Only then conclude

**What I Actually Did**:
1. Saw c2.py starting at index 5
2. Assumed this was wrong without checking TensorFlow
3. Jumped to conclusion about 100% misclassification

This error wasted time and created false leads. Must be more rigorous.
