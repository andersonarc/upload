# Input Encoding Issues in jupyter/spinnaker.py

## CRITICAL BUG #1: Probabilistic spike generation (line 254)
```python
if np.clip((spike_trains[t, i] / 1.3) * scale, 0.0, 1.0) > np.random.rand():
    times.append(float(t * timestep))
```
**Problem**: Uses `np.random.rand()` to stochastically generate spikes
**Impact**: Non-reproducible - same input produces different spike trains each run
**Expected**: Deterministic spike generation for debugging

## CRITICAL BUG #2: Division by 1.3 (line 254)  
```python
(spike_trains[t, i] / 1.3) * scale
```
**Problem**: Reduces all spike probabilities by 23% 
**Impact**: If probability = 1.0, becomes 0.769 - 23% chance of missing spikes
**Expected**: Use spike probabilities as-is, or justify the scaling

## CRITICAL BUG #3: Only first sample used (line 266)
```python
spike_times = create_spike_times(dataset['spike_probabilities'][0], scale=1.0)
```
**Problem**: Always uses sample index [0], regardless of which digit is being tested
**Impact**: Testing "digit 4" vs "digit 9" actually uses the SAME input every time
**Expected**: Select appropriate sample index based on which digit to test

## Additional observations:
- Line 825: Empty spike trains get spike at 600ms (after simulation ends)
- Weight scaling (line 803, 850) appears CORRECT - matches TensorFlow normalization

---

# Readout Decoding Analysis

## Code structure (lines 917-940):
```python
gid2train = {}  # Map global neuron ID to spike train
for i, item in enumerate(output_nnpols.items()):
    readout = readouts[i]
    key, lids = item
    gids = ps2g[key]
    spiketrains = readout.get_data('spikes').segments[0].spiketrains
    for lid, spiketrain in zip(lids, spiketrains):
        gid = gids[lid]
        gid2train[gid] = spiketrain

votes = np.zeros(10)
for i in range(0, 10):
    start = i*30
    end = (1+i)*30
    keys = network['output'][start:end]  # 30 neurons per class
    for key in keys:
        spiketrain = gid2train[key]
        mask = (spiketrain > 50) & (spiketrain < 200)
        count = mask.sum()
        votes[i] += count
```

## POTENTIAL ISSUE: Response window mismatch
- **Line 937**: Hardcoded response window `50-200ms`
- **Line 229**: Dataset has `response_window` field loaded from HDF5
- **Problem**: Hardcoded window may not match dataset's actual response window
- **Recommendation**: Use `dataset['response_window']` instead of hardcoded values

## Ordering assumption:
- Assumes readout neurons are ordered by class (first 30 = class 0, etc.)
- This comes from HDF5 file `readout/neuron_ids`
- **Cannot verify without seeing HDF5 file structure**
- If ordering is wrong, votes are assigned to wrong classes

## Mapping appears correct:
- Global ID mapping through `gid2train` looks correct
- Vote counting logic is straightforward
- Response window spike counting is correct (if window is correct)
