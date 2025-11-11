# Population Mapping Bug Analysis

## The Smoking Gun

Neurons that fired:
- GID 5883 → should be class 7
- GID 37514 → should be class 7
- GID 37955 → should be class 8

**Expected**: Class 2 neurons should fire (for digit 2 input)

**Conclusion**: Wrong neurons are receiving LGN input! Population mapping is scrambling connectivity.

## Hypothesis: Array Index ≠ Global ID

Looking at `v1_compute_initial_mappings` (line 450-457):

```python
for gid, pid in enumerate(neurons):
    if pid not in p2g:
        p2g[pid] = []
    lid = len(p2g[pid])
    p2g[pid].append(gid)  # Appending array index as GID!
```

**Critical assumption**: `enumerate(neurons)` treats array index as global ID.

**This is only correct if**: neurons[i] corresponds to neuron with global ID = i

But looking at how neurons are loaded (line 97):
```python
network['neurons'] = np.array(file['neurons/node_type_ids'])
```

**What if node_type_ids is NOT ordered by global ID?**

## The Fix Needed

Instead of:
```python
for gid, pid in enumerate(neurons):  # gid = index, pid = type
```

Should be:
```python
for gid in range(len(neurons)):  # Explicitly use index as GID
    pid = neurons[gid]  # Get type for this neuron
```

OR, if h5 has actual global IDs stored somewhere:
```python
gids = np.array(file['neurons/node_gids'])  # If this exists
for i, (gid, pid) in enumerate(zip(gids, neurons)):
    ...
```

## Testing

Need to verify: Does NEST assume neurons array index = global ID?

Looking at nest_glif.py line 83-122, NEST uses:
```python
v1 = nest.Create('glif_psc', len(neurons))  # Creates neurons 1 to len(neurons)
```

Then sets parameters by iterating:
```python
for ntype in unique_types:
    mask = neurons == ntype
    indices = np.where(mask)[0]  # Gets array indices where type matches
    for idx in indices:
        v1[int(idx)].set(params)  # Sets params for neuron at index idx
```

So NEST uses v1[idx] where idx is the array index! This means:
- NEST neuron at position idx has GID = idx+1 (NEST starts at 1)
- Parameters come from neurons[idx]

**NEST assumes array index = GID (minus 1 for 0-indexing)**

So SpiNNaker should too! The current code is correct in that assumption.

## So Where's The Bug?

If the assumption is correct, the bug must be in:
1. **Synapse remapping** (lgn_group_synapses or v1_group_synapses)
2. **Population creation** (create_V1 or create_LGN)
3. **Projection creation** (how FromListConnector interprets local IDs)

Need to add more diagnostics to trace synaptic connectivity.
