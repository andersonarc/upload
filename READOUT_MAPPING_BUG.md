# CRITICAL BUG: Readout Mapping Incorrect!

## The Issue

**Line 1044 in spynnaker_newclass.py assumes spike trains are in same order as indices**

```python
for lid, spiketrain in zip(lids, spiketrains):
    gid = gids[lid]  # WRONG: assumes spiketrains[i] corresponds to lids[i]
    gid2train[gid] = spiketrain
```

If `PopulationView.get_data()` returns spike trains in a **different order** than the indices passed to the view, this mapping is completely wrong!

## Evidence

Neo/PyNN spike trains have annotations (like `source_id`) to identify which neuron they belong to. The correct approach is to **use these annotations** instead of assuming order.

## Why This Causes Uniform Random Distribution

With scrambled readout mapping:
- Class 0 neurons might be reading from Class 5, 7, 2, etc. neurons
- Each class reads a random mix of actual output neurons
- Result: All classes get similar random vote counts

This explains:
- ✅ Uniform distribution across all classes
- ✅ No selectivity/differentiation
- ✅ Classification fails completely

## The Fix

Use spike train annotations to correctly map back to global IDs:

```python
for i, item in enumerate(output_nnpols.items()):
    readout = readouts[i]
    key, lids = item
    gids = ps2g[key]
    spiketrains = readout.get_data('spikes').segments[0].spiketrains

    # Correct mapping using annotations
    for spiketrain in spiketrains:
        # Get the actual source neuron ID from annotations
        source_id = spiketrain.annotations['source_id']  # Or 'channel_id'
        # source_id is the LOCAL ID within the view
        # Need to map: view_local_id → population_local_id → global_id

        # If source_id is the index within the PopulationView:
        lid = lids[source_id]  # Map view index to population local ID
        gid = gids[lid]  # Map to global ID
        gid2train[gid] = spiketrain
```

OR, if PopulationView indices map directly to population indices:

```python
for spiketrain in spiketrains:
    # source_id might already be the population local ID
    lid = spiketrain.annotations['source_id']
    gid = gids[lid]
    gid2train[gid] = spiketrain
```

## Testing Required

Need to check:
1. What annotation field contains the neuron ID (`source_id`, `channel_id`, etc.)
2. Whether it's the PopulationView index or population index
3. Verify the mapping is correct after fix

This bug combined with background weight issue would completely break classification!
