# LGN Grouping Correction

**Date**: 2025-11-10
**Status**: CORRECTING PREVIOUS ERROR

---

## Previous Claim (WRONG)

In `PP_ROUTING_DEEP_ANALYSIS.md`, I claimed:

> **Issue #1: LGN Grouping Merges Dissimilar Neurons 🔴 CRITICAL**
>
> `lgn_group_similar()` with 15% threshold merges LGN neurons with different connectivity, causing WRONG neurons to receive input.

**This claim is INCORRECT.**

---

## Why I Was Wrong

I misunderstood how PyNN's Population and FromListConnector work together.

### Key Insight 1: Individual Spike Times Per Neuron

**Lines 856-866** (create_LGN):
```python
LGN_x = sim.Population(
    len(lgns),
    sim.SpikeSourceArray,
    cellparams={
        'spike_times': [spike_times[lgn] if len(spike_times[lgn]) > 0 else [1000.0] for lgn in lgns]
    },
    label=f'LGN_{i}'
)
```

The `spike_times` parameter is a **list of lists** - one per neuron. Each neuron gets its **individual** spike train.

If `lgns = [100, 101, 102]`, then:
- Local ID 0 → spike_times[100]
- Local ID 1 → spike_times[101]
- Local ID 2 → spike_times[102]

### Key Insight 2: Local ID Mapping Preserves Identity

**Lines 654-657** (lgn_group_similar):
```python
l2pl = {}
for pid, item in enumerate(tm2l.items()):
    tgtpols, lgns = item
    for lid, lgn in enumerate(lgns):
        l2pl[lgn] = (pid, lid)
```

Each LGN GID is assigned a unique (population_id, local_id) pair. The local ID is simply the position in the merged list.

### Key Insight 3: FromListConnector Uses Local IDs

**Lines 691, 699** (lgn_group_synapses):
```python
lgn_pid, lgn_lid = l2pl[src_gid]
# ...
synpols[synkey].append(np.hstack([[lgn_lid, tgt_lid], syn[2:]]))
```

**Line 895** (create_LGN):
```python
sim.Projection(LGN[lgn_pid], V1[tgt_key],
    sim.FromListConnector(syn[:, [S.SRC, S.TGT, S.WHT]], column_names=['weight']),
    receptor_type=receptor_type)
```

The FromListConnector receives a list of `[src_lid, tgt_lid, weight]` triplets. Each triplet specifies:
- Which neuron in the source population (by local ID)
- Connects to which neuron in the target population (by local ID)
- With what weight

---

## Complete Example

**Scenario**: Two LGN neurons with different connectivity are merged:

**Original**:
- LGN 100 → V1 neurons {A, B, C}
- LGN 101 → V1 neurons {A, B, D} (different!)

**After lgn_group_similar(threshold=0.15)**:
- Merged into population X
- LGN 100 → (pid=X, lid=0)
- LGN 101 → (pid=X, lid=1)

**Population Creation**:
```python
Population X with cellparams={
    'spike_times': [spike_times[100], spike_times[101]]
}
```
- Neuron 0 fires according to LGN 100's spike train
- Neuron 1 fires according to LGN 101's spike train

**Synapses**:
```python
# Original:
# LGN 100 → A, B, C
# LGN 101 → A, B, D

# Converted to local IDs:
# [0, A_local, weight]  # LGN 100 (local 0) → A
# [0, B_local, weight]  # LGN 100 (local 0) → B
# [0, C_local, weight]  # LGN 100 (local 0) → C
# [1, A_local, weight]  # LGN 101 (local 1) → A
# [1, B_local, weight]  # LGN 101 (local 1) → B
# [1, D_local, weight]  # LGN 101 (local 1) → D
```

**Result**:
- When neuron 0 (LGN 100) spikes → A, B, C receive input (CORRECT)
- When neuron 1 (LGN 101) spikes → A, B, D receive input (CORRECT)
- **Per-neuron connectivity is PRESERVED**

---

## What lgn_group_similar Actually Does

**Purpose**: Memory optimization, not functional grouping.

PyNN requires creating Population objects, which have overhead. By grouping LGN neurons with similar (but not necessarily identical) target patterns, the code reduces:
- Number of Population objects
- Number of Projection objects
- Memory and initialization time

**Key**: The grouping does NOT broadcast spikes - each neuron maintains:
- Its own spike train
- Its own synaptic connections (via local IDs in FromListConnector)

**The 15% threshold** controls how aggressively to merge:
- Lower threshold → more populations, less memory savings
- Higher threshold → fewer populations, more memory savings
- But connectivity is preserved regardless of threshold

---

## Why This Is Actually Clever

The code uses PyNN's capabilities correctly:

1. **SpikeSourceArray** with per-neuron `spike_times` parameter
2. **FromListConnector** with local IDs for per-synapse specification
3. **Population merging** for memory optimization

This is a **valid and efficient** implementation technique.

---

## Apology

I jumped to a conclusion without fully understanding PyNN's API. The user correctly pointed out:

> "lgn_group_similar since it should support varying output populations"

I should have verified how PyNN handles individual neurons within merged populations before claiming a bug.

---

## Corrected Bug Status

| Issue | Status |
|-------|--------|
| ~~LGN grouping merges dissimilar neurons~~ | ❌ **NOT A BUG** - Connectivity preserved via local IDs |
| ASC scaling | ✅ **CONFIRMED BUG** - User verified unnormalized values |
| Weight scaling | ⚠️ **LIKELY BUG** - Same pattern as ASC, needs verification |

---

## Lessons Learned

1. **Verify before concluding** - User's "try to disprove it first" principle
2. **Understand the framework** - PyNN's Population/FromListConnector semantics matter
3. **Read code carefully** - Line 863 clearly shows per-neuron spike_times
4. **Don't assume broadcast semantics** - FromListConnector is per-synapse, not per-population

---

## Confidence Assessment

**Previous claim confidence**: 90% (WRONG)
**Corrected understanding confidence**: 100% (code is clear)

The LGN grouping code is **CORRECT** and represents good engineering practice for memory optimization in PyNN.
