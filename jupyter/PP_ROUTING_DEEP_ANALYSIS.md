# Population-Projection Routing: Deep Code Analysis

**Date**: 2025-11-10
**Context**: User reported network malfunctioning even AFTER ASC fix
**Question**: "Has PP code been checked properly?"

---

## Code Flow Overview

### Step 1: Initial V1 Mappings (lines 437-457)
```python
p2g = v1_compute_initial_mappings(network['neurons'])
```

**Logic**:
- Iterates GIDs in order: 0, 1, 2, ...
- Groups by PID (neuron type)
- Assigns LID sequentially within each PID
- Returns p2g[PID] = [GID0, GID1, ...] in ascending order

**Analysis**: ✅ **CORRECT** - Preserves GID order

### Step 2: Split Large Populations (lines 459-494)
```python
ps2g, g2psl = v1_compute_split_mappings(p2g, target=256)
```

**Logic**:
- Splits populations > 256 neurons into subpopulations
- Each subpopulation: max 256 neurons
- LIDs reset per subpopulation (0-255, 0-255, ...)
- Creates g2psl[GID] = (PID, SUBPID, LID)

**Example**:
```
PID 5 has 1000 neurons → 4 subpopulations
  (5, 0): GIDs 0-255, LIDs 0-255
  (5, 1): GIDs 256-511, LIDs 0-255
  (5, 2): GIDs 512-767, LIDs 0-255
  (5, 3): GIDs 768-999, LIDs 0-231
```

**Analysis**: ✅ **CORRECT** - Proper chunking logic

### Step 3: V1 Synapse Grouping (lines 506-541)
```python
v1_synpols = v1_group_synapses(network['recurrent'], g2psl)
```

**Logic**:
```python
for syn in synapses:
    src_gid = syn[S.SRC]
    tgt_gid = syn[S.TGT]

    src_pid, src_subpid, src_lid = g2psl[src_gid]
    tgt_pid, tgt_subpid, tgt_lid = g2psl[tgt_gid]

    synkey = ((src_pid, src_subpid), (tgt_pid, tgt_subpid))
    synpols[synkey].append([src_lid, tgt_lid, weight, receptor, id, delay])
```

**Analysis**: ✅ **CORRECT** - Proper GID→LID conversion, data preserved

### Step 4: LGN Grouping (lines 552-666)

**Step 4a: lgn_group_exact()** (lines 552-587)
```python
t2l = lgn_group_exact(network['input'], g2psl)
```

**Logic**:
- Groups LGN neurons by their TARGET populations
- l2t[lgn_gid] = set of (PID, SUBPID) targets
- t2l[target_set] = list of LGN GIDs

**Example**:
```
LGN 100 → targets (5,0), (5,1), (7,0)
LGN 101 → targets (5,0), (5,1), (7,0)  # Same targets!
LGN 102 → targets (5,0), (7,0)         # Different targets

Result:
  t2l[((5,0), (5,1), (7,0))] = [100, 101]
  t2l[((5,0), (7,0))] = [102]
```

**Analysis**: ✅ **CORRECT** - Groups by exact target match

**Step 4b: lgn_group_similar()** (lines 590-662, called 3 times)
```python
tm2l_1, l2pl_1 = lgn_group_similar(t2l, threshold=0.15)
tm2l_2, l2pl_2 = lgn_group_similar(tm2l_1, threshold=0.15)
tm2l, l2pl = lgn_group_similar(tm2l_2, threshold=0.15)
```

**Logic**:
- Merges similar target sets if difference < 15%
- Creates l2pl[lgn_gid] = (LGN_PID, LGN_LID)

**⚠️ POTENTIAL ISSUE**: Merges dissimilar LGN neurons into same population!

**Example of WRONG merging**:
```
LGN 100 → targets A, B, C       (3 targets)
LGN 101 → targets A, B, C, D    (4 targets)

Merged targets: A, B, C, D      (4 targets)
Delta: 4 - 3 = 1
Delta fraction: 1/3 = 0.33 > 0.15

Should NOT merge, but what if:

LGN 100 → targets A, B, C, D, E, F, G, H, I, J  (10 targets)
LGN 101 → targets A, B, C, D, E, F, G, H, I, K  (10 targets, 1 different)

Merged: 11 targets
Delta: 1
Delta fraction: 1/10 = 0.10 < 0.15

WILL MERGE! But LGN 100 connects to J, LGN 101 connects to K!
```

**CRITICAL**: Merged LGN populations will send spikes to DIFFERENT targets!

### Step 5: LGN Synapse Grouping (lines 673-710)
```python
lgn_synpols = lgn_group_synapses(network['input'], l2pl, g2psl)
```

**Logic**:
```python
for syn in input_synapses:
    src_gid = syn[S.SRC]  # LGN GID
    tgt_gid = syn[S.TGT]  # V1 GID

    tgt_pid, tgt_subpid, tgt_lid = g2psl[tgt_gid]
    lgn_pid, lgn_lid = l2pl[src_gid]

    synkey = (lgn_pid, (tgt_pid, tgt_subpid))
    synpols[synkey].append([lgn_lid, tgt_lid, weight, receptor, id])
```

**Analysis**: ✅ **CORRECT** mapping logic, BUT uses potentially WRONG l2pl from merging

### Step 6: V1 Projection Creation (lines 806-848)
```python
for synkey, syn in v1_synpols.items():
    src_key, tgt_key = synkey
    vsc = network['glif3'][tgt_key[0], G.VSC]
    syn[:, S.WHT] *= vsc / 1000.0  # ← WEIGHT SCALING
    sim.Projection(V1[src_key], V1[tgt_key],
                   sim.FromListConnector(syn[:, [S.SRC, S.TGT, S.WHT, S.DLY]]))
```

**⚠️ WEIGHT SCALING ISSUE**:
- Multiplies by voltage_scale / 1000
- Correct IF H5 has normalized weights (from trained checkpoint)
- WRONG IF H5 has unnormalized weights (from initial weights)

### Step 7: LGN Projection Creation (lines 856-896)
```python
for synkey, syn in lgn_synpols.items():
    lgn_pid, tgt_key = synkey
    vsc = network['glif3'][tgt_key[0], G.VSC]
    syn[:, S.WHT] *= vsc / 1000.0  # ← WEIGHT SCALING
    sim.Projection(LGN[lgn_pid], V1[tgt_key],
                   sim.FromListConnector(syn[:, [S.SRC, S.TGT, S.WHT]]))
```

**⚠️ SAME WEIGHT SCALING ISSUE** as V1

---

## Identified Issues

### Issue #1: LGN Grouping Merges Dissimilar Neurons 🔴 CRITICAL

**Problem**: `lgn_group_similar()` with 15% threshold merges LGN neurons with different connectivity

**Impact**:
- LGN population broadcasts spikes to merged neuron set
- But individual LGN neurons should only connect to their specific targets
- Result: WRONG LGN neurons receive input

**Example**:
```
Original:
  LGN 100 → V1 neurons {0, 5, 10}
  LGN 101 → V1 neurons {0, 5, 11}  (11 instead of 10!)

After merging (15% threshold passes):
  LGN population [100, 101] → V1 neurons {0, 5, 10, 11}

When LGN 100 spikes:
  V1 neurons 0, 5, 10, 11 ALL receive input
  But neuron 11 should NOT receive from LGN 100!
```

**Severity**: **CRITICAL** - Fundamentally wrong connectivity

**Where it happens**: Lines 664-666 (called 3 times!)

**Fix**: Either:
1. Use threshold=0.0 (exact matching only)
2. Create individual spike sources per LGN neuron (memory intensive)
3. Use more sophisticated grouping that preserves per-neuron connectivity

### Issue #2: Weight Scaling Depends on H5 Format ⚠️ UNCERTAIN

**Problem**: Lines 841, 888 multiply by `voltage_scale / 1000`

**Two scenarios**:
- IF H5 has normalized weights (from checkpoint): CORRECT
- IF H5 has unnormalized weights (from initial): WRONG (too large)

**Status**: Cannot verify without H5 file or c2.py logs

### Issue #3: ASC Scaling (Already Identified) ⚠️ UNCERTAIN

**Problem**: Lines 123-124 multiply by `voltage_scale / 1000`

**Status**: User confirmed ASC values are unnormalized → BUG CONFIRMED
**Fix needed**: Change to `/= 1000.0` (no voltage_scale multiplication)

---

## Verification Needed

### Test 1: LGN Grouping Impact

**Hypothesis**: LGN merging causes wrong connectivity

**Test**:
```python
# Modify lines 664-666:
# OLD:
tm2l_1, l2pl_1 = lgn_group_similar(t2l, threshold=0.15)
tm2l_2, l2pl_2 = lgn_group_similar(tm2l_1, threshold=0.15)
tm2l, l2pl = lgn_group_similar(tm2l_2, threshold=0.15)

# NEW:
tm2l, l2pl = t2l, {}  # Use exact grouping only, no merging
for lgn_gid in range(17400):
    if lgn_gid in l2t:
        tgtkey = tuple(l2t[lgn_gid])
        lgn_pid = list(t2l.keys()).index(tgtkey)
        lgn_lid = t2l[tgtkey].index(lgn_gid)
        l2pl[lgn_gid] = (lgn_pid, lgn_lid)
```

**Expected**: If LGN merging is the bug, accuracy should improve dramatically

### Test 2: Weight Format Verification

**Option A**: Check c2.py logs for "Using TRAINED" vs "Using UNTRAINED"

**Option B**: Inspect H5 file directly (verify_asc_normalization.py also checks weights)

**Option C**: Test both scenarios:
```python
# Version 1: Assume normalized (keep current)
syn[:, S.WHT] *= vsc / 1000.0

# Version 2: Assume unnormalized (remove voltage_scale)
syn[:, S.WHT] /= 1000.0
```

---

## Confidence Assessment

| Issue | Confidence | Impact |
|-------|-----------|--------|
| **LGN merging bug** | 90% | 🔴 CRITICAL - Wrong connectivity |
| **Weight scaling** | 50% | ⚠️ HIGH - Depends on H5 format |
| **ASC scaling** | 95% | 🔴 CRITICAL - Confirmed by user |

---

## Recommended Fix Priority

### Priority 1: Fix ASC Scaling (CONFIRMED BUG)
```python
# class.py lines 123-124
network['glif3'][:, G.AA0] /= 1000.0  # NOT *= VSC / 1000
network['glif3'][:, G.AA1] /= 1000.0
```

### Priority 2: Test LGN Grouping Without Merging
```python
# class.py lines 664-666
# Comment out lgn_group_similar calls
# Use exact grouping only
```

### Priority 3: Verify Weight Format
```python
# Option A: Check H5 file
python jupyter/verify_asc_normalization.py

# Option B: Test both weight scaling approaches
```

---

## Conclusion

**PP Routing Analysis**: Found **CRITICAL BUG** in LGN grouping

The `lgn_group_similar()` function merges LGN neurons with different connectivity patterns, causing WRONG neurons to receive input spikes.

Combined with ASC scaling bug (confirmed) and potential weight scaling bug (unconfirmed), this explains why network completely fails.

**Next steps**:
1. Fix ASC scaling (confirmed bug)
2. Disable LGN merging (test exact grouping only)
3. Verify weight format in H5 file
4. Test with all fixes applied

**Expected outcome**: With all three bugs fixed, SpiNNaker should achieve ~80% accuracy (matching TensorFlow).
