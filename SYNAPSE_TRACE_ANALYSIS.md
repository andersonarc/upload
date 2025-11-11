# Detailed Synapse Trace Analysis

## The Problem

GID 37955 (class 8) fired 9 spikes when digit 2 was input, but class 2 neurons (indices 60-89 in output array) should have fired instead.

## Hypothesis

Either:
1. LGN synapses meant for class 2 output neurons are connecting to class 8 neurons, OR
2. The population mapping/splitting is scrambling the neuron correspondence

## Tracing Through the Mapping

### Step 1: Initial Population Mapping (v1_compute_initial_mappings)

```python
for gid, pid in enumerate(neurons):  # gid = array index (0-based)
    if pid not in p2g:
        p2g[pid] = []
    p2g[pid].append(gid)  # Append GID to population
```

**Example:**
- If neurons[37955] = 8 (neuron type 8)
- Then p2g[8].append(37955)
- Say GID 37955 is the 150th neuron of type 8, so it's at position 149 in p2g[8]

### Step 2: Population Splitting (v1_compute_split_mappings)

```python
for pid, gids in p2g.items():
    # Split into chunks of 256
    for subpid in range(n_split):
        start = subpid * 256
        end = (subpid + 1) * 256
        subgids = gids[start:end]
        ps2g[(pid, subpid)] = subgids

        for lid, gid in enumerate(subgids):
            g2psl[gid] = (pid, subpid, lid)
```

**Example:**
- If GID 37955 is at position 149 in p2g[8]
- And p2g[8] has 300 neurons (indices 0-299)
- Then: ps2g[(8, 0)] = p2g[8][0:256] (first 256 neurons)
-       ps2g[(8, 1)] = p2g[8][256:300] (remaining 44 neurons)
- GID 37955 at position 149 goes into ps2g[(8, 0)]
- Local ID = 149
- **g2psl[37955] = (8, 0, 149)**

### Step 3: Population Creation (create_V1)

```python
for key, gids in ps2g.items():
    pid, subpid = key
    V1[key] = sim.Population(
        len(gids),
        GLIF3Curr,
        cellparams=G2D(glif3s[pid]),  # Parameters for neuron type 'pid'
        ...
    )
```

**Example:**
- V1[(8, 0)] = sim.Population(256, ..., cellparams for type 8)
- Neuron at index 149 in V1[(8, 0)] corresponds to GID ps2g[(8, 0)][149] = 37955 ✓

### Step 4: LGN Synapse Remapping (lgn_group_synapses)

Say there's an LGN synapse in the h5 file:
- syn = [lgn_500, 37955, 0.05, 0, ...]  (LGN 500 → V1 37955)

```python
src_gid = 500
tgt_gid = 37955

# Map LGN source
lgn_pid, lgn_lid = l2pl[500]  # Say (10, 25)

# Map V1 target
tgt_pid, tgt_subpid, tgt_lid = g2psl[37955]  # = (8, 0, 149)

# Create remapped synapse
synkey = (10, (8, 0))
synpol = [25, 149, 0.05, ...]  # [lgn_lid, v1_lid, weight, ...]
```

### Step 5: Projection Creation (create_LGN)

```python
synkey = (10, (8, 0))
lgn_pid, tgt_key = synkey  # lgn_pid=10, tgt_key=(8,0)

sim.Projection(
    LGN[10],           # Source: LGN population 10
    V1[(8, 0)],        # Target: V1 population (8,0)
    sim.FromListConnector([[25, 149, 0.05]]),  # [src_lid, tgt_lid, weight]
    ...
)
```

**Question:** Does this projection correctly connect:
- LGN neuron at index 25 in LGN[10] → V1 neuron at index 149 in V1[(8, 0)]?

**For this to work:**
- LGN[10] at index 25 must correspond to LGN GID 500
- V1[(8, 0)] at index 149 must correspond to V1 GID 37955

### Step 6: LGN Population Creation

```python
for i, lgns in enumerate(tm2l.values()):
    LGN_x = sim.Population(
        len(lgns),
        sim.SpikeSourceArray,
        cellparams={
            'spike_times': [spike_times[lgn] for lgn in lgns]
        },
        ...
    )
    LGN.append(LGN_x)
```

And l2pl was created:
```python
for pid, item in enumerate(tm2l.items()):
    tgtpols, lgns = item
    for lid, lgn in enumerate(lgns):
        l2pl[lgn] = (pid, lid)
```

**Critical Check:**
- When pid=10 in l2pl creation: lgns is some array, say [450, 480, ..., 500, ...]
- If LGN 500 is at position 25 in this array: l2pl[500] = (10, 25) ✓
- When i=10 in LGN creation: lgns is the same array (must be from tm2l.values()[10])
- LGN[10] is created with spike_times[450], spike_times[480], ..., spike_times[500], ...
- Neuron at index 25 gets spike_times[500] ✓

**But wait!** This assumes `tm2l.items()` and `tm2l.values()` iterate in the same order.

In Python 3.7+, dict ordering is preserved, so this should be fine...

**UNLESS** there's an issue with the iteration across multiple calls to `lgn_group_similar`!

## The Bug: Multiple lgn_group_similar Calls

```python
t2l = lgn_group_exact(network['input'], g2psl)
tm2l_1, l2pl_1 = lgn_group_similar(t2l, threshold=0.15)
tm2l_2, l2pl_2 = lgn_group_similar(tm2l_1, threshold=0.15)  # Uses tm2l_1 as input!
tm2l, l2pl = lgn_group_similar(tm2l_2, threshold=0.15)      # Uses tm2l_2 as input!
```

The third call creates `l2pl` by enumerating `tm2l_2`, not `tm2l`!

Wait no, let me re-read lgn_group_similar... It returns both tm2l and l2pl, and l2pl is created from the tm2l that it returns.

Actually, looking more carefully at lgn_group_similar:
- It takes an input dict (e.g., tm2l_1)
- It merges similar populations into a new dict (tm2l)
- It creates l2pl by enumerating the NEW dict (tm2l)

So after the third call:
- `tm2l` is the final merged dict
- `l2pl` maps LGN GIDs to (pid, lid) based on `tm2l`

This should be consistent with the LGN population creation that uses `tm2l`.

## Alternative Theory: Dict Iteration Order Bug

Even though Python 3.7+ preserves dict order, there might be a mismatch if:
- `.items()` and `.values()` are called on the dict at different times
- The dict is modified between calls

But `tm2l` is created once and then used for both l2pl creation and LGN population creation, so this shouldn't be an issue.

## Next Steps

Need to add diagnostic code to verify:
1. **V1 population correspondence:** Does V1[(pid, subpid)][lid] correspond to ps2g[(pid, subpid)][lid]?
2. **LGN population correspondence:** Does LGN[pid][lid] correspond to the correct LGN GID?
3. **Synapse connectivity:** Print out a few synapses targeting class 2 output neurons and verify where they actually connect

The diagnostic should print:
- For GID 37955: What population/lid is it in? What are the first few GIDs in that population?
- For class 2 output neurons: What LGN synapses target them? Where do those synapses actually connect after remapping?
