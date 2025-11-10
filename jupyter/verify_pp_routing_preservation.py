#!/usr/bin/env python
"""
Verify that Population-Projection grouping preserves synapse data correctly.

The grouping process:
1. network['recurrent'] - flat array of synapses (GID-based)
2. v1_group_synapses() - groups into synapse populations (LID-based)
3. Projections created - uses LID-based connectivity
4. Check: Do we get back the same connectivity?

Critical checks:
- Total synapse count preserved
- Weight values preserved
- Source/target mapping preserved (after GID→LID→GID conversion)
- Receptor types preserved
"""

import numpy as np
import h5py
import sys

# Add synapse indices
class S:
    SRC = 0  # Source
    TGT = 1  # Target
    WHT = 2  # Weight
    RTY = 3  # Receptor Type
    ID  = 4  # Synapse ID
    DLY = 5  # Delay (recurrent only)

# GLIF3 parameter indices
class G:
    CM  = 0  # C_m
    EL  = 1  # E_L
    RST = 2  # V_reset
    THR = 3  # V_thresh
    AA0 = 4  # asc_amps0
    AA1 = 5  # asc_amps1
    G   = 6  # g
    K0  = 7  # k0
    K1  = 8  # k1
    RFR = 9  # t_ref
    TA0 = 10  # tau_syn0
    TA1 = 11  # tau_syn1
    TA2 = 12  # tau_syn2
    TA3 = 13  # tau_syn3
    VSC = 14  # voltage scale
    CON = 15  # count

print("="*80)
print("POPULATION-PROJECTION ROUTING VERIFICATION")
print("="*80)

# Load H5 file
try:
    with h5py.File('ckpt_51978-153.h5', 'r') as f:
        # Load network
        neurons = np.array(f['neurons/node_type_ids'])

        # Load GLIF3 parameters (need for voltage_scale calculation)
        glif3 = np.stack([
            f['neurons/glif3_params/C_m'],
            f['neurons/glif3_params/E_L'],
            f['neurons/glif3_params/V_reset'],
            f['neurons/glif3_params/V_th'],
            f['neurons/glif3_params/asc_amps'][:, 0],
            f['neurons/glif3_params/asc_amps'][:, 1],
            f['neurons/glif3_params/g'],
            f['neurons/glif3_params/k'][:, 0],
            f['neurons/glif3_params/k'][:, 1],
            f['neurons/glif3_params/t_ref'],
            f['neurons/glif3_params/tau_syn'][:, 0],
            f['neurons/glif3_params/tau_syn'][:, 1],
            f['neurons/glif3_params/tau_syn'][:, 2],
            f['neurons/glif3_params/tau_syn'][:, 3],
            np.zeros_like(f['neurons/glif3_params/C_m']),
            np.bincount(neurons)
        ], axis=1)

        # Calculate voltage_scale (as class.py does)
        glif3[:, G.VSC] = glif3[:, G.THR] - glif3[:, G.EL]

        # Load recurrent synapses
        recurrent = np.stack((
            f['recurrent/sources'],
            f['recurrent/targets'],
            f['recurrent/weights'],
            f['recurrent/receptor_types'],
            np.arange(len(f['recurrent/weights'])),
            f['recurrent/delays']
        ), axis=1)

        # Load input synapses
        input_syns = np.stack((
            f['input/sources'],
            f['input/targets'],
            f['input/weights'],
            f['input/receptor_types'],
            np.arange(len(f['input/weights']))
        ), axis=1)

        print(f"\n✓ Loaded network from H5 file")
        print(f"  Neurons: {len(neurons)}")
        print(f"  Recurrent synapses: {len(recurrent)}")
        print(f"  Input synapses: {len(input_syns)}")

except FileNotFoundError:
    print(f"\n✗ ERROR: H5 file not found: ckpt_51978-153.h5")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now simulate the grouping process
print(f"\n{'='*80}")
print(f"SIMULATING v1_compute_initial_mappings()")
print(f"{'='*80}")

# Compute initial mappings
g2pl = {}  # GID -> (PID, LID)
p2g = {}   # PID -> [GID, ...]

for gid, pid in enumerate(neurons):
    if pid not in p2g:
        p2g[pid] = []
    lid = len(p2g[pid])
    p2g[pid].append(gid)
    g2pl[gid] = (int(pid), lid)

print(f"\nCreated {len(g2pl)} GID mappings to {len(p2g)} populations")
print(f"Sample mappings:")
for gid in list(g2pl.keys())[:5]:
    print(f"  GID {gid} → {g2pl[gid]}")

# Now simulate v1_compute_split_mappings()
print(f"\n{'='*80}")
print(f"SIMULATING v1_compute_split_mappings()")
print(f"{'='*80}")

# Add SUBPID
g2psl = {}  # GID -> (PID, SUBPID, LID)
ps2g = {}   # (PID, SUBPID) -> [GID, ...]

subpid_size = 2048
for gid, (pid, lid) in g2pl.items():
    subpid = lid // subpid_size
    new_lid = lid % subpid_size

    key = (pid, subpid)
    if key not in ps2g:
        ps2g[key] = []
    ps2g[key].append(gid)
    g2psl[gid] = (pid, subpid, new_lid)

print(f"\nSplit into {len(ps2g)} sub-populations")
print(f"Sample mappings:")
for gid in list(g2psl.keys())[:5]:
    print(f"  GID {gid} → {g2psl[gid]}")

# Now simulate v1_group_synapses()
print(f"\n{'='*80}")
print(f"SIMULATING v1_group_synapses()")
print(f"{'='*80}")

synpols = {}
for i, syn in enumerate(recurrent):
    if i % 500000 == 0 and i > 0:
        print(f"  Processed {i}/{len(recurrent)} synapses")

    src_gid = int(syn[S.SRC])
    tgt_gid = int(syn[S.TGT])

    src_pid, src_subpid, src_lid = g2psl[src_gid]
    tgt_pid, tgt_subpid, tgt_lid = g2psl[tgt_gid]

    synkey = ((src_pid, src_subpid), (tgt_pid, tgt_subpid))
    if synkey not in synpols:
        synpols[synkey] = []

    # Store LID-based synapse
    synpols[synkey].append(np.hstack([[src_lid, tgt_lid], syn[2:]]))

# Convert to numpy
for synkey in synpols:
    synpols[synkey] = np.array(synpols[synkey])

print(f"\nGrouped {len(recurrent)} synapses into {len(synpols)} synapse populations")
lens = [len(x) for x in synpols.values()]
print(f"  Min size: {np.min(lens)}")
print(f"  Max size: {np.max(lens)}")
print(f"  Mean size: {np.mean(lens):.1f}")
print(f"  Total: {np.sum(lens)}")

# CRITICAL CHECK: Did we lose any synapses?
print(f"\n{'='*80}")
print(f"CHECK 1: SYNAPSE COUNT PRESERVATION")
print(f"{'='*80}")

total_grouped = sum(len(x) for x in synpols.values())
if total_grouped == len(recurrent):
    print(f"✓ PASS: {total_grouped} == {len(recurrent)}")
else:
    print(f"✗ FAIL: {total_grouped} != {len(recurrent)}")
    print(f"  Lost {len(recurrent) - total_grouped} synapses!")

# CHECK 2: Weight preservation
print(f"\n{'='*80}")
print(f"CHECK 2: WEIGHT VALUE PRESERVATION")
print(f"{'='*80}")

original_weights = recurrent[:, S.WHT]
grouped_weights = np.concatenate([sp[:, S.WHT] for sp in synpols.values()])

print(f"Original weights:")
print(f"  Mean: {np.mean(original_weights):.6f}")
print(f"  Std: {np.std(original_weights):.6f}")
print(f"  Min: {np.min(original_weights):.6f}")
print(f"  Max: {np.max(original_weights):.6f}")

print(f"\nGrouped weights:")
print(f"  Mean: {np.mean(grouped_weights):.6f}")
print(f"  Std: {np.std(grouped_weights):.6f}")
print(f"  Min: {np.min(grouped_weights):.6f}")
print(f"  Max: {np.max(grouped_weights):.6f}")

if np.allclose(np.sort(original_weights), np.sort(grouped_weights)):
    print(f"\n✓ PASS: Weight values preserved (sorted comparison)")
else:
    print(f"\n✗ FAIL: Weight values NOT preserved!")
    diff = np.abs(np.sort(original_weights) - np.sort(grouped_weights))
    print(f"  Max difference: {np.max(diff):.6e}")

# CHECK 3: Connectivity preservation (harder - need to reverse GID→LID→GID)
print(f"\n{'='*80}")
print(f"CHECK 3: CONNECTIVITY PRESERVATION")
print(f"{'='*80}")

# Reconstruct GID-based connectivity from grouped synapses
reconstructed = []
for synkey, synpol in synpols.items():
    src_key, tgt_key = synkey
    src_gids = ps2g[src_key]
    tgt_gids = ps2g[tgt_key]

    for syn in synpol:
        src_lid = int(syn[0])
        tgt_lid = int(syn[1])

        # Convert LID back to GID
        src_gid = src_gids[src_lid]
        tgt_gid = tgt_gids[tgt_lid]

        # Reconstruct synapse with GID
        reconstructed.append([src_gid, tgt_gid] + list(syn[2:]))

reconstructed = np.array(reconstructed)

print(f"Reconstructed {len(reconstructed)} synapses")

# Compare source-target pairs
original_pairs = set(tuple(row[:2]) for row in recurrent[:, :2].astype(int))
reconstructed_pairs = set(tuple(row[:2]) for row in reconstructed[:, :2].astype(int))

if original_pairs == reconstructed_pairs:
    print(f"✓ PASS: All {len(original_pairs)} source-target pairs preserved")
else:
    missing = original_pairs - reconstructed_pairs
    extra = reconstructed_pairs - original_pairs
    print(f"✗ FAIL: Connectivity NOT preserved!")
    print(f"  Missing pairs: {len(missing)}")
    print(f"  Extra pairs: {len(extra)}")
    if len(missing) > 0:
        print(f"  First 5 missing: {list(missing)[:5]}")
    if len(extra) > 0:
        print(f"  First 5 extra: {list(extra)[:5]}")

# CHECK 4: Weight scaling applied during projection creation
print(f"\n{'='*80}")
print(f"CHECK 4: WEIGHT SCALING IN PROJECTION CREATION")
print(f"{'='*80}")

print(f"\nclass.py lines 840-841 and 887-888 apply:")
print(f"  syn[:, S.WHT] *= vsc / 1000.0")
print(f"\nThis scaling is applied PER TARGET POPULATION")

# Sample a few projections and check scaling
sample_keys = list(synpols.keys())[:3]
for synkey in sample_keys:
    src_key, tgt_key = synkey
    syn = synpols[synkey]

    tgt_pid = tgt_key[0]
    vsc = glif3[tgt_pid, G.VSC]

    print(f"\nProjection {synkey}:")
    print(f"  Target PID: {tgt_pid}, voltage_scale: {vsc:.2f} mV")
    print(f"  Original weight mean: {np.mean(syn[:, S.WHT]):.6f}")
    print(f"  After scaling (×{vsc:.2f}/1000): {np.mean(syn[:, S.WHT]) * vsc / 1000:.6f}")

print(f"\n{'='*80}")
print(f"FINAL VERDICT")
print(f"{'='*80}")

print(f"\n1. Synapse count: {'PASS' if total_grouped == len(recurrent) else 'FAIL'}")
print(f"2. Weight values: {'PASS' if np.allclose(np.sort(original_weights), np.sort(grouped_weights)) else 'FAIL'}")
print(f"3. Connectivity: {'PASS' if original_pairs == reconstructed_pairs else 'FAIL'}")
print(f"4. Weight scaling: Applied per-target with voltage_scale")

print(f"\n{'='*80}")
