#!/usr/bin/env python3
"""
Check if population mapping preserves global IDs correctly.

We know GID 37955 (class 8) fired when we expected class 2.
Let's trace where GID 37955 ends up in the population structure.
"""

import numpy as np
import h5py

print("Loading checkpoint...")
# Load network
with h5py.File('ckpt_51978-153.h5', 'r') as file:
    neurons = np.array(file['neurons/node_type_ids'])
    output_neurons = np.array(file['readout/neuron_ids'])

    # Load LGN synapses
    lgn_src = np.array(file['input/sources'])
    lgn_tgt = np.array(file['input/targets'])
    lgn_wht = np.array(file['input/weights'])
    lgn_rty = np.array(file['input/receptor_types'])

    # Stack into synapse array: [src, tgt, weight, receptor_type]
    lgn_synapses = np.column_stack([lgn_src, lgn_tgt, lgn_wht, lgn_rty])

print(f"Total neurons: {len(neurons)}")
print(f"Output neurons: {len(output_neurons)}")
print(f"LGN synapses: {len(lgn_synapses)}")

# Check GID 37955
gid = 37955
if gid < len(neurons):
    neuron_type = neurons[gid]
    print(f"\nGID {gid}:")
    print(f"  Type (from neurons array): {neuron_type}")

    # Is it an output neuron?
    if gid in output_neurons:
        idx = np.where(output_neurons == gid)[0][0]
        expected_class = idx // 30
        print(f"  Is output neuron: YES")
        print(f"  Position in output array: {idx}")
        print(f"  Expected class: {expected_class}")
    else:
        print(f"  Is output neuron: NO")
else:
    print(f"GID {gid} out of range!")

# Now trace through the population mapping logic
# (same as spynnaker_newclass.py)
p2g = {}
for gid_idx, pid in enumerate(neurons):
    if pid not in p2g:
        p2g[pid] = []
    p2g[pid].append(gid_idx)

print(f"\nPopulation mapping (p2g): {len(p2g)} neuron types")

# Check which population GID 37955 is in
for pid, gids in p2g.items():
    if gid in gids:
        pos = gids.index(gid)
        print(f"\nGID {gid} is in population {pid}:")
        print(f"  Position in population: {pos} / {len(gids)}")
        print(f"  This population has {len(gids)} neurons")
        break

# Now split populations
target = 256
g2psl = {}
ps2g = {}

for pid, gids in p2g.items():
    if len(gids) > target:
        n_split = int(np.ceil(len(gids) / target))
    else:
        n_split = 1

    for subpid in range(n_split):
        key = (pid, subpid)
        start = subpid * target
        end = (subpid + 1) * target
        subgids = gids[start:end]
        ps2g[key] = subgids

        for lid, g in enumerate(subgids):
            g2psl[g] = (pid, subpid, lid)

# Check GID 37955 mapping
if gid in g2psl:
    pid, subpid, lid = g2psl[gid]
    print(f"\nAfter splitting:")
    print(f"  GID {gid} → population ({pid}, {subpid}), local ID {lid}")
    print(f"  This subpopulation has {len(ps2g[(pid, subpid)])} neurons")
    print(f"  Local ID {lid} should correspond to global ID {ps2g[(pid, subpid)][lid]}")
    if ps2g[(pid, subpid)][lid] == gid:
        print(f"  ✓ Mapping is CORRECT")
    else:
        print(f"  ✗ Mapping is WRONG! lid {lid} maps to GID {ps2g[(pid, subpid)][lid]}")
else:
    print(f"\nGID {gid} not found in g2psl!")

# Check if class 2 output neurons are where we expect
print(f"\n=== CLASS 2 OUTPUT NEURONS ===")
class2_output = output_neurons[60:90]
print(f"Class 2 should have these {len(class2_output)} neurons:")
for i, gid in enumerate(class2_output[:5]):
    if gid in g2psl:
        pid, subpid, lid = g2psl[gid]
        print(f"  GID {gid:5d} → pop ({pid:2d}, {subpid}), lid {lid:3d}")
    else:
        print(f"  GID {gid:5d} → NOT IN MAPPING!")

# Now trace LGN synapses targeting class 2 neurons
print(f"\n=== LGN SYNAPSES TO CLASS 2 NEURONS ===")
class2_set = set(class2_output)
class2_lgn_synapses = []

for i, syn in enumerate(lgn_synapses):
    tgt_gid = int(syn[1])  # Target is column 1
    if tgt_gid in class2_set:
        class2_lgn_synapses.append((i, syn))
        if len(class2_lgn_synapses) >= 10:  # Get first 10
            break

print(f"Found {len(class2_lgn_synapses)} LGN synapses targeting class 2 neurons (first 10):")
for syn_idx, syn in class2_lgn_synapses[:3]:  # Print first 3
    src_gid = int(syn[0])
    tgt_gid = int(syn[1])
    weight = syn[2]

    print(f"\nSynapse {syn_idx}: LGN {src_gid:5d} → V1 {tgt_gid:5d}, weight={weight:.6f}")

    # Check target mapping
    if tgt_gid in g2psl:
        tgt_pid, tgt_subpid, tgt_lid = g2psl[tgt_gid]
        print(f"  Target V1 {tgt_gid} → population ({tgt_pid}, {tgt_subpid}), local ID {tgt_lid}")

        # Verify round-trip
        if tgt_lid < len(ps2g[(tgt_pid, tgt_subpid)]):
            reconstructed = ps2g[(tgt_pid, tgt_subpid)][tgt_lid]
            if reconstructed == tgt_gid:
                print(f"  ✓ Mapping correct: ps2g[({tgt_pid}, {tgt_subpid})][{tgt_lid}] = {reconstructed}")
            else:
                print(f"  ✗ ERROR: ps2g[({tgt_pid}, {tgt_subpid})][{tgt_lid}] = {reconstructed} ≠ {tgt_gid}")
        else:
            print(f"  ✗ ERROR: lid {tgt_lid} out of range for population ({tgt_pid}, {tgt_subpid}) with {len(ps2g[(tgt_pid, tgt_subpid)])} neurons")

# Now check: Are there LGN synapses targeting GID 37955 that shouldn't be there?
print(f"\n=== LGN SYNAPSES TO GID 37955 (class 8 neuron that fired) ===")
synapses_to_37955 = []
for i, syn in enumerate(lgn_synapses):
    if int(syn[1]) == 37955:
        synapses_to_37955.append((i, syn))

print(f"Found {len(synapses_to_37955)} LGN synapses targeting GID 37955:")
for syn_idx, syn in synapses_to_37955[:3]:
    src_gid = int(syn[0])
    weight = syn[2]
    print(f"  Synapse {syn_idx}: LGN {src_gid:5d} → 37955, weight={weight:.6f}")

# ============================================================================
# KEY DIAGNOSTIC: Check LGN population creation order
# ============================================================================
print(f"\n" + "="*80)
print("CHECKING LGN POPULATION CREATION (lgn_group_exact + lgn_group_similar)")
print("="*80)

# Simulate lgn_group_exact
print("\nStep 1: lgn_group_exact - group LGN by target populations")
l2t = {}  # LGN GID -> set of target populations
for syn in lgn_synapses:
    lgn_gid = int(syn[0])
    tgt_gid = int(syn[1])

    if lgn_gid not in l2t:
        l2t[lgn_gid] = set()

    # Get target population
    if tgt_gid in g2psl:
        pid, subpid, _ = g2psl[tgt_gid]
        l2t[lgn_gid].add((pid, subpid))

# Invert: target pattern -> list of LGN GIDs
t2l = {}
for lgn_gid, tgtpols in l2t.items():
    tgtkey = tuple(sorted(tgtpols))  # Sort for consistency
    if tgtkey not in t2l:
        t2l[tgtkey] = []
    t2l[tgtkey].append(lgn_gid)

print(f"  {len(l2t)} LGN neurons split into {len(t2l)} groups by target pattern")
print(f"  Largest group: {max(len(v) for v in t2l.values())} LGN neurons")

# Now create l2pl as spynnaker does
print("\nStep 2: Creating l2pl mapping (as lgn_group_similar does)")
l2pl = {}
for pid, item in enumerate(t2l.items()):
    tgtpols, lgns = item
    for lid, lgn_gid in enumerate(lgns):
        l2pl[lgn_gid] = (pid, lid)

print(f"  Created l2pl with {len(l2pl)} LGN neurons")

# Verify: Check a few LGN neurons that target class 2
print("\nStep 3: Verify LGN mapping for neurons targeting class 2")
lgn_to_class2 = set()
for syn in lgn_synapses[:10000]:  # Check first 10k synapses
    if int(syn[1]) in class2_set:
        lgn_to_class2.add(int(syn[0]))

print(f"  Found {len(lgn_to_class2)} LGN neurons targeting class 2 (from first 10k synapses)")
for lgn_gid in list(lgn_to_class2)[:3]:
    if lgn_gid in l2pl:
        pid, lid = l2pl[lgn_gid]
        print(f"  LGN {lgn_gid:5d} → population {pid}, local ID {lid}")

        # Check: if we iterate t2l.items(), does pid'th item have lgn_gid at position lid?
        items_list = list(t2l.items())
        if pid < len(items_list):
            tgtpols, lgns = items_list[pid]
            if lid < len(lgns):
                actual_lgn = lgns[lid]
                if actual_lgn == lgn_gid:
                    print(f"    ✓ Correct: t2l.items()[{pid}][1][{lid}] = {actual_lgn}")
                else:
                    print(f"    ✗ ERROR: t2l.items()[{pid}][1][{lid}] = {actual_lgn} ≠ {lgn_gid}")
            else:
                print(f"    ✗ ERROR: lid {lid} out of range (population has {len(lgns)} neurons)")

print("="*80)
