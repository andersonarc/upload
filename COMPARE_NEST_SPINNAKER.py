#!/usr/bin/env python3
"""
Compare NEST vs SpiNNaker setup for classes 7-8 neurons.
Find what's different that causes constant firing in SpiNNaker.
"""

import numpy as np
import h5py

print("Loading checkpoint...")
with h5py.File('ckpt_51978-153.h5', 'r') as file:
    neurons = np.array(file['neurons/node_type_ids'])
    output_neurons = np.array(file['readout/neuron_ids'])
    glif3_params = {
        'C_m': np.array(file['neurons/glif3_params/C_m']),
        'E_L': np.array(file['neurons/glif3_params/E_L']),
        'V_reset': np.array(file['neurons/glif3_params/V_reset']),
        'V_th': np.array(file['neurons/glif3_params/V_th']),
        'asc_amps': np.array(file['neurons/glif3_params/asc_amps']),
        'k': np.array(file['neurons/glif3_params/k']),
        'g': np.array(file['neurons/glif3_params/g']),
        't_ref': np.array(file['neurons/glif3_params/t_ref']),
        'tau_syn': np.array(file['neurons/glif3_params/tau_syn'])
    }

    # Load synapses
    lgn_src = np.array(file['input/sources'])
    lgn_tgt = np.array(file['input/targets'])
    lgn_wht = np.array(file['input/weights'])
    lgn_rty = np.array(file['input/receptor_types'])

    rec_src = np.array(file['recurrent/sources'])
    rec_tgt = np.array(file['recurrent/targets'])
    rec_wht = np.array(file['recurrent/weights'])
    rec_rty = np.array(file['recurrent/receptor_types'])
    rec_dly = np.array(file['recurrent/delays'])

    bkg_wht = np.array(file['input/bkg_weights'])

print("\n" + "="*80)
print("COMPARING INPUT TO CLASSES 7-8 VS OTHER CLASSES")
print("="*80)

# Get class 7-8 neurons
class7_gids = set(output_neurons[210:240])
class8_gids = set(output_neurons[240:270])
class78_gids = class7_gids | class8_gids

# Get class 2 neurons (expected to fire for digit 2)
class2_gids = set(output_neurons[60:90])

# Get class 0 neurons (baseline comparison)
class0_gids = set(output_neurons[0:30])

print(f"\nClass 0: {len(class0_gids)} neurons")
print(f"Class 2: {len(class2_gids)} neurons")
print(f"Class 7-8: {len(class78_gids)} neurons")

# Count LGN synapses
def count_synapses(targets, src, tgt, wht, rty):
    """Count synapses targeting a set of neurons by receptor type"""
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    weights = {0: [], 1: [], 2: [], 3: []}

    for i in range(len(src)):
        if tgt[i] in targets:
            r = int(rty[i])
            counts[r] += 1
            weights[r].append(wht[i])

    return counts, weights

print("\n--- LGN SYNAPSES ---")
c0_lgn_counts, c0_lgn_weights = count_synapses(class0_gids, lgn_src, lgn_tgt, lgn_wht, lgn_rty)
c2_lgn_counts, c2_lgn_weights = count_synapses(class2_gids, lgn_src, lgn_tgt, lgn_wht, lgn_rty)
c78_lgn_counts, c78_lgn_weights = count_synapses(class78_gids, lgn_src, lgn_tgt, lgn_wht, lgn_rty)

print(f"\nClass 0 LGN synapses by receptor type:")
for r in range(4):
    if c0_lgn_counts[r] > 0:
        print(f"  Receptor {r}: {c0_lgn_counts[r]} synapses, mean weight: {np.mean(c0_lgn_weights[r]):.6f}")

print(f"\nClass 2 LGN synapses by receptor type:")
for r in range(4):
    if c2_lgn_counts[r] > 0:
        print(f"  Receptor {r}: {c2_lgn_counts[r]} synapses, mean weight: {np.mean(c2_lgn_weights[r]):.6f}")

print(f"\nClass 7-8 LGN synapses by receptor type:")
for r in range(4):
    if c78_lgn_counts[r] > 0:
        print(f"  Receptor {r}: {c78_lgn_counts[r]} synapses, mean weight: {np.mean(c78_lgn_weights[r]):.6f}")

print("\n--- RECURRENT SYNAPSES ---")
c0_rec_counts, c0_rec_weights = count_synapses(class0_gids, rec_src, rec_tgt, rec_wht, rec_rty)
c2_rec_counts, c2_rec_weights = count_synapses(class2_gids, rec_src, rec_tgt, rec_wht, rec_rty)
c78_rec_counts, c78_rec_weights = count_synapses(class78_gids, rec_src, rec_tgt, rec_wht, rec_rty)

print(f"\nClass 0 recurrent synapses by receptor type:")
for r in range(4):
    if c0_rec_counts[r] > 0:
        print(f"  Receptor {r}: {c0_rec_counts[r]} synapses, mean weight: {np.mean(c0_rec_weights[r]):.6f}")

print(f"\nClass 2 recurrent synapses by receptor type:")
for r in range(4):
    if c2_rec_counts[r] > 0:
        print(f"  Receptor {r}: {c2_rec_counts[r]} synapses, mean weight: {np.mean(c2_rec_weights[r]):.6f}")

print(f"\nClass 7-8 recurrent synapses by receptor type:")
for r in range(4):
    if c78_rec_counts[r] > 0:
        print(f"  Receptor {r}: {c78_rec_counts[r]} synapses, mean weight: {np.mean(c78_rec_weights[r]):.6f}")

print("\n--- BACKGROUND WEIGHTS ---")
# Background weights are stored as: neuron_id * 4 + receptor_type
def get_bkg_weights(gids):
    """Get background weights for a set of neurons"""
    weights_by_receptor = {0: [], 1: [], 2: [], 3: []}
    for gid in gids:
        for r in range(4):
            idx = gid * 4 + r
            if idx < len(bkg_wht):
                weights_by_receptor[r].append(bkg_wht[idx])
    return weights_by_receptor

c0_bkg = get_bkg_weights(class0_gids)
c2_bkg = get_bkg_weights(class2_gids)
c78_bkg = get_bkg_weights(class78_gids)

print(f"\nClass 0 background weights by receptor type:")
for r in range(4):
    if len(c0_bkg[r]) > 0:
        print(f"  Receptor {r}: mean={np.mean(c0_bkg[r]):.6f}, std={np.std(c0_bkg[r]):.6f}")

print(f"\nClass 2 background weights by receptor type:")
for r in range(4):
    if len(c2_bkg[r]) > 0:
        print(f"  Receptor {r}: mean={np.mean(c2_bkg[r]):.6f}, std={np.std(c2_bkg[r]):.6f}")

print(f"\nClass 7-8 background weights by receptor type:")
for r in range(4):
    if len(c78_bkg[r]) > 0:
        print(f"  Receptor {r}: mean={np.mean(c78_bkg[r]):.6f}, std={np.std(c78_bkg[r]):.6f}")

print("\n--- NEURON PARAMETERS ---")
# Get neuron types for each class
c0_types = [neurons[gid] for gid in class0_gids]
c2_types = [neurons[gid] for gid in class2_gids]
c78_types = [neurons[gid] for gid in class78_gids]

print(f"\nClass 0 neuron types: {sorted(set(c0_types))}")
print(f"Class 2 neuron types: {sorted(set(c2_types))}")
print(f"Class 7-8 neuron types: {sorted(set(c78_types))}")

# Compare voltage scale (V_th - E_L)
def get_voltage_scales(gids):
    scales = []
    for gid in gids:
        ntype = neurons[gid]
        vsc = glif3_params['V_th'][ntype] - glif3_params['E_L'][ntype]
        scales.append(vsc)
    return scales

c0_vsc = get_voltage_scales(class0_gids)
c2_vsc = get_voltage_scales(class2_gids)
c78_vsc = get_voltage_scales(class78_gids)

print(f"\nVoltage scales (V_th - E_L):")
print(f"  Class 0: mean={np.mean(c0_vsc):.2f} mV, range=[{np.min(c0_vsc):.2f}, {np.max(c0_vsc):.2f}]")
print(f"  Class 2: mean={np.mean(c2_vsc):.2f} mV, range=[{np.min(c2_vsc):.2f}, {np.max(c2_vsc):.2f}]")
print(f"  Class 7-8: mean={np.mean(c78_vsc):.2f} mV, range=[{np.min(c78_vsc):.2f}, {np.max(c78_vsc):.2f}]")

print("\n" + "="*80)
print("ANALYSIS: What makes classes 7-8 different?")
print("="*80)

# Compare total excitatory vs inhibitory input
print("\nExcitatory (receptors 0,2) vs Inhibitory (receptors 1,3) synapse counts:")
for name, lgn_c, rec_c in [("Class 0", c0_lgn_counts, c0_rec_counts),
                            ("Class 2", c2_lgn_counts, c2_rec_counts),
                            ("Class 7-8", c78_lgn_counts, c78_rec_counts)]:
    exc = lgn_c[0] + lgn_c[2] + rec_c[0] + rec_c[2]
    inh = lgn_c[1] + lgn_c[3] + rec_c[1] + rec_c[3]
    print(f"  {name}: {exc} excitatory, {inh} inhibitory (ratio {exc/inh:.2f})")
