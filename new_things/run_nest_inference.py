#!/usr/bin/env python
# coding: utf-8

"""
Complete NEST Simulator inference script for Mouse V1 Cortex Model
Ported from SpiNNaker implementation
"""

import os
import h5py
import numpy as np
import pyNN.nest as sim

np.random.seed(1)

# Set target index (which sample to process)
TARGET_INDEX = int(os.environ.get('TARGET_INDEX', 0))

print("=" * 80)
print(f"NEST Simulator V1 Cortex Inference - Sample {TARGET_INDEX}")
print("=" * 80)

# ============================================================================
# Index Classes
# ============================================================================

class S:  # Synapse indices
    SRC, TGT, WHT, RTY, ID, DLY = 0, 1, 2, 3, 4, 5

class G:  # GLIF3 parameter indices
    CM, EL, RST, THR, AA0, AA1, G, K0, K1 = 0, 1, 2, 3, 4, 5, 6, 7, 8
    RFR, TA0, TA1, TA2, TA3, VSC, CON = 9, 10, 11, 12, 13, 14, 15

class I:  # IF_curr_exp parameter indices
    CM, TAU, VRT, RST, THR, RFR, EXC, INH, OFT, CON = range(10)

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_network(path):
    """Load network structure and parameters from HDF5."""
    print(f"Loading network from {path}...")
    with h5py.File(path, 'r') as file:
        network = {}
        network['neurons'] = np.array(file['neurons/node_type_ids'])

        # Load GLIF3 parameters
        network['glif3'] = np.stack([
            file['neurons/glif3_params/C_m'],
            file['neurons/glif3_params/E_L'],
            file['neurons/glif3_params/V_reset'],
            file['neurons/glif3_params/V_th'],
            file['neurons/glif3_params/asc_amps'][:, 0],
            file['neurons/glif3_params/asc_amps'][:, 1],
            file['neurons/glif3_params/g'],
            file['neurons/glif3_params/k'][:, 0],
            file['neurons/glif3_params/k'][:, 1],
            file['neurons/glif3_params/t_ref'],
            file['neurons/glif3_params/tau_syn'][:, 0],
            file['neurons/glif3_params/tau_syn'][:, 1],
            file['neurons/glif3_params/tau_syn'][:, 2],
            file['neurons/glif3_params/tau_syn'][:, 3],
            np.zeros_like(file['neurons/glif3_params/C_m']),
            np.bincount(network['neurons'])
        ], axis=1)

        # Scale parameters
        network['glif3'][:, G.CM] /= 1000.0  # pF -> nF
        network['glif3'][:, G.G] /= 1000.0   # nS -> uS
        network['glif3'][:, G.VSC] = network['glif3'][:, G.THR] - network['glif3'][:, G.EL]
        network['glif3'][:, G.AA0] /= 1000.0  # pA -> nA
        network['glif3'][:, G.AA1] /= 1000.0  # pA -> nA

        # Load synapses
        network['recurrent'] = np.stack((
            file['recurrent/sources'],
            file['recurrent/targets'],
            file['recurrent/weights'],
            file['recurrent/receptor_types'],
            np.arange(len(file['recurrent/weights'])),
            file['recurrent/delays']
        ), axis=1)

        network['input'] = np.stack((
            file['input/sources'],
            file['input/targets'],
            file['input/weights'],
            file['input/receptor_types'],
            np.arange(len(file['input/weights']))
        ), axis=1)

        network['output'] = np.array(file['readout/neuron_ids'])

    print(f"  Neurons: {len(network['neurons'])}")
    print(f"  Recurrent synapses: {len(network['recurrent'])}")
    print(f"  Input synapses: {len(network['input'])}")
    print(f"  Output neurons: {len(network['output'])}")
    return network

def load_dataset(path, target_idx=0, n_samples=1):
    """Load spike train dataset from HDF5."""
    print(f"Loading dataset from {path}...")
    with h5py.File(path, 'r') as file:
        dataset = {
            'spike_probabilities': np.array(file['spike_trains'][target_idx:target_idx+n_samples]),
            'response_window': np.array(file['response_window']),
            'labels': np.array(file['labels'][target_idx:target_idx+n_samples])
        }
    print(f"  Samples: {dataset['spike_probabilities'].shape[0]}")
    print(f"  Sequence length: {dataset['spike_probabilities'].shape[1]} ms")
    print(f"  LGN neurons: {dataset['spike_probabilities'].shape[2]}")
    print(f"  Label: {dataset['labels'][0]}")
    return dataset

def create_spike_times(spike_trains, timestep=1.0, scale=1.0):
    """Convert spike probability trains to spike times."""
    lgn_size = spike_trains.shape[1]
    spike_times = []

    for i in range(lgn_size):
        times = []
        for t in range(spike_trains.shape[0]):
            if np.clip((spike_trains[t, i] / 1.3) * scale, 0.0, 1.0) > np.random.rand():
                times.append(float(t * timestep))
        spike_times.append(times if len(times) > 0 else [1000.0])  # Dummy spike if empty

    return spike_times

# ============================================================================
# Parameter Conversion
# ============================================================================

def glif3_to_if_curr_exp(network):
    """Convert GLIF3 parameters to IF_curr_exp."""
    glif3 = network['glif3']
    network['ice'] = np.stack([
        glif3[:, G.CM],                  # C_m
        glif3[:, G.CM] / glif3[:, G.G],  # tau_m
        glif3[:, G.EL],                  # V_rest
        glif3[:, G.RST],                 # V_reset
        glif3[:, G.THR],                 # V_thresh
        glif3[:, G.RFR],                 # tau_refrac
        glif3[:, G.TA0],                 # tau_syn_E
        glif3[:, G.TA2],                 # tau_syn_I
        np.zeros(len(glif3)),            # i_offset
        glif3[:, G.CON]                  # count
    ], axis=1)

def ice_to_dict(i):
    """Convert IF_curr_exp parameters to dictionary."""
    return {
        'cm': i[I.CM],
        'i_offset': i[I.OFT],
        'tau_m': i[I.TAU],
        'tau_refrac': i[I.RFR],
        'tau_syn_E': i[I.EXC],
        'tau_syn_I': i[I.INH],
        'v_reset': i[I.RST],
        'v_rest': i[I.VRT],
        'v_thresh': i[I.THR]
    }

# ============================================================================
# Population Grouping
# ============================================================================

def compute_population_mappings(neurons, target_size=256):
    """Compute population mappings with splitting."""
    # Initial grouping by neuron type
    p2g = {}
    for gid, pid in enumerate(neurons):
        if pid not in p2g:
            p2g[pid] = []
        p2g[pid].append(gid)

    print(f"Initial: {len(neurons)} neurons in {len(p2g)} types")

    # Split large populations
    ps2g, g2psl = {}, {}
    for pid, gids in p2g.items():
        n_split = max(1, int(np.ceil(len(gids) / target_size)))
        for subpid in range(n_split):
            key = (pid, subpid)
            start, end = subpid * target_size, (subpid + 1) * target_size
            subgids = gids[start:end]
            ps2g[key] = subgids
            for lid, gid in enumerate(subgids):
                g2psl[gid] = (pid, subpid, lid)

    print(f"Split: {len(ps2g)} subpopulations")
    return ps2g, g2psl

def group_synapses(syns, g2psl, synapse_type="V1"):
    """Group synapses by source and target populations."""
    synpols = {}
    for i, syn in enumerate(syns):
        if i % 500000 == 0 and i > 0:
            print(f"  {i}/{len(syns)} synapses processed")

        src_gid, tgt_gid = int(syn[S.SRC]), int(syn[S.TGT])
        src_pid, src_subpid, src_lid = g2psl[src_gid]
        tgt_pid, tgt_subpid, tgt_lid = g2psl[tgt_gid]

        synkey = ((src_pid, src_subpid), (tgt_pid, tgt_subpid))
        if synkey not in synpols:
            synpols[synkey] = []
        synpols[synkey].append(np.hstack([[src_lid, tgt_lid], syn[2:]]))

    for key in synpols:
        synpols[key] = np.array(synpols[key])

    print(f"{synapse_type} synapses: {len(syns)} -> {len(synpols)} groups")
    return synpols

def group_lgn(input_syns, g2psl, threshold=0.15):
    """Group LGN neurons by connectivity similarity."""
    # Group by exact targets
    l2t = {}
    for syn in input_syns:
        lgn, tgt = int(syn[S.SRC]), int(syn[S.TGT])
        if lgn not in l2t:
            l2t[lgn] = set()
        pid, subpid, _ = g2psl[tgt]
        l2t[lgn].add((pid, subpid))

    t2l = {}
    for lgn, targets in l2t.items():
        key = tuple(targets)
        if key not in t2l:
            t2l[key] = []
        t2l[key].append(lgn)

    print(f"LGN exact grouping: {len(l2t)} -> {len(t2l)} groups")

    # Iterative merging by similarity
    for iteration in range(3):
        tm2l, used = {}, []
        for i, (tgtpols, lgns) in enumerate(t2l.items()):
            if i in used:
                continue
            used.append(i)

            found = False
            for j, (other_tgtpols, other_lgns) in enumerate(t2l.items()):
                if j in used:
                    continue

                merged = tuple(set(tgtpols + other_tgtpols))
                if (len(merged) - len(tgtpols)) / len(tgtpols) < threshold:
                    tm2l[merged] = np.hstack([lgns, other_lgns])
                    used.append(j)
                    found = True
                    break

            if not found:
                tm2l[tgtpols] = np.array(lgns)

        print(f"  Iteration {iteration+1}: {len(t2l)} -> {len(tm2l)} groups")
        t2l = tm2l

    # Create reverse mapping
    l2pl = {}
    for pid, (_, lgns) in enumerate(t2l.items()):
        for lid, lgn in enumerate(lgns):
            l2pl[lgn] = (pid, lid)

    return t2l, l2pl

def group_lgn_synapses(syns, l2pl, g2psl):
    """Group LGN->V1 synapses."""
    synpols = {}
    for i, syn in enumerate(syns):
        if i % 500000 == 0 and i > 0:
            print(f"  {i}/{len(syns)} synapses processed")

        src_gid, tgt_gid = int(syn[S.SRC]), int(syn[S.TGT])
        lgn_pid, lgn_lid = l2pl[src_gid]
        tgt_pid, tgt_subpid, tgt_lid = g2psl[tgt_gid]

        synkey = (lgn_pid, (tgt_pid, tgt_subpid))
        if synkey not in synpols:
            synpols[synkey] = []
        synpols[synkey].append(np.hstack([[lgn_lid, tgt_lid], syn[2:]]))

    for key in synpols:
        synpols[key] = np.array(synpols[key])

    print(f"LGN synapses: {len(syns)} -> {len(synpols)} groups")
    return synpols

# ============================================================================
# Network Creation
# ============================================================================

def create_v1_populations(ice_params, ps2g, v1_synpols, glif3_params):
    """Create V1 neuron populations and connections."""
    print("Creating V1 populations...")
    V1 = {}

    for key, gids in ps2g.items():
        pid, subpid = key
        V1[key] = sim.Population(
            len(gids),
            sim.IF_curr_exp,
            cellparams=ice_to_dict(ice_params[pid]),
            label=f'V1_{pid}_{subpid}'
        )
        V1[key].record(['spikes', 'v'])

    print(f"  Created {len(V1)} populations")

    # Create projections
    print("Creating V1 projections...")
    n_proj = 0
    for synkey, syn in v1_synpols.items():
        src_key, tgt_key = synkey

        # Scale weights
        vsc = glif3_params[tgt_key[0], G.VSC]
        syn_copy = syn.copy()
        syn_copy[:, S.WHT] *= vsc / 1000.0

        # Determine receptor type
        receptor_type = 'excitatory' if np.mean(syn[:, S.RTY]) < 1.5 else 'inhibitory'

        sim.Projection(
            V1[src_key], V1[tgt_key],
            sim.FromListConnector(syn_copy[:, [S.SRC, S.TGT, S.WHT, S.DLY]]),
            receptor_type=receptor_type
        )
        n_proj += 1

        if n_proj % 1000 == 0:
            print(f"  {n_proj}/{len(v1_synpols)} projections")

    print(f"  Created {n_proj} projections")
    return V1

def create_lgn_populations(V1, spike_times, tm2l, lgn_synpols, glif3_params):
    """Create LGN input populations and connections."""
    print("Creating LGN populations...")
    LGN = []

    for i, (_, lgns) in enumerate(tm2l.items()):
        spike_times_list = [spike_times[lgn] for lgn in lgns]
        LGN.append(sim.Population(
            len(lgns),
            sim.SpikeSourceArray,
            cellparams={'spike_times': spike_times_list},
            label=f'LGN_{i}'
        ))
        LGN[i].record(['spikes'])

    print(f"  Created {len(LGN)} populations")

    # Create projections
    print("Creating LGN projections...")
    n_proj = 0
    for synkey, syn in lgn_synpols.items():
        lgn_pid, tgt_key = synkey

        # Scale weights
        vsc = glif3_params[tgt_key[0], G.VSC]
        syn_copy = syn.copy()
        syn_copy[:, S.WHT] *= vsc / 1000.0

        receptor_type = 'excitatory' if np.mean(syn[:, S.RTY]) < 1.5 else 'inhibitory'

        sim.Projection(
            LGN[lgn_pid], V1[tgt_key],
            sim.FromListConnector(syn_copy[:, [S.SRC, S.TGT, S.WHT]], column_names=['weight']),
            receptor_type=receptor_type
        )
        n_proj += 1

        if n_proj % 1000 == 0:
            print(f"  {n_proj}/{len(lgn_synpols)} projections")

    print(f"  Created {n_proj} projections")
    return LGN

def create_readout_views(output_neurons, ps2g, g2psl, V1):
    """Create views for readout neurons."""
    nnpols = {}
    for gid in output_neurons:
        pid, subpid, lid = g2psl[gid]
        key = (pid, subpid)
        if key not in nnpols:
            nnpols[key] = []
        nnpols[key].append(lid)

    readouts = []
    for key, lids in nnpols.items():
        view = sim.PopulationView(V1[key], lids)
        view.record(['spikes', 'v'])
        readouts.append((key, lids, view))

    print(f"Created {len(readouts)} readout views")
    return readouts, nnpols

# ============================================================================
# Analysis
# ============================================================================

def analyze_results(readouts, nnpols, output_neurons, ps2g, label):
    """Analyze simulation results and compute predictions."""
    print("\n" + "=" * 80)
    print("Analyzing results...")

    # Build spike train map
    gid2train = {}
    for key, lids, view in readouts:
        gids = ps2g[key]
        spiketrains = view.get_data('spikes').segments[0].spiketrains
        for lid, spiketrain in zip(lids, spiketrains):
            gid = gids[lid]
            gid2train[gid] = spiketrain

    # Compute votes for each class (30 neurons per class)
    time_windows = [
        ('50-200 ms', 50, 200),
        ('50-150 ms', 50, 150),
        ('50-100 ms (target)', 50, 100),
    ]

    for window_name, t_start, t_end in time_windows:
        votes = np.zeros(10)
        for class_idx in range(10):
            start_idx = class_idx * 30
            end_idx = (class_idx + 1) * 30
            keys = output_neurons[start_idx:end_idx]

            for key in keys:
                if key in gid2train:
                    spiketrain = gid2train[key]
                    mask = (spiketrain >= t_start) & (spiketrain < t_end)
                    votes[class_idx] += mask.sum()

        sorted_order = votes.argsort()
        print(f"\n{window_name}:")
        print(f"  Votes: {votes}")
        print(f"  Prediction order: {sorted_order}")
        print(f"  Top prediction: {sorted_order[-1]}, Expected: {label}")
        print(f"  Correct: {sorted_order[-1] == label}")

    print("=" * 80)

# ============================================================================
# Main Execution
# ============================================================================

def main():
    # Load data
    network = load_network('ckpt_51978-153.h5')
    dataset = load_dataset('mnist.h5', target_idx=TARGET_INDEX, n_samples=1)

    # Convert parameters
    glif3_to_if_curr_exp(network)

    # Create spike times
    spike_times = create_spike_times(dataset['spike_probabilities'][0], scale=1.0)
    label = dataset['labels'][0]
    print(f"\nProcessing sample with label: {label}")

    # Compute mappings
    print("\nComputing population mappings...")
    ps2g, g2psl = compute_population_mappings(network['neurons'])

    print("Grouping V1 synapses...")
    v1_synpols = group_synapses(network['recurrent'], g2psl, "V1")

    print("Grouping LGN neurons...")
    tm2l, l2pl = group_lgn(network['input'], g2psl)

    print("Grouping LGN synapses...")
    lgn_synpols = group_lgn_synapses(network['input'], l2pl, g2psl)

    # Setup simulation
    print("\n" + "=" * 80)
    print("Setting up NEST simulation...")
    sim.setup(timestep=1.0)

    # Create network
    V1 = create_v1_populations(network['ice'], ps2g, v1_synpols, network['glif3'])
    LGN = create_lgn_populations(V1, spike_times, tm2l, lgn_synpols, network['glif3'])
    readouts, nnpols = create_readout_views(network['output'], ps2g, g2psl, V1)

    # Run simulation
    print("\n" + "=" * 80)
    print("Running simulation for 1000 ms...")
    sim.run(1000)
    print("Simulation complete!")

    # Analyze results
    analyze_results(readouts, nnpols, network['output'], ps2g, label)

    # Cleanup
    sim.end()
    print("\nDone!")

if __name__ == '__main__':
    main()
