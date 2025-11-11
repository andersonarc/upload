#!/usr/bin/env python
# coding: utf-8

# # Simulating Mouse V1 Cortex Model on NEST Simulator
# Ported from SpiNNaker implementation

# Import the required libraries.
import os
import h5py
import logging
import numpy as np
import pyNN.nest as sim  # Changed from pyNN.spiNNaker
import matplotlib.pyplot as plt

np.random.seed(1)
import sys

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# What we're doing today
target_now = int(os.environ.get('TARGET_INDEX', 0))  # Default to 0 if not set
logfile = open(f'ot_nest_{target_now}.log', 'a')
sys.stdout = Tee(sys.stdout, logfile)
sys.stderr = Tee(sys.stderr, logfile)

print("=" * 80)
print("NEST Simulator Version of V1 Cortex Inference")
print("=" * 80)

# Synapse array indices
class S():
    SRC = 0 # Source
    TGT = 1 # Target
    WHT = 2 # Weight
    RTY = 3 # Receptor Type
    ID  = 4 # Synapse ID
    DLY = 5 # Delay (recurrent only)

# GLIF3 parameter indices
class G():
    CM  = 0 # C_m
    EL  = 1 # E_L
    RST = 2 # V_reset
    THR = 3 # V_thresh
    AA0 = 4 # asc_amps0
    AA1 = 5 # asc_amps1
    G   = 6 # g
    K0  = 7 # k0
    K1  = 8 # k1
    RFR = 9 # t_ref
    TA0 = 10 # tau_syn0
    TA1 = 11 # tau_syn1
    TA2 = 12 # tau_syn2
    TA3 = 13 # tau_syn3
    VSC = 14 # voltage scale
    CON = 15 # count

# IF_curr_exp parameter indices
class I():
    CM  = 0 # C_m
    TAU = 1 # tau_m
    VRT = 2 # V_rest
    RST = 3 # V_reset
    THR = 4 # V_thresh
    RFR = 5 # tau_refrac
    EXC = 6 # tau_syn_E
    INH = 7 # tau_syn_I
    OFT = 8 # i_offset
    CON = 9 # count


# Load V1 cortex network from the HDF5 file.
def load_network(path):
    with h5py.File(path, 'r') as file:
        network = {}

        # Load neurons
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

        # Scale
        network['glif3'][:, G.CM]  /= 1000.0 # pF -> nF
        network['glif3'][:, G.G]   /= 1000.0 # nS -> uS
        network['glif3'][:, G.VSC] = network['glif3'][:, G.THR] - network['glif3'][:, G.EL] # voltage scale
        network['glif3'][:, G.AA0] /= 1000.0 # pA -> nA
        network['glif3'][:, G.AA1] /= 1000.0 # pA -> nA

        # Load recurrent synapses
        network['recurrent'] = np.stack((
            file['recurrent/sources'],
            file['recurrent/targets'],
            file['recurrent/weights'],
            file['recurrent/receptor_types'],
            np.arange(len(file['recurrent/weights'])),
            file['recurrent/delays']
        ), axis=1)

        # Load input synapses
        network['input'] = np.stack((
            file['input/sources'],
            file['input/targets'],
            file['input/weights'],
            file['input/receptor_types'],
            np.arange(len(file['input/weights']))
        ), axis=1)

        # Load background weights from checkpoint if available
        if 'input/bkg_weights' in file:
            network['bkg_weights'] = np.array(file['input/bkg_weights'])
            print(f"Loaded background weights: {network['bkg_weights'].shape}")
        else:
            network['bkg_weights'] = None
            print("No background weights found in file")

        # Load output neurons
        network['output'] = np.array(file['readout/neuron_ids'])

        return network

network = load_network('ckpt_51978-153.h5')
print(f"Network loaded successfully")
print(f"Neurons: {len(network['neurons'])}, Recurrent synapses: {len(network['recurrent'])}, Input synapses: {len(network['input'])}")

# Converts GLIF3 parameters to IF_curr_exp
def glif32ice(network):
    glif3 = network['glif3']
    network['ice'] = np.stack([
        glif3[:, G.CM],                 # C_m
        glif3[:, G.CM] / glif3[:, G.G], # tau_m
        glif3[:, G.EL],                 # V_rest
        glif3[:, G.RST],                # V_reset
        glif3[:, G.THR],                # V_thresh
        glif3[:, G.RFR],                # tau_refrac
        glif3[:, G.TA0],                # tau_syn_E
        glif3[:, G.TA2],                # tau_syn_I
        np.zeros(len(glif3)),           # i_offset
        glif3[:, G.CON]                 # count
    ], axis=1)

# Do the conversion
glif32ice(network)
print("Converted GLIF3 to IF_curr_exp parameters")

# Converts to dictionary
def I2D(i):
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

# Initial mappings for the V1 populations
def v1_compute_initial_mappings(neurons):
    g2pl = {} # GID -> PID, LID
    p2g  = {} # PID -> GID

    for gid, pid in enumerate(neurons):
        if pid not in p2g:
            p2g[pid] = []
        lid = len(p2g[pid])
        p2g[pid].append(gid)
        g2pl[gid] = (int(pid), lid)

    lens = [len(x) for x in p2g.values()]
    print(f'{len(neurons)} grouped into {len(lens)} populations of size {np.min(lens)} min, {np.max(lens)} max, {np.mean(lens)} mean, {np.sum(lens)} total.')

    return p2g

# Split mappings to reduce density
def v1_compute_split_mappings(p2g, target=256):
    g2psl = {}
    ps2g  = {}

    for pid, gids in p2g.items():
        if len(gids) > target:
            n_split = int(np.ceil(len(gids) / target))
        else:
            n_split = 1

        for subpid in range(n_split):
            key = (pid, subpid)
            if key not in ps2g:
                start     = subpid * target
                end       = (subpid + 1) * target
                subgids   = gids[start:end]
                ps2g[key] = subgids

                for lid, gid in enumerate(subgids):
                    g2psl[gid] = (pid, subpid, lid)

    lens = [len(x) for x in ps2g.values()]
    print(f'{len(p2g)} populations split into {len(lens)} populations+subpopulations of size {np.min(lens)} min, {np.max(lens)} max, {np.mean(lens):.2f} mean, {np.sum(lens)} total.')

    return ps2g, g2psl

p2g = v1_compute_initial_mappings(network['neurons'])
ps2g, g2psl = v1_compute_split_mappings(p2g)

# Groups V1 synapses using the mappings
def v1_group_synapses(syns, g2psl):
    synpols = {}

    for i, syn in enumerate(syns):
        if i % 500000 == 0:
            print(f'{i} done out of {len(syns)}')

        src_gid = int(syn[S.SRC])
        tgt_gid = int(syn[S.TGT])

        src_pid, src_subpid, src_lid = g2psl[src_gid]
        tgt_pid, tgt_subpid, tgt_lid = g2psl[tgt_gid]

        synkey = ((src_pid, src_subpid), (tgt_pid, tgt_subpid))
        if synkey not in synpols:
            synpols[synkey] = []

        synpols[synkey].append(np.hstack([[src_lid, tgt_lid], syn[2:]]))

    for synkey, synpol in synpols.items():
        synpols[synkey] = np.array(synpol)

    lens = [len(x) for x in synpols.values()]
    print(f'{len(syns)} synapses grouped into {len(lens)} synapse populations')

    return synpols

v1_synpols = v1_group_synapses(network['recurrent'], g2psl)

# Groups the LGN into populations by exact target matches
def lgn_group_exact(syns, g2psl):
    t2l = {}
    l2t = {}

    for syn in syns:
        lgn = int(syn[S.SRC])
        tgt = int(syn[S.TGT])

        if lgn not in l2t:
            l2t[lgn] = set()

        pid, subpid, _ = g2psl[tgt]
        l2t[lgn].add((pid, subpid))

    for lgn, tgtpols in l2t.items():
        tgtkey = tuple(tgtpols)
        if tgtkey not in t2l:
            t2l[tgtkey] = []
        t2l[tgtkey].append(lgn)

    lens = [len(x) for x in t2l.values()]
    print(f'{len(l2t)} LGN neurons split into {len(lens)} populations')

    return t2l

# Groups together similar LGN populations
def lgn_group_similar(t2l, threshold=0.15):
    tm2l = {}
    used = []
    unmatched = 0

    for i, item in enumerate(t2l.items()):
        if i in used:
            continue

        if i % 100 == 0:
            print(f'{len(used)} used, {len(t2l) - len(used)} left')

        used.append(i)

        found = False
        tgtpols, lgn = item
        own_length = len(tgtpols)

        for target_i, target_item in enumerate(t2l.items()):
            if target_i in used:
                continue

            target_tgtpols, target_lgn = target_item
            merged = tuple(set(tgtpols + target_tgtpols))
            delta  = len(merged) - own_length
            delta_fraction = delta / own_length

            if delta_fraction < threshold:
                if merged in tm2l:
                    tm2l[merged] = np.hstack([tm2l[merged], lgn, target_lgn])
                else:
                    tm2l[merged] = np.hstack([lgn, target_lgn])
                used.append(target_i)
                found = True
                break

        if not found:
            unmatched += 1
            tm2l[tgtpols] = np.array(lgn)

    lens = [len(x) for x in tm2l.values()]
    print(f'{len(t2l)} populations merged into {len(lens)} populations')

    l2pl = {}
    for pid, item in enumerate(tm2l.items()):
        tgtpols, lgns = item
        for lid, lgn in enumerate(lgns):
            l2pl[lgn] = (pid, lid)

    return tm2l, l2pl

t2l = lgn_group_exact(network['input'], g2psl)
tm2l_1, l2pl_1 = lgn_group_similar(t2l, threshold=0.15)
tm2l_2, l2pl_2 = lgn_group_similar(tm2l_1, threshold=0.15)
tm2l, l2pl = lgn_group_similar(tm2l_2, threshold=0.15)

# Now group LGN synapses into synapse populations
def lgn_group_synapses(syns, l2pl, g2psl):
    synpols = {}

    for i, syn in enumerate(syns):
        if i % 500000 == 0:
            print(f'{i} done out of {len(syns)}')

        src_gid = int(syn[S.SRC])
        tgt_gid = int(syn[S.TGT])

        tgt_pid, tgt_subpid, tgt_lid = g2psl[tgt_gid]
        lgn_pid, lgn_lid = l2pl[src_gid]

        synkey = (lgn_pid, (tgt_pid, tgt_subpid))
        if synkey not in synpols:
            synpols[synkey] = []

        synpols[synkey].append(np.hstack([[lgn_lid, tgt_lid], syn[2:]]))

    for synkey, synpol in synpols.items():
        synpols[synkey] = np.array(synpol)

    lens = [len(x) for x in synpols.values()]
    print(f'{len(syns)} synapses grouped into {len(lens)} synapse populations')

    return synpols

lgn_synpols = lgn_group_synapses(network['input'], l2pl, g2psl)

# Converts neurons to populations
def nn2pol(nn, g2psl):
    nnpols = {}

    for gid in nn:
        pid, subpid, lid = g2psl[gid]
        key = (pid, subpid)

        if key not in nnpols:
            nnpols[key] = []
        nnpols[key].append(lid)

    for key, nnpol in nnpols.items():
        nnpols[key] = np.array(nnpol)

    lens = [len(x) for x in nnpols.values()]
    print(f'{len(nn)} neurons linked to {len(lens)} populations')

    return nnpols

output_nnpols = nn2pol(network['output'], g2psl)

print("\n" + "=" * 80)
print("Setting up NEST simulation...")
print("=" * 80)

def setup():
    # Configure the simulation
    sim.setup(timestep=1.0)
    # NEST doesn't need neurons_per_core setting like SpiNNaker
    print("NEST simulation configured with 1ms timestep")

def create_V1(ice_params, ps2g, v1_synapses):
    # Create the V1 IF_curr_exp populations
    V1 = {}
    for key, gids in ps2g.items():
        pid, subpid = key
        V1_N = sim.Population(
            len(gids),
            sim.IF_curr_exp,
            cellparams=I2D(ice_params[pid]),
            label=f'V1_{pid}_{subpid}'
        )
        V1[key] = V1_N

    V1_n_proj = 0

    # Process synapse populations
    for synkey, syn in v1_synapses.items():
        src_key, tgt_key = synkey

        # Scale weights
        vsc = network['glif3'][int(tgt_key[0]), G.VSC]
        syn[:, S.WHT] *= vsc / 1000.0

        # Determine receptor type
        if np.all(syn[:, S.RTY] == 0) or np.all(syn[:, S.RTY] == 2):
            receptor_type = 'excitatory'
        elif np.all(syn[:, S.RTY] == 1) or np.all(syn[:, S.RTY] == 3):
            receptor_type = 'inhibitory'
        else:
            # Mixed receptor types - use most common
            receptor_type = 'excitatory' if np.mean(syn[:, S.RTY]) < 1.5 else 'inhibitory'

        sim.Projection(V1[src_key], V1[tgt_key],
                      sim.FromListConnector(syn[:, [S.SRC, S.TGT, S.WHT, S.DLY]]),
                      receptor_type=receptor_type)
        V1_n_proj += 1

    print(f'V1 populations created: {len(V1)}, projections: {V1_n_proj}')

    for pop in V1.values():
        if pop is not None:
            pop.record(['spikes', 'v'])

    return V1, len(V1), V1_n_proj

def create_LGN(V1, spike_times, tm2l, lgn_synapses):
    """Create LGN populations with spike source arrays.

    Note: spike_times should be provided or this will create dummy populations.
    For testing, you can create spike_times from your dataset."""

    LGN = []

    # Create dummy spike times if not provided
    if spike_times is None:
        print("WARNING: No spike times provided, creating empty spike sources")
        spike_times = [[1000.0] for _ in range(max([max(lgns) for lgns in tm2l.values()]) + 1)]

    for i, lgns in enumerate(tm2l.values()):
        LGN_x = sim.Population(
            len(lgns),
            sim.SpikeSourceArray,
            cellparams={
                'spike_times': [spike_times[lgn] if lgn < len(spike_times) and len(spike_times[lgn]) > 0 else [1000.0] for lgn in lgns]
            },
            label=f'LGN_{i}'
        )
        LGN.append(LGN_x)

    LGN_n_proj = 0

    for synkey, syn in lgn_synapses.items():
        lgn_pid, tgt_key = synkey

        # Scale weights
        vsc = network['glif3'][int(tgt_key[0]), G.VSC]
        syn[:, S.WHT] *= vsc / 1000.0

        # Determine receptor type
        if np.all(syn[:, S.RTY] == 0) or np.all(syn[:, S.RTY] == 2):
            receptor_type = 'excitatory'
        elif np.all(syn[:, S.RTY] == 1) or np.all(syn[:, S.RTY] == 3):
            receptor_type = 'inhibitory'
        else:
            receptor_type = 'excitatory' if np.mean(syn[:, S.RTY]) < 1.5 else 'inhibitory'

        sim.Projection(LGN[lgn_pid], V1[tgt_key],
                      sim.FromListConnector(syn[:, [S.SRC, S.TGT, S.WHT]], column_names=['weight']),
                      receptor_type=receptor_type)
        LGN_n_proj += 1

    print(f'LGN populations created: {len(LGN)}, projections: {LGN_n_proj}')

    for pop in LGN:
        if pop is not None:
            pop.record(['spikes'])

    return LGN, len(LGN), LGN_n_proj

def create_readouts(output_nnpols, V1):
    readouts = []
    for key, lids in output_nnpols.items():
        view = sim.PopulationView(V1[key], lids)
        view.record(['spikes', 'v'])
        readouts.append(view)

    print(f'Readout neurons selected: {len(readouts)} views')
    return readouts

# Setup the simulation
setup()
V1, V1_n_pop, V1_n_proj = create_V1(network['ice'], ps2g, v1_synpols)
LGN, LGN_n_pop, LGN_n_proj = create_LGN(V1, None, tm2l, lgn_synpols)  # Note: spike_times=None for now
readouts = create_readouts(output_nnpols, V1)

print("\n" + "=" * 80)
print(f'Network created successfully:')
print(f'  V1: {V1_n_pop} populations, {V1_n_proj} projections')
print(f'  LGN: {LGN_n_pop} populations, {LGN_n_proj} projections')
print(f'  Readout views: {len(readouts)}')
print("=" * 80)
print("\nNOTE: To run simulation, you need to:")
print("1. Load spike times from your dataset (e.g., spikes-128.h5)")
print("2. Call create_LGN with proper spike_times")
print("3. Run: sim.run(1000)")
print("4. Extract results from readouts")
print("=" * 80)

logfile.close()
