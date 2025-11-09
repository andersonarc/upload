#!/usr/bin/env python
# coding: utf-8

# # Simulating Mouse V1 Cortex Model on SpiNNaker

# Import the required libraries.

# In[1]:


import os
import h5py
import logging
import numpy as np
import pyNN.spiNNaker as sim
import matplotlib.pyplot as plt
from python_models8.neuron.builds.glif3_curr import GLIF3Curr
np.random.seed(1)


# We define synapses, GLIF3 and IF_curr_exp parameters as 2D numpy arrays, indexed via shorthands.

# In[2]:


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

# In[3]:


# Loads the model
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
        network['glif3'][:, G.AA0] /= 1000.0 # pA -> nA
        network['glif3'][:, G.AA1] /= 1000.0 # pA -> nA
        network['glif3'][:, G.VSC] = network['glif3'][:, G.THR] - network['glif3'][:, G.EL] # voltage scale

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

        # Load output neurons
        network['output'] = np.array(file['readout/neuron_ids'])

        return network

network = load_network('ckpt_51978-77.h5')
print(network['recurrent'][0:5])
print(network['input'][0:5])
print(np.mean(network['recurrent'][:, S.WHT]))
print(np.std(network['recurrent'][:, S.WHT]))
print(np.mean(network['input'][:, S.WHT]))
print(np.std(network['input'][:, S.WHT]))


# Inspect the network.

# In[4]:


# Draws a histogram of the data
def histogram(data, samples=8192, bins='auto', range=None, xlabel='', ylabel='Count', title=''):
    plt.figure(figsize=(10, 6))
    if len(data) < samples:
        hist = data
    else:
        hist = np.random.choice(data, samples)
    plt.hist(hist, bins=bins, range=None, color='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Draws a scatter plot of the data
def scatter(x, y, xlabel='', ylabel='Count', title='', fmt='k.'):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, fmt)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Inspects the neural network
def inspect_network(network):
    print(f'{len(network['neurons'])} neurons with {len(network['glif3'])} classes and {np.shape(network['glif3'])[1]} parameters per class')
    print(f'{len(network['recurrent'])} recurrent synapses, {len(network['input'])} input synapses, {len(network['output'])} readout neurons')

    # Plot input synapses by source
    a = network['input']
    a = a[a[:, S.SRC].argsort()]
    grouped = np.split(a[:, ...], np.unique(a[:, S.SRC], return_index=True)[1][1:])
    y = [len(v) for v in grouped]
    x = [v[0, S.SRC] for v in grouped]
    scatter(x, y, xlabel='Input source neuron ID', title='Input synapses by source')

    # Plot input synapses by target
    a = network['input']
    a = a[a[:, S.TGT].argsort()]
    grouped = np.split(a[:, ...], np.unique(a[:, S.TGT], return_index=True)[1][1:])
    y = [len(v) for v in grouped]
    x = [v[0, S.TGT] for v in grouped]
    scatter(x, y, xlabel='Input target neuron ID', title='Input synapses by target')

    # Plot recurrent synapses by source
    a = network['recurrent']
    a = a[a[:, S.SRC].argsort()]
    grouped = np.split(a[:, ...], np.unique(a[:, S.SRC], return_index=True)[1][1:])
    y = [len(v) for v in grouped]
    x = [v[0, S.SRC] for v in grouped]
    scatter(x, y, xlabel='Recurrent source neuron ID', title='Recurrent synapses by source')

    # Plot recurrent synapses by target
    a = network['recurrent']
    a = a[a[:, S.TGT].argsort()]
    grouped = np.split(a[:, ...], np.unique(a[:, S.TGT], return_index=True)[1][1:])
    y = [len(v) for v in grouped]
    x = [v[0, S.TGT] for v in grouped]
    scatter(x, y, xlabel='Recurrent target neuron ID', title='Recurrent synapses by target')

    # Plot input synapse weights
    a = network['input']
    x = a[:, S.ID]
    y = a[:, S.WHT]
    scatter(x, y, xlabel='Input synapse ID', ylabel='Weight', title='Input synapse weights', fmt='k-')

    # Plot recurrent synapse weights
    a = network['input']
    x = a[:, S.ID]
    y = a[:, S.WHT]
    scatter(x, y, xlabel='Recurrent synapse ID', ylabel='Weight', title='Recurrent synapse weights', fmt='k-')

inspect_network(network)


# Load the dataset.

# In[5]:


# Inspects the dataset
def inspect_dataset(dataset):
    print(f'{np.shape(dataset['spike_probabilities'])[0]} samples with {np.shape(dataset['spike_probabilities'])[2]} spike trains per sample and {np.shape(dataset['spike_probabilities'])[1]} ms sequence length')
    print(f'average probability: {np.mean(np.mean(dataset['spike_probabilities']))}')
    print(f'response window between {dataset['response_window'][0]} and {dataset['response_window'][1]} ms')
    print(f'labels: {dataset['labels']}')

# Loads the dataset
def load_dataset(path, trim=16):
    with h5py.File(path, 'r') as file:
        return {
            'spike_probabilities': np.array(file['spike_trains'][:trim]),
            'response_window': np.array(file['response_window']),
            'labels': np.array(file['labels'][:trim])
        }

# Creates the spike times

def create_spike_times(spike_trains, timestep=1.0, scale=1.0):
    lgn_size = spike_trains.shape[1]
    spike_times = []
    x = []
    y = []

    for i in range(lgn_size):
        x.append(i)
        y.append(np.sum(spike_trains[:, i]))
        times = []
        for t in range(spike_trains.shape[0]):
            if np.clip((spike_trains[t, i] / 1.3) * scale, 0.0, 1.0) > np.random.rand():
                times.append(float(t * timestep))
        spike_times.append(times)

    scatter(x, y)
    return spike_times

# Load and inspect the dataset
dataset = load_dataset('spikes-128.h5')
inspect_dataset(dataset)

# Create spike trains
spike_times = create_spike_times(dataset['spike_probabilities'][0], scale=1.0)


# Convert GLIF3 to IF_curr_exp.

# In[6]:


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

#def glif32ice(network):
#    glif3 = network['glif3']
#    n_types = len(glif3)
#    
#    # Use standard IF_curr_exp parameters that are known to work
#    network['ice'] = np.stack([
#        np.full(n_types, 1.0),      # C_m (nF)
#        np.full(n_types, 20.0),     # tau_m (ms)
#        np.full(n_types, -65.0),    # V_rest (mV)
#        np.full(n_types, -65.0),    # V_reset (mV)
#        np.full(n_types, -60.0),    # V_thresh (mV)
#        np.full(n_types, 2.0),      # tau_refrac (ms)
#        np.full(n_types, 5.0),      # tau_syn_E (ms)
#        np.full(n_types, 5.0),      # tau_syn_I (ms)
#        np.full(n_types, 0.0),      # i_offset (nA)
#        glif3[:, G.CON]             # count (keep original)
#    ], axis=1)

# Do the conversion
glif32ice(network)
print(network['glif3'][0:5])


# Weak synapse pruning.
def prune(synapses, n_neurons, threshold=256, syn_dir=S.TGT):
    dw = []
    counts = np.zeros(n_neurons, dtype=np.int32)

    # Calculate synapses per neuron
    for synapse in synapses:
        counts[int(synapse[syn_dir])] += 1

    # Determine overthreshold neuron indices
    otn_indices = np.nonzero(counts > threshold)[0]
    print(f'{len(otn_indices)} neurons over threshold')

    # Determine how many to drop
    n_drop_each = counts[otn_indices] - threshold
    n_drop_sum = np.sum(n_drop_each)
    if n_drop_sum == 0:
        print('nothing to drop')
        return

    # Mask of relevant synapses
    mask = np.isin(synapses[:, syn_dir], otn_indices)
    ots = synapses[mask]

    # Group by syn_dir
    ots = ots[ots[:, syn_dir].argsort()]
    ots = np.split(ots[:, ...], np.unique(ots[:, syn_dir], return_index=True)[1][1:])

    # Remove connections with the least weight based on ndrop
    to_drop = np.zeros(n_drop_sum)
    offset = 0
    for tgt_idx, syn in enumerate(ots):
        # How many to drop for this one
        tgt_n_drop = n_drop_each[tgt_idx]

        # Sort this neuron's synapses by weight
        syn_sorted = syn[syn[:, S.WHT].argsort()]

        # Drop the first n-drop
        syn_drop = syn_sorted[0:tgt_n_drop]
        dw.extend(syn_drop[:, S.WHT])

        # Move to results
        to_drop[offset:offset+tgt_n_drop] = syn_drop[:, S.ID]

        # Shift
        offset += tgt_n_drop

    # Do drop
    drop_mask = np.isin(synapses[:, S.ID], to_drop)
    print(f'{len(to_drop)} synapses dropped, max = {np.max(dw)}, min = {np.min(dw)}, mean = {np.mean(dw)}')
    return synapses[~drop_mask]

print(np.mean(network['recurrent'][:, S.WHT]))
print(np.min(network['recurrent'][:, S.WHT]))
print(np.max(network['recurrent'][:, S.WHT]))
print(np.mean(network['input'][:, S.WHT]))
print(np.min(network['input'][:, S.WHT]))
print(np.max(network['input'][:, S.WHT]))
print('-----')
network['recurrent'] = prune(network['recurrent'], len(network['neurons']), threshold=256, syn_dir=S.TGT)
#network['recurrent'] = prune(network['recurrent'], len(network['neurons']), threshold=256, syn_dir=S.SRC)
#prune(network['input'], len(network['neurons']), threshold=256, syn_dir=S.TGT)
network['input'] = prune(network['input'], len(network['neurons']), threshold=256, syn_dir=S.SRC)
print(np.mean(network['recurrent'][:, S.WHT]))
print(np.min(network['recurrent'][:, S.WHT]))
print(np.max(network['recurrent'][:, S.WHT]))
print(np.mean(network['input'][:, S.WHT]))
print(np.min(network['input'][:, S.WHT]))
print(np.max(network['input'][:, S.WHT]))
# Population grouping.
# 
# 1. Recurrent neurons
# Recurrent neurons are already split into populations, however some of these are too large and SpiNNaker struggles with hosting them. Therefore, we must first split the largest neuron populations until we meet either of the 3 criteria: 1) at most N synapses 2) at most K neurons 3) at most X populations. Currently, 2 is implemented as it is the most direct and effective option.
# 
# 2. Input neurons.
# Input neurons (or pseudo-neurons, since they are technically just spike emitters) do not have any grouping by population. This is an issue, since it leaves us with an 'all-or-nothing' approach (17400 or 1 populations), where both are extremely unfeasible. For this reason, we must first 1) group input neurons by target neurons in V1 2) iteratively merge them using a certain similarity measure after exact merging has been done, until there are at most Y input populations.

# In[7]:


# Initial mappings for the V1 populations
def v1_compute_initial_mappings(neurons):
    # Global to Population, Local
    g2pl = {} # GID -> PID, LID
    p2g  = {} # PID -> GID

    # Compute mappings
    for gid, pid in enumerate(neurons):
        if pid not in p2g:
            p2g[pid] = []

        # Local ID
        lid = len(p2g[pid])
        p2g[pid].append(gid)
        g2pl[gid] = (int(pid), lid)

    # Report
    lens = [len(x) for x in p2g.values()]
    print(f'{len(neurons)} grouped into {len(lens)} populations of size {np.min(lens)} min, {np.max(lens)} max, {np.mean(lens)} mean, {np.sum(lens)} total.')

    # Return the mappings
    return p2g # g2pl is not really needed

# Split mappings to reduce density
def v1_compute_split_mappings(p2g, target=256):
    # Any population with more than target neurons will be split
    g2psl = {}
    ps2g  = {}

    # Iterate existing mappings
    for pid, gids in p2g.items():
        if len(gids) > target:
            # Must be split. How many?
            n_split = int(np.ceil(len(gids) / target))
        else:
            # No splitting
            n_split = 1

        # Separate into subpopulations using a double index
        for subpid in range(n_split):           
            # Create subkey
            key = (pid, subpid)
            if key not in ps2g:
                # Put a subslice of GIDs there
                start     = subpid * target
                end       = (subpid + 1) * target
                subgids   = gids[start:end]
                ps2g[key] = subgids

                # Store GIDs in the reverse mapping
                for lid, gid in enumerate(subgids):
                    g2psl[gid] = (pid, subpid, lid)

    # Report
    lens = [len(x) for x in ps2g.values()]
    print(f'{len(p2g)} populations split into {len(lens)} populations+subpopulations of size {np.min(lens)} min, {np.max(lens)} max, {np.mean(lens):.2f} mean, {np.sum(lens)} total.')

    # Return the split mappings
    return ps2g, g2psl

p2g = v1_compute_initial_mappings(network['neurons'])
ps2g, g2psl = v1_compute_split_mappings(p2g)


# Recurrent synapses.

# In[8]:


# Groups V1 synapses using the mappings
def v1_group_synapses(syns, g2psl):
    # Create synapse populations
    synpols = {}

    # Iterate all synapses
    for i, syn in enumerate(syns):
        # Log progress
        if i % 500000 == 0:
            print(f'{i} done out of {len(syns)}')

        # Get GIDs
        src_gid = int(syn[S.SRC])
        tgt_gid = int(syn[S.TGT])

        # Map to (PID, SUBPID, LID)
        src_pid, src_subpid, src_lid = g2psl[src_gid]
        tgt_pid, tgt_subpid, tgt_lid = g2psl[tgt_gid]

        # Generate SYNKEY
        synkey = ((src_pid, src_subpid), (tgt_pid, tgt_subpid))
        if synkey not in synpols:
            synpols[synkey] = []

        # Store the SYNPOL
        synpols[synkey].append(np.hstack([[src_lid, tgt_lid], syn[2:]]))

    # Full conversion to numpy
    for synkey, synpol in synpols.items():
        synpols[synkey] = np.array(synpol)

    # Report
    lens = [len(x) for x in synpols.values()]
    print(f'{len(syns)} done out of {len(syns)}')
    print(f'{len(syns)} synapses grouped into {len(lens)} synapse populations of size {np.min(lens)} min, {np.max(lens)} max, {np.mean(lens):.2f} mean, {np.sum(lens)} total.')

    return synpols

v1_synpols = v1_group_synapses(network['recurrent'], g2psl)


# Now deal with the LGN.

# In[9]:


# Groups the LGN into populations by exact target matches
def lgn_group_exact(syns, g2psl):
    # Target Populations -> LGN GID
    t2l = {}
    l2t = {}

    # Compute mappings
    for syn in syns:
        lgn = int(syn[S.SRC])
        tgt = int(syn[S.TGT])

        # Compute target populations for each synapse
        if lgn not in l2t:
            l2t[lgn] = set()

        # Convert target to PID, SUBPID
        pid, subpid, _ = g2psl[tgt]

        # Store the pair
        l2t[lgn].add((pid, subpid))

    # Now each LGN neuron is mapped to its targets
    # And we can group by them
    for lgn, tgtpols in l2t.items():
        tgtkey = tuple(tgtpols)
        if tgtkey not in t2l:
            t2l[tgtkey] = []

        # Append the LGN neuron to a particular group
        t2l[tgtkey].append(lgn)

    # Report
    neurons = list(l2t.keys())
    lens = [len(x) for x in t2l.values()]
    print(f'{len(l2t)} LGN neurons ({np.min(neurons)} - {np.max(neurons)}) split into {len(lens)} populations of size {np.min(lens)} min, {np.max(lens)} max, {np.mean(lens):.2f} mean, {np.sum(lens)} total.')

    return t2l # l2t is not really needed

# Groups together similar LGN populations
def lgn_group_similar(t2l, threshold=0.15): # % are allowed to be different
    # Target Population Merged -> LGN GID
    tm2l = {}
    used = []
    unmatched = 0
    overlapped = 0

    for i, item in enumerate(t2l.items()):
        # Skip used
        if i in used:
            continue

        # Log progress
        if i % 100 == 0:
            print(f'{len(used)} used, {len(t2l) - len(used)} left, {unmatched} unmatched, {overlapped} overlapped')

        # Mark self as used
        used.append(i)

        # Find similar
        found = False
        tgtpols, lgn = item
        own_length = len(tgtpols)
        for target_i, target_item in enumerate(t2l.items()):
            # Skip used
            if target_i in used:
                continue

            # Unfold target
            target_tgtpols, target_lgn = target_item

            # Measure similarity
            merged = tuple(set(tgtpols + target_tgtpols))
            delta  = len(merged) - own_length # cannot be 0 or negative
            delta_fraction = delta / own_length

            # Compare to threshold
            if delta_fraction < threshold:
                # Can merge!
                if merged in tm2l:
                    # Already exists, avoid overlap
                    overlapped += 1
                    tm2l[merged] = np.hstack([tm2l[merged], lgn, target_lgn])
                else:
                    tm2l[merged] = np.hstack([lgn, target_lgn])

                # Mark target as used
                used.append(target_i)
                found = True
                break

        # Keep populations that did not find a match
        if not found:
            # Copy over
            unmatched += 1
            tm2l[tgtpols] = np.array(lgn)

    # Report
    lens = [len(x) for x in tm2l.values()]
    print(f'{len(used)} used, {len(t2l) - len(used)} left, {unmatched} unmatched, {overlapped} overlapped')
    print(f'{len(t2l)} populations merged into {len(lens)} populations of size {np.min(lens)} min, {np.max(lens)} max, {np.mean(lens):.2f} mean, {np.sum(lens)} total.')

    # Create the reverse mapping
    l2pl = {}
    for pid, item in enumerate(tm2l.items()):
        tgtpols, lgns = item
        for lid, lgn in enumerate(lgns):
            l2pl[lgn] = (pid, lid)

    return tm2l, l2pl

t2l = lgn_group_exact(network['input'], g2psl)
tm2l_1, l2pl_1 = lgn_group_similar(t2l, threshold=0.15)
# Group again.
tm2l_2, l2pl_2 = lgn_group_similar(tm2l_1, threshold=0.15)
# And again.
tm2l, l2pl = lgn_group_similar(tm2l_2, threshold=0.15)


# In[10]:


# Now group LGN synapses into synapse populations
def lgn_group_synapses(syns, l2pl, g2psl):
    # Create synapse populations
    synpols = {}

    # Iterate all synapses
    for i, syn in enumerate(syns):
        # Log progress
        if i % 500000 == 0:
            print(f'{i} done out of {len(syns)}')

        # Get GIDs
        src_gid = int(syn[S.SRC])
        tgt_gid = int(syn[S.TGT])

        # Map to (PID, SUBPID, LID)
        tgt_pid, tgt_subpid, tgt_lid = g2psl[tgt_gid]

        # Map to (LGN_PID, LGN_LID)
        lgn_pid, lgn_lid = l2pl[src_gid]

        # Generate SYNKEY
        synkey = (lgn_pid, (tgt_pid, tgt_subpid))
        if synkey not in synpols:
            synpols[synkey] = []

        # Store the SYNPOL
        synpols[synkey].append(np.hstack([[lgn_lid, tgt_lid], syn[2:]]))

    # Full conversion to numpy
    for synkey, synpol in synpols.items():
        synpols[synkey] = np.array(synpol)

    # Report
    lens = [len(x) for x in synpols.values()]
    print(f'{len(syns)} done out of {len(syns)}')
    print(f'{len(syns)} synapses grouped into {len(lens)} synapse populations of size {np.min(lens)} min, {np.max(lens)} max, {np.mean(lens):.2f} mean, {np.sum(lens)} total.')

    return synpols

lgn_synpols = lgn_group_synapses(network['input'], l2pl, g2psl)


# Readout neurons to populations.

# In[11]:


# Converts neurons to populations
def nn2pol(nn, g2psl):
    # Create neuron populations
    nnpols = {}

    # Iterate neurons
    for gid in nn:
        # Map to (PID, SUBPID, LID)
        pid, subpid, lid = g2psl[gid]
        key = (pid, subpid)

        # Store
        if key not in nnpols:
            nnpols[key] = []
        nnpols[key].append(lid)

    # Full conversion to numpy
    for key, nnpol in nnpols.items():
        nnpols[key] = np.array(nnpol)

    # Report
    lens = [len(x) for x in nnpols.values()]
    print(f'{len(nn)} neurons linked to {len(lens)} populations of size {np.min(lens)} min, {np.max(lens)} max, {np.mean(lens):.2f} mean, {np.sum(lens)} total.')

    return nnpols

output_nnpols = nn2pol(network['output'], g2psl)


# Do the simulation.
# 

# In[12]:


def setup():
    # Configure the simulation
    sim.setup(timestep=1)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 200) # I have tried from 1 to 100

#class G():
#    CM  = 0 # C_m
#    EL  = 1 # E_L
#    RST = 2 # V_reset
#    THR = 3 # V_thresh
#    AA0 = 4 # asc_amps0
#    AA1 = 5 # asc_amps1
#    G   = 6 # g
#    K0  = 7 # k0
#    K1  = 8 # k1
#    RFR = 9 # t_ref
#    TA0 = 10 # tau_syn0
#    TA1 = 11 # tau_syn1
#    TA2 = 12 # tau_syn2
#    TA3 = 13 # tau_syn3
#    CON = 14 # count

def G2D(g):
    """Convert GLIF3 parameters to PyNN GLIF3Curr cell parameters.

    Note: Initial values (v, i_asc_0, i_asc_1) are set separately via initial_values
    parameter as per PyNN specification, not in cellparams.
    """
    return {
        'c_m': g[G.CM],
        'e_l': g[G.EL],
        'v_reset': g[G.RST],
        'v_thresh': g[G.THR],
        'asc_amp_0': g[G.AA0],
        'asc_amp_1': g[G.AA1],
        'g': g[G.G],
        'k0': g[G.K0],
        'k1': g[G.K1],
        't_ref': g[G.RFR],
        'i_offset': 0.0,
        'tau_syn_0': g[G.TA0],
        'tau_syn_1': g[G.TA1],
        'tau_syn_2': g[G.TA2],
        'tau_syn_3': g[G.TA3],
    }

def G2IV(g):
    """Get GLIF3 initial values for state variables.

    Returns initial values dict for PyNN Population initialization.
    Matches TensorFlow initialization in models.py:275
    """
    return {
        'v': g[G.EL],       # Initialize at resting potential
        'i_asc_0': 0.0,     # After-spike current 0 starts at zero
        'i_asc_1': 0.0,     # After-spike current 1 starts at zero
    }

def create_V1(glif3s, ps2g, v1_synapses):
    # Create the V1 IF_curr_exp populations
    V1 = {}
    for key, gids in ps2g.items():
        pid, subpid = key
        size = len(gids)
        V1_N = sim.Population(
            len(gids),
            GLIF3Curr,
            cellparams=G2D(glif3s[pid]),
            initial_values=G2IV(glif3s[pid]),
            neurons_per_core=int(np.min([200, ( 1 / size ) * 1e6])),
            label=f'V1_{pid}_{subpid}'
        )
        V1[key] = V1_N
    V1_n_pop = len(V1)
    V1_n_proj = 0

    # Process synapse populations
    p = 0
    for synkey, syn in v1_synapses.items():
        src_key, tgt_key = synkey

        # Separate by receptor type
        #if np.all(syn[:, S.RTY] == 0) or np.all(syn[:, S.RTY] == 2):
        #    # Excitatory
        #    receptor_type = 'excitatory'
        #elif np.all(syn[:, S.RTY] == 1) or np.all(syn[:, S.RTY] == 3):
        #    # Inhibitory
        #    receptor_type = 'inhibitory'
        #else:
        #    raise RuntimeError(f'inhomogenous projection {syn[:, S.RTY]}')

        # Scale
        vsc = network['glif3'][int(tgt_key[0]), G.VSC]
        syn[:, S.WHT] *= vsc / 1000.0
        if p < 5:
            print(vsc)
            p += 1

        receptor_type = f'synapse_{int(syn[0, S.RTY])}'
        sim.Projection(V1[src_key], V1[tgt_key], sim.FromListConnector(syn[:, [S.SRC, S.TGT, S.WHT, S.DLY]]), receptor_type=receptor_type)
        V1_n_proj += 1

    print(f'V1 populations created and linked, total size {np.sum([p.size for p in V1.values()])}')
    for i, pop in enumerate(V1.values()):
        if pop is not None:
            pop.record(['spikes', 'v'])
    return V1, V1_n_pop, V1_n_proj

def create_LGN(V1, spike_times, tm2l, lgn_synapses):
    LGN = []
    for i, lgns in enumerate(tm2l.values()):
        LGN_x = sim.Population(
            len(lgns),
            sim.SpikeSourceArray,
            cellparams={
                'spike_times': [spike_times[lgn] if len(spike_times[lgn]) > 0 else [600] for lgn in lgns]
            },
            label=f'LGN_{i}'
        )
        LGN.append(LGN_x)
    LGN_n_pop = len(LGN)
    LGN_n_proj = 0

    p = 0
    # Process synapse populations
    for synkey, syn in lgn_synapses.items():
        lgn_pid, tgt_key = synkey

        # Separate by receptor type
        #if np.all(syn[:, S.RTY] == 0) or np.all(syn[:, S.RTY] == 2):
        #    # Excitatory
        #    receptor_type = 'excitatory'
        #elif np.all(syn[:, S.RTY] == 1) or np.all(syn[:, S.RTY] == 3):
        #    # Inhibitory
        #    receptor_type = 'inhibitory'
        #else:
        #    raise RuntimeError(f'inhomogenous projection {syn[:, S.RTY]}')

        # Scale
        vsc = network['glif3'][int(tgt_key[0]), G.VSC]
        syn[:, S.WHT] *= vsc / 1000.0

        if p < 5:
            print(vsc)
            p += 1

        receptor_type = f'synapse_{int(syn[0, S.RTY])}'
        sim.Projection(LGN[lgn_pid], V1[tgt_key], sim.FromListConnector(syn[:, [S.SRC, S.TGT, S.WHT]], column_names=['weight']), receptor_type=receptor_type)
        LGN_n_proj += 1

    print(f'LGN populations created and linked, total size {np.sum([p.size for p in LGN])}')
    for pop in LGN:
        if pop is not None:
            pop.record(['spikes'])
    return LGN, LGN_n_pop, LGN_n_proj

def create_readouts(output_nnpols, V1):
    readouts = []
    for key, lids in output_nnpols.items():
        # Create the readout view
        view = sim.PopulationView(V1[key], lids)
        view.record(['spikes', 'v'])
        readouts.append(view)

    print('Readout neurons selected')
    return readouts

# Setup the simulation
setup()
V1, V1_n_pop, V1_n_proj = create_V1(network['glif3'], ps2g, v1_synpols)
LGN, LGN_n_pop, LGN_n_proj = create_LGN(V1, spike_times, tm2l, lgn_synpols)
readouts = create_readouts(output_nnpols, V1)

# Print statistics
print(f'V1 populations: {V1_n_pop}, V1 projections: {V1_n_proj}')
print(f'LGN populations: {LGN_n_pop}, LGN projections: {LGN_n_proj}')
print(f'Readout views: {len(readouts)}')


# In[13]:


# Run it
sim.run(1000)


# In[14]:


print(f"\nCreated {len(LGN)} LGN populations")
for i, lgn in enumerate(LGN[:10]):
    spikes = lgn.get_data('spikes').segments[0].spiketrains
    total = sum(len(st) for st in spikes)
    print(f"LGN {i}: {lgn.size} neurons, {total} total spikes")

ps2keys = list(ps2g.keys())
for i in range(0):
    key = ps2keys[i]
    if V1[key] is not None:
        v = V1[key].get_data('v').segments[0].analogsignals[0]
        print(f'{i} = {v}')
        #print(f"V1[{i}] voltage range: {v.min():.2f} to {v.max():.2f} mV")


# In[15]:


print(dataset['labels'][0])
gid2train = {}
for i, item in enumerate(output_nnpols.items()):
    readout = readouts[i]
    key, lids = item
    gids = ps2g[key]
    # Get spike trains and map them to correct global IDs
    spiketrains = readout.get_data('spikes').segments[0].spiketrains
    for lid, spiketrain in zip(lids, spiketrains):
        gid = gids[lid]  # Get global ID at local position lid
        gid2train[gid] = spiketrain
print(len(gid2train))
# Now slice
votes = np.zeros(10)
for i in range(0, 10):
    start = i*30
    end = (1+i)*30
    keys = network['output'][start:end]
    for key in keys:
        spiketrain = gid2train[key]
        mask = (spiketrain > 50) & (spiketrain < 200)
        count = mask.sum()
        votes[i] += count
print(votes)

# 3   -> 0? [ 1.  8. 12.  5.  4.  8.  7.  2.  5.  3.]
# 2,3 -> 4? [ 1. 12. 12.  4.  8.  2.  9.  3.  1.  1.]
# 3   -> 4? [ 1.  7. 13.  1.  5.  4.  4.  5.  0.  2.]

# trained 6 -> [ 0.  1.  2.  2.  3.  5.  4.  3. 12.  3.]
# trained 4 -> [ 0.  1.  4.  3.  1.  6.  4.  0.  6.  0.]
# trained 4 -> [ 0.  1.  3.  1.  0.  5.  4.  2.  ...   ]
# 0
# 300
# [ 0.  3.  5.  0.  2. 10.  8.  5. 13. 12.]
# 0
# 300
# [  0.   6.  12.   1.   5.   9.   5.   9. 189.  10.]
# 4
# 300
# [  7.   9.  12.   0.   6.   2.  12.   6. 204.   1.]
# 8
# 300
# [  1.   5.  23.  10.  16.   9.  10.   6. 184.   4.]

# After fix

# 8
# 300
# [ 1.  0. 56.  0. 54.  0.  4.  0.  0.  0.]


# In[16]:


y = (dataset['spike_probabilities'][0][:, 1000])
x = np.arange(0, len(y))
#print(y)
#scatter(x, np.log(1 - (y/1.3)) * -1000)


# In[17]:


print(f"Recurrent weights - positive: {np.sum(network['recurrent'][:, S.WHT] > 0)}, negative: {np.sum(network['recurrent'][:, S.WHT] < 0)}")
print(f"Input weights - positive: {np.sum(network['input'][:, S.WHT] > 0)}, negative: {np.sum(network['input'][:, S.WHT] < 0)}")


# In[18]:


print("\nReadout neuron activity across full simulation:")
oso = np.sort(network['output'])
for class_idx in range(10):
    start = class_idx * 30
    end = (class_idx + 1) * 30
    keys = network['output'][start:end]
    total_spikes = 0
    for key in keys:
        if key in gid2train:
            total_spikes += len(gid2train[key])
    print(f"Class {class_idx}: {total_spikes} total spikes (all time)")

# Real answer: 0

# No /1000 scaling on main weights
# Readout neuron activity across full simulation:
# Class 0: 0 total spikes (all time)
# Class 1: 0 total spikes (all time)
# Class 2: 0 total spikes (all time)
# Class 3: 0 total spikes (all time)
# Class 4: 0 total spikes (all time)
# Class 5: 0 total spikes (all time)
# Class 6: 0 total spikes (all time)
# Class 7: 10 total spikes (all time)
# Class 8: 0 total spikes (all time)
# Class 9: 0 total spikes (all time)

# /1000 scaling on main weights
# Readout neuron activity across full simulation:
# Class 0: 4 total spikes (all time)
# Class 1: 2 total spikes (all time)
# Class 2: 1 total spikes (all time)
# Class 3: 2 total spikes (all time)
# Class 4: 162 total spikes (all time)
# Class 5: 10 total spikes (all time)
# Class 6: 4 total spikes (all time)
# Class 7: 5 total spikes (all time)
# Class 8: 2 total spikes (all time)
# Class 9: 185 total spikes (all time)

# Input scaled down by /1.3
# Readout neuron activity across full simulation:
# Class 0: 2 total spikes (all time)
# Class 1: 0 total spikes (all time)
# Class 2: 0 total spikes (all time)
# Class 3: 0 total spikes (all time)
# Class 4: 135 total spikes (all time)
# Class 5: 0 total spikes (all time)
# Class 6: 0 total spikes (all time)
# Class 7: 0 total spikes (all time)
# Class 8: 0 total spikes (all time)
# Class 9: 168 total spikes (all time)

# E_L used
# Readout neuron activity across full simulation:
# Class 0: 0 total spikes (all time)
# Class 1: 0 total spikes (all time)
# Class 2: 0 total spikes (all time)
# Class 3: 0 total spikes (all time)
# Class 4: 0 total spikes (all time)
# Class 5: 0 total spikes (all time)
# Class 6: 0 total spikes (all time)
# Class 7: 0 total spikes (all time)
# Class 8: 0 total spikes (all time)
# Class 9: 161 total spikes (all time)

# Readout neuron activity across full simulation:
# Class=6
# Class 0: 0 total spikes (all time)
# Class 1: 0 total spikes (all time)
# Class 2: 0 total spikes (all time)
# Class 3: 0 total spikes (all time)
# Class 4: 1 total spikes (all time)
# Class 5: 0 total spikes (all time)
# Class 6: 0 total spikes (all time)
# Class 7: 1 total spikes (all time)
# Class 8: 0 total spikes (all time)
# Class 9: 0 total spikes (all time)


# In[19]:


# What neuron types are the readout neurons?
readout_types = network['neurons'][network['output']]
print(f"Readout neuron types: {np.unique(readout_types, return_counts=True)}")


# In[ ]:




