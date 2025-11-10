#!/usr/bin/env python
"""NEST GLIF3 Implementation - Proper neuron model for 80%+ accuracy"""

import os
import h5py
import numpy as np
import nest

np.random.seed(1)
TARGET_INDEX = int(os.environ.get('TARGET_INDEX', 0))

print("=" * 80)
print(f"NEST GLIF3 V1 Inference - Sample {TARGET_INDEX}")
print("=" * 80)

os.chdir('/home/user/upload/new_things')

# Load network
print("Loading network...")
with h5py.File('ckpt_51978-153.h5', 'r') as f:
    neurons = np.array(f['neurons/node_type_ids'])

    # Load GLIF3 parameters directly from h5
    glif_params = {}
    for key in f['neurons/glif3_params'].keys():
        glif_params[key] = np.array(f['neurons/glif3_params/' + key])

    recurrent = np.array([
        f['recurrent/sources'][:],
        f['recurrent/targets'][:],
        f['recurrent/weights'][:],
        f['recurrent/delays'][:],
        f['recurrent/receptor_types'][:]
    ]).T

    input_syns = np.array([
        f['input/sources'][:],
        f['input/targets'][:],
        f['input/weights'][:],
        f['input/receptor_types'][:]
    ]).T

    output_neurons = np.array(f['readout/neuron_ids'], dtype=int)

print(f"  Neurons: {len(neurons)}, Recurrent: {len(recurrent)}, Input: {len(input_syns)}")

# Load spikes
with h5py.File('mnist.h5', 'r') as f:
    spike_trains = np.array(f['spike_trains'])
    labels = np.array(f['labels'])

label = labels[TARGET_INDEX]
print(f"  Label: {label}")

# Create spike times by sampling from spike probabilities
print("Creating spike times...")
sample_spikes = spike_trains[TARGET_INDEX]
spike_times = {}

# Sample spikes stochastically (like newclass.py)
for neuron_idx in range(sample_spikes.shape[1]):
    times = []
    for t_idx in range(sample_spikes.shape[0]):
        prob = np.clip((sample_spikes[t_idx, neuron_idx] / 1.3), 0.0, 1.0)
        if prob > np.random.rand():
            times.append(float(t_idx + 1.0))
    if len(times) > 0:
        spike_times[neuron_idx] = times

print(f"  Active LGN: {len(spike_times)}")

# Setup NEST
print("Setting up NEST...")
nest.ResetKernel()
nest.resolution = 1.0

# Create V1 neurons with GLIF model
print("Creating V1 with GLIF3 neurons...")
v1 = nest.Create('glif_psc', len(neurons))

# Set GLIF3 parameters by neuron type
print("Setting GLIF3 parameters...")
unique_types = np.unique(neurons)
for ntype in unique_types:
    mask = neurons == ntype
    indices = np.where(mask)[0]

    # Get parameters for this neuron type
    C_m = float(glif_params['C_m'][ntype])
    E_L = float(glif_params['E_L'][ntype])
    V_reset = float(glif_params['V_reset'][ntype])
    V_th = float(glif_params['V_th'][ntype])
    asc_amps = tuple(float(x) for x in glif_params['asc_amps'][ntype])
    g = float(glif_params['g'][ntype])
    asc_decay = tuple(float(x) for x in glif_params['k'][ntype])  # k → asc_decay
    t_ref = float(glif_params['t_ref'][ntype])
    tau_syn = tuple(float(x) for x in glif_params['tau_syn'][ntype])

    # GLIF3 configuration (LIF_ASC: after-spike currents enabled)
    params = {
        'C_m': C_m,
        'E_L': E_L,
        'V_reset': V_reset,
        'V_th': V_th,
        'V_m': E_L,  # Initial voltage = leak potential (not V_reset!)
        'g': g,
        't_ref': t_ref,
        'tau_syn': tau_syn,
        'asc_amps': asc_amps,
        'asc_decay': asc_decay,
        'after_spike_currents': True,  # Enable GLIF3 ASC
        'spike_dependent_threshold': False,  # GLIF3 doesn't use this
        'adapting_threshold': False  # GLIF3 doesn't use this
    }

    # Apply to all neurons of this type
    for idx in indices:
        v1[int(idx)].set(params)

# Create LGN
print("Creating LGN...")
lgn = nest.Create('spike_generator', spike_trains.shape[2])
for i, times in spike_times.items():
    lgn[i].set({'spike_times': times})

print("Pre-computing global IDs...")
v1_gids = np.array([n.global_id for n in v1])
lgn_gids = np.array([n.global_id for n in lgn])

# Connect LGN -> V1 (VECTORIZED)
print("Connecting LGN -> V1...")
src_arr = input_syns[:, 0].astype(int)
tgt_arr = input_syns[:, 1].astype(int)
w_arr = input_syns[:, 2]
rtype_arr = input_syns[:, 3].astype(int)

# Calculate voltage scale for each target neuron
vsc = np.array([glif_params['V_th'][neurons[i]] - glif_params['E_L'][neurons[i]] for i in range(len(neurons))])

# Scale weights by voltage scale (NOT /1000 - NEST uses pA, SpiNNaker uses nA)
w_scaled = w_arr * vsc[tgt_arr]

# Filter
mask = np.abs(w_scaled) > 1e-10
src_filt = src_arr[mask]
tgt_filt = tgt_arr[mask]
w_filt = w_scaled[mask]
rtype_filt = rtype_arr[mask] + 1  # Convert 0,1,2,3 to 1,2,3,4 for NEST

# Connect
lgn_conn_gids = lgn_gids[src_filt]
v1_conn_gids = v1_gids[tgt_filt]
delays_lgn = np.ones(len(w_filt))

nest.Connect(lgn_conn_gids.tolist(), v1_conn_gids.tolist(),
             conn_spec='one_to_one',
             syn_spec={'weight': w_filt, 'delay': delays_lgn, 'receptor_type': rtype_filt.tolist()})

print(f"  Connected {len(w_filt)} LGN synapses")

# Connect V1 recurrent (VECTORIZED)
print("Connecting V1 recurrent...")
src_arr = recurrent[:, 0].astype(int)
tgt_arr = recurrent[:, 1].astype(int)
w_arr = recurrent[:, 2]
d_arr = recurrent[:, 3]
rtype_arr = recurrent[:, 4].astype(int)

# Scale weights by voltage scale (NOT /1000 - NEST uses pA, SpiNNaker uses nA)
w_scaled = w_arr * vsc[tgt_arr]

# Filter
mask = np.abs(w_scaled) > 1e-10
src_filt = src_arr[mask]
tgt_filt = tgt_arr[mask]
w_filt = w_scaled[mask]
d_filt = np.maximum(d_arr[mask], 1.0)
rtype_filt = rtype_arr[mask] + 1  # Convert 0,1,2,3 to 1,2,3,4 for NEST

print(f"Building {len(src_filt)} connections...")
v1_src_gids = v1_gids[src_filt]
v1_tgt_gids = v1_gids[tgt_filt]

print(f"Connecting {len(v1_src_gids)} synapses...")
nest.Connect(v1_src_gids.tolist(), v1_tgt_gids.tolist(),
             conn_spec='one_to_one',
             syn_spec={'weight': w_filt, 'delay': d_filt, 'receptor_type': rtype_filt.tolist()})

print(f"  Connected {len(v1_src_gids)} recurrent synapses")

# Record spikes
print("Setting up recording...")
spike_rec = nest.Create('spike_recorder')
for oid in output_neurons:
    nest.Connect(v1[int(oid)], spike_rec)

# Run
print("\n" + "=" * 80)
print("Running 1000ms simulation...")
nest.Simulate(1000.0)
print("Done!")

# Analyze
print("\n" + "=" * 80)
print("Analyzing results...")

events = spike_rec.get('events')
senders, times = events['senders'], events['times']

# Create mapping from global ID to digit class
sender_to_class = {}
for i, oid in enumerate(output_neurons):
    sender_to_class[v1_gids[oid]] = i // 30

# Count votes in time windows
for window_name, t_start, t_end in [
    ('50-200ms', 50, 200),
    ('50-100ms (target)', 50, 100),
]:
    votes = np.zeros(10)
    for sender, time in zip(senders, times):
        if t_start <= time < t_end and sender in sender_to_class:
            votes[sender_to_class[sender]] += 1

    prediction = np.argmax(votes)
    print(f"\n{window_name}:")
    print(f"  Votes: {votes}")
    print(f"  Prediction: {prediction}, Expected: {label}")
    print(f"  Correct: {prediction == label}")

print("=" * 80)
