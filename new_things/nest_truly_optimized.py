#!/usr/bin/env python
"""TRULY Optimized NEST - Full NumPy vectorization"""

import os
import h5py
import numpy as np
import nest

np.random.seed(1)
TARGET_INDEX = int(os.environ.get('TARGET_INDEX', 0))

print("=" * 80)
print(f"TRULY OPTIMIZED NEST V1 Inference - Sample {TARGET_INDEX}")
print("=" * 80)

os.chdir('/home/user/upload/new_things')

# Load network
print("Loading network...")
with h5py.File('ckpt_51978-153.h5', 'r') as f:
    neurons = np.array(f['neurons/node_type_ids'])
    glif3 = np.stack([
        f['neurons/glif3_params/C_m'][:] / 1000.0,  # pF -> nF
        f['neurons/glif3_params/E_L'][:],
        f['neurons/glif3_params/V_reset'][:],
        f['neurons/glif3_params/V_th'][:],
        f['neurons/glif3_params/g'][:] / 1000.0,  # nS -> uS
    ], axis=1)

    vsc = glif3[:, 3] - glif3[:, 1]  # V_th - E_L
    tau_m = glif3[:, 0] / glif3[:, 4]

    recurrent = np.array([
        f['recurrent/sources'][:],
        f['recurrent/targets'][:],
        f['recurrent/weights'][:],
        f['recurrent/delays'][:]
    ]).T

    input_syns = np.array([
        f['input/sources'][:],
        f['input/targets'][:],
        f['input/weights'][:]
    ]).T

    output_neurons = np.array(f['readout/neuron_ids'], dtype=int)

print(f"  Neurons: {len(neurons)}, Recurrent: {len(recurrent)}, Input: {len(input_syns)}")

# Load spikes
with h5py.File('mnist.h5', 'r') as f:
    spike_trains = np.array(f['spike_trains'])
    labels = np.array(f['labels'])

label = labels[TARGET_INDEX]
print(f"  Label: {label}")

# Create spike times
print("Creating spike times...")
sample_spikes = spike_trains[TARGET_INDEX]
spike_indices = np.where(sample_spikes > 0)
spike_times = {}
for t_idx, neuron_idx in zip(*spike_indices):
    t = max(float(t_idx + 1), 1.0)
    if neuron_idx not in spike_times:
        spike_times[neuron_idx] = []
    spike_times[neuron_idx].append(t)

print(f"  Active LGN: {len(spike_times)}")

# Setup NEST
print("Setting up NEST...")
nest.ResetKernel()
nest.resolution = 1.0

# Create V1 neurons
print("Creating V1...")
v1 = nest.Create('iaf_psc_exp', len(neurons))

# Set parameters by type (batch)
print("Setting neuron parameters...")
unique_types = np.unique(neurons)
for ntype in unique_types:
    mask = neurons == ntype
    indices = np.where(mask)[0]
    g = glif3[ntype]

    params = {
        'C_m': float(g[0]),
        'tau_m': float(tau_m[ntype]),
        'E_L': float(g[1]),
        'V_reset': float(g[2]),
        'V_th': float(g[3]),
        't_ref': 2.0,
        'tau_syn_ex': 2.0,
        'tau_syn_in': 5.0,
        'V_m': float(g[2])
    }

    for idx in indices:
        v1[int(idx)].set(params)

# Create LGN
print("Creating LGN...")
lgn = nest.Create('spike_generator', spike_trains.shape[2])
for i, times in spike_times.items():
    lgn[i].set({'spike_times': times})

print("Pre-computing global IDs...")
# Pre-compute all global IDs as numpy arrays (KEY OPTIMIZATION!)
v1_gids = np.array([n.global_id for n in v1])
lgn_gids = np.array([n.global_id for n in lgn])

# Connect LGN -> V1 (VECTORIZED!)
print("Connecting LGN -> V1...")
src_arr = input_syns[:, 0].astype(int)
tgt_arr = input_syns[:, 1].astype(int)
w_arr = input_syns[:, 2]

# Vectorized weight scaling
w_scaled = w_arr * vsc[neurons[tgt_arr]] / 1000.0

# Filter
mask = np.abs(w_scaled) > 1e-10
src_filt = src_arr[mask]
tgt_filt = tgt_arr[mask]
w_filt = w_scaled[mask]

# Use pre-computed GIDs (FAST!)
lgn_conn_gids = lgn_gids[src_filt]
v1_conn_gids = v1_gids[tgt_filt]

# Connect
delays_lgn = np.ones(len(w_filt))
nest.Connect(lgn_conn_gids.tolist(), v1_conn_gids.tolist(),
             conn_spec='one_to_one',
             syn_spec={'weight': w_filt, 'delay': delays_lgn})

print(f"  Connected {len(w_filt)} LGN synapses")

# Connect V1 recurrent (VECTORIZED!)
print("Connecting V1 recurrent...")
src_arr = recurrent[:, 0].astype(int)
tgt_arr = recurrent[:, 1].astype(int)
w_arr = recurrent[:, 2]
d_arr = recurrent[:, 3]

# Vectorized weight scaling
w_scaled = w_arr * vsc[neurons[tgt_arr]] / 1000.0

# Filter
mask = np.abs(w_scaled) > 1e-10
src_filt = src_arr[mask]
tgt_filt = tgt_arr[mask]
w_filt = w_scaled[mask]
d_filt = np.maximum(d_arr[mask], 1.0)

print(f"Building {len(src_filt)} connections...")
# Use pre-computed GIDs (FAST!)
v1_src_gids = v1_gids[src_filt]
v1_tgt_gids = v1_gids[tgt_filt]

print(f"Connecting {len(v1_src_gids)} synapses...")
nest.Connect(v1_src_gids.tolist(), v1_tgt_gids.tolist(),
             conn_spec='one_to_one',
             syn_spec={'weight': w_filt, 'delay': d_filt})

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
