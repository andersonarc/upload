#!/usr/bin/env python3
"""
NEST V1 Inference - Optimized with proper NEST connectors
"""
import os
import h5py
import numpy as np
import nest

np.random.seed(1)
TARGET_INDEX = int(os.environ.get('TARGET_INDEX', 0))

print("=" * 80)
print(f"NEST V1 Inference (Optimized) - Sample {TARGET_INDEX}")
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
    spike_probs = np.array(f['spikes'])
    labels = np.array(f['labels'])

label = labels[TARGET_INDEX]
print(f"  Label: {label}")

# Create spike times
print("Creating spike times...")
sample_spikes = spike_probs[TARGET_INDEX]
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

# Set neuron parameters by type (batch)
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
        't_ref': 2.0,  # Default refrac
        'tau_syn_ex': 2.0,
        'tau_syn_in': 5.0,
        'V_m': float(g[2])
    }

    for idx in indices:
        v1[int(idx)].set(params)

# Create LGN
print("Creating LGN...")
lgn = nest.Create('spike_generator', spike_probs.shape[1])
for i, times in spike_times.items():
    lgn[i].set({'spike_times': times})

# Connect LGN -> V1 using from_list connector
print("Connecting LGN -> V1...")
# Build connection list as NumPy arrays (fast)
src_arr = input_syns[:, 0].astype(int)
tgt_arr = input_syns[:, 1].astype(int)
w_arr = input_syns[:, 2]

# Vectorized weight scaling
ntype_vec = neurons[tgt_arr]
w_scaled = w_arr * vsc[ntype_vec] / 1000.0

# Filter
mask = np.abs(w_scaled) > 1e-10
src_filt = src_arr[mask]
tgt_filt = tgt_arr[mask]
w_filt = w_scaled[mask]

# Use NEST's from_list connector with global IDs
lgn_gids = lgn[src_filt.tolist()].tolist()
v1_gids = v1[tgt_filt.tolist()].tolist()

# Connect using one_to_one (optimized in NEST)
nest.Connect(lgn_gids, v1_gids,
             conn_spec='one_to_one',
             syn_spec={'weight': w_filt.tolist(), 'delay': 1.0})

print(f"  Connected {len(w_filt)} LGN synapses")

# Connect V1 recurrent
print("Connecting V1 recurrent...")
src_arr = recurrent[:, 0].astype(int)
tgt_arr = recurrent[:, 1].astype(int)
w_arr = recurrent[:, 2]
d_arr = recurrent[:, 3]

# Vectorized weight scaling
ntype_vec = neurons[tgt_arr]
w_scaled = w_arr * vsc[ntype_vec] / 1000.0

# Filter
mask = np.abs(w_scaled) > 1e-10
src_filt = src_arr[mask]
tgt_filt = tgt_arr[mask]
w_filt = w_scaled[mask]
d_filt = np.maximum(d_arr[mask], 1.0)

print(f"Building connections for {len(src_filt)} synapses...")

# Use NEST's from_list connector
v1_src = v1[src_filt.tolist()].tolist()
v1_tgt = v1[tgt_filt.tolist()].tolist()

print(f"Connecting {len(v1_src)} synapses...")
nest.Connect(v1_src, v1_tgt,
             conn_spec='one_to_one',
             syn_spec={'weight': w_filt.tolist(), 'delay': d_filt.tolist()})

print(f"  Connected {len(v1_src)} recurrent synapses")

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
events = spike_rec.events
spike_senders = events['senders']

# Count spikes per output neuron
output_gids = v1[output_neurons.tolist()].tolist()
vote_counts = np.zeros(10)
for sender in spike_senders:
    try:
        digit = output_gids.index(sender)
        vote_counts[digit] += 1
    except ValueError:
        pass

prediction = np.argmax(vote_counts)

print(f"Output neuron spike counts: {vote_counts}")
print(f"Predicted: {prediction}, Expected: {label}")
print(f"Correct: {prediction == label}")
print("=" * 80)
