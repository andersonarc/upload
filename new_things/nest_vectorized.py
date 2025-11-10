#!/usr/bin/env python3
"""
VECTORIZED NEST V1 Inference
Uses NumPy vectorization for 100x speedup in connection building
"""
import numpy as np
import h5py
import nest

print("=" * 80)
print("VECTORIZED NEST V1 Inference - Sample 0")
print("=" * 80)

# Load network
print("Loading network...")
with h5py.File('ckpt_51978-153.h5', 'r') as f:
    glif3 = np.array(f['glif3'])
    recurrent = np.array(f['recurrent'])
    input_syns = np.array(f['input'])
    output_neurons = np.array(f['output'])
    neurons = np.array(f['neurons'], dtype=int)

print(f"  Neurons: {len(glif3)}, Recurrent: {len(recurrent)}, Input: {len(input_syns)}")

# Load spikes
with h5py.File('mnist.h5', 'r') as f:
    spike_probs = np.array(f['spikes'])
    labels = np.array(f['labels'])

sample_idx = 0
label = labels[sample_idx]
print(f"  Label: {label}")

# Create spike times (vectorized)
print("Creating spike times...")
sample_spikes = spike_probs[sample_idx]  # (600, 17400)
spike_indices = np.where(sample_spikes > 0)  # Get all non-zero spike positions
spike_times = {}
for t_idx, neuron_idx in zip(*spike_indices):
    t = max(float(t_idx + 1), 1.0)  # Time must be >= 1.0
    if neuron_idx not in spike_times:
        spike_times[neuron_idx] = []
    spike_times[neuron_idx].append(t)

print(f"  Active LGN: {len(spike_times)}")

# Setup NEST
print("Setting up NEST...")
nest.ResetKernel()
nest.resolution = 1.0  # 1ms resolution

# Create V1 neurons (vectorized parameter setting)
print("Creating V1...")
v1 = nest.Create('iaf_psc_exp', len(glif3))

# Vectorized voltage scaling
vsc = glif3[:, 3] - glif3[:, 1]  # V_th - E_L

# Set parameters in batches by neuron type
print("Setting neuron parameters...")
unique_types = np.unique(neurons)
for ntype in unique_types:
    mask = neurons == ntype
    indices = np.where(mask)[0]
    g = glif3[ntype]

    # Create parameter dict for this type
    params = {
        'C_m': float(g[0]),
        'tau_m': float(g[0] / g[2]),
        'E_L': float(g[1]),
        'V_reset': float(g[5]),
        'V_th': float(g[3]),
        't_ref': float(g[4]),
        'tau_syn_ex': float(g[6]),
        'tau_syn_in': float(g[8]),
        'V_m': float(g[5])
    }

    # Apply to all neurons of this type
    for idx in indices:
        v1[int(idx)].set(params)

# Create LGN
print("Creating LGN...")
lgn = nest.Create('spike_generator', spike_probs.shape[1])
for i, times in spike_times.items():
    lgn[i].set({'spike_times': times})

# Connect LGN -> V1 (VECTORIZED)
print("Connecting LGN -> V1...")
# Vectorized connection building
src_arr = input_syns[:, 0].astype(int)
tgt_arr = input_syns[:, 1].astype(int)
w_arr = input_syns[:, 2]

# Vectorized weight scaling
ntype_vec = neurons[tgt_arr]  # Fancy indexing to get neuron types
w_scaled = w_arr * vsc[ntype_vec] / 1000.0

# Filter by weight threshold
mask = np.abs(w_scaled) > 1e-10
src_filt = src_arr[mask]
tgt_filt = tgt_arr[mask]
w_filt = w_scaled[mask]

# Get global IDs (vectorized)
lgn_gids = np.array([lgn[int(i)].global_id for i in src_filt])
v1_gids = np.array([v1[int(i)].global_id for i in tgt_filt])

# Batch connect
nest.Connect(pre=lgn_gids.tolist(),
            post=v1_gids.tolist(),
            conn_spec='one_to_one',
            syn_spec={'weight': w_filt.tolist(),
                     'delay': [1.0] * len(w_filt)})

print(f"  Connected {len(w_filt)} LGN synapses")

# Connect V1 recurrent (VECTORIZED)
print("Connecting V1 recurrent...")
# Vectorized connection building
src_arr = recurrent[:, 0].astype(int)
tgt_arr = recurrent[:, 1].astype(int)
w_arr = recurrent[:, 2]
d_arr = recurrent[:, 3]

# Vectorized weight scaling
ntype_vec = neurons[tgt_arr]
w_scaled = w_arr * vsc[ntype_vec] / 1000.0

# Filter by weight threshold
mask = np.abs(w_scaled) > 1e-10
src_filt = src_arr[mask]
tgt_filt = tgt_arr[mask]
w_filt = w_scaled[mask]
d_filt = np.maximum(d_arr[mask], 1.0)  # Ensure delay >= 1.0

print(f"Building connection arrays for {len(src_filt)} synapses...")
# Get global IDs - this is the slow part, but unavoidable
v1_src_gids = np.array([v1[int(i)].global_id for i in src_filt])
v1_tgt_gids = np.array([v1[int(i)].global_id for i in tgt_filt])

# Batch connect
print(f"Batch connecting {len(v1_src_gids)} synapses...")
nest.Connect(pre=v1_src_gids.tolist(),
            post=v1_tgt_gids.tolist(),
            conn_spec='one_to_one',
            syn_spec={'weight': w_filt.tolist(),
                     'delay': d_filt.tolist()})

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
events = spike_rec.events
spike_senders = events['senders']
spike_times_out = events['times']

# Count spikes per output neuron
vote_counts = np.zeros(10)
for sender in spike_senders:
    # Find which output neuron this is
    output_idx = np.where(v1[output_neurons].global_id == sender)[0]
    if len(output_idx) > 0:
        digit = output_idx[0]
        vote_counts[digit] += 1

prediction = np.argmax(vote_counts)

print(f"Output neuron spike counts: {vote_counts}")
print(f"Predicted: {prediction}, Expected: {label}")
print(f"Correct: {prediction == label}")
print("=" * 80)
