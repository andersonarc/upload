#!/usr/bin/env python
"""Optimized NEST Simulator - batch connections"""

import os
import h5py
import numpy as np
import nest

np.random.seed(1)
TARGET_INDEX = int(os.environ.get('TARGET_INDEX', 0))

print("=" * 80)
print(f"OPTIMIZED NEST V1 Inference - Sample {TARGET_INDEX}")
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
    spike_probs = np.array(f['spike_trains'][TARGET_INDEX])
    label = int(f['labels'][TARGET_INDEX])
print(f"  Label: {label}")

# Create spike times
print("Creating spike times...")
spike_times = {}
for i in range(spike_probs.shape[1]):
    times = []
    for t in range(spike_probs.shape[0]):
        if np.clip(spike_probs[t, i] / 1.3, 0, 1) > np.random.rand():
            times.append(float(max(t, 1)))
    if len(times) > 0:
        spike_times[i] = times

print(f"  Active LGN: {len(spike_times)}")

# Setup NEST
print("Setting up NEST...")
nest.ResetKernel()
nest.resolution = 1.0
nest.print_time = False

# Create V1 - batch set parameters
print("Creating V1...")
v1 = nest.Create('iaf_psc_exp', len(neurons))
for i in range(len(neurons)):
    ntype = neurons[i]
    v1[i].set({
        'C_m': float(glif3[ntype, 0] * 1000.0),
        'tau_m': float(tau_m[ntype]),
        'V_reset': float(glif3[ntype, 2]),
        'E_L': float(glif3[ntype, 1]),
        'V_th': float(glif3[ntype, 3]),
        't_ref': 2.0,
        'tau_syn_ex': 5.0,
        'tau_syn_in': 5.0,
        'V_m': float(glif3[ntype, 1])
    })

# Create LGN
print("Creating LGN...")
lgn = nest.Create('spike_generator', spike_probs.shape[1])
for i in range(len(lgn)):
    if i in spike_times:
        lgn[i].set({'spike_times': spike_times[i]})

# Connect LGN -> V1 in batch
print("Connecting LGN -> V1...")
lgn_conns = []
for i, syn in enumerate(input_syns):
    src, tgt, weight = int(syn[0]), int(syn[1]), float(syn[2])
    ntype = neurons[tgt]
    w = weight * vsc[ntype] / 1000.0
    if abs(w) > 1e-10:
        lgn_conns.append({
            'source': lgn[src].global_id,
            'target': v1[tgt].global_id,
            'weight': w,
            'delay': 1.0
        })

nest.Connect(pre=[c['source'] for c in lgn_conns],
            post=[c['target'] for c in lgn_conns],
            conn_spec='one_to_one',
            syn_spec={'weight': [c['weight'] for c in lgn_conns],
                     'delay': [c['delay'] for c in lgn_conns]})

print(f"  Connected {len(lgn_conns)} LGN synapses")

# Connect V1 recurrent in batch
print("Connecting V1 recurrent...")
v1_conns = []
for i, syn in enumerate(recurrent):
    if i % 2000000 == 0 and i > 0:
        print(f"  {i}/{len(recurrent)}")

    src, tgt, weight, delay = int(syn[0]), int(syn[1]), float(syn[2]), float(syn[3])
    ntype = neurons[tgt]
    w = weight * vsc[ntype] / 1000.0
    if abs(w) > 1e-10:
        v1_conns.append({
            'source': v1[src].global_id,
            'target': v1[tgt].global_id,
            'weight': w,
            'delay': max(delay, 1.0)
        })

print(f"Batch connecting {len(v1_conns)} synapses...")
nest.Connect(pre=[c['source'] for c in v1_conns],
            post=[c['target'] for c in v1_conns],
            conn_spec='one_to_one',
            syn_spec={'weight': [c['weight'] for c in v1_conns],
                     'delay': [c['delay'] for c in v1_conns]})

print(f"  Connected {len(v1_conns)} recurrent synapses")

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

sender_to_class = {}
for i, oid in enumerate(output_neurons):
    sender_to_class[v1[oid].global_id] = i // 30

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
