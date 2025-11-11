#!/usr/bin/env python
"""Fast NEST Simulator inference - uses native NEST API for speed"""

import os
import h5py
import numpy as np
import nest

np.random.seed(1)

TARGET_INDEX = int(os.environ.get('TARGET_INDEX', 0))

print("=" * 80)
print(f"FAST NEST Simulator V1 Inference - Sample {TARGET_INDEX}")
print("=" * 80)

# Load network
print("Loading network...")
import os
os.chdir('/home/user/upload/new_things')
with h5py.File('ckpt_51978-153.h5', 'r') as f:
    neurons = np.array(f['neurons/node_type_ids'])
    glif3 = np.stack([
        f['neurons/glif3_params/C_m'][:] / 1000.0,  # pF -> nF
        f['neurons/glif3_params/E_L'][:],
        f['neurons/glif3_params/V_reset'][:],
        f['neurons/glif3_params/V_th'][:],
        f['neurons/glif3_params/g'][:] / 1000.0,  # nS -> uS
    ], axis=1)

    # Calculate voltage scale
    vsc = glif3[:, 3] - glif3[:, 1]  # V_th - E_L

    # Convert to IF_curr_exp: tau_m = C_m / g
    tau_m = glif3[:, 0] / glif3[:, 4]

    recurrent = np.stack([
        f['recurrent/sources'][:],
        f['recurrent/targets'][:],
        f['recurrent/weights'][:],
        f['recurrent/delays'][:]
    ], axis=1)

    input_syns = np.stack([
        f['input/sources'][:],
        f['input/targets'][:],
        f['input/weights'][:]
    ], axis=1)

    output_neurons = np.array(f['readout/neuron_ids'])

print(f"  Neurons: {len(neurons)}")
print(f"  Recurrent: {len(recurrent)}")
print(f"  Input: {len(input_syns)}")

# Load spikes
print("Loading spikes...")
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
            times.append(float(max(t, 1)))  # Avoid t=0, use t>=1
    if len(times) > 0:
        spike_times[i] = times

print(f"  Active LGN neurons: {len(spike_times)}")

# Setup NEST
print("\nSetting up NEST...")
nest.ResetKernel()
nest.resolution = 1.0
nest.print_time = False
nest.overwrite_files = True

# Create V1 neurons - one big population
print("Creating V1 neurons...")
v1_params = {}
for neuron_id in range(len(neurons)):
    ntype = neurons[neuron_id]
    v1_params[neuron_id] = {
        'C_m': float(glif3[ntype, 0] * 1000.0),  # nF -> pF
        'tau_m': float(tau_m[ntype]),
        'V_reset': float(glif3[ntype, 2]),
        'E_L': float(glif3[ntype, 1]),
        'V_th': float(glif3[ntype, 3]),
        't_ref': 2.0,
        'tau_syn_ex': 5.0,
        'tau_syn_in': 5.0,
        'V_m': float(glif3[ntype, 1])
    }

v1 = nest.Create('iaf_psc_exp', len(neurons))

# Set parameters
print("Setting neuron parameters...")
for i in range(len(v1)):
    v1[i].set(v1_params[i])

# Create LGN
print("Creating LGN spike sources...")
lgn = nest.Create('spike_generator', spike_probs.shape[1])
for i in range(len(lgn)):
    if i in spike_times:
        lgn[i].set({'spike_times': spike_times[i]})

# Connect LGN -> V1
print("Connecting LGN -> V1...")
for syn in input_syns[:]:
    src, tgt, weight = int(syn[0]), int(syn[1]), float(syn[2])
    ntype = neurons[tgt]
    weight_scaled = weight * vsc[ntype] / 1000.0

    if weight_scaled != 0:
        nest.Connect(lgn[src], v1[tgt], syn_spec={
            'weight': weight_scaled,
            'delay': 1.0
        })

# Connect V1 recurrent
print("Connecting V1 recurrent...")
for i, syn in enumerate(recurrent):
    if i % 1000000 == 0 and i > 0:
        print(f"  {i}/{len(recurrent)}")

    src, tgt, weight, delay = int(syn[0]), int(syn[1]), float(syn[2]), float(syn[3])
    ntype = neurons[tgt]
    weight_scaled = weight * vsc[ntype] / 1000.0

    if weight_scaled != 0:
        nest.Connect(v1[src], v1[tgt], syn_spec={
            'weight': weight_scaled,
            'delay': max(delay, 1.0)  # Must be >= resolution
        })

# Record spikes from output neurons
print("Setting up recording...")
spike_recorder = nest.Create('spike_recorder')
nest.Connect(v1[[int(i) for i in output_neurons]], spike_recorder)

# Run simulation
print("\n" + "=" * 80)
print("Running 1000ms simulation...")
nest.Simulate(1000.0)
print("Done!")

# Analyze results
print("\n" + "=" * 80)
print("Analyzing results...")

events = spike_recorder.get('events')
senders = events['senders']
times = events['times']

# Map senders to classes
sender_to_class = {}
for i, oid in enumerate(output_neurons):
    sender_to_class[v1[int(oid)].global_id] = i // 30

# Count votes
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
