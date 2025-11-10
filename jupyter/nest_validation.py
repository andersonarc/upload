#!/usr/bin/env python3
"""
NEST GLIF3 Validation Script

Purpose: Validate SpiNNaker GLIF3 implementation by comparing with NEST's native GLIF model.

Strategy:
1. Use NEST's glif_psc model with same parameters as SpiNNaker
2. Feed same input spike trains (from spikes-128.h5)
3. Use same weights (from ckpt_51978-153.h5)
4. Compare output spikes with SpiNNaker and TensorFlow

This isolates whether bugs are in:
- GLIF3 implementation (if NEST != SpiNNaker)
- OR routing/connectivity (if NEST == SpiNNaker but both != TensorFlow)
"""

import numpy as np
import h5py
import sys

# Check NEST availability
try:
    import nest
    NEST_AVAILABLE = True
except ImportError:
    print("=" * 80)
    print("ERROR: NEST simulator not available")
    print("=" * 80)
    print("\nTo install NEST:")
    print("  conda install -c conda-forge nest-simulator")
    print("OR:")
    print("  Build from source: https://nest-simulator.readthedocs.io/")
    print("\nThis script cannot run without NEST, but demonstrates the approach.")
    print("=" * 80)
    NEST_AVAILABLE = False
    sys.exit(1)

# File paths
H5_FILE = 'ckpt_51978-153.h5'
SPIKES_FILE = 'spikes-128.h5'

# Test configuration
TEST_SAMPLES = [10, 50, 90, 100]  # Failed and "working" samples
SIMULATION_TIME = 100.0  # ms
TIMESTEP = 1.0  # ms
SEED = 1  # Match class.py seed

# Network structure (from class.py)
LGN_SIZE = 17400
V1_SIZE = 51978
OUTPUT_SIZE = 300
N_POPULATIONS = 111  # Number of neuron types (from H5)

print("=" * 80)
print("NEST GLIF3 Validation")
print("=" * 80)

# =============================================================================
# Step 1: Load network parameters from H5
# =============================================================================

print("\nStep 1: Loading network parameters...")

try:
    with h5py.File(H5_FILE, 'r') as f:
        # GLIF3 parameters (per population type)
        C_m = np.array(f['neurons/glif3_params/C_m'])          # pF
        E_L = np.array(f['neurons/glif3_params/E_L'])          # mV
        V_reset = np.array(f['neurons/glif3_params/V_reset'])  # mV
        V_th = np.array(f['neurons/glif3_params/V_th'])        # mV
        asc_amps = np.array(f['neurons/glif3_params/asc_amps'])  # pA (UNNORMALIZED!)
        k = np.array(f['neurons/glif3_params/k'])              # 1/ms
        g = np.array(f['neurons/glif3_params/g'])              # nS
        tau_syn = np.array(f['neurons/glif3_params/tau_syn'])  # ms

        # Neuron type assignments
        node_type_ids = np.array(f['neurons/node_type_ids'])

        # Connectivity
        rec_sources = np.array(f['recurrent/sources'])
        rec_targets = np.array(f['recurrent/targets'])
        rec_weights = np.array(f['recurrent/weights'])
        rec_receptors = np.array(f['recurrent/receptors'])
        rec_delays = np.array(f['recurrent/delays'])

        inp_sources = np.array(f['input/sources'])
        inp_targets = np.array(f['input/targets'])
        inp_weights = np.array(f['input/weights'])
        inp_receptors = np.array(f['input/receptors'])

        # Output neuron IDs
        output_neurons = np.array(f['output/neurons'])

    print(f"  ✓ Loaded {len(C_m)} neuron types")
    print(f"  ✓ Loaded {len(rec_sources)} recurrent synapses")
    print(f"  ✓ Loaded {len(inp_sources)} input synapses")
    print(f"  ✓ Loaded {len(output_neurons)} output neurons")

except FileNotFoundError:
    print(f"ERROR: H5 file not found: {H5_FILE}")
    print("Please download from HuggingFace repository")
    sys.exit(1)

# =============================================================================
# Step 2: Parameter Conversion (H5 format → NEST format)
# =============================================================================

print("\nStep 2: Converting parameters to NEST format...")

# CRITICAL: Fix ASC scaling bug (from Phase 6 analysis)
# H5 contains UNNORMALIZED values in pA
# NEST expects nA
# WRONG (class.py bug): asc_nA = asc_pA * voltage_scale / 1000
# CORRECT: asc_nA = asc_pA / 1000

voltage_scale = V_th - E_L  # mV

print(f"\n  ASC Scaling Verification:")
print(f"    H5 asc_amps format: UNNORMALIZED (pA)")
print(f"    NEST expects: nA")
print(f"    Voltage scale range: [{voltage_scale.min():.2f}, {voltage_scale.max():.2f}] mV")
print(f"    ")
print(f"    SpiNNaker (BUGGY):  asc_nA = asc_pA * voltage_scale / 1000")
print(f"    NEST (CORRECT):     asc_nA = asc_pA / 1000")
print(f"    ")
print(f"    This means SpiNNaker ASC is ~{voltage_scale.mean():.0f}x too large!")

# Convert parameters for NEST
asc_amps_nA = asc_amps / 1000.0  # CORRECT conversion (pA -> nA)
C_m_nF = C_m / 1000.0  # pF -> nF
g_uS = g / 1000.0  # nS -> uS

print(f"\n  Parameter ranges:")
print(f"    C_m: [{C_m_nF.min():.4f}, {C_m_nF.max():.4f}] nF")
print(f"    E_L: [{E_L.min():.2f}, {E_L.max():.2f}] mV")
print(f"    V_th: [{V_th.min():.2f}, {V_th.max():.2f}] mV")
print(f"    asc_amps[0]: [{asc_amps_nA[:, 0].min():.4f}, {asc_amps_nA[:, 0].max():.4f}] nA")
print(f"    asc_amps[1]: [{asc_amps_nA[:, 1].min():.4f}, {asc_amps_nA[:, 1].max():.4f}] nA")
print(f"    k[0]: [{k[:, 0].min():.4f}, {k[:, 0].max():.4f}] 1/ms")
print(f"    k[1]: [{k[:, 1].min():.4f}, {k[:, 1].max():.4f}] 1/ms")

# =============================================================================
# Step 3: Initialize NEST
# =============================================================================

print("\nStep 3: Initializing NEST...")

nest.ResetKernel()
nest.SetKernelStatus({
    'resolution': TIMESTEP,
    'rng_seed': SEED,
    'print_time': False
})

print(f"  ✓ NEST kernel initialized")
print(f"    Resolution: {TIMESTEP} ms")
print(f"    Seed: {SEED}")

# =============================================================================
# Step 4: Create neurons
# =============================================================================

print("\nStep 4: Creating NEST GLIF neurons...")

# NEST GLIF model parameters (from web search results)
# glif_psc model with after-spike currents enabled

neurons = []
neuron_gids = []

for gid in range(V1_SIZE):
    # Get neuron type
    ntype = node_type_ids[gid]

    # Create GLIF neuron with parameters for this type
    # Note: NEST parameter names may differ slightly
    params = {
        'C_m': float(C_m_nF[ntype]),
        'E_L': float(E_L[ntype]),
        'V_reset': float(V_reset[ntype]),
        'V_th': float(V_th[ntype]),
        'g_L': float(g_uS[ntype]),  # Leak conductance
        't_ref': 2.0,  # Refractory period (default, not in H5)
        'V_m': float(E_L[ntype]),  # Initial voltage
        'after_spike_currents': True,
        'asc_amps': [float(asc_amps_nA[ntype, 0]), float(asc_amps_nA[ntype, 1])],
        'k': [float(k[ntype, 0]), float(k[ntype, 1])],
        'tau_syn': [float(tau_syn[ntype, 0]), float(tau_syn[ntype, 1]),
                    float(tau_syn[ntype, 2]), float(tau_syn[ntype, 3])],
    }

    neuron = nest.Create('glif_psc', params=params)
    neurons.append(neuron)
    neuron_gids.append(gid)

    if gid % 10000 == 0:
        print(f"  Created {gid}/{V1_SIZE} neurons...")

print(f"  ✓ Created {V1_SIZE} GLIF neurons")

# =============================================================================
# Step 5: Run simulation for each test sample
# =============================================================================

print("\nStep 5: Running simulations...")

# Load input spikes
try:
    with h5py.File(SPIKES_FILE, 'r') as f:
        # spikes-128.h5 contains spike trains for 128 samples
        # Shape: (128 samples, 100 timesteps, 17400 LGN neurons)
        spike_probs = np.array(f['spikes'])

    print(f"  ✓ Loaded spike probabilities: {spike_probs.shape}")

except FileNotFoundError:
    print(f"ERROR: Spikes file not found: {SPIKES_FILE}")
    print("Please download from HuggingFace repository")
    sys.exit(1)

for sample_idx in TEST_SAMPLES:
    print(f"\n  Sample {sample_idx}:")

    # Reset network state
    nest.ResetNetwork()

    # Generate LGN spike trains (same as class.py)
    np.random.seed(SEED)  # Reset seed for determinism

    lgn_spike_times = []
    for lgn_i in range(LGN_SIZE):
        times = []
        for t in range(spike_probs.shape[1]):
            # Same Poisson sampling as class.py (with 1.3 removal)
            prob = np.clip(spike_probs[sample_idx, t, lgn_i] / 1.3, 0.0, 1.0)
            if prob > np.random.rand():
                times.append(float(t * TIMESTEP))
        lgn_spike_times.append(times if len(times) > 0 else [SIMULATION_TIME + 100.0])

    # Create LGN spike sources
    lgn_sources = []
    for i in range(LGN_SIZE):
        source = nest.Create('spike_generator', params={'spike_times': lgn_spike_times[i]})
        lgn_sources.append(source)

        if i % 5000 == 0:
            print(f"    Created {i}/{LGN_SIZE} LGN sources...")

    # Connect LGN to V1
    for i, (src, tgt, weight, receptor) in enumerate(zip(inp_sources, inp_targets, inp_weights, inp_receptors)):
        nest.Connect(lgn_sources[src], neurons[tgt],
                     syn_spec={'weight': weight, 'receptor_type': int(receptor) + 1})

        if i % 100000 == 0:
            print(f"    Connected {i}/{len(inp_sources)} input synapses...")

    # Connect V1 recurrent
    for i, (src, tgt, weight, receptor, delay) in enumerate(zip(rec_sources, rec_targets, rec_weights, rec_receptors, rec_delays)):
        nest.Connect(neurons[src], neurons[tgt],
                     syn_spec={'weight': weight, 'delay': delay, 'receptor_type': int(receptor) + 1})

        if i % 500000 == 0:
            print(f"    Connected {i}/{len(rec_sources)} recurrent synapses...")

    # Record output neurons
    spike_recorder = nest.Create('spike_recorder')
    for out_gid in output_neurons:
        nest.Connect(neurons[out_gid], spike_recorder)

    # Run simulation
    print(f"    Running simulation for {SIMULATION_TIME} ms...")
    nest.Simulate(SIMULATION_TIME)

    # Get spikes
    events = nest.GetStatus(spike_recorder, 'events')[0]
    spike_times = events['times']
    spike_senders = events['senders']

    # Decode output (same logic as class.py)
    votes = np.zeros(10)
    for class_idx in range(10):
        start = class_idx * 30
        end = (class_idx + 1) * 30
        class_neurons = output_neurons[start:end]

        for neuron_gid in class_neurons:
            # Find spikes from this neuron in response window [50, 100] ms
            neuron_spikes = spike_times[spike_senders == neuron_gid]
            response_spikes = neuron_spikes[(neuron_spikes > 50.0) & (neuron_spikes < 100.0)]
            votes[class_idx] += len(response_spikes)

    predicted_class = np.argmax(votes)

    print(f"    Predicted class: {predicted_class}")
    print(f"    Vote distribution: {votes}")
    print(f"    Total output spikes: {len(spike_times)}")
    print(f"    Response window spikes: {int(votes.sum())}")

# =============================================================================
# Step 6: Summary and Comparison
# =============================================================================

print("\n" + "=" * 80)
print("NEST Validation Complete")
print("=" * 80)

print("\nNext steps:")
print("1. Compare NEST outputs with SpiNNaker outputs (from class.py)")
print("2. Compare with TensorFlow outputs (ground truth)")
print("3. If NEST matches TF but SpiNNaker doesn't:")
print("   → GLIF3 implementation bug in SpiNNaker")
print("4. If NEST matches SpiNNaker (both wrong):")
print("   → Bug is in routing, weights, or other aspect")
print("\nKey finding from this analysis:")
print("  ASC scaling bug WILL cause different results:")
print(f"    SpiNNaker uses: asc_nA = asc_pA * {voltage_scale.mean():.1f} / 1000")
print(f"    NEST uses:      asc_nA = asc_pA / 1000")
print(f"    Difference: {voltage_scale.mean():.1f}x")

print("\n" + "=" * 80)
