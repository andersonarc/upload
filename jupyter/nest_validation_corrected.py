#!/usr/bin/env python3
"""
NEST GLIF3 Validation Script - CORRECTED VERSION

Validates SpiNNaker GLIF3 implementation against NEST's native glif_psc model.
Uses actual checkpoint parameters and proper NEST API.
"""

import numpy as np
import sys

# Check NEST availability
try:
    import nest
    print(f"NEST version: {nest.version()}")
except ImportError:
    print("ERROR: NEST not installed")
    print("Install with: conda install -c conda-forge nest-simulator")
    sys.exit(1)

print("=" * 80)
print("NEST GLIF3 Validation")
print("=" * 80)

# Configuration
SIMULATION_TIME = 100.0  # ms
TIMESTEP = 0.1  # ms (NEST uses finer resolution than PyNN's 1ms)
SEED = 1

# Initialize NEST
nest.ResetKernel()
nest.resolution = TIMESTEP
nest.rng_seed = SEED
nest.print_time = False

print(f"\nNEST Configuration:")
print(f"  Resolution: {TIMESTEP} ms")
print(f"  Simulation time: {SIMULATION_TIME} ms")
print(f"  Seed: {SEED}")

# =============================================================================
# Load Parameters from Checkpoint
# =============================================================================

print("\n" + "=" * 80)
print("Loading Parameters from Checkpoint")
print("=" * 80)

try:
    import tensorflow as tf
    reader = tf.train.load_checkpoint('../training_code/ckpt_51978-153')

    # Load neuron parameters (voltage-normalized in checkpoint)
    E_L_norm = reader.get_tensor('model/layer_with_weights-1/cell/e_l/.ATTRIBUTES/VARIABLE_VALUE')
    V_th_norm = reader.get_tensor('model/layer_with_weights-1/cell/v_th/.ATTRIBUTES/VARIABLE_VALUE')
    V_reset_norm = reader.get_tensor('model/layer_with_weights-1/cell/v_reset/.ATTRIBUTES/VARIABLE_VALUE')

    # Load denormalization parameters
    voltage_offset = reader.get_tensor('model/layer_with_weights-1/cell/voltage_offset/.ATTRIBUTES/VARIABLE_VALUE')
    voltage_scale = reader.get_tensor('model/layer_with_weights-1/cell/voltage_scale/.ATTRIBUTES/VARIABLE_VALUE')

    # Denormalize voltages
    E_L = E_L_norm * voltage_scale + voltage_offset
    V_th = V_th_norm * voltage_scale + voltage_offset
    V_reset = V_reset_norm * voltage_scale + voltage_offset

    # Load other parameters
    C_m_pF = reader.get_tensor('model/layer_with_weights-1/cell/current_factor/.ATTRIBUTES/VARIABLE_VALUE') * 1000  # Convert to pF
    g_nS = reader.get_tensor('model/layer_with_weights-1/cell/param_g/.ATTRIBUTES/VARIABLE_VALUE')
    t_ref = reader.get_tensor('model/layer_with_weights-1/cell/t_ref/.ATTRIBUTES/VARIABLE_VALUE')

    # ASC parameters (these are the TRAINED values from checkpoint)
    asc_amps_raw = reader.get_tensor('model/layer_with_weights-1/cell/asc_amps/.ATTRIBUTES/VARIABLE_VALUE')  # voltage-normalized
    asc_amps_pA = asc_amps_raw * voltage_scale[:, np.newaxis]  # Denormalize to pA

    k_values = reader.get_tensor('model/layer_with_weights-1/cell/param_k/.ATTRIBUTES/VARIABLE_VALUE')  # 1/ms (negative)
    asc_decay = -k_values  # NEST uses positive decay rates

    # Synapse time constants
    syn_decay = reader.get_tensor('model/layer_with_weights-1/cell/syn_decay/.ATTRIBUTES/VARIABLE_VALUE')
    # Convert decay to tau: tau = -dt / ln(decay)
    tau_syn = -TIMESTEP / np.log(syn_decay)

    print(f"\nLoaded {len(E_L)} neuron types")
    print(f"\nSample parameters (type 0):")
    print(f"  C_m: {C_m_pF[0]:.2f} pF")
    print(f"  E_L: {E_L[0]:.2f} mV")
    print(f"  V_th: {V_th[0]:.2f} mV")
    print(f"  V_reset: {V_reset[0]:.2f} mV")
    print(f"  g: {g_nS[0]:.4f} nS")
    print(f"  t_ref: {t_ref[0]:.2f} ms")
    print(f"  asc_amps: [{asc_amps_pA[0,0]:.4f}, {asc_amps_pA[0,1]:.4f}] pA")
    print(f"  asc_decay: [{asc_decay[0,0]:.6f}, {asc_decay[0,1]:.6f}] 1/ms")
    print(f"  tau_syn: {tau_syn[0]} ms")

except ImportError:
    print("TensorFlow not available - using hardcoded test parameters")
    # Use first neuron type from your earlier printout
    C_m_pF = np.array([105.8])
    E_L = np.array([-77.28])
    V_th = np.array([-53.37])
    V_reset = np.array([-77.28])
    g_nS = np.array([7.18])
    t_ref = np.array([3.4])
    asc_amps_pA = np.array([[-12.60, -159.82]])  # Already in pA
    asc_decay = np.array([[0.003, 0.100]])  # Positive decay rates
    tau_syn = np.array([[5.5, 8.5, 2.8, 5.8]])

# =============================================================================
# Create Test Neurons
# =============================================================================

print("\n" + "=" * 80)
print("Creating NEST GLIF3 Neurons")
print("=" * 80)

# Create a small test network (not full 51978 neurons - too slow)
N_TEST = 100  # Test with 100 neurons

neurons = []
neuron_types = np.random.randint(0, len(C_m_pF), N_TEST)  # Random neuron types

for i in range(N_TEST):
    ntype = neuron_types[i]

    # NEST glif_psc parameters
    params = {
        # Membrane parameters
        'C_m': float(C_m_pF[ntype]),
        'E_L': float(E_L[ntype]),
        'V_th': float(V_th[ntype]),
        'V_reset': float(V_reset[ntype]),
        'g': float(g_nS[ntype]),
        't_ref': float(t_ref[ntype]),
        'V_m': float(E_L[ntype]),  # Initial voltage

        # GLIF3 features (Model 3: LIF with reset and ASC)
        'spike_dependent_threshold': False,  # GLIF3 doesn't have adaptive threshold
        'after_spike_currents': True,  # GLIF3 HAS after-spike currents
        'adapting_threshold': False,  # GLIF3 doesn't adapt threshold

        # After-spike current parameters
        'asc_init': [0.0, 0.0],  # Initial ASC values
        'asc_amps': [float(asc_amps_pA[ntype, 0]), float(asc_amps_pA[ntype, 1])],
        'asc_decay': [float(asc_decay[ntype, 0]), float(asc_decay[ntype, 1])],
        'asc_r': [0.0, 0.0],  # Fraction coefficients (not used in GLIF3)

        # Synaptic time constants (4 receptor types)
        'tau_syn': [float(tau_syn[ntype, 0]), float(tau_syn[ntype, 1]),
                    float(tau_syn[ntype, 2]), float(tau_syn[ntype, 3])],
    }

    try:
        neuron = nest.Create('glif_psc', 1, params=params)
        neurons.append(neuron)
    except nest.kernel.NESTErrors.BadProperty as e:
        print(f"\nError creating neuron {i}: {e}")
        print(f"Parameters: {params}")
        sys.exit(1)

    if (i + 1) % 20 == 0:
        print(f"  Created {i+1}/{N_TEST} neurons...")

print(f"\n✓ Created {N_TEST} GLIF3 neurons")

# =============================================================================
# Create Test Input
# =============================================================================

print("\n" + "=" * 80)
print("Creating Test Input")
print("=" * 80)

# Create Poisson spike sources to drive network
spike_sources = nest.Create('poisson_generator', N_TEST, params={'rate': 50.0})  # 50 Hz input

# Connect with random weights
for i, (source, target) in enumerate(zip(spike_sources, neurons)):
    weight = np.random.uniform(0.5, 2.0)  # Test weights in pA
    nest.Connect(source, target, syn_spec={'weight': weight, 'receptor_type': 1})  # Receptor 1 = first synapse

print(f"✓ Created {N_TEST} Poisson spike sources")
print(f"✓ Connected to neurons with random weights")

# =============================================================================
# Record Activity
# =============================================================================

print("\n" + "=" * 80)
print("Running Simulation")
print("=" * 80)

# Record spikes
spike_recorder = nest.Create('spike_recorder')
nest.Connect(neurons, spike_recorder)

# Record voltages (first 10 neurons only - too much data otherwise)
multimeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': 1.0})
nest.Connect(multimeter, neurons[:10])

# Run simulation
print(f"Simulating {SIMULATION_TIME} ms...")
nest.Simulate(SIMULATION_TIME)
print("✓ Simulation complete")

# =============================================================================
# Analyze Results
# =============================================================================

print("\n" + "=" * 80)
print("Results")
print("=" * 80)

# Get spike data
events = spike_recorder.get('events')
spike_times = events['times']
spike_senders = events['senders']

total_spikes = len(spike_times)
neurons_that_fired = len(np.unique(spike_senders))
mean_rate = total_spikes / (N_TEST * SIMULATION_TIME / 1000.0)  # Hz

print(f"\nSpike Statistics:")
print(f"  Total spikes: {total_spikes}")
print(f"  Neurons that fired: {neurons_that_fired}/{N_TEST}")
print(f"  Mean firing rate: {mean_rate:.2f} Hz")

if total_spikes > 0:
    print(f"\n  ✓ Network is ACTIVE")

    # Show spike distribution
    spike_counts = np.bincount(spike_senders.astype(int), minlength=max(spike_senders.astype(int))+1)
    print(f"\n  Spike count distribution:")
    print(f"    Min: {spike_counts.min()}")
    print(f"    Max: {spike_counts.max()}")
    print(f"    Mean: {spike_counts.mean():.2f}")
else:
    print(f"\n  ✗ Network is SILENT - check parameters!")

# Get voltage traces
vm_events = multimeter.get('events')
print(f"\nVoltage traces recorded for {len(neurons[:10])} neurons")

print("\n" + "=" * 80)
print("Validation Complete")
print("=" * 80)

print("\nNext steps:")
print("1. Compare with SpiNNaker simulation")
print("2. Check if activity levels match")
print("3. Verify neuron dynamics are similar")
print("4. If NEST works but SpiNNaker doesn't → SpiNNaker GLIF3 bug")
print("5. If both fail → parameter issue")
print("6. If both work → routing/connectivity issue")

print("\n" + "=" * 80)
