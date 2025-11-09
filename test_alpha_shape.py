#!/usr/bin/env python3
"""
Test to verify alpha synapse shape vs exponential.
Alpha synapses should show delayed, rounded peak.
"""

import pyNN.spiNNaker as sim
import matplotlib.pyplot as plt
import numpy as np

# Setup
sim.setup(timestep=1.0)

# Create single GLIF3 neuron
neuron = sim.Population(
    1,
    sim.extra_models.GLIF3Curr(
        c_m=1.0,
        e_l=-70.0,
        v_reset=-70.0,
        v_thresh=-50.0,  # High threshold so it doesn't spike
        asc_amp_0=0.0,
        asc_amp_1=0.0,
        g=0.01,  # Low conductance for longer time constant
        k0=0.2,
        k1=0.05,
        t_ref=2.0,
        i_offset=0.0,
        tau_syn_0=10.0,  # Test synapse 0
        tau_syn_1=10.0,
        tau_syn_2=10.0,
        tau_syn_3=10.0
    ),
    label="test_neuron"
)

# Single spike at t=10ms
spike_source = sim.Population(
    1,
    sim.SpikeSourceArray(spike_times=[[10.0]]),
    label="spike"
)

# Connect with weight 1.0
sim.Projection(
    spike_source,
    neuron,
    sim.OneToOneConnector(),
    synapse_type=sim.StaticSynapse(weight=5.0, delay=1.0),
    receptor_type="excitatory"
)

# Record membrane voltage
neuron.record("v")

# Run
sim.run(100)

# Get voltage trace
data = neuron.get_data("v")
v_trace = data.segments[0].filter(name='v')[0]
times = v_trace.times.magnitude
voltages = v_trace.magnitude.flatten()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(times, voltages)
plt.axvline(10, color='r', linestyle='--', label='Input spike')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane voltage (mV)')
plt.title('Alpha synapse response (should show delayed, rounded peak)')
plt.legend()
plt.grid(True)
plt.savefig('alpha_synapse_test.png', dpi=150, bbox_inches='tight')
print("Saved plot to alpha_synapse_test.png")

# Find peak time and value
spike_idx = np.argmin(np.abs(times - 10))
post_spike = voltages[spike_idx:]
post_times = times[spike_idx:]
peak_idx = np.argmax(post_spike)
peak_time = post_times[peak_idx]
peak_voltage = post_spike[peak_idx]

print(f"\nAlpha synapse analysis:")
print(f"  Input spike at: 10.0 ms")
print(f"  Peak voltage: {peak_voltage:.2f} mV")
print(f"  Peak time: {peak_time:.1f} ms")
print(f"  Delay to peak: {peak_time - 10:.1f} ms")
print(f"\nExpected for alpha with tau=10ms: peak ~10ms after input")
print(f"Expected for exponential: peak immediately at input")

sim.end()
