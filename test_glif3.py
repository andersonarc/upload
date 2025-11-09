#!/usr/bin/env python3
"""
Simple test script for GLIF3 neuron model on SpiNNaker.
Tests basic compilation and execution without DMA errors.
"""

import pyNN.spiNNaker as sim

# Setup
sim.setup(timestep=1.0)

# Create a small population of GLIF3 neurons
print("Creating GLIF3 population...")
glif3_pop = sim.Population(
    10,  # Small population for testing
    sim.extra_models.GLIF3Curr(
        c_m=1.0,
        e_l=-70.0,
        v_reset=-70.0,
        v_thresh=-50.0,
        asc_amp_0=0.0,
        asc_amp_1=0.0,
        g=0.05,
        k0=0.2,
        k1=0.05,
        t_ref=2.0,
        i_offset=0.1,  # Small DC current to elicit spikes
        tau_syn_0=5.0,
        tau_syn_1=5.0,
        tau_syn_2=5.0,
        tau_syn_3=5.0
    ),
    label="glif3_test"
)

# Create a simple spike source for input
print("Creating spike source...")
spike_source = sim.Population(
    5,
    sim.SpikeSourceArray(spike_times=[[10.0, 20.0, 30.0]]),
    label="spike_source"
)

# Connect spike source to GLIF3 neurons via synapse type 0
print("Creating connections...")
sim.Projection(
    spike_source,
    glif3_pop,
    sim.AllToAllConnector(),
    synapse_type=sim.StaticSynapse(weight=0.5, delay=1.0),
    receptor_type="excitatory"
)

# Record spikes
glif3_pop.record("spikes")

# Run simulation
print("Running simulation for 100ms...")
sim.run(100)

# Get results
spikes = glif3_pop.get_data("spikes")
print(f"Recorded {len(spikes.segments[0].spiketrains)} spike trains")
for i, spiketrain in enumerate(spikes.segments[0].spiketrains[:3]):  # Show first 3
    print(f"  Neuron {i}: {len(spiketrain)} spikes at times {list(spiketrain)[:5]}...")

# Cleanup
sim.end()

print("\n✓ Test completed successfully - no DMA errors!")
