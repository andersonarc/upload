#!/usr/bin/env python3
"""
Minimal test to verify GLIF3 and weight scaling work correctly

This creates a tiny network: 1 LGN neuron -> 1 V1 neuron
Tests different weight values to find the right scaling

Run this BEFORE trying the full network simulation.
"""

import pyNN.spiNNaker as sim
import numpy as np
from python_models8.neuron.builds.glif3_curr import GLIF3Curr

def test_single_connection(weight_nA, label="Test"):
    """
    Test a single LGN -> V1 connection with a specific weight

    Args:
        weight_nA: Weight in nanoamps
        label: Description for this test

    Returns:
        Number of spikes produced by V1 neuron
    """
    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"Weight: {weight_nA:.6f} nA")
    print(f"{'='*60}")

    # Setup
    sim.setup(timestep=1.0)

    # Create single LGN neuron that spikes regularly
    spike_times = [50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70]  # 10 spikes in 20ms
    lgn = sim.Population(
        1,
        sim.SpikeSourceArray,
        {'spike_times': spike_times},
        label='LGN'
    )

    # Create single V1 neuron with typical GLIF3 parameters
    # These are approximate values from the first neuron type in the dataset
    v1_params = {
        'c_m': 0.1058,           # nF (converted from pF)
        'e_l': -77.28,           # mV
        'v_reset': -77.28,       # mV
        'v_thresh': -53.37,      # mV
        'asc_amp_0': -0.0126,    # nA (converted from pA)
        'asc_amp_1': -0.1598,    # nA
        'g': 0.00718,            # uS (converted from nS)
        'k0': 0.003,             # 1/ms
        'k1': 0.0001,            # 1/ms
        't_ref': 3.4,            # ms
        'i_offset': 0.0,         # nA
        'v': -77.28,             # Initial voltage = E_L
        'tau_syn_0': 5.5,        # ms
        'tau_syn_1': 8.5,        # ms
        'tau_syn_2': 2.8,        # ms
        'tau_syn_3': 5.8,        # ms
    }

    v1 = sim.Population(
        1,
        GLIF3Curr,
        v1_params,
        label='V1'
    )

    # Connect with specified weight
    # Use receptor 0 (excitatory synapse with tau=5.5ms)
    sim.Projection(
        lgn, v1,
        sim.OneToOneConnector(),
        receptor_type='synapse_0',
        synapse_type=sim.StaticSynapse(weight=weight_nA)
    )

    # Record
    lgn.record(['spikes'])
    v1.record(['spikes', 'v'])

    # Run for 200ms
    print("Running simulation...")
    sim.run(200)

    # Get results
    lgn_spikes = lgn.get_data('spikes').segments[0].spiketrains[0]
    v1_spikes = v1.get_data('spikes').segments[0].spiketrains[0]
    v1_voltage = v1.get_data('v').segments[0].analogsignals[0]

    # Analysis
    print(f"\nResults:")
    print(f"  LGN spikes: {len(lgn_spikes)} at times {list(lgn_spikes)}")
    print(f"  V1 spikes: {len(v1_spikes)}" + (f" at times {list(v1_spikes)}" if len(v1_spikes) > 0 else ""))
    print(f"  V1 voltage range: [{float(v1_voltage.min()):.2f}, {float(v1_voltage.max()):.2f}] mV")
    print(f"  V1 threshold: {v1_params['v_thresh']:.2f} mV")

    if len(v1_spikes) > 0:
        print(f"  ✓ SUCCESS: V1 neuron spiked!")
        print(f"  First spike at: {v1_spikes[0]:.1f} ms")
        print(f"  Latency: {v1_spikes[0] - lgn_spikes[0]:.1f} ms")
    else:
        gap = v1_params['v_thresh'] - float(v1_voltage.max())
        print(f"  ✗ FAIL: No V1 spikes")
        print(f"  Gap to threshold: {gap:.2f} mV")

        if gap > 10:
            print(f"  → Weight too small (voltage barely moved)")
        elif gap > 0.1:
            print(f"  → Weight too small (close but not enough)")
        else:
            print(f"  → Something else wrong (voltage at threshold but no spike?)")

    # Cleanup
    sim.end()

    return len(v1_spikes)


def run_weight_sweep():
    """
    Test a range of weights to find what works
    """
    print(f"\n{'#'*60}")
    print(f"# MINIMAL GLIF3 WEIGHT SCALING TEST")
    print(f"# Testing different weight values to find correct scaling")
    print(f"{'#'*60}")

    # Test weights across several orders of magnitude
    test_weights = [
        (0.001, "Very small (scenario C)"),
        (0.01, "Small"),
        (0.1, "Medium-small"),
        (1.0, "Medium"),
        (10.0, "Medium-large"),
        (20.0, "Large (scenario A)"),
        (50.0, "Very large"),
        (100.0, "Huge"),
    ]

    results = []
    for weight, description in test_weights:
        n_spikes = test_single_connection(weight, description)
        results.append((weight, n_spikes))

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Weight (nA)':<15} {'Spikes':<10} Status")
    print(f"{'-'*60}")

    for weight, n_spikes in results:
        status = "✓ Works" if n_spikes > 0 else "✗ No response"
        print(f"{weight:<15.6f} {n_spikes:<10} {status}")

    # Recommendation
    working_weights = [w for w, s in results if s > 0]
    if working_weights:
        min_working = min(working_weights)
        max_working = max(working_weights)
        print(f"\n💡 RECOMMENDATION:")
        print(f"  Working weight range: [{min_working:.6f}, {max_working:.6f}] nA")
        print(f"  For full network, weights should be in this range.")

        # Map back to scenarios
        if 0.001 <= min_working < 0.01:
            print(f"  → This suggests SCENARIO C is correct (÷(vsc*1000))")
        elif 0.01 <= min_working < 1:
            print(f"  → This suggests weights need moderate scaling")
        elif 1 <= min_working < 10:
            print(f"  → This suggests SCENARIO B might be correct (*vsc/1000)")
        elif 10 <= min_working:
            print(f"  → This suggests SCENARIO A is correct (÷1000 only)")
    else:
        print(f"\n❌ PROBLEM:")
        print(f"  No weights produced spikes!")
        print(f"  Possible issues:")
        print(f"    - GLIF3 model implementation problem")
        print(f"    - Synapse type not working")
        print(f"    - Parameters incorrect")
        print(f"  Check GLIF3 model and synapse configuration.")


def test_known_good_weight():
    """
    Test with a weight that SHOULD definitely work
    Based on biological EPSCs of ~50-100 pA with 10-20 coincident spikes needed to fire
    """
    print(f"\n{'#'*60}")
    print(f"# BIOLOGICAL PLAUSIBILITY TEST")
    print(f"{'#'*60}")
    print(f"\nBiological reasoning:")
    print(f"  - Typical EPSC: 50-100 pA = 0.05-0.1 nA")
    print(f"  - Typical neuron needs 10-20 mV depolarization to fire")
    print(f"  - With C_m ~ 0.1 nF, Q = C*V, so need ~1-2 pC charge")
    print(f"  - With tau_syn ~ 5ms, integrated current ~ I*tau")
    print(f"  - So need I*tau*N ~ 2pC, where N is number of inputs")
    print(f"  - For 10 inputs: I ~ 2pC / (5ms * 10) = 0.04 nA")
    print(f"\n  → Testing with weight = 0.05 nA (biologically plausible)")

    test_single_connection(0.05, "Biologically plausible weight")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick test with a few weights
        test_single_connection(0.001, "Scenario C")
        test_single_connection(0.02, "Scenario B")
        test_single_connection(20.0, "Scenario A")
        test_known_good_weight()
    else:
        # Full weight sweep
        run_weight_sweep()
        test_known_good_weight()

    print(f"\n{'#'*60}")
    print(f"# TEST COMPLETE")
    print(f"# Use results to determine correct weight scaling for full network")
    print(f"{'#'*60}\n")
