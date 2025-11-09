#!/usr/bin/env python3
"""
Systematic debugging script for Mouse V1 SpiNNaker simulation
Tests different weight scaling scenarios and provides detailed diagnostics
"""

import numpy as np
import h5py
import sys

def analyze_h5_file(h5_path):
    """Analyze the H5 file to understand weight statistics and units"""
    print(f"="*80)
    print(f"ANALYZING H5 FILE: {h5_path}")
    print(f"="*80)

    with h5py.File(h5_path, 'r') as f:
        # Network info
        print(f"\n📊 NETWORK STATISTICS:")
        n_neurons = len(f['neurons/node_type_ids'])
        print(f"  Neurons: {n_neurons}")
        print(f"  Recurrent synapses: {len(f['recurrent/weights'])}")
        print(f"  Input synapses: {len(f['input/weights'])}")

        # Weight statistics
        rec_weights = np.array(f['recurrent/weights'])
        inp_weights = np.array(f['input/weights'])

        print(f"\n⚖️  RECURRENT WEIGHTS:")
        print(f"  Mean: {np.mean(rec_weights):.6f}")
        print(f"  Std:  {np.std(rec_weights):.6f}")
        print(f"  Min:  {np.min(rec_weights):.6f}")
        print(f"  Max:  {np.max(rec_weights):.6f}")
        print(f"  Positive: {np.sum(rec_weights > 0)} ({100*np.sum(rec_weights > 0)/len(rec_weights):.1f}%)")
        print(f"  Negative: {np.sum(rec_weights < 0)} ({100*np.sum(rec_weights < 0)/len(rec_weights):.1f}%)")
        print(f"  Zero:     {np.sum(rec_weights == 0)}")

        print(f"\n⚖️  INPUT WEIGHTS:")
        print(f"  Mean: {np.mean(inp_weights):.6f}")
        print(f"  Std:  {np.std(inp_weights):.6f}")
        print(f"  Min:  {np.min(inp_weights):.6f}")
        print(f"  Max:  {np.max(inp_weights):.6f}")
        print(f"  Positive: {np.sum(inp_weights > 0)} ({100*np.sum(inp_weights > 0)/len(inp_weights):.1f}%)")
        print(f"  Negative: {np.sum(inp_weights < 0)} ({100*np.sum(inp_weights < 0)/len(inp_weights):.1f}%)")

        # GLIF3 parameters
        print(f"\n🔬 GLIF3 PARAMETERS (first neuron type):")
        cm = np.array(f['neurons/glif3_params/C_m'])[0]
        el = np.array(f['neurons/glif3_params/E_L'])[0]
        vth = np.array(f['neurons/glif3_params/V_th'])[0]
        vrst = np.array(f['neurons/glif3_params/V_reset'])[0]
        g = np.array(f['neurons/glif3_params/g'])[0]

        print(f"  C_m:      {cm:.4f} pF")
        print(f"  E_L:      {el:.4f} mV")
        print(f"  V_th:     {vth:.4f} mV")
        print(f"  V_reset:  {vrst:.4f} mV")
        print(f"  g:        {g:.4f} nS")

        voltage_scale = vth - el
        print(f"  Voltage scale (V_th - E_L): {voltage_scale:.4f} mV")

        # Estimate what weight values should be after conversion
        print(f"\n🧮 WEIGHT CONVERSION ESTIMATES:")
        print(f"  If multiplied by voltage_scale/1000:")
        print(f"    Mean recurrent: {np.mean(rec_weights) * voltage_scale / 1000:.6f} nA")
        print(f"    Mean input:     {np.mean(inp_weights) * voltage_scale / 1000:.6f} nA")
        print(f"  If divided by voltage_scale*1000:")
        print(f"    Mean recurrent: {np.mean(rec_weights) / (voltage_scale * 1000):.9f} nA")
        print(f"    Mean input:     {np.mean(inp_weights) / (voltage_scale * 1000):.9f} nA")
        print(f"  If only divided by 1000:")
        print(f"    Mean recurrent: {np.mean(rec_weights) / 1000:.6f} nA")
        print(f"    Mean input:     {np.mean(inp_weights) / 1000:.6f} nA")

        # Check receptor types
        print(f"\n🔌 RECEPTOR TYPE DISTRIBUTION:")
        rec_rtypes = np.array(f['recurrent/receptor_types'])
        inp_rtypes = np.array(f['input/receptor_types'])

        for rtype in range(4):
            rec_count = np.sum(rec_rtypes == rtype)
            inp_count = np.sum(inp_rtypes == rtype)
            rec_mean_w = np.mean(rec_weights[rec_rtypes == rtype]) if rec_count > 0 else 0
            inp_mean_w = np.mean(inp_weights[inp_rtypes == rtype]) if inp_count > 0 else 0
            print(f"  Receptor {rtype}:")
            print(f"    Recurrent: {rec_count:8d} synapses, mean weight: {rec_mean_w:+.6f}")
            print(f"    Input:     {inp_count:8d} synapses, mean weight: {inp_mean_w:+.6f}")

def compare_weight_scaling_scenarios(h5_path):
    """Compare different weight scaling approaches"""
    print(f"\n" + "="*80)
    print(f"WEIGHT SCALING SCENARIOS")
    print(f"="*80)

    with h5py.File(h5_path, 'r') as f:
        # Get a sample of weights and voltage scales
        inp_weights = np.array(f['input/weights'][:1000])  # Sample
        inp_targets = np.array(f['input/targets'][:1000])
        node_type_ids = np.array(f['neurons/node_type_ids'])

        # Compute voltage scales for each neuron type
        vth = np.array(f['neurons/glif3_params/V_th'])
        el = np.array(f['neurons/glif3_params/E_L'])
        voltage_scales = vth - el

        # Get voltage scale for sample synapses (based on target neuron type)
        sample_voltage_scales = voltage_scales[node_type_ids[inp_targets]]

        print(f"\n📋 Testing on {len(inp_weights)} sample input synapses")
        print(f"   Original mean weight: {np.mean(inp_weights):.6f}")

        scenarios = {
            "A: No voltage_scale (÷1000 only)": inp_weights / 1000.0,
            "B: Current (* vsc/1000)": inp_weights * sample_voltage_scales / 1000.0,
            "C: Inverse TF (÷(vsc*1000))": inp_weights / (sample_voltage_scales * 1000.0),
            "D: Match TF units (* vsc)": inp_weights * sample_voltage_scales,
            "E: No scaling": inp_weights,
        }

        print(f"\n🎯 SCALING SCENARIOS:")
        for name, scaled_weights in scenarios.items():
            print(f"\n  {name}")
            print(f"    Mean:  {np.mean(scaled_weights):.9f}")
            print(f"    Std:   {np.std(scaled_weights):.9f}")
            print(f"    Range: [{np.min(scaled_weights):.9f}, {np.max(scaled_weights):.9f}]")

            # Estimate if this would cause neurons to spike
            # Rough estimate: if mean synaptic current * number of inputs > threshold
            mean_current = np.mean(np.abs(scaled_weights))
            print(f"    Mean |current|: {mean_current:.9f} nA")
            print(f"    Would need ~{20.0/mean_current:.0f} simultaneous inputs to reach ~20mV depolarization")

def check_tensorflow_model_loading(checkpoint_dir):
    """Check if TensorFlow model was actually loaded in c2.py"""
    import os

    print(f"\n" + "="*80)
    print(f"TENSORFLOW CHECKPOINT CHECK")
    print(f"="*80)

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if 'ckpt' in f]
    print(f"\n📂 Checkpoint directory: {checkpoint_dir}")
    print(f"   Files found: {len(checkpoint_files)}")
    for f in checkpoint_files[:10]:  # Show first 10
        print(f"     - {f}")

    # Check for latest checkpoint
    latest = None
    index_files = [f for f in checkpoint_files if f.endswith('.index')]
    if index_files:
        latest = max(index_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
        print(f"\n✓ Latest checkpoint: {latest}")
    else:
        print(f"\n❌ No .index files found - checkpoint may not have loaded correctly")

def generate_test_recommendations():
    """Generate specific testing recommendations"""
    print(f"\n" + "="*80)
    print(f"RECOMMENDED TESTS")
    print(f"="*80)

    tests = [
        {
            "name": "Test 1: Minimal Network",
            "description": "Create minimal test with 1 LGN → 1 V1 neuron",
            "code": """
# Create single LGN neuron spiking at 100Hz
lgn = sim.Population(1, sim.SpikeSourceArray,
                     {'spike_times': [10, 20, 30, 40, 50]})

# Create single V1 neuron with known parameters
v1 = sim.Population(1, GLIF3Curr, {...})

# Create connection with known weight
sim.Projection(lgn, v1, sim.OneToOneConnector(),
               synapse_type=..., receptor_type='synapse_0',
               weight=0.1)  # Test different values

# Verify V1 spikes in response
"""
        },
        {
            "name": "Test 2: Weight Scaling Sweep",
            "description": "Run simulation with different weight multipliers",
            "code": """
for scale_factor in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    syn[:, S.WHT] = original_weights * scale_factor
    # Run simulation
    # Count output spikes
    # Record which scale factor produces reasonable activity
"""
        },
        {
            "name": "Test 3: Voltage Monitoring",
            "description": "Monitor V1 voltages to see if they're changing",
            "code": """
# For each V1 population, record voltage
V1[key].record(['v', 'spikes'])

# After simulation
v_data = V1[key].get_data('v')
print(f\"Voltage range: {v_data.min()} to {v_data.max()}")
print(f\"Threshold: {network['glif3'][pid, G.THR]}")
print(f\"Max below threshold: {v_data.max() - threshold}")
"""
        },
        {
            "name": "Test 4: TensorFlow Comparison",
            "description": "Run same input through TensorFlow model",
            "code": """
# Load TensorFlow model
# Feed same spike trains
# Compare:
#   - V1 spike counts
#   - Voltage trajectories
#   - Response timing
#   - Which neurons spike
"""
        },
    ]

    for i, test in enumerate(tests, 1):
        print(f"\n{test['name']}")
        print(f"  {test['description']}")
        print(f"  Code snippet:")
        for line in test['code'].strip().split('\n'):
            print(f"  {line}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python debug_weight_scaling.py <path_to_network.h5> [checkpoint_dir]")
        print("\nExample: python debug_weight_scaling.py ckpt_51978-77.h5 checkpoints")
        sys.exit(1)

    h5_path = sys.argv[1]
    checkpoint_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # Run analyses
    analyze_h5_file(h5_path)
    compare_weight_scaling_scenarios(h5_path)

    if checkpoint_dir:
        check_tensorflow_model_loading(checkpoint_dir)

    generate_test_recommendations()

    print(f"\n" + "="*80)
    print(f"ANALYSIS COMPLETE")
    print(f"="*80)
    print(f"\n💡 Key recommendations:")
    print(f"  1. Test weight scaling scenarios A, B, C with minimal network")
    print(f"  2. Add voltage monitoring to verify neurons are responding to input")
    print(f"  3. Verify TensorFlow checkpoint was loaded correctly")
    print(f"  4. Compare SpiNNaker outputs with TensorFlow inference")
    print(f"\n📄 See ANALYSIS_AND_FINDINGS.md for detailed analysis")
