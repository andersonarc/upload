"""
Proposed fixes for Mouse V1 SpiNNaker simulation

This file contains corrected versions of the critical functions from the Jupyter notebook
with detailed explanations of the changes.
"""

import numpy as np

# ============================================================================
# FIX 1: WEIGHT SCALING
# ============================================================================

def create_V1_FIXED(glif3s, ps2g, v1_synapses, network, test_mode='scenario_B'):
    """
    Fixed version of create_V1 with correct weight scaling

    Args:
        test_mode: Which weight scaling to use
            'scenario_A': No voltage_scale (÷1000 only) - TEST THIS FIRST
            'scenario_B': Current approach (* vsc/1000) - Original (likely wrong)
            'scenario_C': Inverse of TensorFlow (÷(vsc*1000))
            'scenario_D': Match TensorFlow exactly (÷vsc, no /1000)
            'scenario_E': TensorFlow with unit conversion (÷vsc then *1000 for pA→nA)
    """
    import pyNN.spiNNaker as sim
    from python_models8.neuron.builds.glif3_curr import GLIF3Curr

    # Import indices from notebook
    class G:
        CM=0; EL=1; RST=2; THR=3; AA0=4; AA1=5; G=6; K0=7; K1=8; RFR=9
        TA0=10; TA1=11; TA2=12; TA3=13; VSC=14; CON=15
    class S:
        SRC=0; TGT=1; WHT=2; RTY=3; ID=4; DLY=5

    def G2D(g):
        return {
            'c_m': g[G.CM], 'e_l': g[G.EL], 'v_reset': g[G.RST],
            'v_thresh': g[G.THR], 'asc_amp_0': g[G.AA0], 'asc_amp_1': g[G.AA1],
            'g': g[G.G], 'k0': g[G.K0], 'k1': g[G.K1], 't_ref': g[G.RFR],
            'i_offset': 0.0, 'v': g[G.EL],
            'tau_syn_0': g[G.TA0], 'tau_syn_1': g[G.TA1],
            'tau_syn_2': g[G.TA2], 'tau_syn_3': g[G.TA3],
        }

    # Create V1 populations
    V1 = {}
    for key, gids in ps2g.items():
        pid, subpid = key
        size = len(gids)
        V1_N = sim.Population(
            len(gids), GLIF3Curr, cellparams=G2D(glif3s[pid]),
            neurons_per_core=int(np.min([200, (1 / size) * 1e6])),
            label=f'V1_{pid}_{subpid}'
        )
        V1[key] = V1_N

    V1_n_proj = 0
    weight_stats = {'min': [], 'max': [], 'mean': [], 'std': []}

    # Process synapse populations with CORRECTED weight scaling
    for synkey, syn in v1_synapses.items():
        src_key, tgt_key = synkey

        # Get voltage scale for target neuron type
        vsc = network['glif3'][int(tgt_key[0]), G.VSC]

        # APPLY WEIGHT SCALING BASED ON TEST MODE
        original_weights = syn[:, S.WHT].copy()

        if test_mode == 'scenario_A':
            # No voltage_scale multiplication - just unit conversion
            # Hypothesis: H5 weights are already in correct units, just need pA→nA
            syn[:, S.WHT] = original_weights / 1000.0

        elif test_mode == 'scenario_B':
            # Original (current) approach
            syn[:, S.WHT] = original_weights * vsc / 1000.0

        elif test_mode == 'scenario_C':
            # Inverse of what TensorFlow did
            # TF divides by vsc, so we should too (not multiply)
            syn[:, S.WHT] = original_weights / (vsc * 1000.0)

        elif test_mode == 'scenario_D':
            # Match TensorFlow exactly: weights are normalized, undo that
            # In TF: w_normalized = w_original / vsc
            # To recover: w_original = w_normalized * vsc
            # But then need unit conversion
            syn[:, S.WHT] = original_weights / vsc  # Undo normalization

        elif test_mode == 'scenario_E':
            # TensorFlow stored as pA, normalized by vsc
            # Recover original pA values, then convert to nA
            syn[:, S.WHT] = (original_weights * vsc) / 1000.0

        else:
            raise ValueError(f"Unknown test_mode: {test_mode}")

        # Collect statistics
        weight_stats['min'].append(np.min(syn[:, S.WHT]))
        weight_stats['max'].append(np.max(syn[:, S.WHT]))
        weight_stats['mean'].append(np.mean(syn[:, S.WHT]))
        weight_stats['std'].append(np.std(syn[:, S.WHT]))

        # Create projection with correct receptor type
        receptor_type = f'synapse_{int(syn[0, S.RTY])}'
        sim.Projection(
            V1[src_key], V1[tgt_key],
            sim.FromListConnector(syn[:, [S.SRC, S.TGT, S.WHT, S.DLY]]),
            receptor_type=receptor_type
        )
        V1_n_proj += 1

    # Print weight statistics for this mode
    print(f"\n{'='*80}")
    print(f"Weight scaling mode: {test_mode}")
    print(f"{'='*80}")
    print(f"Recurrent weight statistics:")
    print(f"  Min:  {np.min(weight_stats['min']):.9f} nA")
    print(f"  Max:  {np.max(weight_stats['max']):.9f} nA")
    print(f"  Mean: {np.mean(weight_stats['mean']):.9f} nA")
    print(f"  Std:  {np.mean(weight_stats['std']):.9f} nA")

    # Estimate expected PSC
    avg_weight = np.mean(weight_stats['mean'])
    print(f"\nEstimated post-synaptic current per spike: {avg_weight:.9f} nA")
    print(f"With ~100 inputs, total current: {avg_weight * 100:.6f} nA")

    # Set up recording
    print(f"\nV1 populations created: {len(V1)}")
    print(f"V1 projections: {V1_n_proj}")
    for pop in V1.values():
        if pop is not None:
            pop.record(['spikes', 'v'])

    return V1, len(V1), V1_n_proj


def create_LGN_FIXED(V1, spike_times, tm2l, lgn_synapses, network, test_mode='scenario_B'):
    """
    Fixed version of create_LGN with correct weight scaling
    Same modes as create_V1_FIXED
    """
    import pyNN.spiNNaker as sim

    class G:
        CM=0; EL=1; RST=2; THR=3; AA0=4; AA1=5; G=6; K0=7; K1=8; RFR=9
        TA0=10; TA1=11; TA2=12; TA3=13; VSC=14; CON=15
    class S:
        SRC=0; TGT=1; WHT=2; RTY=3; ID=4

    # Create LGN populations
    LGN = []
    for i, lgns in enumerate(tm2l.values()):
        LGN_x = sim.Population(
            len(lgns), sim.SpikeSourceArray,
            cellparams={'spike_times': [spike_times[lgn] if len(spike_times[lgn]) > 0 else [600] for lgn in lgns]},
            label=f'LGN_{i}'
        )
        LGN.append(LGN_x)

    LGN_n_proj = 0
    weight_stats = {'min': [], 'max': [], 'mean': [], 'std': []}

    # Process synapse populations with CORRECTED weight scaling
    for synkey, syn in lgn_synapses.items():
        lgn_pid, tgt_key = synkey

        # Get voltage scale for target neuron type
        vsc = network['glif3'][int(tgt_key[0]), G.VSC]

        # APPLY WEIGHT SCALING BASED ON TEST MODE (same as V1)
        original_weights = syn[:, S.WHT].copy()

        if test_mode == 'scenario_A':
            syn[:, S.WHT] = original_weights / 1000.0
        elif test_mode == 'scenario_B':
            syn[:, S.WHT] = original_weights * vsc / 1000.0
        elif test_mode == 'scenario_C':
            syn[:, S.WHT] = original_weights / (vsc * 1000.0)
        elif test_mode == 'scenario_D':
            syn[:, S.WHT] = original_weights / vsc
        elif test_mode == 'scenario_E':
            syn[:, S.WHT] = (original_weights * vsc) / 1000.0

        # Collect statistics
        weight_stats['min'].append(np.min(syn[:, S.WHT]))
        weight_stats['max'].append(np.max(syn[:, S.WHT]))
        weight_stats['mean'].append(np.mean(syn[:, S.WHT]))
        weight_stats['std'].append(np.std(syn[:, S.WHT]))

        # Create projection
        receptor_type = f'synapse_{int(syn[0, S.RTY])}'
        sim.Projection(
            LGN[lgn_pid], V1[tgt_key],
            sim.FromListConnector(syn[:, [S.SRC, S.TGT, S.WHT]], column_names=['weight']),
            receptor_type=receptor_type
        )
        LGN_n_proj += 1

    # Print weight statistics
    print(f"\nInput weight statistics:")
    print(f"  Min:  {np.min(weight_stats['min']):.9f} nA")
    print(f"  Max:  {np.max(weight_stats['max']):.9f} nA")
    print(f"  Mean: {np.mean(weight_stats['mean']):.9f} nA")
    print(f"  Std:  {np.mean(weight_stats['std']):.9f} nA")

    print(f"\nLGN populations created: {len(LGN)}")
    print(f"LGN projections: {LGN_n_proj}")
    for pop in LGN:
        if pop is not None:
            pop.record(['spikes'])

    return LGN, len(LGN), LGN_n_proj


# ============================================================================
# FIX 2: DIAGNOSTIC FUNCTIONS
# ============================================================================

def analyze_simulation_results(V1, LGN, network, output_nnpols, ps2g, dataset):
    """
    Comprehensive analysis of simulation results
    """
    class S:
        SRC=0; TGT=1; WHT=2; RTY=3; ID=4; DLY=5

    print(f"\n{'='*80}")
    print(f"SIMULATION RESULTS ANALYSIS")
    print(f"={'*80}")

    # 1. LGN Activity
    print(f"\n1. LGN INPUT ACTIVITY:")
    total_lgn_spikes = 0
    for i, lgn in enumerate(LGN[:10]):  # Show first 10
        spikes = lgn.get_data('spikes').segments[0].spiketrains
        n_spikes = sum(len(st) for st in spikes)
        total_lgn_spikes += n_spikes
        if i < 10:
            print(f"  LGN_{i}: {lgn.size} neurons, {n_spikes} total spikes")
    print(f"  Total LGN spikes (first 10 pops): {total_lgn_spikes}")

    # 2. V1 Activity
    print(f"\n2. V1 NETWORK ACTIVITY:")
    total_v1_spikes = 0
    v1_voltage_stats = []

    ps2keys = list(ps2g.keys())
    for i, key in enumerate(ps2keys[:10]):  # First 10 populations
        if V1[key] is not None:
            # Spike count
            spikes = V1[key].get_data('spikes').segments[0].spiketrains
            n_spikes = sum(len(st) for st in spikes)
            total_v1_spikes += n_spikes

            # Voltage stats
            v = V1[key].get_data('v').segments[0].analogsignals[0]
            v_min, v_max = float(v.min()), float(v.max())
            v_voltage_stats.append((v_min, v_max))

            pid = key[0]
            threshold = network['glif3'][pid, 3]  # THR index

            if i < 5:  # Show details for first 5
                print(f"  V1_{key}: {V1[key].size} neurons, {n_spikes} spikes")
                print(f"    Voltage: [{v_min:.2f}, {v_max:.2f}] mV, Threshold: {threshold:.2f} mV")
                print(f"    Gap to threshold: {threshold - v_max:.2f} mV")

    print(f"  Total V1 spikes (first 10 pops): {total_v1_spikes}")

    # 3. Readout Activity
    print(f"\n3. READOUT NEURON ACTIVITY:")
    gid2train = {}
    for i, item in enumerate(output_nnpols.items()):
        key, lids = item
        gids = ps2g[key]
        # Get spike trains
        try:
            readout = sim.PopulationView(V1[key], lids)
            spiketrains = readout.get_data('spikes').segments[0].spiketrains
            for lid, spiketrain in zip(lids, spiketrains):
                gid = gids[lid]
                gid2train[gid] = spiketrain
        except:
            pass

    # Count votes
    votes = np.zeros(10)
    response_start = 50
    response_end = 200

    for class_idx in range(10):
        start = class_idx * 30
        end = (class_idx + 1) * 30
        keys = network['output'][start:end]
        for key in keys:
            if key in gid2train:
                spiketrain = gid2train[key]
                mask = (spiketrain > response_start) & (spiketrain < response_end)
                votes[class_idx] += mask.sum()

    print(f"  Label: {dataset['labels'][0]}")
    print(f"  Response window: [{response_start}, {response_end}] ms")
    print(f"  Votes by class: {votes}")
    print(f"  Predicted class: {np.argmax(votes)}")
    print(f"  Total readout spikes: {np.sum(votes)}")

    # 4. Diagnosis
    print(f"\n4. DIAGNOSIS:")
    if total_lgn_spikes == 0:
        print(f"  ❌ CRITICAL: No LGN spikes! Check spike generation.")
    else:
        print(f"  ✓ LGN is active ({total_lgn_spikes} spikes)")

    if total_v1_spikes == 0:
        print(f"  ❌ CRITICAL: No V1 spikes!")
        print(f"  Possible causes:")
        print(f"    - Weights too small (check weight scaling)")
        print(f"    - Voltages not reaching threshold (check voltage stats)")
        print(f"    - Synaptic connections not working")

        if v1_voltage_stats:
            max_v_reached = max(v_max for _, v_max in v1_voltage_stats)
            print(f"  Max voltage reached: {max_v_reached:.2f} mV")
            if max_v_reached < -40:
                print(f"  → Voltages barely moving. LIKELY: Weights too small!")
    else:
        print(f"  ✓ V1 is active ({total_v1_spikes} spikes)")

        if np.sum(votes) == 0:
            print(f"  ❌ No readout spikes in response window")
            print(f"  Try different response window or check readout connectivity")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Example of how to use the fixed functions in the notebook
    """
    usage = '''
# In your Jupyter notebook, replace the create_V1 and create_LGN calls with:

# TEST SCENARIO A (most likely fix)
setup()
V1, V1_n_pop, V1_n_proj = create_V1_FIXED(
    network['glif3'], ps2g, v1_synpols, network, test_mode='scenario_A'
)
LGN, LGN_n_pop, LGN_n_proj = create_LGN_FIXED(
    V1, spike_times, tm2l, lgn_synpols, network, test_mode='scenario_A'
)
readouts = create_readouts(output_nnpols, V1)

# Run simulation
sim.run(1000)

# Analyze results
analyze_simulation_results(V1, LGN, network, output_nnpols, ps2g, dataset)

# If scenario A doesn't work, try B, C, D, E in order
'''
    print(usage)

if __name__ == "__main__":
    example_usage()
