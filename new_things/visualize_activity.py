#!/usr/bin/env python
"""Visualize network activity from NEST GLIF3 simulation with advanced analysis"""

import os
import sys
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from scipy.special import softmax
from multiprocessing import Pool, cpu_count

# ============================================================================
# Configuration
# ============================================================================

CHECKPOINT_FILE = 'ckpt_51978-153_NEW.h5'
MNIST_FILE = 'mnist24.h5'
SIMULATION_TIME = 1000.0
TIME_WINDOW = (50, 200)
ANIMATION_BIN_SIZE = 5  # ms
ANIMATION_FPS = 10

def get_untrained_checkpoint(checkpoint_path):
    """Get corresponding untrained checkpoint path"""
    base, ext = os.path.splitext(checkpoint_path)
    return f"{base}.untrained{ext}"

# Styling
#plt.style.use('dark_background')
CMAP = 'hot'
SPIKE_COLOR = 'black'
OUTPUT_NEURON_COLOR = 'lime'


# ============================================================================
# Data Loading
# ============================================================================

def load_checkpoint(checkpoint_path, is_trained=True):
    """Load network structure and parameters from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        is_trained: Whether this is a trained network (affects weight processing)
    """
    with h5py.File(checkpoint_path, 'r') as f:
        neurons = np.array(f['neurons/node_type_ids'])
        output_neurons = np.array(f['readout/neuron_ids'], dtype=int)
        
        glif_params = {}
        for key in f['neurons/glif3_params'].keys():
            glif_params[key] = np.array(f['neurons/glif3_params/' + key])
        
        recurrent = np.array([
            f['recurrent/sources'][:],
            f['recurrent/targets'][:],
            f['recurrent/weights'][:],
            f['recurrent/delays'][:],
            f['recurrent/receptor_types'][:]
        ]).T
        
        input_syns = np.array([
            f['input/sources'][:],
            f['input/targets'][:],
            f['input/weights'][:],
            f['input/receptor_types'][:]
        ]).T
        
        bkg_weights = np.array(f['input/bkg_weights'])
        
        # Load readout weights if available
        readout_weights = None
        readout_biases = None
        if 'readout/weights' in f:
            readout_weights = np.array(f['readout/weights'])
        if 'readout/biases' in f:
            readout_biases = np.array(f['readout/biases'])
    
    return {
        'neurons': neurons,
        'output_neurons': output_neurons,
        'glif_params': glif_params,
        'recurrent': recurrent,
        'input_syns': input_syns,
        'bkg_weights': bkg_weights,
        'readout_weights': readout_weights,
        'readout_biases': readout_biases,
        'is_trained': is_trained
    }


def load_mnist_data(mnist_path):
    """Load MNIST spike train data"""
    with h5py.File(mnist_path, 'r') as f:
        spike_trains = np.array(f['spike_trains'])
        labels = np.array(f['labels'])
    return spike_trains, labels


# ============================================================================
# NEST Simulation
# ============================================================================

def create_spike_times(sample_spikes, seed=None):
    """Convert spike probabilities to spike times"""
    if seed is not None:
        np.random.seed(seed)
    
    spike_times = {}
    for neuron_idx in range(sample_spikes.shape[1]):
        times = []
        for t_idx in range(sample_spikes.shape[0]):
            prob = np.clip((sample_spikes[t_idx, neuron_idx] / 1.3), 0.0, 1.0)
            if prob > np.random.rand():
                times.append(float(t_idx + 1.0))
        if len(times) > 0:
            spike_times[neuron_idx] = times
    
    return spike_times


def setup_nest_network(network_data, spike_times):
    """Setup NEST network with given parameters and input spikes
    
    CRITICAL: Trained weights are multiplied by voltage scale.
              Untrained weights are NOT multiplied by voltage scale.
    """
    import nest
    
    neurons = network_data['neurons']
    glif_params = network_data['glif_params']
    recurrent = network_data['recurrent']
    input_syns = network_data['input_syns']
    bkg_weights = network_data['bkg_weights']
    is_trained = network_data['is_trained']
    
    # Reset NEST
    nest.ResetKernel()
    nest.resolution = 1.0
    
    # Create V1 neurons
    v1 = nest.Create('glif_psc', len(neurons))
    
    # Set GLIF3 parameters
    unique_types = np.unique(neurons)
    for ntype in unique_types:
        mask = neurons == ntype
        indices = np.where(mask)[0]
        
        params = {
            'C_m': float(glif_params['C_m'][ntype]),
            'E_L': float(glif_params['E_L'][ntype]),
            'V_reset': float(glif_params['V_reset'][ntype]),
            'V_th': float(glif_params['V_th'][ntype]),
            'V_m': float(glif_params['E_L'][ntype]),
            'g': float(glif_params['g'][ntype]),
            't_ref': float(glif_params['t_ref'][ntype]),
            'tau_syn': tuple(float(x) for x in glif_params['tau_syn'][ntype]),
            'asc_amps': tuple(float(x) for x in glif_params['asc_amps'][ntype]),
            'asc_decay': tuple(float(x) for x in glif_params['k'][ntype]),
            'after_spike_currents': True,
            'spike_dependent_threshold': False,
            'adapting_threshold': False
        }
        
        for idx in indices:
            v1[int(idx)].set(params)
    
    # Calculate voltage scales (only used for TRAINED networks)
    vsc = np.array([glif_params['V_th'][neurons[i]] - glif_params['E_L'][neurons[i]]
                    for i in range(len(neurons))])
    
    # Create background noise
    bkg_generators = nest.Create('poisson_generator', 10)
    for gen in bkg_generators:
        gen.set({'rate': 10.0, 'start': 0.0, 'stop': SIMULATION_TIME})
    
    v1_gids = np.array([n.global_id for n in v1])
    
    # Connect background noise
    for receptor_idx in range(4):
        weights = bkg_weights[:, receptor_idx]
        mask = np.abs(weights) > 1e-10
        if np.sum(mask) == 0:
            continue
        
        neuron_indices = np.where(mask)[0]
        w_base = weights[mask] / 10.0
        
        # CRITICAL: Apply voltage scaling only for TRAINED networks
        if is_trained:
            w_scaled = w_base * vsc[neuron_indices]
        else:
            w_scaled = w_base
        
        gids_filt = v1_gids[neuron_indices]
        
        for gen in bkg_generators:
            nest.Connect([gen.global_id] * len(gids_filt), gids_filt.tolist(),
                        conn_spec='one_to_one',
                        syn_spec={'weight': w_scaled.tolist(),
                                 'delay': np.ones(len(gids_filt)),
                                 'receptor_type': receptor_idx + 1})
    
    # Create LGN input
    n_lgn = max(spike_times.keys()) + 1 if spike_times else 0
    lgn = nest.Create('spike_generator', n_lgn)
    for i, times in spike_times.items():
        lgn[i].set({'spike_times': times})
    
    lgn_gids = np.array([n.global_id for n in lgn])
    
    # Connect LGN -> V1
    src_arr = input_syns[:, 0].astype(int)
    tgt_arr = input_syns[:, 1].astype(int)
    w_arr = input_syns[:, 2]
    rtype_arr = input_syns[:, 3].astype(int)
    
    # CRITICAL: Apply voltage scaling only for TRAINED networks
    if is_trained:
        w_scaled = w_arr * vsc[tgt_arr]
    else:
        w_scaled = w_arr
    
    mask = np.abs(w_scaled) > 1e-10
    
    nest.Connect(lgn_gids[src_arr[mask]].tolist(),
                v1_gids[tgt_arr[mask]].tolist(),
                conn_spec='one_to_one',
                syn_spec={'weight': w_scaled[mask],
                         'delay': np.ones(np.sum(mask)),
                         'receptor_type': (rtype_arr[mask] + 1).tolist()})
    
    # Connect V1 recurrent
    src_arr = recurrent[:, 0].astype(int)
    tgt_arr = recurrent[:, 1].astype(int)
    w_arr = recurrent[:, 2]
    d_arr = recurrent[:, 3]
    rtype_arr = recurrent[:, 4].astype(int)
    
    # CRITICAL: Apply voltage scaling only for TRAINED networks
    if is_trained:
        w_scaled = w_arr * vsc[tgt_arr]
    else:
        w_scaled = w_arr
    
    mask = np.abs(w_scaled) > 1e-10
    
    nest.Connect(v1_gids[src_arr[mask]].tolist(),
                v1_gids[tgt_arr[mask]].tolist(),
                conn_spec='one_to_one',
                syn_spec={'weight': w_scaled[mask],
                         'delay': np.maximum(d_arr[mask], 1.0),
                         'receptor_type': (rtype_arr[mask] + 1).tolist()})
    
    # Record spikes
    spike_rec = nest.Create('spike_recorder')
    nest.Connect(v1, spike_rec)
    
    return v1, v1_gids, spike_rec


def run_simulation(network_data, spike_times, verbose=True):
    """Run NEST simulation and return spike data"""
    import nest
    
    if verbose:
        print(f"  Running {SIMULATION_TIME}ms simulation...")
    
    v1, v1_gids, spike_rec = setup_nest_network(network_data, spike_times)
    
    nest.Simulate(SIMULATION_TIME)
    
    # Extract spikes
    events = spike_rec.get('events')
    senders, times = events['senders'], events['times']
    
    # Convert global IDs to neuron indices
    gid_to_idx = {gid: idx for idx, gid in enumerate(v1_gids)}
    neuron_ids = np.array([gid_to_idx[s] for s in senders])
    
    if verbose:
        print(f"  Total spikes: {len(times)}")
    
    return neuron_ids, times


def _run_single_trial_worker(args):
    """Worker function for parallel trial execution"""
    trial, base_seed, sample_spikes, network_data = args
    
    # Force NEST reset in worker process
    import nest
    nest.ResetKernel()
    
    seed = base_seed + trial
    spike_times = create_spike_times(sample_spikes, seed=seed)
    neuron_ids, times = run_simulation(network_data, spike_times, verbose=False)
    spike_counts = np.bincount(neuron_ids, minlength=len(network_data['neurons']))
    
    print(f"    [Trial {trial+1}] Complete", flush=True)
    return neuron_ids, times, spike_counts


def run_multiple_trials(network_data, sample_spikes, n_trials=10, base_seed=42, n_jobs=None):
    """Run simulation multiple times and return averaged results"""
    if n_jobs is None:
        n_jobs = min(cpu_count(), n_trials, 4)  # Cap at 4 for NEST stability
    
    print(f"    Running {n_trials} trials using {n_jobs} parallel workers...")
    
    # Prepare arguments for parallel execution
    args_list = [(trial, base_seed, sample_spikes, network_data) for trial in range(n_trials)]
    
    try:
        if n_jobs > 1:
            with Pool(processes=n_jobs) as pool:
                results = pool.map(_run_single_trial_worker, args_list)
        else:
            results = [_run_single_trial_worker(args) for args in args_list]
    except Exception as e:
        print(f"    WARNING: Parallel trials failed: {e}")
        print(f"    Falling back to sequential...")
        results = [_run_single_trial_worker(args) for args in args_list]
    
    # Unpack results
    all_neuron_ids = []
    all_times = []
    all_spike_counts = []
    
    for neuron_ids, times, spike_counts in results:
        all_neuron_ids.append(neuron_ids)
        all_times.append(times)
        all_spike_counts.append(spike_counts)
    
    avg_spike_counts = np.mean(all_spike_counts, axis=0)
    std_spike_counts = np.std(all_spike_counts, axis=0)
    
    return {
        'avg_counts': avg_spike_counts,
        'std_counts': std_spike_counts,
        'all_neuron_ids': all_neuron_ids,
        'all_times': all_times,
        'all_spike_counts': all_spike_counts
    }


# ============================================================================
# Readout and Prediction
# ============================================================================

def compute_prediction(neuron_ids, times, network_data, time_window=TIME_WINDOW):
    """Compute network prediction from output neuron activity"""
    output_neurons = network_data['output_neurons']
    readout_weights = network_data['readout_weights']
    readout_biases = network_data['readout_biases']
    
    # Filter spikes to time window
    mask = (times >= time_window[0]) & (times < time_window[1])
    filtered_ids = neuron_ids[mask]
    
    # Count spikes for output neurons
    output_spikes = np.zeros(len(output_neurons))
    for i, out_id in enumerate(output_neurons):
        output_spikes[i] = np.sum(filtered_ids == out_id)
    
    # Compute logits
    if readout_weights is not None and readout_biases is not None:
        logits = readout_weights @ output_spikes + readout_biases
    else:
        # Fallback: use spike counts directly as logits
        logits = np.zeros(10)
        for i in range(10):
            start = i*30
            end = (i+1)*30
            logits[i] = np.sum(output_spikes[start:end])
    
    # Compute probabilities
    probs = logits / np.sum(logits)
    prediction = np.argmax(logits)
    confidence = probs[prediction]
    
    return prediction, confidence, probs


# ============================================================================
# Visualization Utilities
# ============================================================================

def create_position_mapping(n_neurons):
    """Create 2D grid position mapping for neurons"""
    grid_size = int(np.ceil(np.sqrt(n_neurons)))
    positions = {}
    for i in range(n_neurons):
        row = i // grid_size
        col = i % grid_size
        positions[i] = (col, grid_size - row - 1)
    return positions, grid_size


def filter_to_window(neuron_ids, times, time_window):
    """Filter spike data to time window"""
    mask = (times >= time_window[0]) & (times < time_window[1])
    return neuron_ids[mask], times[mask]


# ============================================================================
# Static Visualizations
# ============================================================================

def visualize_spatial_raster(neuron_ids, times, network_data, label, 
                             time_window=TIME_WINDOW, prediction=None, 
                             confidence=None, untrained=False):
    """Spatial activity and raster plot"""
    neurons = network_data['neurons']
    output_neurons = network_data['output_neurons']
    
    positions, grid_size = create_position_mapping(len(neurons))
    filtered_ids, filtered_times = filter_to_window(neuron_ids, times, time_window)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Spatial activity heatmap
    spike_counts = np.zeros((grid_size, grid_size))
    for nid in filtered_ids:
        if nid in positions:
            x, y = positions[nid]
            spike_counts[y, x] += 1
    
    im = ax1.imshow(spike_counts, cmap=CMAP, aspect='auto', interpolation='nearest')
    
    network_type = 'untrained' if untrained else 'trained'
    title = f'Spatial activity for digit {label} in the {network_type} network'
    if prediction is not None:
        title += f'\npredicted: {prediction} ({confidence:.1%})'
    ax1.set_title(title, fontsize=12, pad=10)
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    plt.colorbar(im, ax=ax1, label='Spike count')
    
    # Highlight output neurons
    for out_id in output_neurons:
        if out_id in positions:
            x, y = positions[out_id]
            rect = Rectangle((x-0.5, y-0.5), 1, 1, fill=False,
                           edgecolor=OUTPUT_NEURON_COLOR, linewidth=1, alpha=0.6)
            ax1.add_patch(rect)
    
    # Raster plot
    ax2.scatter(filtered_times, filtered_ids, s=0.5, c=SPIKE_COLOR, alpha=0.6)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Neuron ID')
    ax2.set_title(f'Network activity, {len(filtered_times)} spikes total')
    ax2.set_xlim(time_window)
    ax2.set_ylim(-10, len(neurons) + 10)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    return fig


def visualize_population_activity(neuron_ids, times, network_data, label,
                                  time_window=TIME_WINDOW, untrained=False):
    """Activity breakdown by neuron population"""
    neurons = network_data['neurons']
    filtered_ids, _ = filter_to_window(neuron_ids, times, time_window)
    
    unique_pops = np.unique(neurons)
    pop_counts = []
    
    for pop_id in unique_pops:
        mask = neurons == pop_id
        pop_neuron_ids = np.where(mask)[0]
        count = np.sum(np.isin(filtered_ids, pop_neuron_ids))
        pop_counts.append(count)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars = ax.bar(range(len(pop_counts)), pop_counts, color='cyan', alpha=0.7)
    
    network_type = 'untrained' if untrained else 'trained'
    title = f'Population activity for digit {label} in the {network_type} network'
    ax.set_title(title)
    ax.set_xlabel('Population ID')
    ax.set_ylabel('Spike count')
    ax.set_xticks(range(0, len(pop_counts), max(1, len(pop_counts)//20)))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_activity_diff(neuron_ids1, times1, neuron_ids2, times2, 
                            network_data, label1, label2, time_window=TIME_WINDOW,
                            untrained=False):
    """Visualize difference in activity between two samples"""
    neurons = network_data['neurons']
    positions, grid_size = create_position_mapping(len(neurons))
    
    # Get spike counts for each sample
    filtered_ids1, _ = filter_to_window(neuron_ids1, times1, time_window)
    filtered_ids2, _ = filter_to_window(neuron_ids2, times2, time_window)
    
    spike_counts1 = np.zeros((grid_size, grid_size))
    spike_counts2 = np.zeros((grid_size, grid_size))
    
    for nid in filtered_ids1:
        if nid in positions:
            x, y = positions[nid]
            spike_counts1[y, x] += 1
    
    for nid in filtered_ids2:
        if nid in positions:
            x, y = positions[nid]
            spike_counts2[y, x] += 1
    
    diff = spike_counts1 - spike_counts2
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    network_type = 'untrained' if untrained else 'trained'
    
    # Sample 1
    im1 = ax1.imshow(spike_counts1, cmap=CMAP, aspect='auto', interpolation='nearest')
    ax1.set_title(f'Activity for digit {label1} in the {network_type} network')
    plt.colorbar(im1, ax=ax1)
    
    # Sample 2
    im2 = ax2.imshow(spike_counts2, cmap=CMAP, aspect='auto', interpolation='nearest')
    ax2.set_title(f'Activity for digit {label2} in the {network_type} network')
    plt.colorbar(im2, ax=ax2)
    
    # Difference
    vmax = np.abs(diff).max()
    im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                     vmin=-vmax, vmax=vmax)
    title = f'Difference between digit {label1} and digit {label2} in the {network_type} network'
    ax3.set_title(title)
    plt.colorbar(im3, ax=ax3, label='Spike count difference')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
    
    plt.tight_layout()
    return fig


def visualize_diff_histogram(neuron_ids1, times1, neuron_ids2, times2,
                             network_data, label1, label2, 
                             time_window=TIME_WINDOW, untrained=False):
    """Histogram of spike count differences per neuron"""
    neurons = network_data['neurons']
    
    filtered_ids1, _ = filter_to_window(neuron_ids1, times1, time_window)
    filtered_ids2, _ = filter_to_window(neuron_ids2, times2, time_window)
    
    counts1 = np.bincount(filtered_ids1, minlength=len(neurons))
    counts2 = np.bincount(filtered_ids2, minlength=len(neurons))
    diff = counts1 - counts2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(diff, bins=50, color='cyan', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    
    network_type = 'untrained' if untrained else 'trained'
    title = f'Spike count difference distribution for digits {label1} and {label2} in the {network_type} network'
    ax.set_title(title)
    ax.set_xlabel('Spike count difference')
    ax.set_ylabel('Number of neurons')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_averaged_results(avg_results, network_data, label, 
                               time_window=TIME_WINDOW, untrained=False):
    """Visualize averaged results from multiple trials"""
    avg_counts = avg_results['avg_counts']
    std_counts = avg_results['std_counts']
    neurons = network_data['neurons']
    positions, grid_size = create_position_mapping(len(neurons))
    
    # Create spatial maps
    avg_map = np.zeros((grid_size, grid_size))
    std_map = np.zeros((grid_size, grid_size))
    
    for nid, (avg, std) in enumerate(zip(avg_counts, std_counts)):
        if nid in positions:
            x, y = positions[nid]
            avg_map[y, x] = avg
            std_map[y, x] = std
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Average activity
    im1 = ax1.imshow(avg_map, cmap=CMAP, aspect='auto', interpolation='nearest')
    network_type = 'untrained' if untrained else 'trained'
    title = f'Average spike count for digit {label} in the {network_type} network'
    ax1.set_title(title)
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    plt.colorbar(im1, ax=ax1, label='Average spike count')
    
    # Standard deviation
    im2 = ax2.imshow(std_map, cmap='viridis', aspect='auto', interpolation='nearest')
    ax2.set_title(f'Spike count standard deviation for digit {label} in the {network_type} network')
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    plt.colorbar(im2, ax=ax2, label='Standard deviation')
    
    plt.tight_layout()
    return fig


def visualize_trained_vs_untrained(neuron_ids_trained, times_trained,
                                   neuron_ids_untrained, times_untrained,
                                   network_data, label, time_window=TIME_WINDOW):
    """Compare activity between trained and untrained networks"""
    neurons = network_data['neurons']
    positions, grid_size = create_position_mapping(len(neurons))
    
    # Filter to time window
    filtered_ids_t, _ = filter_to_window(neuron_ids_trained, times_trained, time_window)
    filtered_ids_u, _ = filter_to_window(neuron_ids_untrained, times_untrained, time_window)
    
    # Create spike count maps
    spikes_trained = np.zeros((grid_size, grid_size))
    spikes_untrained = np.zeros((grid_size, grid_size))
    
    for nid in filtered_ids_t:
        if nid in positions:
            x, y = positions[nid]
            spikes_trained[y, x] += 1
    
    for nid in filtered_ids_u:
        if nid in positions:
            x, y = positions[nid]
            spikes_untrained[y, x] += 1
    
    diff = spikes_trained - spikes_untrained
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Trained network
    im1 = ax1.imshow(spikes_trained, cmap=CMAP, aspect='auto', interpolation='nearest')
    ax1.set_title(f'Trained network activity for digit {label}')
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    plt.colorbar(im1, ax=ax1, label='Spike count')
    
    # Untrained network
    im2 = ax2.imshow(spikes_untrained, cmap=CMAP, aspect='auto', interpolation='nearest')
    ax2.set_title(f'Untrained network activity for digit {label}')
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    plt.colorbar(im2, ax=ax2, label='Spike count')
    
    # Difference (Trained - Untrained)
    vmax = np.abs(diff).max()
    im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                     vmin=-vmax, vmax=vmax)
    ax3.set_title(f'Difference between trained and untrained network activity for digit {label}')
    ax3.set_xlabel('X position')
    ax3.set_ylabel('Y position')
    plt.colorbar(im3, ax=ax3, label='Spike count difference')
    
    plt.tight_layout()
    return fig


def visualize_trained_vs_untrained_histogram(neuron_ids_trained, times_trained,
                                             neuron_ids_untrained, times_untrained,
                                             network_data, label, time_window=TIME_WINDOW):
    """Histogram comparing trained vs untrained spike counts"""
    neurons = network_data['neurons']
    
    filtered_ids_t, _ = filter_to_window(neuron_ids_trained, times_trained, time_window)
    filtered_ids_u, _ = filter_to_window(neuron_ids_untrained, times_untrained, time_window)
    
    counts_t = np.bincount(filtered_ids_t, minlength=len(neurons))
    counts_u = np.bincount(filtered_ids_u, minlength=len(neurons))
    diff = counts_t - counts_u
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Trained distribution
    ax1.hist(counts_t, bins=50, color='cyan', alpha=0.7, edgecolor='white')
    ax1.set_title(f'Spike count distribution for digit {label} in the trained network')
    ax1.set_xlabel('Spikes per neuron')
    ax1.set_ylabel('Number of neurons')
    ax1.grid(True, alpha=0.3)
    
    # Untrained distribution
    ax2.hist(counts_u, bins=50, color='orange', alpha=0.7, edgecolor='white')
    ax2.set_title(f'Spike count distribution for digit {label} in the untrained network')
    ax2.set_xlabel('Spikes per neuron')
    ax2.set_ylabel('Number of neurons')
    ax2.grid(True, alpha=0.3)
    
    # Difference distribution
    ax3.hist(diff, bins=50, color='lime', alpha=0.7, edgecolor='white')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax3.set_title(f'Difference distribution for digit {label} between trained and untrained networks')
    ax3.set_xlabel('Spike count difference')
    ax3.set_ylabel('Number of neurons')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Scatter comparison
    ax4.scatter(counts_u, counts_t, s=1, c='cyan', alpha=0.5)
    max_val = max(counts_u.max(), counts_t.max())
    ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Equal')
    ax4.set_xlabel('Untrained spike count')
    ax4.set_ylabel('Trained spike count')
    ax4.set_title('Neuron-by-neuron comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_trained_vs_untrained_metrics(metrics_trained, metrics_untrained, label):
    """Compare key metrics between trained and untrained networks"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract comparable metrics
    metric_names = ['total_spikes', 'active_neurons', 'firing_rate', 'sparsity']
    trained_vals = [metrics_trained[m] for m in metric_names]
    untrained_vals = [metrics_untrained[m] for m in metric_names]
    
    # Bar comparison
    x = np.arange(len(metric_names))
    width = 0.35
    
    ax1.bar(x - width/2, trained_vals, width, label='Trained', color='cyan', alpha=0.7)
    ax1.bar(x + width/2, untrained_vals, width, label='Untrained', color='orange', alpha=0.7)
    ax1.set_ylabel('Value')
    ax1.set_title(f'Key metrics comparison for digit {label}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Population firing rates comparison
    pop_keys_t = sorted([k for k in metrics_trained.keys() if k.startswith('pop_')])
    pop_keys_u = sorted([k for k in metrics_untrained.keys() if k.startswith('pop_')])
    
    if pop_keys_t and pop_keys_u:
        pop_ids = [int(k.split('_')[1]) for k in pop_keys_t]
        pop_rates_t = [metrics_trained[k] for k in pop_keys_t]
        pop_rates_u = [metrics_untrained[k] for k in pop_keys_u]
        
        x_pop = np.arange(len(pop_ids))
        ax2.plot(x_pop, pop_rates_t, 'o-', label='Trained', color='cyan', linewidth=2)
        ax2.plot(x_pop, pop_rates_u, 's-', label='Untrained', color='orange', linewidth=2)
        ax2.set_xlabel('Population ID')
        ax2.set_ylabel('Firing rate (Hz)')
        ax2.set_title('Population firing rates')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Relative differences
    relative_diffs = []
    diff_labels = []
    for name in metric_names:
        if untrained_vals[metric_names.index(name)] > 0:
            rel_diff = (trained_vals[metric_names.index(name)] - 
                       untrained_vals[metric_names.index(name)]) / untrained_vals[metric_names.index(name)] * 100
            relative_diffs.append(rel_diff)
            diff_labels.append(name)
    
    colors = ['lime' if d > 0 else 'red' for d in relative_diffs]
    ax3.barh(diff_labels, relative_diffs, color=colors, alpha=0.7)
    ax3.axvline(0, color='white', linestyle='--', linewidth=2)
    ax3.set_xlabel('Relative difference %')
    ax3.set_title('Relative change between trained and untrained weights')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Summary text
    summary_text = f"Label: {label}\n\n"
    summary_text += f"Total spikes:\n  Trained: {metrics_trained['total_spikes']}\n"
    summary_text += f"  Untrained: {metrics_untrained['total_spikes']}\n\n"
    summary_text += f"Active neurons:\n  Trained: {metrics_trained['active_neurons']}\n"
    summary_text += f"  Untrained: {metrics_untrained['active_neurons']}\n\n"
    summary_text += f"Average firing rate:\n  Trained: {metrics_trained['firing_rate']:.2f} Hz\n"
    summary_text += f"  Untrained: {metrics_untrained['firing_rate']:.2f} Hz\n\n"
    summary_text += f"Sparsity:\n  Trained: {metrics_trained['sparsity']:.3f}\n"
    summary_text += f"  Untrained: {metrics_untrained['sparsity']:.3f}"
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', alpha=0.8, edgecolor='cyan'))
    ax4.axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================================
# Animation
# ============================================================================

def create_animation(neuron_ids, times, network_data, label, output_file,
                    prediction=None, confidence=None, untrained=False):
    """Create animation with variable speed (slower during critical window)"""
    neurons = network_data['neurons']
    output_neurons = network_data['output_neurons']
    positions, grid_size = create_position_mapping(len(neurons))
    
    # Create time bins
    time_bins = np.arange(0, SIMULATION_TIME, ANIMATION_BIN_SIZE)
    
    # Pre-compute spike maps
    spike_maps = []
    max_spikes = 0
    for t_start in time_bins:
        t_end = t_start + ANIMATION_BIN_SIZE
        mask = (times >= t_start) & (times < t_end)
        bin_ids = neuron_ids[mask]
        
        spike_map = np.zeros((grid_size, grid_size))
        for nid in bin_ids:
            if nid in positions:
                x, y = positions[nid]
                spike_map[y, x] += 1
        
        spike_maps.append(spike_map)
        max_spikes = max(max_spikes, spike_map.max())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(spike_maps[0], cmap=CMAP, aspect='auto',
                  interpolation='nearest', vmin=0, vmax=max_spikes)
    
    network_type = 'untrained' if untrained else 'trained'
    title = f'Network activity for digit {label} in the {network_type} network'
    if prediction is not None:
        title += f'\npredicted: {prediction} ({confidence:.1%})'
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    plt.colorbar(im, ax=ax, label=f'Spikes per {ANIMATION_BIN_SIZE} ms')
    
    # Highlight output neurons
    for out_id in output_neurons:
        if out_id in positions:
            x, y = positions[out_id]
            rect = Rectangle((x-0.5, y-0.5), 1, 1, fill=False,
                           edgecolor=OUTPUT_NEURON_COLOR, linewidth=1, alpha=0.6)
            ax.add_patch(rect)
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       verticalalignment='top', fontsize=14, color='white',
                       bbox=dict(boxstyle='round', alpha=0.7))
    
    # Variable interval based on time window
    def get_interval(frame):
        t_start = time_bins[frame]
        if TIME_WINDOW[0] <= t_start < TIME_WINDOW[1]:
            return 300  # Slower (300ms per frame)
        else:
            return 50   # Faster (50ms per frame)
    
    intervals = [get_interval(i) for i in range(len(time_bins))]
    
    def animate(frame):
        im.set_data(spike_maps[frame])
        t_start = time_bins[frame]
        t_end = t_start + ANIMATION_BIN_SIZE
        
        # Highlight critical window
        if TIME_WINDOW[0] <= t_start < TIME_WINDOW[1]:
            time_text.set_text(f'{int(t_start)}-{int(t_end)}ms ⚡ CRITICAL')
            time_text.set_bbox(dict(boxstyle='round', facecolor='red', alpha=0.8))
        else:
            time_text.set_text(f'{int(t_start)}-{int(t_end)}ms')
            time_text.set_bbox(dict(boxstyle='round', alpha=0.7))
        
        return [im, time_text]
    
    # Create animation with variable speed
    anim = animation.FuncAnimation(fig, animate, frames=len(time_bins),
                                  interval=100, blit=True)
    
    print(f"  Saving animation to {output_file}...")
    
    # Custom writer with variable frame duration
    writer = animation.FFMpegWriter(fps=ANIMATION_FPS, bitrate=2000)
    anim.save(output_file, writer=writer, dpi=100)
    
    plt.close()
    print(f"  Animation saved!")


def create_diff_animation(neuron_ids1, times1, neuron_ids2, times2,
                         network_data, label1, label2, output_file, untrained=False):
    """Animate activity difference between two samples"""
    neurons = network_data['neurons']
    positions, grid_size = create_position_mapping(len(neurons))
    
    time_bins = np.arange(0, SIMULATION_TIME, ANIMATION_BIN_SIZE)
    
    # Pre-compute difference maps
    diff_maps = []
    max_abs_diff = 0
    
    for t_start in time_bins:
        t_end = t_start + ANIMATION_BIN_SIZE
        
        mask1 = (times1 >= t_start) & (times1 < t_end)
        mask2 = (times2 >= t_start) & (times2 < t_end)
        
        spike_map1 = np.zeros((grid_size, grid_size))
        spike_map2 = np.zeros((grid_size, grid_size))
        
        for nid in neuron_ids1[mask1]:
            if nid in positions:
                x, y = positions[nid]
                spike_map1[y, x] += 1
        
        for nid in neuron_ids2[mask2]:
            if nid in positions:
                x, y = positions[nid]
                spike_map2[y, x] += 1
        
        diff = spike_map1 - spike_map2
        diff_maps.append(diff)
        max_abs_diff = max(max_abs_diff, np.abs(diff).max())
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(diff_maps[0], cmap='RdBu_r', aspect='auto',
                  interpolation='nearest', vmin=-max_abs_diff, vmax=max_abs_diff)
    
    network_type = 'untrained' if untrained else 'trained'
    title = f'Activity difference for digits {label1} and {label2} in the {network_type} network'
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    plt.colorbar(im, ax=ax, label='Spike count difference')
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       verticalalignment='top', fontsize=14, color='white',
                       bbox=dict(boxstyle='round', alpha=0.7))
    
    def animate(frame):
        im.set_data(diff_maps[frame])
        t_start = time_bins[frame]
        t_end = t_start + ANIMATION_BIN_SIZE
        
        if TIME_WINDOW[0] <= t_start < TIME_WINDOW[1]:
            time_text.set_text(f'{int(t_start)}-{int(t_end)}ms ⚡ CRITICAL')
            time_text.set_bbox(dict(boxstyle='round', facecolor='red', alpha=0.8))
        else:
            time_text.set_text(f'{int(t_start)}-{int(t_end)}ms')
            time_text.set_bbox(dict(boxstyle='round', alpha=0.7))
        
        return [im, time_text]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(time_bins),
                                  interval=100, blit=True)
    
    print(f"  Saving diff animation to {output_file}...")
    writer = animation.FFMpegWriter(fps=ANIMATION_FPS, bitrate=2000)
    anim.save(output_file, writer=writer, dpi=100)
    plt.close()
    print(f"  Diff animation saved!")


def create_trained_vs_untrained_animation(neuron_ids_trained, times_trained,
                                         neuron_ids_untrained, times_untrained,
                                         network_data, label, output_file):
    """Animate comparison between trained and untrained networks"""
    neurons = network_data['neurons']
    positions, grid_size = create_position_mapping(len(neurons))
    
    time_bins = np.arange(0, SIMULATION_TIME, ANIMATION_BIN_SIZE)
    
    # Pre-compute maps for both networks
    trained_maps = []
    untrained_maps = []
    diff_maps = []
    max_spikes = 0
    max_abs_diff = 0
    
    for t_start in time_bins:
        t_end = t_start + ANIMATION_BIN_SIZE
        
        mask_t = (times_trained >= t_start) & (times_trained < t_end)
        mask_u = (times_untrained >= t_start) & (times_untrained < t_end)
        
        spike_map_t = np.zeros((grid_size, grid_size))
        spike_map_u = np.zeros((grid_size, grid_size))
        
        for nid in neuron_ids_trained[mask_t]:
            if nid in positions:
                x, y = positions[nid]
                spike_map_t[y, x] += 1
        
        for nid in neuron_ids_untrained[mask_u]:
            if nid in positions:
                x, y = positions[nid]
                spike_map_u[y, x] += 1
        
        diff = spike_map_t - spike_map_u
        
        trained_maps.append(spike_map_t)
        untrained_maps.append(spike_map_u)
        diff_maps.append(diff)
        
        max_spikes = max(max_spikes, spike_map_t.max(), spike_map_u.max())
        max_abs_diff = max(max_abs_diff, np.abs(diff).max())
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Trained network
    im1 = ax1.imshow(trained_maps[0], cmap=CMAP, aspect='auto',
                    interpolation='nearest', vmin=0, vmax=max_spikes)
    ax1.set_title(f'Trained network activity for digit {label}', fontsize=12)
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    plt.colorbar(im1, ax=ax1, label=f'Spikes per {ANIMATION_BIN_SIZE} ms')
    
    # Untrained network
    im2 = ax2.imshow(untrained_maps[0], cmap=CMAP, aspect='auto',
                    interpolation='nearest', vmin=0, vmax=max_spikes)
    ax2.set_title(f'Untrained network activity for digit {label}', fontsize=12)
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    plt.colorbar(im2, ax=ax2, label=f'Spikes per {ANIMATION_BIN_SIZE} ms')
    
    # Difference
    im3 = ax3.imshow(diff_maps[0], cmap='RdBu_r', aspect='auto',
                    interpolation='nearest', vmin=-max_abs_diff, vmax=max_abs_diff)
    ax3.set_title('Difference between trained and untrained networks', fontsize=12)
    ax3.set_xlabel('X position')
    ax3.set_ylabel('Y position')
    plt.colorbar(im3, ax=ax3, label='Spike count difference')
    
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14, color='white',
                        bbox=dict(boxstyle='round', alpha=0.7))
    
    def animate(frame):
        im1.set_data(trained_maps[frame])
        im2.set_data(untrained_maps[frame])
        im3.set_data(diff_maps[frame])
        
        t_start = time_bins[frame]
        t_end = t_start + ANIMATION_BIN_SIZE
        
        if TIME_WINDOW[0] <= t_start < TIME_WINDOW[1]:
            time_text.set_text(f'{int(t_start)}-{int(t_end)}ms ⚡ CRITICAL WINDOW')
            time_text.set_bbox(dict(boxstyle='round', facecolor='red', alpha=0.8))
        else:
            time_text.set_text(f'{int(t_start)}-{int(t_end)}ms')
            time_text.set_bbox(dict(boxstyle='round', alpha=0.7))
        
        return [im1, im2, im3, time_text]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(time_bins),
                                  interval=100, blit=True)
    
    print(f"  Saving trained vs untrained animation to {output_file}...")
    writer = animation.FFMpegWriter(fps=ANIMATION_FPS, bitrate=3000)
    anim.save(output_file, writer=writer, dpi=100)
    plt.close()
    print(f"  Trained vs untrained animation saved!")


# ============================================================================
# Weight Analysis
# ============================================================================

def normalize_weights_to_distribution(weights, target_mean=0.0, target_std=1.0):
    """Normalize weights to match target distribution"""
    current_mean = np.mean(weights)
    current_std = np.std(weights)
    if current_std > 0:
        normalized = (weights - current_mean) / current_std
        normalized = normalized * target_std + target_mean
    else:
        normalized = weights
    return normalized


def compare_weight_distributions(network_data_trained, network_data_untrained):
    """Compare weight distributions between trained and untrained networks"""
    # Extract weights
    w_trained_rec = network_data_trained['recurrent'][:, 2]
    w_untrained_rec = network_data_untrained['recurrent'][:, 2]
    
    w_trained_input = network_data_trained['input_syns'][:, 2]
    w_untrained_input = network_data_untrained['input_syns'][:, 2]
    
    # Normalize both to same distribution
    w_trained_rec_norm = normalize_weights_to_distribution(w_trained_rec)
    w_untrained_rec_norm = normalize_weights_to_distribution(w_untrained_rec)
    
    w_trained_input_norm = normalize_weights_to_distribution(w_trained_input)
    w_untrained_input_norm = normalize_weights_to_distribution(w_untrained_input)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Recurrent weights - Raw
    axes[0, 0].hist(w_trained_rec, bins=100, alpha=0.7, label='Trained', color='cyan')
    axes[0, 0].hist(w_untrained_rec, bins=100, alpha=0.7, label='Untrained', color='orange')
    axes[0, 0].set_title('Recurrent weights (raw)')
    axes[0, 0].set_xlabel('Weight value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Recurrent weights - Normalized
    axes[0, 1].hist(w_trained_rec_norm, bins=100, alpha=0.7, label='Trained (norm)', color='cyan')
    axes[0, 1].hist(w_untrained_rec_norm, bins=100, alpha=0.7, label='Untrained (norm)', color='orange')
    axes[0, 1].set_title('Recurrent weights (normalized)')
    axes[0, 1].set_xlabel('Weight value')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Recurrent weights - Scatter comparison
    axes[0, 2].scatter(w_untrained_rec_norm, w_trained_rec_norm, s=0.5, alpha=0.3, color='lime')
    lim = max(abs(w_trained_rec_norm).max(), abs(w_untrained_rec_norm).max())
    axes[0, 2].plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Equal')
    axes[0, 2].set_xlabel('Untrained (normalized)')
    axes[0, 2].set_ylabel('Trained (normalized)')
    axes[0, 2].set_title('Recurrent weight correlation')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Input weights - Raw
    axes[1, 0].hist(w_trained_input, bins=100, alpha=0.7, label='Trained', color='cyan')
    axes[1, 0].hist(w_untrained_input, bins=100, alpha=0.7, label='Untrained', color='orange')
    axes[1, 0].set_title('Input weights (raw)')
    axes[1, 0].set_xlabel('Weight value')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Input weights - Normalized
    axes[1, 1].hist(w_trained_input_norm, bins=100, alpha=0.7, label='Trained (norm)', color='cyan')
    axes[1, 1].hist(w_untrained_input_norm, bins=100, alpha=0.7, label='Untrained (norm)', color='orange')
    axes[1, 1].set_title('Input weights (normalized)')
    axes[1, 1].set_xlabel('Weight value')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Input weights - Scatter comparison
    axes[1, 2].scatter(w_untrained_input_norm, w_trained_input_norm, s=0.5, alpha=0.3, color='lime')
    lim = max(abs(w_trained_input_norm).max(), abs(w_untrained_input_norm).max())
    axes[1, 2].plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Equal')
    axes[1, 2].set_xlabel('Untrained (normalized)')
    axes[1, 2].set_ylabel('Trained (normalized)')
    axes[1, 2].set_title('Input weight correlation')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compute_weight_statistics(network_data_trained, network_data_untrained):
    """Compute statistical comparison of weights"""
    w_trained_rec = network_data_trained['recurrent'][:, 2]
    w_untrained_rec = network_data_untrained['recurrent'][:, 2]
    w_trained_input = network_data_trained['input_syns'][:, 2]
    w_untrained_input = network_data_untrained['input_syns'][:, 2]
    
    # Normalize
    w_trained_rec_norm = normalize_weights_to_distribution(w_trained_rec)
    w_untrained_rec_norm = normalize_weights_to_distribution(w_untrained_rec)
    w_trained_input_norm = normalize_weights_to_distribution(w_trained_input)
    w_untrained_input_norm = normalize_weights_to_distribution(w_untrained_input)
    
    # Compute correlations
    corr_rec = np.corrcoef(w_trained_rec_norm, w_untrained_rec_norm)[0, 1]
    corr_input = np.corrcoef(w_trained_input_norm, w_untrained_input_norm)[0, 1]
    
    stats = {
        'recurrent': {
            'trained_mean': np.mean(w_trained_rec),
            'trained_std': np.std(w_trained_rec),
            'untrained_mean': np.mean(w_untrained_rec),
            'untrained_std': np.std(w_untrained_rec),
            'correlation_normalized': corr_rec
        },
        'input': {
            'trained_mean': np.mean(w_trained_input),
            'trained_std': np.std(w_trained_input),
            'untrained_mean': np.mean(w_untrained_input),
            'untrained_std': np.std(w_untrained_input),
            'correlation_normalized': corr_input
        }
    }
    
    return stats


# ============================================================================
# Digit Similarity Analysis
# ============================================================================

def _analyze_pair_worker(args):
    """Worker function for parallel digit pair analysis"""
    pair_name, digits, spike_trains, labels, network_data, seed = args
    digit1, digit2 = digits
    
    # Force NEST reset in worker process
    import nest
    nest.ResetKernel()
    
    print(f"    [Worker] Processing {pair_name}...", flush=True)
    
    # Find samples for each digit
    samples_digit1 = np.where(labels == digit1)[0]
    samples_digit2 = np.where(labels == digit2)[0]
    
    if len(samples_digit1) == 0 or len(samples_digit2) == 0:
        return None
    
    # Use first sample of each digit
    sample1 = samples_digit1[0]
    sample2 = samples_digit2[0]
    
    # Run simulations
    spike_times1 = create_spike_times(spike_trains[sample1], seed=seed)
    spike_times2 = create_spike_times(spike_trains[sample2], seed=seed + 1000)
    
    neuron_ids1, times1 = run_simulation(network_data, spike_times1, verbose=False)
    neuron_ids2, times2 = run_simulation(network_data, spike_times2, verbose=False)
    
    # Compute metrics
    metrics1 = compute_metrics(neuron_ids1, times1, network_data)
    metrics2 = compute_metrics(neuron_ids2, times2, network_data)
    
    # Compute activity difference
    neurons = network_data['neurons']
    filtered_ids1, _ = filter_to_window(neuron_ids1, times1, TIME_WINDOW)
    filtered_ids2, _ = filter_to_window(neuron_ids2, times2, TIME_WINDOW)
    
    counts1 = np.bincount(filtered_ids1, minlength=len(neurons))
    counts2 = np.bincount(filtered_ids2, minlength=len(neurons))
    
    # Similarity metrics
    diff = counts1 - counts2
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    correlation = np.corrcoef(counts1, counts2)[0, 1]
    
    print(f"    [Worker] Completed {pair_name} | Correlation: {correlation:.3f}", flush=True)
    
    return pair_name, {
        'digits': (digit1, digit2),
        'samples': (sample1, sample2),
        'neuron_ids1': neuron_ids1,
        'times1': times1,
        'neuron_ids2': neuron_ids2,
        'times2': times2,
        'metrics1': metrics1,
        'metrics2': metrics2,
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'counts1': counts1,
        'counts2': counts2
    }


def analyze_digit_similarity(network_data, spike_trains, labels, digit_pairs, 
                             n_trials=5, seed=42, n_jobs=None):
    """Analyze network response to similar vs dissimilar digit pairs"""
    if n_jobs is None:
        n_jobs = min(cpu_count(), len(digit_pairs), 4)  # Cap at 4 for NEST stability
    
    print(f"    Analyzing {len(digit_pairs)} digit pairs using {n_jobs} parallel workers...")
    
    # Prepare arguments for parallel execution
    args_list = [(pair_name, digits, spike_trains, labels, network_data, seed)
                 for pair_name, digits in digit_pairs.items()]
    
    try:
        if n_jobs > 1:
            with Pool(processes=n_jobs) as pool:
                results_list = pool.map(_analyze_pair_worker, args_list)
        else:
            results_list = [_analyze_pair_worker(args) for args in args_list]
    except Exception as e:
        print(f"    WARNING: Parallel analysis failed: {e}")
        print(f"    Falling back to sequential...")
        results_list = [_analyze_pair_worker(args) for args in args_list]
    
    # Convert to dictionary
    results = {}
    for result in results_list:
        if result is not None:
            pair_name, data = result
            results[pair_name] = data
            print(f"  {pair_name}: Digit {data['digits'][0]} vs {data['digits'][1]} | " +
                  f"Correlation: {data['correlation']:.3f}")
    
    return results


def visualize_similarity_analysis(similarity_results, network_data):
    """Visualize similarity analysis results"""
    pair_names = list(similarity_results.keys())
    n_pairs = len(pair_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MSE comparison
    mse_values = [similarity_results[name]['mse'] for name in pair_names]
    colors = ['lime' if 'similar' in name.lower() else 'red' for name in pair_names]
    axes[0, 0].bar(range(n_pairs), mse_values, color=colors, alpha=0.7)
    axes[0, 0].set_xticks(range(n_pairs))
    axes[0, 0].set_xticklabels(pair_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Mean squared error')
    axes[0, 0].set_title('Activity difference (MSE)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # MAE comparison
    mae_values = [similarity_results[name]['mae'] for name in pair_names]
    axes[0, 1].bar(range(n_pairs), mae_values, color=colors, alpha=0.7)
    axes[0, 1].set_xticks(range(n_pairs))
    axes[0, 1].set_xticklabels(pair_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Mean absolute error')
    axes[0, 1].set_title('Activity difference (MAE)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Correlation comparison
    corr_values = [similarity_results[name]['correlation'] for name in pair_names]
    axes[1, 0].bar(range(n_pairs), corr_values, color=colors, alpha=0.7)
    axes[1, 0].axhline(0, color='white', linestyle='--', linewidth=1)
    axes[1, 0].set_xticks(range(n_pairs))
    axes[1, 0].set_xticklabels(pair_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_title('Activity pattern correlation')
    axes[1, 0].set_ylim(-0.1, 1.0)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Summary text
    summary_text = "SIMILARITY ANALYSIS\n\n"
    for name in pair_names:
        res = similarity_results[name]
        d1, d2 = res['digits']
        summary_text += f"{name}:\n"
        summary_text += f"  Digits: {d1} vs {d2}\n"
        summary_text += f"  Correlation: {res['correlation']:.3f}\n"
        summary_text += f"  MAE: {res['mae']:.2f}\n\n"
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', alpha=0.8, edgecolor='cyan'))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(neuron_ids, times, network_data, time_window=TIME_WINDOW):
    """Compute various network metrics"""
    neurons = network_data['neurons']
    filtered_ids, filtered_times = filter_to_window(neuron_ids, times, time_window)
    
    window_duration = (time_window[1] - time_window[0]) / 1000.0  # in seconds
    
    metrics = {
        'total_spikes': len(filtered_times),
        'active_neurons': len(np.unique(filtered_ids)),
        'firing_rate': len(filtered_times) / (len(neurons) * window_duration),  # Hz
        'sparsity': len(np.unique(filtered_ids)) / len(neurons),
    }
    
    # Population-specific firing rates
    unique_pops = np.unique(neurons)
    pop_rates = {}
    for pop_id in unique_pops:
        mask = neurons == pop_id
        pop_neuron_ids = np.where(mask)[0]
        pop_spikes = np.sum(np.isin(filtered_ids, pop_neuron_ids))
        pop_rate = pop_spikes / (np.sum(mask) * window_duration)
        pop_rates[f'pop_{pop_id}_rate'] = pop_rate
    
    metrics.update(pop_rates)
    
    return metrics


def print_metrics(metrics, label, prediction=None, confidence=None):
    """Print metrics in a formatted way"""
    print(f"\n{'='*60}")
    print(f"METRICS - Label: {label}")
    if prediction is not None:
        print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
    print(f"{'='*60}")
    print(f"Total spikes:      {metrics['total_spikes']}")
    print(f"Active neurons:    {metrics['active_neurons']}")
    print(f"Firing rate:       {metrics['firing_rate']:.2f} Hz")
    print(f"Sparsity:          {metrics['sparsity']:.3f}")
    
    pop_keys = [k for k in metrics.keys() if k.startswith('pop_')]
    if pop_keys:
        print(f"\nPopulation firing rates (Hz):")
        for key in sorted(pop_keys):
            pop_id = key.split('_')[1]
            print(f"  Pop {pop_id:3s}: {metrics[key]:6.2f}")
    print(f"{'='*60}\n")


# ============================================================================
# Auto Mode
# ============================================================================

def auto_select_samples(labels, n_samples_per_digit=1):
    """Automatically select representative samples for each digit"""
    selected = {}
    for digit in range(10):
        indices = np.where(labels == digit)[0]
        if len(indices) > 0:
            # Select first n_samples_per_digit
            selected[digit] = indices[:n_samples_per_digit].tolist()
    return selected


def create_auto_summary(all_results, network_data, network_name):
    """Create comprehensive summary visualization for auto mode"""
    n_digits = len(all_results)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Confusion matrix
    ax_conf = fig.add_subplot(gs[0, 0])
    conf_matrix = np.zeros((10, 10))
    for true_digit, result in all_results.items():
        pred_digit = result['prediction']
        conf_matrix[true_digit, pred_digit] += 1
    
    im = ax_conf.imshow(conf_matrix, cmap='hot', aspect='auto', interpolation='nearest')
    ax_conf.set_xlabel('Predicted digit')
    ax_conf.set_ylabel('True digit')
    ax_conf.set_title(f'Confusion matrix for the {network_name} network')
    ax_conf.set_xticks(range(10))
    ax_conf.set_yticks(range(10))
    plt.colorbar(im, ax=ax_conf)
    
    # Confidence per digit
    ax_conf_val = fig.add_subplot(gs[0, 1])
    digits = sorted(all_results.keys())
    confidences = [all_results[d]['confidence'] for d in digits]
    colors = ['lime' if all_results[d]['prediction'] == d else 'red' for d in digits]
    ax_conf_val.bar(digits, confidences, color=colors, alpha=0.7)
    ax_conf_val.set_xlabel('Digit')
    ax_conf_val.set_ylabel('Confidence')
    ax_conf_val.set_title('Prediction confidence per digit')
    ax_conf_val.set_ylim(0, 1)
    ax_conf_val.grid(True, alpha=0.3, axis='y')
    
    # Metrics comparison
    ax_metrics = fig.add_subplot(gs[0, 2])
    metric_names = ['total_spikes', 'active_neurons', 'firing_rate']
    metric_values = {m: [] for m in metric_names}
    for digit in digits:
        for m in metric_names:
            metric_values[m].append(all_results[digit]['metrics'][m])
    
    x = np.arange(len(digits))
    width = 0.25
    for i, m in enumerate(metric_names):
        normalized = np.array(metric_values[m]) / max(metric_values[m])
        ax_metrics.bar(x + i*width, normalized, width, label=m, alpha=0.7)
    
    ax_metrics.set_xlabel('Digit')
    ax_metrics.set_ylabel('Normalized value')
    ax_metrics.set_title('Network metrics per digit')
    ax_metrics.set_xticks(x + width)
    ax_metrics.set_xticklabels(digits)
    ax_metrics.legend()
    ax_metrics.grid(True, alpha=0.3, axis='y')
    
    # Probability distributions for each digit
    for idx, digit in enumerate(digits):
        row = 1 + idx // 3
        col = idx % 3
        if row >= 3:
            break
        
        ax = fig.add_subplot(gs[row, col])
        probs = all_results[digit]['probs']
        pred = all_results[digit]['prediction']
        
        colors_prob = ['lime' if i == digit else 'red' if i == pred else 'cyan' 
                      for i in range(len(probs))]
        ax.bar(range(len(probs)), probs, color=colors_prob, alpha=0.7)
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.set_title(f'Digit {digit}, predicted: {pred} ({all_results[digit]["confidence"]:.2%})')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Summary statistics
    ax_summary = fig.add_subplot(gs[2, 2])
    correct = sum(1 for d, r in all_results.items() if r['prediction'] == d)
    accuracy = correct / len(all_results)
    avg_conf = np.mean([r['confidence'] for r in all_results.values()])
    avg_spikes = np.mean([r['metrics']['total_spikes'] for r in all_results.values()])
    avg_rate = np.mean([r['metrics']['firing_rate'] for r in all_results.values()])
    
    summary_text = f"{network_name} network summary\n\n"
    summary_text += f"Samples analyzed: {len(all_results)}\n"
    summary_text += f"Accuracy:         {accuracy:.1%} ({correct}/{len(all_results)})\n"
    summary_text += f"Avg confidence:   {avg_conf:.3f}\n"
    summary_text += f"Avg spikes:       {avg_spikes:.1f}\n"
    summary_text += f"Avg firing rate:  {avg_rate:.2f} Hz\n\n"
    summary_text += "Legend:\n"
    summary_text += "  Lime = Correct\n"
    summary_text += "  Red  = Incorrect"
    
    ax_summary.text(0.1, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=12, verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', alpha=0.8, 
                           edgecolor='cyan', linewidth=2))
    ax_summary.axis('off')
    
    fig.suptitle(f'Comprehensive analysis for the {network_name} network',
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def _process_sample_worker(args):
    """Worker function for parallel sample processing"""
    digit, sample_idx, spike_trains, network_data, seed = args
    
    # Force NEST to reset in this worker process
    import nest
    nest.ResetKernel()
    
    print(f"  [Worker] Starting digit {digit} (sample {sample_idx})...", flush=True)
    
    spike_times = create_spike_times(spike_trains[sample_idx], seed=seed)
    neuron_ids, times = run_simulation(network_data, spike_times, verbose=False)
    prediction, confidence, probs = compute_prediction(neuron_ids, times, network_data)
    metrics = compute_metrics(neuron_ids, times, network_data)
    
    print(f"  [Worker] Completed digit {digit}, predicted: {prediction} ({confidence:.3f})", flush=True)
    
    return digit, {
        'sample_idx': sample_idx,
        'neuron_ids': neuron_ids,
        'times': times,
        'prediction': prediction,
        'confidence': confidence,
        'probs': probs,
        'metrics': metrics
    }


def run_auto_mode(args, spike_trains, labels):
    """Run comprehensive automatic analysis"""
    print("\n" + "="*70)
    print("AUTO MODE - Comprehensive Analysis")
    
    # Limit parallelism for NEST stability
    max_workers = min(cpu_count(), 10, 6)  # Cap at 6 workers for NEST
    print(f"Using {max_workers} parallel workers (CPU count: {cpu_count()})")
    print("Note: Limited to 6 workers for NEST simulator stability")
    print("="*70)
    
    # Auto-select samples
    print("\nAuto-selecting samples...")
    selected_samples = auto_select_samples(labels, n_samples_per_digit=1)
    print(f"Selected samples: {selected_samples}")
    
    # Determine networks to analyze
    networks_to_run = []
    if args.compare_untrained:
        trained_path = args.checkpoint
        untrained_path = get_untrained_checkpoint(args.checkpoint)
        if not os.path.exists(untrained_path):
            print(f"ERROR: Untrained checkpoint not found: {untrained_path}")
            return
        print(f"\nLoading checkpoints:")
        print(f"  Trained: {trained_path}")
        print(f"  Untrained: {untrained_path}")
        network_data_trained = load_checkpoint(trained_path, is_trained=True)
        network_data_untrained = load_checkpoint(untrained_path, is_trained=False)
        networks_to_run.append(('trained', network_data_trained))
        networks_to_run.append(('untrained', network_data_untrained))
    elif args.untrained:
        untrained_path = get_untrained_checkpoint(args.checkpoint)
        if not os.path.exists(untrained_path):
            print(f"ERROR: Untrained checkpoint not found: {untrained_path}")
            return
        print(f"\nLoading UNTRAINED checkpoint: {untrained_path}")
        network_data_untrained = load_checkpoint(untrained_path, is_trained=False)
        networks_to_run.append(('untrained', network_data_untrained))
    else:
        print(f"\nLoading TRAINED checkpoint: {args.checkpoint}")
        network_data_trained = load_checkpoint(args.checkpoint, is_trained=True)
        networks_to_run.append(('trained', network_data_trained))
    
    # Run analysis for each network
    for network_name, network_data in networks_to_run:
        print(f"\n{'='*70}")
        print(f"ANALYZING {network_name.upper()} NETWORK")
        print(f"{'='*70}")
        
        # Prepare sample list
        args_list = [(digit, sample_indices[0], spike_trains, network_data, args.seed)
                     for digit, sample_indices in selected_samples.items()]
        
        # Process samples with timeout
        print(f"\n  Processing {len(args_list)} samples with {max_workers} workers...")
        print(f"  Progress will be shown below:")
        
        try:
            if max_workers > 1:
                with Pool(processes=max_workers) as pool:
                    # Use map_async with timeout
                    result = pool.map_async(_process_sample_worker, args_list)
                    # Wait with timeout (2 minutes per sample)
                    results_list = result.get(timeout=120 * len(args_list))
            else:
                results_list = [_process_sample_worker(a) for a in args_list]
        except Exception as e:
            print(f"\n  WARNING: Parallel processing issue: {e}")
            print(f"  Falling back to sequential processing...")
            results_list = []
            for a in args_list:
                try:
                    result = _process_sample_worker(a)
                    results_list.append(result)
                except Exception as worker_error:
                    print(f"  ERROR processing digit {a[0]}: {worker_error}")
                    continue
        
        # Convert to dictionary
        all_results = {}
        for digit, result in results_list:
            all_results[digit] = result
            print(f"    Digit {digit}: Predicted {result['prediction']} " +
                  f"(confidence: {result['confidence']:.3f}) | True: {digit}")
        
        # Generate summary statistics
        print(f"\n  Generating summary statistics for {network_name}...")
        correct = sum(1 for d, r in all_results.items() if r['prediction'] == d)
        accuracy = correct / len(all_results)
        avg_confidence = np.mean([r['confidence'] for r in all_results.values()])
        
        print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(all_results)})")
        print(f"  Average confidence: {avg_confidence:.3f}")
        
        # Create confusion matrix visualization
        fig = create_auto_summary(all_results, network_data, network_name)
        fig.savefig(f'{args.output_dir}/auto_summary_{network_name}.png',
                   dpi=150)
        plt.close(fig)
        print(f"  Saved: auto_summary_{network_name}.png")
        
        # Create per-digit visualizations if requested
        if not args.no_animation:
            for digit, result in all_results.items():
                sample_idx = result['sample_idx']
                fig = visualize_spatial_raster(result['neuron_ids'], result['times'],
                                              network_data, digit,
                                              prediction=result['prediction'],
                                              confidence=result['confidence'],
                                              untrained=(network_name=='untrained'))
                fig.savefig(f'{args.output_dir}/auto_{network_name}_digit{digit}_sample{sample_idx}.png',
                           dpi=150)
                plt.close(fig)
    
    # If comparing trained vs untrained, do additional comparisons
    if args.compare_untrained:
        print(f"\n{'='*70}")
        print("TRAINED vs UNTRAINED COMPARISON")
        print(f"{'='*70}")
        
        # Similarity analysis
        digit_pairs = {
            'Similar_1vs7': (1, 7),
            'Similar_3vs8': (3, 8),
            'Similar_4vs9': (4, 9),
            'Dissimilar_0vs1': (0, 1),
            'Dissimilar_2vs7': (2, 7),
        }
        
        # Get network data from the tuple
        network_data_trained = networks_to_run[0][1]
        network_data_untrained = networks_to_run[1][1]
        
        print("\nRunning similarity analysis...")
        results_trained = analyze_digit_similarity(network_data_trained, spike_trains,
                                                  labels, digit_pairs, seed=args.seed)
        results_untrained = analyze_digit_similarity(network_data_untrained, spike_trains,
                                                    labels, digit_pairs, seed=args.seed)
        
        fig = visualize_similarity_analysis(results_trained, network_data_trained)
        fig.savefig(f'{args.output_dir}/auto_similarity_trained.png',
                   dpi=150)
        plt.close(fig)
        
        fig = visualize_similarity_analysis(results_untrained, network_data_untrained)
        fig.savefig(f'{args.output_dir}/auto_similarity_untrained.png',
                   dpi=150)
        plt.close(fig)
        
        # Weight comparison
        print("\nComparing weight distributions...")
        fig = compare_weight_distributions(network_data_trained, network_data_untrained)
        fig.savefig(f'{args.output_dir}/auto_weight_comparison.png',
                   dpi=150)
        plt.close(fig)
        
        stats = compute_weight_statistics(network_data_trained, network_data_untrained)
        print(f"  Recurrent correlation (normalized): {stats['recurrent']['correlation_normalized']:.3f}")
        print(f"  Input correlation (normalized):     {stats['input']['correlation_normalized']:.3f}")
    
    print(f"\n{'='*70}")
    print("AUTO MODE COMPLETE")
    print(f"{'='*70}")
    print(f"Output saved to: {args.output_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize NEST GLIF3 network activity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sample analysis
  python %(prog)s 0
  
  # Compare two samples
  python %(prog)s 0 --sample2 5
  
  # Compare trained vs untrained
  python %(prog)s 0 --compare-untrained
  
  # Similarity analysis
  python %(prog)s --similarity-analysis --compare-untrained
  
  # Auto mode - comprehensive analysis
  python %(prog)s --auto --compare-untrained --checkpoint ckpt_51978-215.h5 --mnist mnist64.h5
        """)
    
    parser.add_argument('sample1', type=int, nargs='?', default=None,
                       help='First sample index (optional with --similarity-analysis or --auto)')
    parser.add_argument('--sample2', type=int, default=None, 
                       help='Second sample for comparison')
    parser.add_argument('--trials', type=int, default=1,
                       help='Number of trials to average (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--untrained', action='store_true',
                       help='Use untrained weights instead of trained')
    parser.add_argument('--compare-untrained', action='store_true',
                       help='Compare trained vs untrained networks')
    parser.add_argument('--similarity-analysis', action='store_true',
                       help='Run similarity analysis on digit pairs')
    parser.add_argument('--auto', action='store_true',
                       help='Auto mode: comprehensive analysis on all digits')
    parser.add_argument('--no-animation', action='store_true',
                       help='Skip animation generation')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory (default: current dir)')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_FILE,
                       help=f'Checkpoint file (default: {CHECKPOINT_FILE})')
    parser.add_argument('--mnist', type=str, default=MNIST_FILE,
                       help=f'MNIST data file (default: {MNIST_FILE})')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MNIST data
    print("Loading MNIST data...")
    spike_trains, labels = load_mnist_data(args.mnist)
    
    # ========================================================================
    # Auto Mode
    # ========================================================================
    if args.auto:
        run_auto_mode(args, spike_trains, labels)
        return
    parser = argparse.ArgumentParser(description='Visualize NEST GLIF3 network activity')
    parser.add_argument('sample1', type=int, nargs='?', default=None,
                       help='First sample index (optional if using --similarity-analysis)')
    parser.add_argument('--sample2', type=int, default=None, 
                       help='Second sample for comparison')
    parser.add_argument('--trials', type=int, default=1,
                       help='Number of trials to average (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--untrained', action='store_true',
                       help='Use untrained weights instead of trained')
    parser.add_argument('--compare-untrained', action='store_true',
                       help='Compare trained vs untrained networks')
    parser.add_argument('--similarity-analysis', action='store_true',
                       help='Run similarity analysis on digit pairs')
    parser.add_argument('--no-animation', action='store_true',
                       help='Skip animation generation')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory (default: current dir)')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_FILE,
                       help=f'Checkpoint file (default: {CHECKPOINT_FILE})')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MNIST data
    print("Loading MNIST data...")
    spike_trains, labels = load_mnist_data(MNIST_FILE)
    # ========================================================================
    # Similarity Analysis Mode
    # ========================================================================
    if args.similarity_analysis:
        print("\n" + "="*70)
        print("SIMILARITY ANALYSIS MODE")
        n_jobs = min(cpu_count(), 8, 4)  # Cap at 4 for NEST stability
        print(f"Using {n_jobs} parallel workers (limited for NEST stability)")
        print("="*70)
        
        # Define digit pairs
        digit_pairs = {
            'Similar_1vs7': (1, 7),
            'Similar_3vs8': (3, 8),
            'Similar_4vs9': (4, 9),
            'Similar_6vs8': (6, 8),
            'Dissimilar_0vs1': (0, 1),
            'Dissimilar_1vs6': (1, 6),
            'Dissimilar_2vs7': (2, 7),
            'Dissimilar_3vs5': (3, 5)
        }
        
        if args.compare_untrained:
            # Compare similarity analysis between trained and untrained
            trained_path = args.checkpoint
            untrained_path = get_untrained_checkpoint(args.checkpoint)
            
            print(f"\nLoading TRAINED network: {trained_path}")
            network_data_trained = load_checkpoint(trained_path, is_trained=True)
            
            print(f"Loading UNTRAINED network: {untrained_path}")
            network_data_untrained = load_checkpoint(untrained_path, is_trained=False)
            
            print("\nAnalyzing TRAINED network...")
            results_trained = analyze_digit_similarity(network_data_trained, spike_trains, 
                                                      labels, digit_pairs, seed=args.seed)            
            print("\nAnalyzing UNTRAINED network...")
            results_untrained = analyze_digit_similarity(network_data_untrained, spike_trains,
                                                        labels, digit_pairs, seed=args.seed)            
            print("\nGenerating similarity analysis visualizations...")
            
            fig = visualize_similarity_analysis(results_trained, network_data_trained)
            fig.savefig(f'{args.output_dir}/similarity_analysis_trained.png', 
                       dpi=150)
            plt.close(fig)
            print(f"  Saved: similarity_analysis_trained.png")
            
            fig = visualize_similarity_analysis(results_untrained, network_data_untrained)
            fig.savefig(f'{args.output_dir}/similarity_analysis_untrained.png',
                       dpi=150)
            plt.close(fig)
            print(f"  Saved: similarity_analysis_untrained.png")
            
            # Weight comparison
            print("\nComparing weight distributions...")
            fig = compare_weight_distributions(network_data_trained, network_data_untrained)
            fig.savefig(f'{args.output_dir}/weight_comparison.png',
                       dpi=150)
            plt.close(fig)
            print(f"  Saved: weight_comparison.png")
            
            stats = compute_weight_statistics(network_data_trained, network_data_untrained)
            print("\nWeight Statistics:")
            print(f"  Recurrent - Correlation (normalized): {stats['recurrent']['correlation_normalized']:.3f}")
            print(f"  Input     - Correlation (normalized): {stats['input']['correlation_normalized']:.3f}")
        else:
            # Single network similarity analysis
            if args.untrained:
                checkpoint_path = get_untrained_checkpoint(args.checkpoint)
                print(f"\nLoading UNTRAINED network: {checkpoint_path}")
                network_data = load_checkpoint(checkpoint_path, is_trained=False)
            else:
                print(f"\nLoading TRAINED network: {args.checkpoint}")
                network_data = load_checkpoint(args.checkpoint, is_trained=True)
            
            print("\nAnalyzing digit pair similarities...")
            results = analyze_digit_similarity(network_data, spike_trains, labels, 
                                             digit_pairs, seed=args.seed, n_jobs=n_jobs)
            
            print("\nGenerating visualizations...")
            fig = visualize_similarity_analysis(results, network_data)
            suffix = 'untrained' if args.untrained else 'trained'
            fig.savefig(f'{args.output_dir}/similarity_analysis_{suffix}.png',
                       dpi=150)
            plt.close(fig)
            print(f"  Saved: similarity_analysis_{suffix}.png")
        
        print("\nSimilarity analysis complete!")
        return
    
    # ========================================================================
    # Standard Analysis Mode (requires sample1)
    # ========================================================================
    if args.sample1 is None:
        parser.error("sample1 is required unless using --similarity-analysis or --auto")
    
    # Determine which checkpoint(s) to use
    if args.untrained:
        checkpoint_path = get_untrained_checkpoint(args.checkpoint)
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Untrained checkpoint not found: {checkpoint_path}")
            return
        print(f"Using UNTRAINED checkpoint: {checkpoint_path}")
        network_data = load_checkpoint(checkpoint_path, is_trained=False)
    elif args.compare_untrained:
        trained_path = args.checkpoint
        untrained_path = get_untrained_checkpoint(args.checkpoint)
        if not os.path.exists(untrained_path):
            print(f"ERROR: Untrained checkpoint not found: {untrained_path}")
            return
        print(f"Comparing TRAINED vs UNTRAINED networks")
        print(f"  Trained: {trained_path}")
        print(f"  Untrained: {untrained_path}")
        network_data_trained = load_checkpoint(trained_path, is_trained=True)
        network_data_untrained = load_checkpoint(untrained_path, is_trained=False)
    else:
        print(f"Using TRAINED checkpoint: {args.checkpoint}")
        network_data = load_checkpoint(args.checkpoint, is_trained=True)
    
    label1 = labels[args.sample1]
    
    # ========================================================================
    # Process sample 1
    # ========================================================================
    if args.compare_untrained:
        print(f"\n{'='*70}")
        print(f"TRAINED NETWORK - Sample {args.sample1} (Label: {label1})")
        print(f"{'='*70}")
        
        # Run trained network
        if args.trials > 1:
            print(f"Running {args.trials} trials...")
            sample_spikes1 = spike_trains[args.sample1]
            avg_results_trained = run_multiple_trials(network_data_trained, 
                                                     sample_spikes1, 
                                                     args.trials, args.seed)
            neuron_ids_trained = avg_results_trained['all_neuron_ids'][0]
            times_trained = avg_results_trained['all_times'][0]
        else:
            spike_times1 = create_spike_times(spike_trains[args.sample1], seed=args.seed)
            neuron_ids_trained, times_trained = run_simulation(network_data_trained, spike_times1)
        
        pred_t, conf_t, probs_t = compute_prediction(neuron_ids_trained, times_trained, 
                                                     network_data_trained)
        metrics_trained = compute_metrics(neuron_ids_trained, times_trained, 
                                         network_data_trained)
        print_metrics(metrics_trained, label1, pred_t, conf_t)
        
        print(f"\n{'='*70}")
        print(f"UNTRAINED NETWORK - Sample {args.sample1} (Label: {label1})")
        print(f"{'='*70}")
        
        # Run untrained network
        if args.trials > 1:
            print(f"Running {args.trials} trials...")
            avg_results_untrained = run_multiple_trials(network_data_untrained,
                                                       sample_spikes1,
                                                       args.trials, args.seed)
            neuron_ids_untrained = avg_results_untrained['all_neuron_ids'][0]
            times_untrained = avg_results_untrained['all_times'][0]
        else:
            spike_times1 = create_spike_times(spike_trains[args.sample1], seed=args.seed)
            neuron_ids_untrained, times_untrained = run_simulation(network_data_untrained, 
                                                                  spike_times1)
        
        pred_u, conf_u, probs_u = compute_prediction(neuron_ids_untrained, times_untrained,
                                                     network_data_untrained)
        metrics_untrained = compute_metrics(neuron_ids_untrained, times_untrained,
                                           network_data_untrained)
        print_metrics(metrics_untrained, label1, pred_u, conf_u)
        
        # Generate comparison visualizations
        print("\nCreating trained vs untrained comparison visualizations...")
        
        fig = visualize_trained_vs_untrained(neuron_ids_trained, times_trained,
                                             neuron_ids_untrained, times_untrained,
                                             network_data_trained, label1)
        fig.savefig(f'{args.output_dir}/trained_vs_untrained_sample{args.sample1}.png',
                   dpi=150)
        plt.close(fig)
        print(f"  Saved: trained_vs_untrained_sample{args.sample1}.png")
        
        fig = visualize_trained_vs_untrained_histogram(neuron_ids_trained, times_trained,
                                                       neuron_ids_untrained, times_untrained,
                                                       network_data_trained, label1)
        fig.savefig(f'{args.output_dir}/trained_vs_untrained_hist_sample{args.sample1}.png',
                   dpi=150)
        plt.close(fig)
        print(f"  Saved: trained_vs_untrained_hist_sample{args.sample1}.png")
        
        fig = visualize_trained_vs_untrained_metrics(metrics_trained, metrics_untrained, label1)
        fig.savefig(f'{args.output_dir}/trained_vs_untrained_metrics_sample{args.sample1}.png',
                   dpi=150)
        plt.close(fig)
        print(f"  Saved: trained_vs_untrained_metrics_sample{args.sample1}.png")
        
        # Weight comparison
        print("\nComparing weight distributions...")
        fig = compare_weight_distributions(network_data_trained, network_data_untrained)
        fig.savefig(f'{args.output_dir}/weight_comparison.png',
                   dpi=150)
        plt.close(fig)
        print(f"  Saved: weight_comparison.png")
        
        if not args.no_animation:
            create_trained_vs_untrained_animation(neuron_ids_trained, times_trained,
                                                 neuron_ids_untrained, times_untrained,
                                                 network_data_trained, label1,
                                                 f'{args.output_dir}/trained_vs_untrained_sample{args.sample1}.mp4')
        
    else:
        # Single network mode (trained OR untrained)
        print(f"\nProcessing Sample {args.sample1} (Label: {label1})")
        
        if args.trials > 1:
            print(f"Running {args.trials} trials...")
            sample_spikes1 = spike_trains[args.sample1]
            avg_results1 = run_multiple_trials(network_data, sample_spikes1, 
                                              args.trials, args.seed)
            
            neuron_ids1 = avg_results1['all_neuron_ids'][0]
            times1 = avg_results1['all_times'][0]
            
            # Visualize averaged results
            fig = visualize_averaged_results(avg_results1, network_data, label1,
                                            untrained=args.untrained)
            fig.savefig(f'{args.output_dir}/averaged_sample{args.sample1}.png', 
                       dpi=150)
            print(f"  Saved: averaged_sample{args.sample1}.png")
            plt.close(fig)
        else:
            spike_times1 = create_spike_times(spike_trains[args.sample1], seed=args.seed)
            neuron_ids1, times1 = run_simulation(network_data, spike_times1)
        
        # Compute prediction and metrics
        prediction1, confidence1, probs1 = compute_prediction(neuron_ids1, times1, network_data)
        metrics1 = compute_metrics(neuron_ids1, times1, network_data)
        print_metrics(metrics1, label1, prediction1, confidence1)
        
        # Generate visualizations for sample 1
        print("Creating visualizations for sample 1...")
        
        fig = visualize_spatial_raster(neuron_ids1, times1, network_data, label1,
                                       prediction=prediction1, confidence=confidence1,
                                       untrained=args.untrained)
        fig.savefig(f'{args.output_dir}/spatial_sample{args.sample1}.png', 
                   dpi=150)
        plt.close(fig)
        
        fig = visualize_population_activity(neuron_ids1, times1, network_data, label1,
                                           untrained=args.untrained)
        fig.savefig(f'{args.output_dir}/population_sample{args.sample1}.png', 
                   dpi=150)
        plt.close(fig)
        
        if not args.no_animation:
            create_animation(neuron_ids1, times1, network_data, label1,
                            f'{args.output_dir}/animation_sample{args.sample1}.mp4',
                            prediction=prediction1, confidence=confidence1,
                            untrained=args.untrained)
        
        # Process sample 2 if requested
        if args.sample2 is not None:
            label2 = labels[args.sample2]
            print(f"\nProcessing Sample {args.sample2} (Label: {label2})")
            
            if args.trials > 1:
                sample_spikes2 = spike_trains[args.sample2]
                avg_results2 = run_multiple_trials(network_data, sample_spikes2,
                                                  args.trials, args.seed + 1000)
                neuron_ids2 = avg_results2['all_neuron_ids'][0]
                times2 = avg_results2['all_times'][0]
            else:
                spike_times2 = create_spike_times(spike_trains[args.sample2], 
                                                 seed=args.seed + 1000)
                neuron_ids2, times2 = run_simulation(network_data, spike_times2)
            
            prediction2, confidence2, probs2 = compute_prediction(neuron_ids2, times2, network_data)
            metrics2 = compute_metrics(neuron_ids2, times2, network_data)
            print_metrics(metrics2, label2, prediction2, confidence2)
            
            # Generate comparison visualizations
            print("Creating comparison visualizations...")
            
            fig = visualize_activity_diff(neuron_ids1, times1, neuron_ids2, times2,
                                         network_data, label1, label2,
                                         untrained=args.untrained)
            fig.savefig(f'{args.output_dir}/diff_sample{args.sample1}_vs_{args.sample2}.png',
                       dpi=150)
            plt.close(fig)
            
            fig = visualize_diff_histogram(neuron_ids1, times1, neuron_ids2, times2,
                                          network_data, label1, label2,
                                          untrained=args.untrained)
            fig.savefig(f'{args.output_dir}/diff_hist_sample{args.sample1}_vs_{args.sample2}.png',
                       dpi=150)
            plt.close(fig)
            
            if not args.no_animation:
                create_diff_animation(neuron_ids1, times1, neuron_ids2, times2,
                                    network_data, label1, label2,
                                    f'{args.output_dir}/diff_animation_sample{args.sample1}_vs_{args.sample2}.mp4',
                                    untrained=args.untrained)
    
    print("\nAll visualizations complete!")
    print(f"Output saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
