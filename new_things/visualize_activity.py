#!/usr/bin/env python
"""Visualize network activity from NEST GLIF3 simulation"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

def load_network(checkpoint_path):
    """Load network structure from checkpoint"""
    with h5py.File(checkpoint_path, 'r') as f:
        neurons = np.array(f['neurons/node_type_ids'])
        output_neurons = np.array(f['readout/neuron_ids'], dtype=int)
    return neurons, output_neurons

def run_simulation_and_record(target_index=0):
    """Run NEST simulation and record all spikes"""
    import nest

    print(f"Running simulation for sample {target_index}...")

    # Load network
    with h5py.File('ckpt_51978-153_NEW.h5', 'r') as f:
        neurons = np.array(f['neurons/node_type_ids'])

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
        output_neurons = np.array(f['readout/neuron_ids'], dtype=int)

    # Load spikes
    with h5py.File('mnist24.h5', 'r') as f:
        spike_trains = np.array(f['spike_trains'])
        labels = np.array(f['labels'])

    label = labels[target_index]
    print(f"  Label: {label}")

    # Create spike times
    sample_spikes = spike_trains[target_index]
    spike_times = {}
    for neuron_idx in range(sample_spikes.shape[1]):
        times = []
        for t_idx in range(sample_spikes.shape[0]):
            prob = np.clip((sample_spikes[t_idx, neuron_idx] / 1.3), 0.0, 1.0)
            if prob > np.random.rand():
                times.append(float(t_idx + 1.0))
        if len(times) > 0:
            spike_times[neuron_idx] = times

    # Setup NEST
    nest.ResetKernel()
    nest.resolution = 1.0

    # Create V1 neurons
    v1 = nest.Create('glif_psc', len(neurons))

    # Set GLIF3 parameters
    unique_types = np.unique(neurons)
    for ntype in unique_types:
        mask = neurons == ntype
        indices = np.where(mask)[0]

        C_m = float(glif_params['C_m'][ntype])
        E_L = float(glif_params['E_L'][ntype])
        V_reset = float(glif_params['V_reset'][ntype])
        V_th = float(glif_params['V_th'][ntype])
        asc_amps = tuple(float(x) for x in glif_params['asc_amps'][ntype])
        g = float(glif_params['g'][ntype])
        asc_decay = tuple(float(x) for x in glif_params['k'][ntype])
        t_ref = float(glif_params['t_ref'][ntype])
        tau_syn = tuple(float(x) for x in glif_params['tau_syn'][ntype])

        params = {
            'C_m': C_m, 'E_L': E_L, 'V_reset': V_reset, 'V_th': V_th,
            'V_m': E_L, 'g': g, 't_ref': t_ref, 'tau_syn': tau_syn,
            'asc_amps': asc_amps, 'asc_decay': asc_decay,
            'after_spike_currents': True,
            'spike_dependent_threshold': False,
            'adapting_threshold': False
        }

        for idx in indices:
            v1[int(idx)].set(params)

    # Calculate voltage scales
    vsc = np.array([glif_params['V_th'][neurons[i]] - glif_params['E_L'][neurons[i]]
                    for i in range(len(neurons))])

    # Create background (10 sources)
    bkg_generators = nest.Create('poisson_generator', 10)
    for gen in bkg_generators:
        gen.set({'rate': 10.0, 'start': 0.0, 'stop': 1000.0})

    v1_gids = np.array([n.global_id for n in v1])

    for receptor_idx in range(4):
        weights = bkg_weights[:, receptor_idx]
        mask = np.abs(weights) > 1e-10
        if np.sum(mask) == 0:
            continue

        neuron_indices = np.where(mask)[0]
        w_filt = weights[mask] / 10.0
        w_scaled = w_filt * vsc[neuron_indices]
        gids_filt = v1_gids[neuron_indices]

        for gen in bkg_generators:
            nest.Connect([gen.global_id] * len(gids_filt), gids_filt.tolist(),
                        conn_spec='one_to_one',
                        syn_spec={'weight': w_scaled.tolist(),
                                 'delay': np.ones(len(gids_filt)),
                                 'receptor_type': receptor_idx + 1})

    # Create LGN
    lgn = nest.Create('spike_generator', spike_trains.shape[2])
    for i, times in spike_times.items():
        lgn[i].set({'spike_times': times})

    lgn_gids = np.array([n.global_id for n in lgn])

    # Connect LGN -> V1
    src_arr = input_syns[:, 0].astype(int)
    tgt_arr = input_syns[:, 1].astype(int)
    w_arr = input_syns[:, 2]
    rtype_arr = input_syns[:, 3].astype(int)

    w_scaled = w_arr * vsc[tgt_arr]
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

    w_scaled = w_arr * vsc[tgt_arr]
    mask = np.abs(w_scaled) > 1e-10

    nest.Connect(v1_gids[src_arr[mask]].tolist(),
                v1_gids[tgt_arr[mask]].tolist(),
                conn_spec='one_to_one',
                syn_spec={'weight': w_scaled[mask],
                         'delay': np.maximum(d_arr[mask], 1.0),
                         'receptor_type': (rtype_arr[mask] + 1).tolist()})

    # Record ALL V1 spikes
    spike_rec = nest.Create('spike_recorder')
    nest.Connect(v1, spike_rec)

    # Run simulation
    print("Running 1000ms simulation...")
    nest.Simulate(1000.0)
    print("Done!")

    # Get spikes
    events = spike_rec.get('events')
    senders, times = events['senders'], events['times']

    # Convert global IDs to neuron indices
    gid_to_idx = {gid: idx for idx, gid in enumerate(v1_gids)}
    neuron_ids = np.array([gid_to_idx[s] for s in senders])

    return neuron_ids, times, neurons, output_neurons, label

def visualize_raster_square(neuron_ids, times, neurons, output_neurons, label,
                            time_window=(50, 150)):
    """Visualize network as a square grid"""
    n_neurons = len(neurons)
    grid_size = int(np.ceil(np.sqrt(n_neurons)))

    # Create neuron position mapping
    positions = {}
    for i in range(n_neurons):
        row = i // grid_size
        col = i % grid_size
        positions[i] = (col, grid_size - row - 1)  # Flip Y for plotting

    # Filter spikes to time window
    mask = (times >= time_window[0]) & (times < time_window[1])
    filtered_ids = neuron_ids[mask]
    filtered_times = times[mask]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Spatial activity
    spike_counts = np.zeros((grid_size, grid_size))
    for nid in filtered_ids:
        if nid in positions:
            x, y = positions[nid]
            spike_counts[y, x] += 1

    im = ax1.imshow(spike_counts, cmap='hot', aspect='auto', interpolation='nearest')
    ax1.set_title(f'Spike Counts ({time_window[0]}-{time_window[1]}ms) - Label: {label}')
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    plt.colorbar(im, ax=ax1, label='Spike count')

    # Highlight output neurons
    for out_id in output_neurons:
        if out_id in positions:
            x, y = positions[out_id]
            rect = Rectangle((x-0.5, y-0.5), 1, 1, fill=False,
                           edgecolor='cyan', linewidth=0.5, alpha=0.3)
            ax1.add_patch(rect)

    # Plot 2: Raster plot
    ax2.scatter(filtered_times, filtered_ids, s=0.5, c='black', alpha=0.5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Neuron ID')
    ax2.set_title(f'Raster Plot - {len(filtered_times)} spikes')
    ax2.set_xlim(time_window)

    plt.tight_layout()
    return fig

def visualize_by_population(neuron_ids, times, neurons, output_neurons, label,
                            time_window=(50, 150)):
    """Visualize activity per population"""
    # Filter to time window
    mask = (times >= time_window[0]) & (times < time_window[1])
    filtered_ids = neuron_ids[mask]

    # Count spikes per population
    unique_pops = np.unique(neurons)
    pop_counts = []
    pop_labels = []

    for pop_id in unique_pops:
        mask = neurons == pop_id
        pop_neuron_ids = np.where(mask)[0]
        count = np.sum(np.isin(filtered_ids, pop_neuron_ids))
        pop_counts.append(count)
        pop_labels.append(f'Pop {pop_id}')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(range(len(pop_counts)), pop_counts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Population ID')
    ax.set_ylabel('Spike count')
    ax.set_title(f'Spikes per Population ({time_window[0]}-{time_window[1]}ms) - Label: {label}')
    ax.set_xticks(range(0, len(pop_counts), max(1, len(pop_counts)//20)))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig

def create_animation(neuron_ids, times, neurons, label, output_file='activity.mp4'):
    """Create animation of network activity over time"""
    n_neurons = len(neurons)
    grid_size = int(np.ceil(np.sqrt(n_neurons)))

    # Create neuron position mapping
    positions = {}
    for i in range(n_neurons):
        row = i // grid_size
        col = i % grid_size
        positions[i] = (col, grid_size - row - 1)

    # Time bins (10ms windows)
    time_bins = np.arange(0, 1000, 10)

    # Pre-compute spike counts for each time bin
    spike_maps = []
    for t_start in time_bins:
        t_end = t_start + 10
        mask = (times >= t_start) & (times < t_end)
        bin_ids = neuron_ids[mask]

        spike_map = np.zeros((grid_size, grid_size))
        for nid in bin_ids:
            if nid in positions:
                x, y = positions[nid]
                spike_map[y, x] += 1
        spike_maps.append(spike_map)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Initial plot
    im = ax.imshow(spike_maps[0], cmap='hot', aspect='auto',
                  interpolation='nearest', vmin=0, vmax=np.max(spike_maps))
    ax.set_title(f'Network Activity - Label: {label}')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    plt.colorbar(im, ax=ax, label='Spikes per 10ms')

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       verticalalignment='top', fontsize=14, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    def animate(frame):
        im.set_data(spike_maps[frame])
        time_text.set_text(f'{time_bins[frame]}-{time_bins[frame]+10}ms')
        return [im, time_text]

    anim = animation.FuncAnimation(fig, animate, frames=len(time_bins),
                                  interval=100, blit=True)

    # Save animation
    print(f"Saving animation to {output_file}...")
    anim.save(output_file, writer='ffmpeg', fps=10, dpi=100)
    print(f"Animation saved!")
    plt.close()

if __name__ == '__main__':
    import sys

    np.random.seed(1)
    target_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Run simulation and get spikes
    neuron_ids, times, neurons, output_neurons, label = run_simulation_and_record(target_index)

    print(f"\nTotal spikes: {len(times)}")
    print(f"Active neurons: {len(np.unique(neuron_ids))}/{len(neurons)}")

    # Create visualizations
    print("\nCreating visualizations...")

    # Square grid visualization
    fig1 = visualize_raster_square(neuron_ids, times, neurons, output_neurons, label,
                                   time_window=(50, 150))
    fig1.savefig(f'activity_square_sample{target_index}.png', dpi=150)
    print(f"  Saved: activity_square_sample{target_index}.png")

    # Population visualization
    fig2 = visualize_by_population(neuron_ids, times, neurons, output_neurons, label,
                                   time_window=(50, 150))
    fig2.savefig(f'activity_population_sample{target_index}.png', dpi=150)
    print(f"  Saved: activity_population_sample{target_index}.png")

    # Animation (optional - takes time)
    if '--animate' in sys.argv:
        create_animation(neuron_ids, times, neurons, label,
                        output_file=f'activity_animation_sample{target_index}.mp4')

    plt.close('all')
    print("\nDone!")
