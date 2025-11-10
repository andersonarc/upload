import os
import numpy as np
import h5py
import tensorflow as tf
import pickle as pkl
import load_sparse

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
tf.config.experimental.set_visible_devices([], 'GPU')


def extract_checkpoint_variables(checkpoint_path):
    """
    Extract all variables from checkpoint using direct extraction.
    Checkpoint data takes priority over network structure.
    """
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_shape_map = reader.get_variable_to_shape_map()

    print("Extracting checkpoint variables...")
    checkpoint_vars = {}

    # Define the variable mappings from checkpoint structure
    variable_map = {
        # Input layer (layer_with_weights-0)
        'input_bkg_weights': 'model/layer_with_weights-0/_bkg_weights/.ATTRIBUTES/VARIABLE_VALUE',
        'input_indices': 'model/layer_with_weights-0/_indices/.ATTRIBUTES/VARIABLE_VALUE',
        'input_weights': 'model/layer_with_weights-0/_weights/.ATTRIBUTES/VARIABLE_VALUE',

        # Recurrent layer weights and connectivity
        'recurrent_indices': 'model/layer_with_weights-1/cell/recurrent_indices/.ATTRIBUTES/VARIABLE_VALUE',
        'recurrent_weight_values': 'model/layer_with_weights-1/cell/recurrent_weight_values/.ATTRIBUTES/VARIABLE_VALUE',

        # GLIF3 neuron parameters from checkpoint
        'asc_amps': 'model/layer_with_weights-1/cell/asc_amps/.ATTRIBUTES/VARIABLE_VALUE',
        'current_factor': 'model/layer_with_weights-1/cell/current_factor/.ATTRIBUTES/VARIABLE_VALUE',
        'decay': 'model/layer_with_weights-1/cell/decay/.ATTRIBUTES/VARIABLE_VALUE',
        'e_l': 'model/layer_with_weights-1/cell/e_l/.ATTRIBUTES/VARIABLE_VALUE',
        'param_g': 'model/layer_with_weights-1/cell/param_g/.ATTRIBUTES/VARIABLE_VALUE',
        'param_k': 'model/layer_with_weights-1/cell/param_k/.ATTRIBUTES/VARIABLE_VALUE',
        'psc_initial': 'model/layer_with_weights-1/cell/psc_initial/.ATTRIBUTES/VARIABLE_VALUE',
        'syn_decay': 'model/layer_with_weights-1/cell/syn_decay/.ATTRIBUTES/VARIABLE_VALUE',
        't_ref': 'model/layer_with_weights-1/cell/t_ref/.ATTRIBUTES/VARIABLE_VALUE',
        'v_reset': 'model/layer_with_weights-1/cell/v_reset/.ATTRIBUTES/VARIABLE_VALUE',
        'v_th': 'model/layer_with_weights-1/cell/v_th/.ATTRIBUTES/VARIABLE_VALUE',
        'voltage_offset': 'model/layer_with_weights-1/cell/voltage_offset/.ATTRIBUTES/VARIABLE_VALUE',
        'voltage_scale': 'model/layer_with_weights-1/cell/voltage_scale/.ATTRIBUTES/VARIABLE_VALUE',
    }

    # Extract each variable
    for key, var_path in variable_map.items():
        if var_path in var_shape_map:
            tensor = reader.get_tensor(var_path)
            checkpoint_vars[key] = tensor
            print(f"  ✓ {key}: {var_shape_map[var_path]}")
        else:
            print(f"  ✗ WARNING: Variable not found: {var_path}")

    # Verify critical shapes
    expected_shapes = {
        'input_indices': (786405, 2),
        'input_weights': (786405,),
        'recurrent_indices': (14441124, 2),
        'recurrent_weight_values': (14441124,),
    }

    print("\nVerifying shapes:")
    for key, expected_shape in expected_shapes.items():
        if key in checkpoint_vars:
            actual_shape = checkpoint_vars[key].shape
            status = "✓" if actual_shape == expected_shape else "✗"
            print(f"  {status} {key}: {actual_shape}")

    return checkpoint_vars


def convert_to_pynn_format(checkpoint_path, data_dir, output_h5, n_neurons=51978, n_input=17400):
    """
    Convert checkpoint to PyNN format - compatible with c2.py output structure.
    Checkpoint data takes priority. Uses network only for metadata not in checkpoint.
    """
    print(f'Loading network structure from {data_dir}')
    input_population, network, bkg_weights = load_sparse.load_billeh(
        n_input=n_input, n_neurons=n_neurons, core_only=False,
        data_dir=data_dir, seed=3000, connected_selection=True,
        n_output=10, neurons_per_output=30,
        use_rand_ini_w=False, use_dale_law=True, use_only_one_type=False,
        use_rand_connectivity=False, scale_w_e=-1,
        localized_readout=True, use_uniform_neuron_type=False
    )

    print(f'\nExtracting variables from {checkpoint_path}')
    checkpoint_vars = extract_checkpoint_variables(checkpoint_path)

    if not checkpoint_vars:
        print("ERROR: No checkpoint variables loaded!")
        return

    print('\nProcessing connectivity from checkpoint')

    # Use checkpoint data for connectivity and weights
    rec_indices = checkpoint_vars['recurrent_indices']
    rec_weights = checkpoint_vars['recurrent_weight_values']
    inp_indices = checkpoint_vars['input_indices']
    inp_weights = checkpoint_vars['input_weights']

    print(f"Recurrent weights: mean={np.mean(rec_weights):.6f}, std={np.std(rec_weights):.6f}")
    print(f"Input weights: mean={np.mean(inp_weights):.6f}, std={np.std(inp_weights):.6f}")

    # Delays from network (not typically in checkpoint)
    rec_delays = network['synapses']['delays']

    # Verify shapes match between checkpoint and network
    if rec_indices.shape != network['synapses']['indices'].shape:
        print(f"WARNING: Recurrent indices shape mismatch - checkpoint: {rec_indices.shape}, network: {network['synapses']['indices'].shape}")
    if inp_indices.shape != input_population['indices'].shape:
        print(f"WARNING: Input indices shape mismatch - checkpoint: {inp_indices.shape}, network: {input_population['indices'].shape}")

    # Convert indices to sources/targets format
    n_receptors = 4
    rec_sources = rec_indices[:, 1]
    rec_targets = rec_indices[:, 0] // n_receptors
    rec_receptor_types = rec_indices[:, 0] % n_receptors

    inp_sources = inp_indices[:, 1]
    inp_targets = inp_indices[:, 0] // n_receptors
    inp_receptor_types = inp_indices[:, 0] % n_receptors

    # Readout neurons from network
    readout_neuron_ids = network['localized_readout_neuron_ids_5']
    for i in range(6, 15):
        key = f'localized_readout_neuron_ids_{i}'
        if key in network:
            readout_neuron_ids = np.concatenate([readout_neuron_ids, network[key]], axis=1)

    # Build GLIF3 params dict from checkpoint (matching c2.py node_params structure)
    print('Building GLIF3 parameters from checkpoint')
    node_params = {
        'asc_amps': checkpoint_vars['asc_amps'],
        'current_factor': checkpoint_vars['current_factor'],
        'decay': checkpoint_vars['decay'],
        'e_l': checkpoint_vars['e_l'],
        'param_g': checkpoint_vars['param_g'],
        'param_k': checkpoint_vars['param_k'],
        'psc_initial': checkpoint_vars['psc_initial'],
        'syn_decay': checkpoint_vars['syn_decay'],
        't_ref': checkpoint_vars['t_ref'],
        'v_reset': checkpoint_vars['v_reset'],
        'v_th': checkpoint_vars['v_th'],
        'voltage_offset': checkpoint_vars['voltage_offset'],
        'voltage_scale': checkpoint_vars['voltage_scale'],
    }

    print(f'Saving to {output_h5}')
    with h5py.File(output_h5, 'w') as f:
        # File attributes (same as c2.py)
        f.attrs['n_neurons'] = n_neurons
        f.attrs['n_input'] = n_input
        f.attrs['n_output'] = 10

        # Neurons group (same structure as c2.py)
        neuron_grp = f.create_group('neurons')
        neuron_grp.create_dataset('node_type_ids', data=network['node_type_ids'])
        neuron_grp.create_dataset('positions_x', data=network['x'])
        neuron_grp.create_dataset('positions_y', data=network['y'])
        neuron_grp.create_dataset('positions_z', data=network['z'])

        # GLIF3 parameters (same structure as c2.py, but from checkpoint)
        params_grp = neuron_grp.create_group('glif3_params')
        for key, val in node_params.items():
            params_grp.create_dataset(key, data=val)

        # Recurrent synapses (same structure as c2.py)
        rec_grp = f.create_group('recurrent')
        rec_grp.create_dataset('sources', data=rec_sources.astype(np.int32))
        rec_grp.create_dataset('targets', data=rec_targets.astype(np.int32))
        rec_grp.create_dataset('weights', data=rec_weights.astype(np.float32))
        rec_grp.create_dataset('receptor_types', data=rec_receptor_types.astype(np.int32))
        rec_grp.create_dataset('delays', data=rec_delays.astype(np.float32))

        # Input synapses (same structure as c2.py)
        inp_grp = f.create_group('input')
        inp_grp.create_dataset('sources', data=inp_sources.astype(np.int32))
        inp_grp.create_dataset('targets', data=inp_targets.astype(np.int32))
        inp_grp.create_dataset('weights', data=inp_weights.astype(np.float32))
        inp_grp.create_dataset('receptor_types', data=inp_receptor_types.astype(np.int32))

        # Readout neurons (same structure as c2.py)
        readout_grp = f.create_group('readout')
        readout_grp.create_dataset('neuron_ids', data=readout_neuron_ids.flatten().astype(np.int32))

        # EXTRA: Background weights (not in c2.py)
        if 'input_bkg_weights' in checkpoint_vars:
            bkg_grp = f.create_group('background')
            bkg_grp.create_dataset('weights', data=checkpoint_vars['input_bkg_weights'].astype(np.float32))
            print(f'  + Background weights: {checkpoint_vars["input_bkg_weights"].shape}')

    print(f'\n✓ Conversion complete:')
    print(f'  - {n_neurons} neurons')
    print(f'  - {len(rec_sources)} recurrent synapses')
    print(f'  - {len(inp_sources)} input synapses')
    print(f'  - GLIF3 parameters from checkpoint')
    print(f'  - Trained weights from checkpoint')
    print(f'  - Compatible with c2.py format + extras')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert TF checkpoint to PyNN format using direct variable extraction')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to TensorFlow checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to network data directory')
    parser.add_argument('--output', type=str, default='network.h5', help='Output HDF5 file path')
    parser.add_argument('--neurons', type=int, default=51978, help='Number of neurons')
    parser.add_argument('--n_input', type=int, default=17400, help='Number of input units')
    args = parser.parse_args()

    convert_to_pynn_format(args.checkpoint, args.data_dir, args.output, args.neurons, args.n_input)
