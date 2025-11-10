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
    Directly extract variables from checkpoint without rebuilding the model.
    This uses the specific variable paths provided by the user.
    """
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_shape_map = reader.get_variable_to_shape_map()

    print("Extracting checkpoint variables...")
    checkpoint_vars = {}

    # Define the variable mappings based on the provided checkpoint structure
    variable_map = {
        # Input layer (layer_with_weights-0)
        'input_bkg_weights': 'model/layer_with_weights-0/_bkg_weights/.ATTRIBUTES/VARIABLE_VALUE',
        'input_indices': 'model/layer_with_weights-0/_indices/.ATTRIBUTES/VARIABLE_VALUE',
        'input_weights': 'model/layer_with_weights-0/_weights/.ATTRIBUTES/VARIABLE_VALUE',
        'input_weights_m': 'model/layer_with_weights-0/_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE',
        'input_weights_v': 'model/layer_with_weights-0/_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE',

        # Recurrent layer (layer_with_weights-1/cell) - GLIF3 neuron parameters
        'asc_amps': 'model/layer_with_weights-1/cell/asc_amps/.ATTRIBUTES/VARIABLE_VALUE',
        'current_factor': 'model/layer_with_weights-1/cell/current_factor/.ATTRIBUTES/VARIABLE_VALUE',
        'decay': 'model/layer_with_weights-1/cell/decay/.ATTRIBUTES/VARIABLE_VALUE',
        'e_l': 'model/layer_with_weights-1/cell/e_l/.ATTRIBUTES/VARIABLE_VALUE',
        'input_weight_positive': 'model/layer_with_weights-1/cell/input_weight_positive/.ATTRIBUTES/VARIABLE_VALUE',
        'param_g': 'model/layer_with_weights-1/cell/param_g/.ATTRIBUTES/VARIABLE_VALUE',
        'param_k': 'model/layer_with_weights-1/cell/param_k/.ATTRIBUTES/VARIABLE_VALUE',
        'psc_initial': 'model/layer_with_weights-1/cell/psc_initial/.ATTRIBUTES/VARIABLE_VALUE',
        'recurrent_indices': 'model/layer_with_weights-1/cell/recurrent_indices/.ATTRIBUTES/VARIABLE_VALUE',
        'recurrent_weight_positive': 'model/layer_with_weights-1/cell/recurrent_weight_positive/.ATTRIBUTES/VARIABLE_VALUE',
        'recurrent_weight_values': 'model/layer_with_weights-1/cell/recurrent_weight_values/.ATTRIBUTES/VARIABLE_VALUE',
        'recurrent_weight_values_m': 'model/layer_with_weights-1/cell/recurrent_weight_values/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE',
        'recurrent_weight_values_v': 'model/layer_with_weights-1/cell/recurrent_weight_values/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE',
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

            # Print first few values for verification
            if tensor.size <= 10:
                print(f"  {key}: {var_shape_map[var_path]} = {tensor.flatten()}")
            else:
                print(f"  {key}: {var_shape_map[var_path]} - sample values: {tensor.flatten()[:2]}")
        else:
            print(f"  WARNING: Variable not found: {var_path}")

    # Verify we got the expected shapes
    expected_shapes = {
        'input_bkg_weights': (207912,),
        'input_indices': (786405, 2),
        'input_weights': (786405,),
        'asc_amps': (51978, 2),
        'current_factor': (51978,),
        'decay': (51978,),
        'recurrent_indices': (14441124, 2),
        'recurrent_weight_values': (14441124,),
        'syn_decay': (51978, 4),
        'psc_initial': (51978, 4),
    }

    print("\nVerifying extracted variable shapes:")
    all_correct = True
    for key, expected_shape in expected_shapes.items():
        if key in checkpoint_vars:
            actual_shape = checkpoint_vars[key].shape
            matches = actual_shape == expected_shape
            status = "✓" if matches else "✗"
            print(f"  {status} {key}: expected {expected_shape}, got {actual_shape}")
            if not matches:
                all_correct = False
        else:
            print(f"  ✗ {key}: MISSING")
            all_correct = False

    if all_correct:
        print("\n✓ All checkpoint variables extracted successfully!")
    else:
        print("\n✗ Some variables have incorrect shapes or are missing")

    return checkpoint_vars


def convert_to_pynn_format(checkpoint_path, data_dir, output_h5, n_neurons=51978, n_input=17400):
    """
    Convert checkpoint to PyNN format using direct variable extraction.
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

    print(f'\nLoading trained weights from {checkpoint_path}')
    checkpoint_vars = extract_checkpoint_variables(checkpoint_path)

    if not checkpoint_vars:
        print("ERROR: No checkpoint variables loaded!")
        return

    print('\nPreparing PyNN conversion')

    # Use checkpoint variables for weights
    rec_indices = checkpoint_vars['recurrent_indices']
    rec_weights = checkpoint_vars['recurrent_weight_values']

    inp_indices = checkpoint_vars['input_indices']
    inp_weights = checkpoint_vars['input_weights']

    print(f"Using trained recurrent weights: mean={np.mean(rec_weights):.6f}, std={np.std(rec_weights):.6f}")
    print(f"Using trained input weights: mean={np.mean(inp_weights):.6f}, std={np.std(inp_weights):.6f}")

    # Get delays from original network (these aren't typically trained)
    rec_delays = network['synapses']['delays']

    # Convert indices to sources/targets format for recurrent synapses
    n_receptors = 4
    rec_sources = rec_indices[:, 1]  # Source neuron indices
    rec_targets = rec_indices[:, 0] // n_receptors  # Target neuron indices
    rec_receptor_types = rec_indices[:, 0] % n_receptors  # Receptor types

    # Convert indices to sources/targets format for input synapses
    inp_sources = inp_indices[:, 1]  # Source input indices
    inp_targets = inp_indices[:, 0] // n_receptors  # Target neuron indices
    inp_receptor_types = inp_indices[:, 0] % n_receptors  # Receptor types

    # Collect readout neurons
    readout_neuron_ids = network['localized_readout_neuron_ids_5']
    for i in range(6, 15):
        key = f'localized_readout_neuron_ids_{i}'
        if key in network:
            readout_neuron_ids = np.concatenate([readout_neuron_ids, network[key]], axis=1)

    # Prepare GLIF3 parameters from checkpoint
    print('\nExtracting GLIF3 parameters from checkpoint')
    glif3_params = {
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
        f.attrs['n_neurons'] = n_neurons
        f.attrs['n_input'] = n_input
        f.attrs['n_output'] = 10

        # Neuron information
        neuron_grp = f.create_group('neurons')
        neuron_grp.create_dataset('node_type_ids', data=network['node_type_ids'])
        neuron_grp.create_dataset('positions_x', data=network['x'])
        neuron_grp.create_dataset('positions_y', data=network['y'])
        neuron_grp.create_dataset('positions_z', data=network['z'])

        # Save GLIF3 parameters from checkpoint
        params_grp = neuron_grp.create_group('glif3_params')
        for key, val in glif3_params.items():
            params_grp.create_dataset(key, data=val)
            print(f"  Saved GLIF3 param: {key} {val.shape}")

        # Save recurrent synapses
        rec_grp = f.create_group('recurrent')
        rec_grp.create_dataset('sources', data=rec_sources.astype(np.int32))
        rec_grp.create_dataset('targets', data=rec_targets.astype(np.int32))
        rec_grp.create_dataset('weights', data=rec_weights.astype(np.float32))
        rec_grp.create_dataset('receptor_types', data=rec_receptor_types.astype(np.int32))
        rec_grp.create_dataset('delays', data=rec_delays.astype(np.float32))

        # Save input synapses
        inp_grp = f.create_group('input')
        inp_grp.create_dataset('sources', data=inp_sources.astype(np.int32))
        inp_grp.create_dataset('targets', data=inp_targets.astype(np.int32))
        inp_grp.create_dataset('weights', data=inp_weights.astype(np.float32))
        inp_grp.create_dataset('receptor_types', data=inp_receptor_types.astype(np.int32))

        # Save background weights
        bkg_grp = f.create_group('background')
        bkg_grp.create_dataset('weights', data=checkpoint_vars['input_bkg_weights'].astype(np.float32))

        # Save readout structure
        readout_grp = f.create_group('readout')
        readout_grp.create_dataset('neuron_ids', data=readout_neuron_ids.flatten().astype(np.int32))

    print(f'\n✓ Conversion complete:')
    print(f'  - {n_neurons} neurons')
    print(f'  - {len(rec_sources)} recurrent synapses')
    print(f'  - {len(inp_sources)} input synapses')
    print(f'  - GLIF3 parameters from checkpoint')
    print(f'  - Trained weights successfully applied')


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
