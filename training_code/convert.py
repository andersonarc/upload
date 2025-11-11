import os
import numpy as np
import h5py
import tensorflow as tf
import pickle as pkl
import load_sparse
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
tf.config.experimental.set_visible_devices([], 'GPU')

def load_tf_checkpoint(checkpoint_path, network):
    """Properly load trained weights from TensorFlow checkpoint"""
    # Recreate the model structure to match the checkpoint
    import classification_tools
    import models

    # Build a minimal model to match the checkpoint structure
    n_input = 17400
    n_output = 10
    seq_len = 600

    # Load the base network (same as during training)
    input_population, network, bkg_weights = load_sparse.load_billeh(
    n_input=n_input, n_neurons=51978, core_only=False,
    data_dir='v1cortex', seed=3000, connected_selection=True,
    n_output=10, neurons_per_output=30,
    use_rand_ini_w=False, use_dale_law=True, use_only_one_type=False,
    use_rand_connectivity=False, scale_w_e=-1,
    localized_readout=True, use_uniform_neuron_type=False)

    # Build the model (same as in multi_training.py)
    model = classification_tools.create_model(
        network, input_population, bkg_weights, seq_len=seq_len, n_input=n_input,
        n_output=n_output, dtype=tf.float32, down_sampled_decode_noise_path='v1cortex/additive_noise.mat',
        input_weight_scale=1.0,
        dampening_factor=0.5, gauss_std=0.28,
        train_recurrent=True, train_input=True,
        use_decoded_noise=True, neuron_output=True,
        output_mode='10class',
        neuron_model='GLIF3', use_dale_law=True, scale=[2, 2]
    )

    # Load the checkpoint
    checkpoint = tf.train.Checkpoint(model=model)

    # Try to restore the checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if tf.train.latest_checkpoint(checkpoint_dir):
        restored = checkpoint.restore(checkpoint_path)
        if restored:
            print(f"Successfully restored checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Could not restore checkpoint from {checkpoint_path}")
            return {}
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        print(np.shape(bkg_weights))
        n_neurons = 51978
        n_receptors = 4
        bkg_weights_by_receptor = bkg_weights.reshape((n_neurons, n_receptors))
        return { 'bkg_weights': bkg_weights_by_receptor }

    # Extract the trained weights
    model_vars = {}

    # Get recurrent weights
    for layer in model.layers:
        if hasattr(layer, 'cell') and hasattr(layer.cell, 'recurrent_weight_values'):
            model_vars['recurrent_weights'] = layer.cell.recurrent_weight_values.numpy()
            print(f"Loaded recurrent weights: {model_vars['recurrent_weights'].shape}")

        # Get input weights
        if hasattr(layer, 'cell') and hasattr(layer.cell, 'input_weight_values'):
            model_vars['input_weights'] = layer.cell.input_weight_values.numpy()
            print(f"Loaded input weights: {model_vars['input_weights'].shape}")

    # Also try direct variable access
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_shape_map = reader.get_variable_to_shape_map()

    print("Available variables in checkpoint:")
    for var_name in sorted(var_shape_map.keys()):
        if '.ATTRIBUTES' not in var_name:
            print(f"  {var_name}: {var_shape_map[var_name]}")

            # Extract weights based on naming patterns
            if 'sparse_recurrent_weights' in var_name:
                if 'recurrent_weights' not in model_vars:
                    model_vars['recurrent_weights'] = reader.get_tensor(var_name)
                    print(f"  -> Loaded as recurrent_weights")
            elif 'sparse_input_weights' in var_name:
                if 'input_weights' not in model_vars:
                    model_vars['input_weights'] = reader.get_tensor(var_name)
                    print(f"  -> Loaded as input_weights")
            elif 'prediction' in var_name and 'kernel' in var_name:
                model_vars['readout_weights'] = reader.get_tensor(var_name)
                print(f"  -> Loaded as readout_weights")
            elif 'prediction' in var_name and 'bias' in var_name:
                model_vars['readout_bias'] = reader.get_tensor(var_name)
                print(f"  -> Loaded as readout_bias")

    # Load background weights from checkpoint
    bkg_weights_var = 'model/layer_with_weights-0/_bkg_weights/.ATTRIBUTES/VARIABLE_VALUE'
    if bkg_weights_var in var_shape_map:
        bkg_weights_flat = reader.get_tensor(bkg_weights_var)
        print(f"Loaded background weights: {bkg_weights_flat.shape}")

        # Split into 4 by receptor type
        n_neurons = 51978
        n_receptors = 4
        bkg_weights_by_receptor = bkg_weights_flat.reshape((n_neurons, n_receptors))
        model_vars['bkg_weights'] = bkg_weights_by_receptor
        print(f"Split into receptor types: {bkg_weights_by_receptor.shape}")

    return model_vars

def convert_to_pynn_format(checkpoint_path, data_dir, output_h5, n_neurons=51978, n_input=17400):
    print(f'Loading network structure from {data_dir}')
    input_population, network, bkg_weights = load_sparse.load_billeh(
    n_input=n_input, n_neurons=n_neurons, core_only=False,
    data_dir=data_dir, seed=3000, connected_selection=True,
    n_output=10, neurons_per_output=30,
    use_rand_ini_w=False, use_dale_law=True, use_only_one_type=False,
    use_rand_connectivity=False, scale_w_e=-1,
    localized_readout=True, use_uniform_neuron_type=False)


    print(f'Loading trained weights from {checkpoint_path}')
    model_vars = load_tf_checkpoint(checkpoint_path, network)

    # Verify weights were loaded
    if not model_vars:
        print("WARNING: No trained weights loaded - using untrained weights!")
    else:
        print("Successfully loaded trained weights:")
        for key, value in model_vars.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

    print('Extracting GLIF3 parameters')
    node_params = network['node_params']

    print('Processing connectivity')

    # Use original variable names and structure
    rec_indices = network['synapses']['indices']

    # Use trained weights if available, otherwise use initial weights
    if 'recurrent_weights' in model_vars:
        rec_weights = model_vars['recurrent_weights']
        print(f"Using TRAINED recurrent weights: mean={np.mean(rec_weights):.6f}, std={np.std(rec_weights):.6f}")
    else:
        rec_weights = network['synapses']['weights']
        print(f"Using UNTRAINED recurrent weights: mean={np.mean(rec_weights):.6f}, std={np.std(rec_weights):.6f}")

    rec_delays = network['synapses']['delays']

    inp_indices = input_population['indices']

    # Use trained input weights if available
    if 'input_weights' in model_vars:
        inp_weights = model_vars['input_weights']
        print(f"Using TRAINED input weights: mean={np.mean(inp_weights):.6f}, std={np.std(inp_weights):.6f}")
    else:
        inp_weights = input_population['weights']
        print(f"Using UNTRAINED input weights: mean={np.mean(inp_weights):.6f}, std={np.std(inp_weights):.6f}")

    # Convert indices to sources/targets format for recurrent synapses
    n_receptors = 4
    rec_sources = rec_indices[:, 1]  # Source neuron indices
    rec_targets = rec_indices[:, 0] // n_receptors  # Target neuron indices
    rec_receptor_types = rec_indices[:, 0] % n_receptors  # Receptor types

    # Convert indices to sources/targets format for input synapses
    inp_sources = inp_indices[:, 1]  # Source input indices
    inp_targets = inp_indices[:, 0] // n_receptors  # Target neuron indices
    inp_receptor_types = inp_indices[:, 0] % n_receptors  # Receptor types

    # Collect readout neurons - keep original structure
    readout_neuron_ids = network['localized_readout_neuron_ids_5']
    for i in range(6, 15):
        key = f'localized_readout_neuron_ids_{i}'
        if key in network:
            readout_neuron_ids = np.concatenate([readout_neuron_ids, network[key]], axis=1)

    print(f'Saving to {output_h5}')
    with h5py.File(output_h5, 'w') as f:
        # Keep original attribute names
        f.attrs['n_neurons'] = n_neurons
        f.attrs['n_input'] = n_input
        f.attrs['n_output'] = 10

        # Keep original group and dataset names
        neuron_grp = f.create_group('neurons')
        neuron_grp.create_dataset('node_type_ids', data=network['node_type_ids'])
        neuron_grp.create_dataset('positions_x', data=network['x'])
        neuron_grp.create_dataset('positions_y', data=network['y'])
        neuron_grp.create_dataset('positions_z', data=network['z'])

        # Keep original parameter names
        params_grp = neuron_grp.create_group('glif3_params')
        for key, val in node_params.items():
            params_grp.create_dataset(key, data=val)

        # Save recurrent synapses in the expected format
        rec_grp = f.create_group('recurrent')
        rec_grp.create_dataset('sources', data=rec_sources.astype(np.int32))
        rec_grp.create_dataset('targets', data=rec_targets.astype(np.int32))
        rec_grp.create_dataset('weights', data=rec_weights.astype(np.float32))
        rec_grp.create_dataset('receptor_types', data=rec_receptor_types.astype(np.int32))
        rec_grp.create_dataset('delays', data=rec_delays.astype(np.float32))

        # Save input synapses in the expected format
        inp_grp = f.create_group('input')
        inp_grp.create_dataset('sources', data=inp_sources.astype(np.int32))
        inp_grp.create_dataset('targets', data=inp_targets.astype(np.int32))
        inp_grp.create_dataset('weights', data=inp_weights.astype(np.float32))
        inp_grp.create_dataset('receptor_types', data=inp_receptor_types.astype(np.int32))

        # Save background weights by receptor type
        if 'bkg_weights' in model_vars:
            print('Saving background weights by receptor type')
            inp_grp.create_dataset('bkg_weights', data=model_vars['bkg_weights'].astype(np.float32))

        # Keep original readout structure
        readout_grp = f.create_group('readout')
        readout_grp.create_dataset('neuron_ids', data=readout_neuron_ids.flatten().astype(np.int32))

        # Add readout weights if available
        if 'readout_weights' in model_vars:
            print('Saving readout weights')
            readout_grp.create_dataset('readout_weights', data=model_vars['readout_weights'])
        if 'readout_bias' in model_vars:
            print('Saving readout bias')
            readout_grp.create_dataset('readout_bias', data=model_vars['readout_bias'])

    print(f'Conversion complete: {n_neurons} neurons, {len(rec_sources)} recurrent synapses, {len(inp_sources)} input synapses')
    if model_vars:
        print("✓ Trained weights successfully applied")
    else:
        print("✗ Using untrained weights - check checkpoint path")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='network.h5')
    parser.add_argument('--neurons', type=int, default=51978)
    parser.add_argument('--n_input', type=int, default=17400)
    args = parser.parse_args()

    convert_to_pynn_format(args.checkpoint, args.data_dir, args.output, args.neurons, args.n_input)
