import os
import numpy as np
import h5py
import tensorflow as tf
import pickle as pkl
import load_sparse

def load_tf_checkpoint(checkpoint_path, network):
    checkpoint = tf.train.Checkpoint()
    model_vars = {}
    
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_shape_map = reader.get_variable_to_shape_map()
    
    for var_name in var_shape_map.keys():
        if 'sparse_recurrent_weights' in var_name:
            model_vars['recurrent_weights'] = reader.get_tensor(var_name)
        elif 'sparse_input_weights' in var_name:
            model_vars['input_weights'] = reader.get_tensor(var_name)
        elif 'projection' in var_name and 'kernel' in var_name:
            model_vars['readout_weights'] = reader.get_tensor(var_name)
        elif 'projection' in var_name and 'bias' in var_name:
            model_vars['readout_bias'] = reader.get_tensor(var_name)
    
    return model_vars

def convert_to_pynn_format(checkpoint_path, data_dir, output_h5, n_neurons=32768, n_input=17400):
    print(f'Loading network structure from {data_dir}')
    input_population, network, bkg_weights = load_sparse.load_billeh(
        n_input=n_input, n_neurons=n_neurons, core_only=False,
        data_dir=data_dir, seed=3000, connected_selection=False,
        n_output=10, neurons_per_output=16)
    
    print(f'Loading trained weights from {checkpoint_path}')
    model_vars = load_tf_checkpoint(checkpoint_path, network)
    
    print('Extracting GLIF3 parameters')
    node_params = network['node_params']
    
    print('Processing connectivity')
    rec_indices = network['synapses']['indices']
    rec_weights = model_vars.get('recurrent_weights', network['synapses']['weights'])
    rec_delays = network['synapses']['delays']
    
    n_receptors = 4
    rec_sources = rec_indices[:, 1] % n_neurons
    rec_targets = rec_indices[:, 0] // n_receptors
    rec_receptor_types = rec_indices[:, 0] % n_receptors
    
    inp_indices = input_population['indices']
    inp_weights = model_vars.get('input_weights', input_population['weights'])
    inp_sources = inp_indices[:, 1]
    inp_targets = inp_indices[:, 0] // n_receptors
    inp_receptor_types = inp_indices[:, 0] % n_receptors
    
    readout_neuron_ids = network['localized_readout_neuron_ids_5']
    for i in range(6, 15):
        key = f'localized_readout_neuron_ids_{i}'
        if key in network:
            readout_neuron_ids = np.concatenate([readout_neuron_ids, network[key]], axis=1)
    
    print(f'Saving to {output_h5}')
    with h5py.File(output_h5, 'w') as f:
        f.attrs['n_neurons'] = n_neurons
        f.attrs['n_input'] = n_input
        f.attrs['n_output'] = 10
        
        neuron_grp = f.create_group('neurons')
        neuron_grp.create_dataset('node_type_ids', data=network['node_type_ids'])
        neuron_grp.create_dataset('positions_x', data=network['x'])
        neuron_grp.create_dataset('positions_y', data=network['y'])
        neuron_grp.create_dataset('positions_z', data=network['z'])
        
        params_grp = neuron_grp.create_group('glif3_params')
        for key, val in node_params.items():
            params_grp.create_dataset(key, data=val)
        
        rec_grp = f.create_group('recurrent')
        rec_grp.create_dataset('sources', data=rec_sources.astype(np.int32))
        rec_grp.create_dataset('targets', data=rec_targets.astype(np.int32))
        rec_grp.create_dataset('weights', data=rec_weights.astype(np.float32))
        rec_grp.create_dataset('delays', data=rec_delays.astype(np.float32))
        rec_grp.create_dataset('receptor_types', data=rec_receptor_types.astype(np.int32))
        
        inp_grp = f.create_group('input')
        inp_grp.create_dataset('sources', data=inp_sources.astype(np.int32))
        inp_grp.create_dataset('targets', data=inp_targets.astype(np.int32))
        inp_grp.create_dataset('weights', data=inp_weights.astype(np.float32))
        inp_grp.create_dataset('receptor_types', data=inp_receptor_types.astype(np.int32))
        
        readout_grp = f.create_group('readout')
        readout_grp.create_dataset('neuron_ids', data=readout_neuron_ids.flatten().astype(np.int32))
        if 'readout_weights' in model_vars:
            readout_grp.create_dataset('weights', data=model_vars['readout_weights'])
            readout_grp.create_dataset('bias', data=model_vars['readout_bias'])
    
    print(f'Conversion complete: {n_neurons} neurons, {len(rec_sources)} recurrent synapses, {len(inp_sources)} input synapses')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='network.h5')
    parser.add_argument('--neurons', type=int, default=32768)
    parser.add_argument('--n_input', type=int, default=17400)
    args = parser.parse_args()
    
    convert_to_pynn_format(args.checkpoint, args.data_dir, args.output, args.neurons, args.n_input)
