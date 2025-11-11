#!/usr/bin/env python
"""Demo script showing how to use visualization with the training code

This script demonstrates:
1. Loading a pre-trained model or checkpoint
2. Running inference with batch size 1
3. Logging and visualizing network activity
4. Saving detailed activity logs

Usage:
    python demo_visualization.py --checkpoint path/to/checkpoint.h5 --data_dir path/to/data
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import training modules
import load_sparse
import classification_tools
import stim_dataset
import models
import visualize_training


def setup_argparse():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize V1 model activity')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory containing GLIF_network')
    parser.add_argument('--neurons', type=int, default=10000,
                        help='Number of neurons in the network')
    parser.add_argument('--n_input', type=int, default=17400,
                        help='Number of input neurons')
    parser.add_argument('--n_output', type=int, default=10,
                        help='Number of output classes')
    parser.add_argument('--seq_len', type=int, default=600,
                        help='Sequence length')

    # Visualization parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (MUST be 1 for visualization)')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--log_dir', type=str, default='visualization_logs',
                        help='Directory to save visualizations')

    # Dataset parameters
    parser.add_argument('--from_lgn', action='store_true',
                        help='Use LGN input')
    parser.add_argument('--sti_intensity', type=float, default=1.0,
                        help='Stimulus intensity')
    parser.add_argument('--im_slice', type=int, default=100,
                        help='Image slice size')
    parser.add_argument('--pre_delay', type=int, default=50,
                        help='Pre-stimulus delay')
    parser.add_argument('--post_delay', type=int, default=150,
                        help='Post-stimulus delay')
    parser.add_argument('--pre_chunks', type=int, default=4,
                        help='Pre-stimulus chunks')
    parser.add_argument('--post_chunks', type=int, default=1,
                        help='Post-stimulus chunks')
    parser.add_argument('--current_input', action='store_true', default=True,
                        help='Use current input')

    # Model construction parameters
    parser.add_argument('--core_only', action='store_true',
                        help='Use core neurons only')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--connected_selection', action='store_true',
                        help='Use connected selection')
    parser.add_argument('--neurons_per_output', type=int, default=30,
                        help='Neurons per output class')
    parser.add_argument('--use_rand_ini_w', action='store_true',
                        help='Use random initial weights')
    parser.add_argument('--use_dale_law', action='store_true',
                        help='Use Dale\'s law constraint')
    parser.add_argument('--use_only_one_type', action='store_true',
                        help='Use only one neuron type')
    parser.add_argument('--use_rand_connectivity', action='store_true',
                        help='Use random connectivity')
    parser.add_argument('--scale_w_e', type=float, default=1.0,
                        help='Excitatory weight scaling')
    parser.add_argument('--localized_readout', action='store_true',
                        help='Use localized readout')
    parser.add_argument('--use_uniform_neuron_type', action='store_true',
                        help='Use uniform neuron type')
    parser.add_argument('--input_weight_scale', type=float, default=1.0,
                        help='Input weight scale')
    parser.add_argument('--dampening_factor', type=float, default=0.3,
                        help='Dampening factor for spike pseudo-derivative')
    parser.add_argument('--gauss_std', type=float, default=0.5,
                        help='Gaussian std for spike pseudo-derivative')
    parser.add_argument('--train_recurrent', action='store_true',
                        help='Train recurrent weights')
    parser.add_argument('--train_input', action='store_true',
                        help='Train input weights')
    parser.add_argument('--use_decoded_noise', action='store_true',
                        help='Use decoded noise')
    parser.add_argument('--neuron_output', type=str, default='spike',
                        help='Neuron output type (spike or voltage)')
    parser.add_argument('--max_delay', type=int, default=5,
                        help='Maximum synaptic delay')
    parser.add_argument('--neuron_model', type=str, default='glif3',
                        help='Neuron model type')
    parser.add_argument('--scale', type=str, default='1.0,1.0',
                        help='Noise scales')

    return parser.parse_args()


def create_model_for_inference(args):
    """Create model for inference with visualization"""

    print("Loading network data...")
    input_population, network, bkg_weights = load_sparse.load_billeh(
        n_input=args.n_input,
        n_neurons=args.neurons,
        core_only=args.core_only,
        data_dir=args.data_dir,
        seed=args.seed,
        connected_selection=args.connected_selection,
        n_output=args.n_output,
        neurons_per_output=args.neurons_per_output,
        use_rand_ini_w=args.use_rand_ini_w,
        use_dale_law=args.use_dale_law,
        use_only_one_type=args.use_only_one_type,
        use_rand_connectivity=args.use_rand_connectivity,
        scale_w_e=args.scale_w_e,
        localized_readout=args.localized_readout,
        use_uniform_neuron_type=args.use_uniform_neuron_type
    )

    noise_scales = [float(a) for a in args.scale.split(',') if a != '']

    print("Creating model...")
    model = classification_tools.create_model(
        network, input_population, bkg_weights,
        seq_len=args.seq_len,
        n_input=args.n_input,
        n_output=args.n_output,
        dtype=tf.float32,
        input_weight_scale=args.input_weight_scale,
        dampening_factor=args.dampening_factor,
        gauss_std=args.gauss_std,
        train_recurrent=args.train_recurrent,
        train_input=args.train_input,
        lRout_pop='all',
        use_decoded_noise=args.use_decoded_noise,
        neuron_output=args.neuron_output,
        L2_factor=0,
        return_state=True,
        max_delay=args.max_delay,
        batch_size=args.batch_size,
        output_mode='10class',
        down_sampled_decode_noise_path=os.path.join(args.data_dir, 'additive_noise.mat'),
        neuron_model=args.neuron_model,
        use_dale_law=args.use_dale_law,
        scale=noise_scales,
    )

    model.build((args.batch_size, args.seq_len, args.n_input))

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        model.load_weights(args.checkpoint)
        print("Checkpoint loaded successfully!")

    return model


def create_dataset(args):
    """Create dataset for visualization"""

    print("Creating dataset...")
    dataset = stim_dataset.generate_pure_classification_data_set_from_generator(
        data_usage=1,  # Test set
        intensity=args.sti_intensity,
        im_slice=args.im_slice,
        pre_delay=args.pre_delay,
        post_delay=args.post_delay,
        current_input=args.current_input,
        dataset='mnist',
        pre_chunks=args.pre_chunks,
        resp_chunks=1,
        from_lgn=args.from_lgn,
        post_chunks=args.post_chunks
    ).take(args.n_samples).batch(args.batch_size)

    return dataset


def visualize_network_activity(model, dataset, args):
    """Run inference and visualize network activity"""

    print(f"Creating activity logger at {args.log_dir}...")
    logger = visualize_training.NetworkActivityLogger(
        log_dir=args.log_dir,
        log_every_n_steps=1  # Log every sample
    )

    # Get the RNN layer to extract intermediate outputs
    rsnn_layer = model.get_layer('rsnn')

    # Create a model that outputs both intermediate and final results
    intermediate_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[rsnn_layer.output, model.output]
    )

    print(f"Visualizing {args.n_samples} samples...")
    sample_count = 0

    for batch in dataset:
        inputs, labels = batch
        sample_count += 1

        print(f"Processing sample {sample_count}/{args.n_samples}...")

        # Run inference
        (spikes, voltages), predictions = intermediate_model.predict(inputs, verbose=0)

        # Extract outputs (assuming you have access to output neurons)
        # For now, we'll use the final predictions as outputs
        outputs = predictions[:, None, :]  # Add time dimension

        # Log and visualize
        logger.log_step(
            inputs=inputs,
            spikes=spikes,
            voltages=voltages,
            outputs=outputs,
            predictions=predictions,
            labels=labels
        )

        # Also create batch prediction visualization
        visualize_training.visualize_batch_predictions(
            spikes=spikes,
            labels=labels,
            predictions=predictions,
            output_dir=os.path.join(args.log_dir, f'sample_{sample_count}')
        )

        # Print prediction results
        pred_class = np.argmax(predictions[0])
        true_class = labels[0].numpy() if hasattr(labels[0], 'numpy') else labels[0]
        print(f"  True: {true_class}, Predicted: {pred_class}, "
              f"Confidence: {predictions[0, pred_class]:.3f}")

        # Log detailed spike statistics
        total_spikes = np.sum(spikes > 0.5)
        mean_rate = np.mean(spikes > 0.5)
        print(f"  Total spikes: {total_spikes}, Mean firing rate: {mean_rate:.4f}")

    # Save summary
    logger.save_summary()
    print(f"\nVisualization complete! Results saved to {args.log_dir}")

    # Log network state
    visualize_training.log_network_state(
        model=model,
        sample_input=inputs,
        output_path=os.path.join(args.log_dir, 'network_state.txt')
    )


def main():
    """Main function"""
    args = setup_argparse()

    # Ensure batch size is 1
    if args.batch_size != 1:
        print("WARNING: Batch size must be 1 for visualization. Setting batch_size=1")
        args.batch_size = 1

    # Create output directory
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 80)
    print("V1 Network Activity Visualization")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Log directory: {args.log_dir}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    # Create model
    model = create_model_for_inference(args)

    # Create dataset
    dataset = create_dataset(args)

    # Visualize activity
    visualize_network_activity(model, dataset, args)

    print("\nDone!")


if __name__ == '__main__':
    main()
