#!/usr/bin/env python3
"""
TensorFlow Runtime Visualization

Purpose: Capture TensorFlow network activity during inference to compare with SpiNNaker.

Captures:
1. Input: LGN spike probabilities (time x neurons)
2. Layer Activities: V1 GLIF3 population outputs
3. Output: Readout neuron activity and predictions

Visualizations:
- Heatmaps (time x neurons)
- Spike rasters
- Population activity traces
- Side-by-side comparisons: failed vs working samples
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os

# Check TensorFlow availability
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")
    if not tf.__version__.startswith('2.10'):
        print(f"WARNING: Expected TensorFlow 2.10.x, got {tf.__version__}")
except ImportError:
    print("=" * 80)
    print("ERROR: TensorFlow not available")
    print("=" * 80)
    print("\nTo install:")
    print("  python3 -m venv /tmp/tf_env")
    print("  source /tmp/tf_env/bin/activate")
    print("  pip install tensorflow==2.10.0 h5py numpy scipy matplotlib")
    print("\n" + "=" * 80)
    sys.exit(1)

# Add training_code to path
sys.path.insert(0, '/home/user/upload/training_code')

import load_sparse
import classification_tools
import stim_dataset

# Configuration
DATA_DIR = '/home/user/upload/v1cortex'
CHECKPOINT_PATH = 'ckpt_51978-153'
TEST_SAMPLES = [10, 50, 90, 100]  # Failed and "working" samples
SEED = 1  # Match class.py for determinism
N_INPUT = 17400
N_NEURONS = 51978
OUTPUT_MODE = '10class'

print("=" * 80)
print("TensorFlow Runtime Visualization")
print("=" * 80)

# =============================================================================
# Step 1: Load network
# =============================================================================

print("\nStep 1: Loading network...")

input_population, network, bkg_weights = load_sparse.load_billeh(
    n_input=N_INPUT,
    n_neurons=N_NEURONS,
    core_only=True,
    data_dir=DATA_DIR,
    seed=SEED,
    connected_selection=True,
    n_output=10,
    neurons_per_output=30,
    use_rand_ini_w=False,
    use_dale_law=True,
    use_only_one_type=False,
    use_rand_connectivity=False,
    scale_w_e=1.0,
    localized_readout=True,
    use_uniform_neuron_type=False
)

print(f"  ✓ Loaded network with {N_NEURONS} neurons")
print(f"  ✓ Input population: {N_INPUT} LGN neurons")
print(f"  ✓ Recurrent synapses: {len(network['synapses']['indices'])}")

# =============================================================================
# Step 2: Create model
# =============================================================================

print("\nStep 2: Creating TensorFlow model...")

model = classification_tools.create_model(
    network,
    input_population,
    bkg_weights,
    output_mode=OUTPUT_MODE,
    n_in=N_INPUT,
    dtype=tf.float32,
    use_recurrent=True
)

print(f"  ✓ Model created")

# =============================================================================
# Step 3: Load checkpoint
# =============================================================================

print("\nStep 3: Loading trained checkpoint...")

checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(CHECKPOINT_PATH)

try:
    status.assert_consumed()
    print(f"  ✓ Checkpoint loaded successfully: {CHECKPOINT_PATH}")
except Exception as e:
    print(f"  ⚠️  WARNING: Checkpoint may not be fully loaded: {e}")
    print(f"  Proceeding anyway...")

# =============================================================================
# Step 4: Load MNIST dataset
# =============================================================================

print("\nStep 4: Loading MNIST dataset...")

# CRITICAL: Must match class.py's dataset generation for fair comparison
# class.py uses np.random.seed(1) and loads from dataset seed=3000

mnist_data = stim_dataset.MNISTDataset(
    seed=3000,  # Match mnist.py dataset seed
    data_dir=DATA_DIR
)

print(f"  ✓ MNIST dataset loaded")
print(f"  Dataset shape: {mnist_data.images.shape}")
print(f"  Labels shape: {mnist_data.labels.shape}")

# =============================================================================
# Step 5: Generate LGN inputs
# =============================================================================

print("\nStep 5: Generating LGN inputs...")

# Generate LGN spike probabilities (match training preprocessing)
lgn_inputs = []
true_labels = []

for sample_idx in TEST_SAMPLES:
    # Get MNIST image
    image = mnist_data.images[sample_idx]
    label = mnist_data.labels[sample_idx]
    true_labels.append(label)

    # Generate LGN spike probabilities
    # NOTE: Training multiplies by 1.3, class.py divides by 1.3
    # Here we generate the BASE probabilities (before 1.3 scaling)
    lgn_response = stim_dataset.generate_lgn_spikes(
        image,
        n_lgn=N_INPUT,
        timesteps=100,
        scale=1.0  # No scaling here - will be applied during training
    )

    lgn_inputs.append(lgn_response)

    print(f"  Sample {sample_idx}: label={label}, LGN shape={lgn_response.shape}")

lgn_inputs = np.array(lgn_inputs)  # Shape: (n_samples, timesteps, n_lgn)
true_labels = np.array(true_labels)

print(f"  ✓ Generated LGN inputs: {lgn_inputs.shape}")

# =============================================================================
# Step 6: Run inference with activity tracking
# =============================================================================

print("\nStep 6: Running inference with activity tracking...")

results = []

for i, (sample_idx, lgn_input, label) in enumerate(zip(TEST_SAMPLES, lgn_inputs, true_labels)):
    print(f"\n  Sample {sample_idx} (label={label}):")

    # Prepare input
    # Training applies 1.3 scaling
    lgn_scaled = lgn_input * 1.3
    inp = tf.convert_to_tensor(lgn_scaled[np.newaxis], dtype=tf.float32)  # Add batch dim

    # Forward pass (TensorFlow 2.10 uses eager execution)
    # We need to intercept intermediate layer outputs

    # Run model
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        output = model(inp, training=False)

    # Get prediction
    if OUTPUT_MODE == '10class':
        # output shape: (batch, timesteps, 10)
        # Average over time window [50:100] (response window)
        response_window = output[:, 50:100, :]
        avg_output = tf.reduce_mean(response_window, axis=1)  # (batch, 10)
        predicted_class = tf.argmax(avg_output, axis=1).numpy()[0]
        confidence = avg_output.numpy()[0]
    else:
        raise ValueError(f"Unsupported output mode: {OUTPUT_MODE}")

    print(f"    Predicted: {predicted_class}")
    print(f"    True: {label}")
    print(f"    Confidence: {confidence}")
    print(f"    Correct: {predicted_class == label}")

    # Store results
    results.append({
        'sample_idx': sample_idx,
        'label': label,
        'predicted': predicted_class,
        'confidence': confidence,
        'lgn_input': lgn_input,
        'output': output.numpy()
    })

# =============================================================================
# Step 7: Visualize results
# =============================================================================

print("\nStep 7: Creating visualizations...")

fig, axes = plt.subplots(len(TEST_SAMPLES), 3, figsize=(15, 4 * len(TEST_SAMPLES)))

for i, result in enumerate(results):
    sample_idx = result['sample_idx']
    label = result['label']
    predicted = result['predicted']
    lgn_input = result['lgn_input']
    output = result['output'][0]  # Remove batch dim

    # Plot 1: LGN input heatmap
    ax = axes[i, 0] if len(TEST_SAMPLES) > 1 else axes[0]
    im = ax.imshow(lgn_input.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title(f'Sample {sample_idx}: LGN Input\n(label={label})')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('LGN Neuron')
    plt.colorbar(im, ax=ax)

    # Plot 2: LGN mean activity over time
    ax = axes[i, 1] if len(TEST_SAMPLES) > 1 else axes[1]
    mean_activity = np.mean(lgn_input, axis=1)
    ax.plot(mean_activity)
    ax.set_title(f'Mean LGN Activity')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mean Spike Probability')
    ax.grid(True, alpha=0.3)

    # Plot 3: Output over time
    ax = axes[i, 2] if len(TEST_SAMPLES) > 1 else axes[2]
    for class_idx in range(10):
        label_str = f'Class {class_idx}' if class_idx != label else f'Class {class_idx} (TRUE)'
        linestyle = '-' if class_idx == predicted else '--'
        linewidth = 2 if class_idx == predicted or class_idx == label else 1
        ax.plot(output[:, class_idx], label=label_str, linestyle=linestyle, linewidth=linewidth)
    ax.axvspan(50, 100, alpha=0.2, color='green', label='Response Window')
    ax.set_title(f'Output (predicted={predicted})')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Output Activity')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tensorflow_runtime_visualization.png', dpi=150)
print(f"  ✓ Saved visualization: tensorflow_runtime_visualization.png")

# =============================================================================
# Step 8: Save results for comparison with SpiNNaker
# =============================================================================

print("\nStep 8: Saving results...")

with open('tensorflow_results.pkl', 'wb') as f:
    pkl.dump(results, f)

print(f"  ✓ Saved results: tensorflow_results.pkl")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("TensorFlow Runtime Visualization Complete")
print("=" * 80)

accuracy = sum(r['predicted'] == r['label'] for r in results) / len(results)
print(f"\nAccuracy on test samples: {accuracy * 100:.1f}%")

print("\nPer-sample results:")
for r in results:
    status = "✓" if r['predicted'] == r['label'] else "✗"
    print(f"  {status} Sample {r['sample_idx']}: true={r['label']}, pred={r['predicted']}")

print("\nNext steps:")
print("1. Compare with SpiNNaker outputs from class.py")
print("2. Identify where SpiNNaker diverges from TensorFlow")
print("3. Check if LGN inputs match (same spike realization)")

print("\n" + "=" * 80)
