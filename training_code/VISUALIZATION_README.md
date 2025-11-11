# Network Activity Visualization and Logging

This document describes the visualization and logging capabilities added to the V1 training code.

## Overview

We've added comprehensive visualization and logging tools to monitor network activity during training and inference, including:

- **Spike raster plots**: Visualize spiking activity across all neurons over time
- **Voltage traces**: Monitor membrane potentials for individual neurons
- **Input activity heatmaps**: Track input neuron activity
- **Population activity breakdowns**: Analyze activity by neuron population
- **Output predictions**: Visualize prediction confidence and accuracy
- **Detailed logging**: Track spikes, voltages, currents, and other state variables

## Files Added

### 1. `visualize_training.py`
Main visualization module containing:
- `NetworkActivityLogger`: Comprehensive activity logging class
- Plotting functions for raster plots, voltage traces, and population activity
- Utilities for creating batch prediction visualizations
- Network state logging functions

### 2. `demo_visualization.py`
Demo script showing how to use the visualization tools:
- Load pre-trained models or checkpoints
- Run inference with batch size 1
- Generate comprehensive visualizations
- Save activity logs for offline analysis

## Setup

### 1. Environment Setup

First, ensure you have miniforge3 installed and create the conda environment:

```bash
# Install miniforge3 (if not already installed)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3

# Create conda environment
conda env create -f conda_environment.yml

# Activate environment
conda activate mouse
```

### 2. Data Setup

Download the GLIF network weights:

```bash
cd training_code
wget https://cloud.tugraz.at/public.php/dav/files/JmDakasAHEqsA9J/GLIF_network.tar.gz
tar -xzf GLIF_network.tar.gz
```

The directory structure should be:
```
training_code/
├── GLIF_network/
│   ├── network/
│   │   ├── v1_node_types.csv
│   │   └── v1_nodes.h5
│   ├── input_dat.pkl
│   └── network_dat.pkl
├── visualize_training.py
├── demo_visualization.py
└── ... (other training files)
```

## Usage

### Basic Usage

Run the demo visualization script:

```bash
python demo_visualization.py \
    --data_dir ./GLIF_network \
    --batch_size 1 \
    --n_samples 10 \
    --log_dir visualization_logs
```

### With Pre-trained Checkpoint

If you have a trained checkpoint:

```bash
python demo_visualization.py \
    --data_dir ./GLIF_network \
    --checkpoint path/to/checkpoint.h5 \
    --batch_size 1 \
    --n_samples 10 \
    --log_dir visualization_logs
```

### In Your Own Code

Import and use the NetworkActivityLogger:

```python
import visualize_training

# Create logger
logger = visualize_training.NetworkActivityLogger(
    log_dir='my_logs',
    log_every_n_steps=100
)

# During training/inference loop:
for batch in dataset:
    inputs, labels = batch

    # Run model
    outputs = model(inputs)

    # Extract intermediate outputs (spikes, voltages)
    rsnn_layer = model.get_layer('rsnn')
    intermediate_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[rsnn_layer.output, model.output]
    )
    (spikes, voltages), predictions = intermediate_model(inputs)

    # Log activity
    logger.log_step(
        inputs=inputs,
        spikes=spikes,
        voltages=voltages,
        outputs=outputs,
        predictions=predictions,
        labels=labels
    )

# Save summary at end
logger.save_summary()
```

## Visualization Outputs

The visualization system creates several types of plots:

### 1. Raster Plots
- Shows spike times for all neurons
- X-axis: Time steps
- Y-axis: Neuron ID
- Black dots indicate spikes

### 2. Voltage Traces
- Displays membrane voltage over time for a subset of neurons
- Red dots mark spike times
- Helps diagnose voltage dynamics

### 3. Input Activity Heatmaps
- Shows input neuron activity over time
- Useful for understanding input patterns
- Color intensity indicates activity level

### 4. Population Activity
- Breaks down spiking activity by neuron population/type
- Bar plot showing total spikes per population
- Helps identify which populations are most active

### 5. Output Activity and Predictions
- Shows output neuron activity over time
- Displays prediction confidence for each class
- Highlights correct class in red

## Batch Size Configuration

**IMPORTANT**: For detailed visualization, you must use `batch_size=1`.

This is because:
- Individual sample analysis requires single samples
- Raster plots need specific spike times per sample
- Voltage traces are sample-specific

To set batch size to 1:

### In `multi_training.py`:
```python
# Change this line:
per_replica_batch_size = flags.batch_size  # Make sure flags.batch_size = 1

# Or directly:
per_replica_batch_size = 1
```

### In `demo_visualization.py`:
```bash
python demo_visualization.py --batch_size 1 ...
```

## Logging Network State

The `log_network_state()` function saves detailed information about:
- Model architecture summary
- Layer details and shapes
- Trainable parameters
- Cell types and state sizes

Usage:
```python
visualize_training.log_network_state(
    model=model,
    sample_input=sample_input,
    output_path='network_state.txt'
)
```

## TensorFlow Print Statements

The `models.py` file contains tf.print() statements in the SparseLayer that log:
- Input shape and values for neuron 0
- Input current shape and values for neuron 0

These are already active and will print during training/inference. To add more logging:

```python
# In BillehColumn.call() method, add:
tf.print('===== SPIKES neuron 0 =====')
tf.print(new_z[:, 0])

tf.print('===== VOLTAGE neuron 0 =====')
tf.print(new_v[:, 0])

tf.print('===== CURRENT neuron 0 =====')
tf.print(input_current[:, 0])
```

## Comparison with visualize_activity.py

The training code visualization is based on the approach in `new_things/visualize_activity.py`, which:
- Uses NEST simulator for spiking network simulation
- Loads H5 checkpoints with GLIF3 neuron parameters
- Creates detailed raster plots and activity heatmaps
- Computes predictions from output neuron spike counts
- Supports multiple trials with parallel execution

Our TensorFlow visualization adapts these concepts for:
- TensorFlow-based differentiable spiking networks
- Real-time logging during training
- Keras model integration
- GPU acceleration

## Example Workflow

1. **Train a model** (or use pre-trained checkpoint):
   ```bash
   python multi_training.py --batch_size 1 --data_dir ./GLIF_network --results_dir results
   ```

2. **Visualize network activity**:
   ```bash
   python demo_visualization.py \
       --data_dir ./GLIF_network \
       --checkpoint results/multi_training/checkpoint.h5 \
       --batch_size 1 \
       --n_samples 10 \
       --log_dir visualizations
   ```

3. **Analyze results**:
   - View raster plots: `visualizations/raster_step_*.png`
   - Check voltage traces: `visualizations/voltages_step_*.png`
   - Review predictions: `visualizations/sample_*/batch_predictions.png`
   - Read network state: `visualizations/network_state.txt`
   - Load saved data: `visualizations/activity_summary.npz`

## Troubleshooting

### Import Errors
Make sure you're in the `training_code` directory and the conda environment is activated:
```bash
cd training_code
conda activate mouse
```

### Memory Issues
If you run out of memory:
- Reduce `n_samples`
- Use `log_every_n_steps` > 1
- Reduce network size (`--neurons`)

### Slow Visualization
If visualization is slow:
- Reduce number of neurons in voltage trace plots
- Increase `log_every_n_steps`
- Use smaller image resolution in plots

## Citation

Based on the V1 model from:
- Chen, Guozhang, Franz Scherr, and Wolfgang Maass. "A data-based large-scale model for primary visual cortex enables brain-like robust and versatile visual processing." Science Advances 8.44 (2022): eabq7592.

Visualization approach inspired by:
- `new_things/visualize_activity.py` (NEST-based visualization)
