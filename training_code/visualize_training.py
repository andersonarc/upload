#!/usr/bin/env python
"""Visualization and logging utilities for training code

This module provides comprehensive visualization and logging capabilities
for the V1 model training, including:
- Network activity tracking (spikes, voltages, currents)
- Input/output logging
- Prediction analysis
- Raster plots and activity heatmaps
- Population activity breakdowns

Based on the visualization approach from new_things/visualize_activity.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import softmax
import tensorflow as tf
import h5py


class NetworkActivityLogger:
    """Logs network activity during training/inference"""

    def __init__(self, log_dir='logs', log_every_n_steps=100):
        """
        Args:
            log_dir: Directory to save logs and visualizations
            log_every_n_steps: Log activity every N steps
        """
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.step_counter = 0
        os.makedirs(log_dir, exist_ok=True)

        # Storage for activity data
        self.spike_history = []
        self.voltage_history = []
        self.input_history = []
        self.output_history = []
        self.prediction_history = []
        self.label_history = []

    def log_step(self, inputs, spikes, voltages, outputs, predictions, labels):
        """Log activity for a single step

        Args:
            inputs: Input tensor (batch, time, features)
            spikes: Spike tensor (batch, time, neurons)
            voltages: Voltage tensor (batch, time, neurons)
            outputs: Output tensor (batch, time, output_neurons)
            predictions: Predicted class probabilities
            labels: True labels
        """
        self.step_counter += 1

        if self.step_counter % self.log_every_n_steps == 0:
            # Convert tensors to numpy
            inputs_np = inputs.numpy() if hasattr(inputs, 'numpy') else inputs
            spikes_np = spikes.numpy() if hasattr(spikes, 'numpy') else spikes
            voltages_np = voltages.numpy() if hasattr(voltages, 'numpy') else voltages
            outputs_np = outputs.numpy() if hasattr(outputs, 'numpy') else outputs
            predictions_np = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
            labels_np = labels.numpy() if hasattr(labels, 'numpy') else labels

            # Store history
            self.spike_history.append(spikes_np)
            self.voltage_history.append(voltages_np)
            self.input_history.append(inputs_np)
            self.output_history.append(outputs_np)
            self.prediction_history.append(predictions_np)
            self.label_history.append(labels_np)

            # Create visualizations
            self._create_visualizations(
                inputs_np, spikes_np, voltages_np, outputs_np,
                predictions_np, labels_np
            )

    def _create_visualizations(self, inputs, spikes, voltages, outputs, predictions, labels):
        """Create and save visualization plots"""
        batch_idx = 0  # Visualize first sample in batch

        # 1. Raster plot
        fig = self._plot_raster(spikes[batch_idx], labels[batch_idx], predictions[batch_idx])
        fig.savefig(os.path.join(self.log_dir, f'raster_step_{self.step_counter}.png'), dpi=150)
        plt.close(fig)

        # 2. Voltage traces
        fig = self._plot_voltage_traces(voltages[batch_idx], spikes[batch_idx],
                                        labels[batch_idx], predictions[batch_idx])
        fig.savefig(os.path.join(self.log_dir, f'voltages_step_{self.step_counter}.png'), dpi=150)
        plt.close(fig)

        # 3. Input activity
        fig = self._plot_input_activity(inputs[batch_idx])
        fig.savefig(os.path.join(self.log_dir, f'input_step_{self.step_counter}.png'), dpi=150)
        plt.close(fig)

        # 4. Population activity
        fig = self._plot_population_activity(spikes[batch_idx], labels[batch_idx])
        fig.savefig(os.path.join(self.log_dir, f'population_step_{self.step_counter}.png'), dpi=150)
        plt.close(fig)

        # 5. Output neuron activity
        fig = self._plot_output_activity(outputs, predictions, labels)
        fig.savefig(os.path.join(self.log_dir, f'output_step_{self.step_counter}.png'), dpi=150)
        plt.close(fig)

        print(f"[Step {self.step_counter}] Saved visualizations to {self.log_dir}")

    def _plot_raster(self, spikes, label, prediction):
        """Create raster plot of spiking activity"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Find spike times and neuron indices
        time_steps, neuron_ids = np.where(spikes > 0.5)

        ax.scatter(time_steps, neuron_ids, s=1, c='black', alpha=0.6)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Neuron ID')
        ax.set_title(f'Network Activity - Label: {label}, Predicted: {prediction.argmax()}')
        ax.set_xlim(0, spikes.shape[0])
        ax.set_ylim(-10, spikes.shape[1] + 10)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        return fig

    def _plot_voltage_traces(self, voltages, spikes, label, prediction, n_neurons=20):
        """Plot voltage traces for a subset of neurons"""
        fig, axes = plt.subplots(4, 5, figsize=(20, 12))
        axes = axes.flatten()

        n_neurons = min(n_neurons, voltages.shape[1])
        neuron_indices = np.linspace(0, voltages.shape[1]-1, n_neurons, dtype=int)

        for i, neuron_idx in enumerate(neuron_indices):
            ax = axes[i]
            v_trace = voltages[:, neuron_idx]
            spike_times = np.where(spikes[:, neuron_idx] > 0.5)[0]

            ax.plot(v_trace, linewidth=0.8)
            if len(spike_times) > 0:
                ax.scatter(spike_times, v_trace[spike_times], c='red', s=20, alpha=0.6)
            ax.set_title(f'Neuron {neuron_idx}', fontsize=9)
            ax.set_ylabel('V (mV)', fontsize=8)
            ax.set_xlabel('Time', fontsize=8)
            ax.grid(True, alpha=0.2)

        fig.suptitle(f'Voltage Traces - Label: {label}, Predicted: {prediction.argmax()}',
                     fontsize=14)
        plt.tight_layout()
        return fig

    def _plot_input_activity(self, inputs):
        """Plot input activity heatmap"""
        fig, ax = plt.subplots(figsize=(14, 6))

        im = ax.imshow(inputs.T, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Input neuron')
        ax.set_title('Input Activity')
        plt.colorbar(im, ax=ax, label='Activity')

        plt.tight_layout()
        return fig

    def _plot_population_activity(self, spikes, label, n_populations=111):
        """Plot activity breakdown by neuron population"""
        fig, ax = plt.subplots(figsize=(14, 6))

        # Compute spike counts per neuron
        spike_counts = np.sum(spikes, axis=0)

        # Group into populations (assuming neurons are ordered by type)
        n_neurons = len(spike_counts)
        neurons_per_pop = n_neurons // n_populations

        pop_counts = []
        for pop_id in range(n_populations):
            start_idx = pop_id * neurons_per_pop
            end_idx = (pop_id + 1) * neurons_per_pop if pop_id < n_populations - 1 else n_neurons
            pop_counts.append(np.sum(spike_counts[start_idx:end_idx]))

        bars = ax.bar(range(len(pop_counts)), pop_counts, color='cyan', alpha=0.7)
        ax.set_title(f'Population Activity - Label: {label}')
        ax.set_xlabel('Population ID')
        ax.set_ylabel('Spike count')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_output_activity(self, outputs, predictions, labels):
        """Plot output neuron activity and predictions"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Aggregate over batch
        mean_output = np.mean(outputs, axis=0)  # (time, output_neurons)

        # Plot 1: Output neuron activity over time
        ax = axes[0]
        im = ax.imshow(mean_output.T, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Output neuron')
        ax.set_title('Output Neuron Activity (batch average)')
        plt.colorbar(im, ax=ax, label='Activity')

        # Plot 2: Prediction distribution
        ax = axes[1]
        mean_predictions = np.mean(predictions, axis=0)  # Average over batch
        x = np.arange(len(mean_predictions))
        bars = ax.bar(x, mean_predictions, color='lime', alpha=0.7)

        # Highlight correct class
        for i, label in enumerate(np.bincount(labels.flatten().astype(int))):
            if label > 0 and i < len(bars):
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(2)

        ax.set_xlabel('Class')
        ax.set_ylabel('Average prediction confidence')
        ax.set_title('Prediction Distribution')
        ax.set_xticks(x)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def save_summary(self):
        """Save summary statistics and aggregated data"""
        summary_path = os.path.join(self.log_dir, 'activity_summary.npz')

        np.savez(summary_path,
                 spike_history=np.array(self.spike_history) if self.spike_history else np.array([]),
                 voltage_history=np.array(self.voltage_history) if self.voltage_history else np.array([]),
                 input_history=np.array(self.input_history) if self.input_history else np.array([]),
                 output_history=np.array(self.output_history) if self.output_history else np.array([]),
                 prediction_history=np.array(self.prediction_history) if self.prediction_history else np.array([]),
                 label_history=np.array(self.label_history) if self.label_history else np.array([]))

        print(f"Saved activity summary to {summary_path}")


def add_logging_to_model(model, log_dir='logs', log_every_n_steps=100):
    """Add activity logging to an existing model

    Args:
        model: Keras model to add logging to
        log_dir: Directory to save logs
        log_every_n_steps: Log every N steps

    Returns:
        logger: NetworkActivityLogger instance
    """
    logger = NetworkActivityLogger(log_dir, log_every_n_steps)
    return logger


def create_logging_callback(logger):
    """Create a Keras callback for logging during training

    Args:
        logger: NetworkActivityLogger instance

    Returns:
        callback: Keras callback
    """
    class LoggingCallback(tf.keras.callbacks.Callback):
        def __init__(self, logger):
            super().__init__()
            self.logger = logger

        def on_batch_end(self, batch, logs=None):
            # This would need to be customized based on how you access
            # intermediate layer outputs during training
            pass

        def on_epoch_end(self, epoch, logs=None):
            self.logger.save_summary()

    return LoggingCallback(logger)


def visualize_batch_predictions(spikes, labels, predictions, output_dir='predictions'):
    """Create visualization of predictions for a batch

    Args:
        spikes: Spike tensor (batch, time, neurons)
        labels: True labels (batch,)
        predictions: Predicted class probabilities (batch, n_classes)
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    batch_size = spikes.shape[0]
    n_classes = predictions.shape[1]

    fig, axes = plt.subplots(2, min(5, batch_size), figsize=(20, 8))
    if batch_size == 1:
        axes = axes[:, None]

    for i in range(min(batch_size, 5)):
        # Raster plot
        ax = axes[0, i]
        time_steps, neuron_ids = np.where(spikes[i] > 0.5)
        ax.scatter(time_steps, neuron_ids, s=0.5, c='black', alpha=0.6)
        ax.set_title(f'Sample {i+1}\nLabel: {labels[i]}')
        ax.set_ylabel('Neuron ID')
        ax.set_xlim(0, spikes.shape[1])

        # Prediction bar plot
        ax = axes[1, i]
        pred_class = predictions[i].argmax()
        colors = ['lime' if j == labels[i] else 'red' if j == pred_class else 'gray'
                  for j in range(n_classes)]
        ax.bar(range(n_classes), predictions[i], color=colors, alpha=0.7)
        ax.set_xlabel('Class')
        ax.set_ylabel('Confidence')
        ax.set_title(f'Pred: {pred_class}')
        ax.set_xticks(range(n_classes))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'batch_predictions.png'), dpi=150)
    plt.close(fig)
    print(f"Saved batch predictions to {output_dir}/batch_predictions.png")


def log_network_state(model, sample_input, output_path='network_state.txt'):
    """Log detailed network state information

    Args:
        model: The trained model
        sample_input: Sample input to run through the network
        output_path: Path to save the log file
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NETWORK STATE SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Model summary
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n")

        # Layer details
        f.write("=" * 80 + "\n")
        f.write("LAYER DETAILS\n")
        f.write("=" * 80 + "\n\n")

        for layer in model.layers:
            f.write(f"Layer: {layer.name} ({layer.__class__.__name__})\n")
            f.write(f"  Input shape: {layer.input_shape}\n")
            f.write(f"  Output shape: {layer.output_shape}\n")

            if hasattr(layer, 'cell'):
                f.write(f"  Cell type: {layer.cell.__class__.__name__}\n")
                if hasattr(layer.cell, 'state_size'):
                    f.write(f"  State size: {layer.cell.state_size}\n")

            if layer.trainable_weights:
                f.write(f"  Trainable weights: {len(layer.trainable_weights)}\n")
                for weight in layer.trainable_weights:
                    f.write(f"    - {weight.name}: {weight.shape}\n")
            f.write("\n")

    print(f"Saved network state to {output_path}")


if __name__ == '__main__':
    print("Visualization module for V1 training code")
    print("Import this module and use NetworkActivityLogger to log network activity")
