# NEST Simulator Port of SpiNNaker V1 Cortex Inference

This directory contains a port of the Mouse V1 Cortex inference code from SpiNNaker to NEST Simulator.

## Files

- `newclass.py` - Original SpiNNaker implementation
- `nest_inference.py` - Basic NEST port (incomplete, for reference)
- `run_nest_inference.py` - **Complete, runnable NEST implementation**
- `ckpt_51978-153.h5` - Neural network weights and structure (290MB, desplit from .split files)
- `mnist.h5` - MNIST spike train dataset (62MB)

## Requirements

The code has been tested with:
- Python 3.11
- NEST Simulator 3.9.0
- PyNN 0.12.4
- h5py
- numpy

## Installation

### Using Conda (Recommended)

```bash
# Install Miniforge3 if not already installed
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3

# Activate conda
source ~/miniforge3/bin/activate

# Install NEST and dependencies
conda install -c conda-forge nest-simulator python=3.11 pyNN h5py numpy
```

## Usage

### Basic Usage

```bash
# Run inference on sample 0 (default)
python run_nest_inference.py

# Run inference on a specific sample
TARGET_INDEX=2 python run_nest_inference.py
```

### Expected Output

The script will:
1. Load the network structure and weights from `ckpt_51978-153.h5`
2. Load spike trains from `mnist.h5`
3. Create V1 cortex populations (~230K neurons)
4. Create LGN input populations
5. Run 1000ms simulation
6. Analyze spike counts in readout neurons
7. Report predicted digit vs. actual label

Example output:
```
================================================================================
NEST Simulator V1 Cortex Inference - Sample 0
================================================================================
Loading network from ckpt_51978-153.h5...
  Neurons: 230924
  Recurrent synapses: 85319133
  Input synapses: 6006972
  ...
Running simulation for 1000 ms...
Simulation complete!

50-100 ms (target):
  Votes: [ 0.  1.  3.  1.  0.  5.  4.  2. 54.  0.]
  Prediction order: [0 1 3 8 4 2 7 6 5 8]
  Top prediction: 8, Expected: 8
  Correct: True
```

## Key Differences from SpiNNaker Version

1. **Simulator**: Uses `pyNN.nest` instead of `pyNN.spiNNaker`
2. **Neuron Model**: Uses standard `IF_curr_exp` instead of custom `GLIF3Curr`
   - GLIF3 parameters are converted to IF_curr_exp using the formula: tau_m = C_m / g
3. **Multi-receptor Synapses**: NEST's IF_curr_exp only supports 2 receptor types (excitatory/inhibitory)
   - The original GLIF3 model used 4 receptor types
   - Receptor types 0,2 -> excitatory; types 1,3 -> inhibitory
4. **No Hardware-Specific Settings**: Removed SpiNNaker-specific configurations like `set_number_of_neurons_per_core`

## File Preparation

### Desplitting the Network File

The `ckpt_51978-153.h5` file was split for version control. To recreate:

```bash
cd new_things
cat ckpt_51978-153.h5.dd_0-50.split \
    ckpt_51978-153.h5.dd_50-100.split \
    ckpt_51978-153.h5.dd_100-150.split \
    ckpt_51978-153.h5.dd_150-200.split \
    ckpt_51978-153.h5.dd_200-250.split \
    ckpt_51978-153.h5.dd_250-rest.split \
    > ckpt_51978-153.h5
```

## Performance Notes

- **Memory**: ~4-6GB RAM required
- **Runtime**: ~5-15 minutes depending on CPU
  - Population creation: 1-2 min
  - Synapse creation: 3-5 min
  - Simulation: 1-5 min
  - Total neurons: ~230K
  - Total synapses: ~91M

## Troubleshooting

### NEST Not Found
```bash
# Make sure NEST is installed
conda list nest-simulator

# Verify installation
python -c "import nest; print('NEST OK')"
```

### PyNN Import Error
```bash
# Install PyNN if missing
pip install pyNN
```

### Memory Error
```bash
# Monitor memory usage
python run_nest_inference.py 2>&1 | tee log.txt

# If out of memory, you may need a machine with more RAM
# or modify the script to reduce network size
```

## References

- [NEST Simulator](https://www.nest-simulator.org/)
- [PyNN Documentation](http://neuralensemble.org/docs/PyNN/)
- Original SpiNNaker implementation: `newclass.py`

## Citation

If you use this code, please cite the original work and acknowledge the NEST Simulator project.
