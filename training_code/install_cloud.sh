#!/usr/bin/bash
# No need for conda since we're already using tensorflow 2.10
# conda create -n tf python
# conda activate tf
# conda install cudnn
# pip install tensorflow-gpu

# Packages
apt install -y git-lfs
pip install matplotlib pandas tqdm numpy scipy
pip install https://github.com/IGITUGraz/SimManager/archive/v0.8.3.zip
pip install git+https://github.com/franzscherr/bmtk@f_dev

# Weights
git clone https://huggingface.co/rnrz/v1cortex
cd v1cortex
git lfs install
git lfs pull
cd ..

# Checkpoint
git clone https://huggingface.co/rnrz/v1cortex-mnist-ckpts-2-7
mv v1cortex-mnist-ckpts-2-7 v1cortex-mnist-ckpt
cd v1cortex-mnist-ckpt
git lfs install
git lfs pull
cd ..

# Move files
mv v1cortex/temporal_kernels.pkl lgn_model
mv v1cortex/spontaneous_firing_rates.pkl lgn_model

mv EA_LGN.h5 .
mv v1cortex/many_small_stimuli.pkl .
mv v1cortex/alternate_small_stimuli.pkl .
