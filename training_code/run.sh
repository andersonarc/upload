#!/usr/bin/bash
python multi_training.py --neurons 32768 --n_epochs 16 --data_dir v1cortex --restore_from v1cortex-mnist-ckpt/results "$@"
