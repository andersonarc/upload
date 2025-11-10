#!/bin/bash
# Quick check of H5 file weight signs

python3 << 'EOF'
import h5py
import numpy as np

with h5py.File('ckpt_51978-153.h5', 'r') as f:
    rec_w = np.array(f['recurrent/weights'])
    inp_w = np.array(f['input/weights'])

    print("RAW H5 FILE CONTENTS:")
    print(f"Recurrent: {len(rec_w)} weights")
    print(f"  Positive: {np.sum(rec_w > 0)}")
    print(f"  Negative: {np.sum(rec_w < 0)}")
    print(f"  Zero: {np.sum(rec_w == 0)}")
    print(f"  Min: {rec_w.min():.6f}")
    print(f"  Max: {rec_w.max():.6f}")

    print(f"\nInput: {len(inp_w)} weights")
    print(f"  Positive: {np.sum(inp_w > 0)}")
    print(f"  Negative: {np.sum(inp_w < 0)}")
    print(f"  Zero: {np.sum(inp_w == 0)}")
    print(f"  Min: {inp_w.min():.6f}")
    print(f"  Max: {inp_w.max():.6f}")

    if np.sum(inp_w < 0) == 0:
        print("\n⚠️  PROBLEM: ALL INPUT WEIGHTS ARE NON-NEGATIVE IN H5!")
        print("This means c2.py didn't apply Dale's law signs correctly")

    if np.sum(rec_w < 0) == 0:
        print("\n⚠️  PROBLEM: ALL RECURRENT WEIGHTS ARE NON-NEGATIVE IN H5!")
        print("This means c2.py didn't apply Dale's law signs correctly")
EOF
