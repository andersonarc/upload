#!/usr/bin/env python
"""
Quick script to inspect the H5 dataset structure
"""

import h5py
import numpy as np

print("Inspecting spikes-128.h5...")

with h5py.File('spikes-128.h5', 'r') as f:
    print("\nAvailable keys:")
    for key in f.keys():
        data = f[key]
        if hasattr(data, 'shape'):
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
        else:
            print(f"  {key}: {data[()]}")

    # Check if there's any spatial information
    print("\nChecking for spatial/LGN metadata...")
    for key in f.attrs.keys():
        print(f"  Attribute {key}: {f.attrs[key]}")
