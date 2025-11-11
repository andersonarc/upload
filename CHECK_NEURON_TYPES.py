#!/usr/bin/env python3
"""
Check what neuron types belong to classes 7 and 8 that are always firing.
"""

import numpy as np
import h5py

print("Loading checkpoint...")
with h5py.File('ckpt_51978-153.h5', 'r') as file:
    neurons = np.array(file['neurons/node_type_ids'])
    output_neurons = np.array(file['readout/neuron_ids'])
    glif3_params = {}
    glif3_params['C_m'] = np.array(file['neurons/glif3_params/C_m'])
    glif3_params['E_L'] = np.array(file['neurons/glif3_params/E_L'])
    glif3_params['V_reset'] = np.array(file['neurons/glif3_params/V_reset'])
    glif3_params['V_th'] = np.array(file['neurons/glif3_params/V_th'])
    glif3_params['asc_amps'] = np.array(file['neurons/glif3_params/asc_amps'])
    glif3_params['k'] = np.array(file['neurons/glif3_params/k'])
    glif3_params['g'] = np.array(file['neurons/glif3_params/g'])
    glif3_params['tau_syn'] = np.array(file['neurons/glif3_params/tau_syn'])

    # Load background weights
    bkg_weights = np.array(file['input/bkg_weights'])

print("\n" + "="*80)
print("CHECKING NEURON TYPES FOR CLASSES 7 AND 8")
print("="*80)

# Class 7 neurons (indices 210-239)
class7_gids = output_neurons[210:240]
class7_types = [neurons[gid] for gid in class7_gids]
print(f"\nClass 7 neurons (indices 210-239):")
print(f"  GIDs: {class7_gids[:5]} ... {class7_gids[-5:]}")
print(f"  Types: {class7_types[:5]} ... {class7_types[-5:]}")
print(f"  Unique types: {np.unique(class7_types)}")

# Class 8 neurons (indices 240-269)
class8_gids = output_neurons[240:270]
class8_types = [neurons[gid] for gid in class8_gids]
print(f"\nClass 8 neurons (indices 240-269):")
print(f"  GIDs: {class8_gids[:5]} ... {class8_gids[-5:]}")
print(f"  Types: {class8_types[:5]} ... {class8_types[-5:]}")
print(f"  Unique types: {np.unique(class8_types)}")

# Check if GID 37955 (known to fire) is in class 8
if 37955 in class8_gids:
    idx = np.where(class8_gids == 37955)[0][0]
    print(f"\n  ✓ GID 37955 is in class 8 at position {idx}")
    print(f"    Type: {neurons[37955]}")

# Compare parameters for these types
print("\n" + "="*80)
print("COMPARING PARAMETERS")
print("="*80)

all_output_types = [neurons[gid] for gid in output_neurons]
type_counts = {}
for t in all_output_types:
    type_counts[t] = type_counts.get(t, 0) + 1

print(f"\nOutput neuron type distribution:")
for ntype in sorted(type_counts.keys()):
    print(f"  Type {ntype:3d}: {type_counts[ntype]:3d} neurons")

# Check if class 7-8 types are overrepresented
class7_8_types = set(class7_types + class8_types)
print(f"\nTypes in classes 7-8: {sorted(class7_8_types)}")

# Check parameters for type 96 (GID 37955)
if 96 < len(glif3_params['V_th']):
    print(f"\nParameters for type 96 (GID 37955's type):")
    print(f"  C_m = {glif3_params['C_m'][96]:.2f}")
    print(f"  E_L = {glif3_params['E_L'][96]:.2f} mV")
    print(f"  V_reset = {glif3_params['V_reset'][96]:.2f} mV")
    print(f"  V_th = {glif3_params['V_th'][96]:.2f} mV")
    print(f"  Voltage scale (V_th - E_L) = {glif3_params['V_th'][96] - glif3_params['E_L'][96]:.2f} mV")
    print(f"  asc_amps = {glif3_params['asc_amps'][96]}")
    print(f"  tau_syn = {glif3_params['tau_syn'][96]}")
    print(f"  g = {glif3_params['g'][96]:.6f}")

# Check background weights
print(f"\nBackground weights for type 96:")
# Background weights are stored per neuron, 4 receptors per neuron
start_idx = 37955 * 4
if start_idx + 4 <= len(bkg_weights):
    print(f"  GID 37955 background weights: {bkg_weights[start_idx:start_idx+4]}")
