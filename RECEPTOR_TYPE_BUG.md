# CRITICAL BUG: Receptor Type Index Mismatch

## The Bug

**NEST** consistently adds +1 to ALL receptor types from the h5 file:

```python
# nest_glif.py line 196 (LGN synapses):
rtype_filt = rtype_arr[mask] + 1  # Convert 0,1,2,3 to 1,2,3,4 for NEST

# nest_glif.py line 226 (Recurrent synapses):
rtype_filt = rtype_arr[mask] + 1  # Convert 0,1,2,3 to 1,2,3,4 for NEST

# nest_glif.py line 167 (Background):
'receptor_type': receptor_idx + 1
```

**NEST uses receptor types: 1, 2, 3, 4**

**SpiNNaker** uses raw receptor type values from h5:

```python
# spynnaker_newclass.py line 962 (Recurrent):
receptor_type = f'synapse_{int(syn[0, S.RTY])}'  # Values: 0, 1, 2, 3

# spynnaker_newclass.py line 1013 (LGN):
receptor_type = f'synapse_{int(syn[0, S.RTY])}'  # Values: 0, 1, 2, 3

# spynnaker_newclass.py line 1083 (Background):
receptor_type = f'synapse_{receptor_idx}'  # Values: 0, 1, 2, 3
```

**SpiNNaker uses receptor types: 0, 1, 2, 3**

## Why This Breaks Everything

If GLIF3 expects 1-indexed receptor types (like NEST's glif_psc), but SpiNNaker passes 0-indexed values, then:

- h5 receptor 0 → SpiNNaker sends to receptor 0 → **WRONG** (should be receptor 1)
- h5 receptor 1 → SpiNNaker sends to receptor 1 → **WRONG** (should be receptor 2)
- h5 receptor 2 → SpiNNaker sends to receptor 2 → **WRONG** (should be receptor 3)
- h5 receptor 3 → SpiNNaker sends to receptor 3 → **WRONG** (should be receptor 4)

If receptor types map to:
- Receptor 1: Excitatory fast (tau_syn[0])
- Receptor 2: Inhibitory fast (tau_syn[1])
- Receptor 3: Excitatory slow (tau_syn[2])
- Receptor 4: Inhibitory slow (tau_syn[3])

Then SpiNNaker's off-by-one error causes:
- **Excitatory synapses → hit wrong receptor or get dropped**
- **Inhibitory synapses → hit excitatory receptors**
- **Complete scrambling of E/I balance**

This would cause exactly what we see:
- Random/constant firing regardless of input
- Wrong classes active
- Classification failure

## The Fix

In `spynnaker_newclass.py`, add +1 to all receptor types:

```python
# V1 recurrent (line 962):
receptor_type = f'synapse_{int(syn[0, S.RTY]) + 1}'

# LGN synapses (line 1013):
receptor_type = f'synapse_{int(syn[0, S.RTY]) + 1}'

# Background (line 1083):
receptor_type = f'synapse_{receptor_idx + 1}'
```

This will make SpiNNaker use receptor types 1, 2, 3, 4 matching NEST.

## Verification Needed

Need to check if SpiNNaker's GLIF3 implementation expects:
- 0-indexed receptors (0, 1, 2, 3) → then NEST is doing unnecessary +1
- 1-indexed receptors (1, 2, 3, 4) → then SpiNNaker has the bug

Based on NEST adding +1 explicitly with the comment "Convert 0,1,2,3 to 1,2,3,4 for NEST",
it's likely that glif_psc uses 1-indexed receptors, and SpiNNaker's GLIF3 should too.
