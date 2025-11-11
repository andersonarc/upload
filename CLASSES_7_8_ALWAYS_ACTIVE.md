# Classes 7-8 Always Active Regardless of Input

## The Problem

User reports classes 7-8 are highly active regardless of what digit is shown:

```
50-100 ms (target)
[1. 0. 0. 0. 0. 0. 0. 6. 9. 0.]
TARGET votes:
[1 2 3 4 6 5 9 0 7 8]
EXPECTED: 9

50-600 ms (overall)
[ 1.  0.  0.  0.  0.  0.  0. 30. 19.  0.]
```

For digit 9 input, class 9 should be active, but classes 7-8 dominate with 30 and 19 spikes.
This pattern occurs for **all inputs** - classes 7-8 are always active.

## What We've Ruled Out

✓ **Population mapping is correct** - Verified GID → (pid, subpid, lid) → GID round-trips correctly
✓ **LGN mapping is correct** - Verified LGN population creation matches l2pl
✓ **Synapse remapping is correct** - LGN synapses target correct V1 neurons
✓ **Neuron types are normal** - Classes 7-8 have same mix of types as other classes
✓ **Output neuron ordering is correct** - Confirmed by NEST working with same h5 file

## Possible Root Causes

### 1. Receptor Type Mismatch
**Hypothesis:** Excitatory synapses might be mapped to inhibitory receptors or vice versa.

**Evidence:**
- Code uses `receptor_type = f'synapse_{int(syn[0, S.RTY])}'`
- Creates receptor types: `synapse_0`, `synapse_1`, `synapse_2`, `synapse_3`
- GLIF3 has 4 tau_syn values (one per receptor)

**Check:**
- Are receptors 0-1 excitatory and 2-3 inhibitory (or some other mapping)?
- Does the C implementation correctly handle these?
- Are weights applied with correct polarity?

### 2. Weight Scaling Still Wrong

**Hypothesis:** The 1.66× PSC normalization fix overcorrected, or there's another scaling issue.

**Evidence:**
- User fixed PSC normalization in C code
- Now get 15 total spikes (was 0 before fix, was 267 before that)
- But wrong classes fire

**Check:**
- Is the 1.66× factor still in spynnaker_newclass.py? (Yes, lines 851, 902)
- Combined with C fix, this might be double-correcting
- Try removing the 1.66× test scaling

### 3. Initial Conditions Wrong

**Hypothesis:** Classes 7-8 neurons start at high voltage or have wrong initial ASC values.

**Evidence:**
- V_m initial is set to E_L (line 926: `G2IV(glif3s[pid])`)
- Should match NEST (line 109: `V_m = E_L`)

**Check:**
- Are initial ASC values correct?
- Are initial PSC values correct?
- Do classes 7-8 neurons have different initial conditions somehow?

### 4. Background Input Timing

**Hypothesis:** Background Poisson sources are providing constant drive throughout simulation.

**Evidence:**
- Background rate: 10 sources @ 10Hz each
- Should be same for all neurons

**Check:**
- Are classes 7-8 receiving more background connections?
- Are background weights scaled correctly for different neuron types?
- Background synapses use same receptor_type mapping as others

### 5. Spike Timing / Synaptic Delays

**Hypothesis:** Synaptic delays or spike timing are causing temporal misalignment.

**Evidence:**
- LGN spike times are loaded correctly (same h5 file works in NEST)
- Recurrent delays might be applied incorrectly

**Check:**
- Are delays in LGN synapses being dropped? (line 909: only weight, no delay)
- Are recurrent delays correct? (line 958: includes delay)
- Does SpiNNaker's execution model handle delays differently?

### 6. PSC Rise Dynamics Bug

**Hypothesis:** The alpha-function PSC implementation has a subtle timing bug.

**Evidence:**
- C implementation uses rise/decay model
- Peak-at-tau normalization was wrong, now fixed
- But temporal dynamics might still be off

**Check:**
- Does PSC peak at correct time?
- Is the rise phase too fast/slow?
- Are PSCs being integrated at the right timestep in the neuron update?

## Diagnostic Steps

1. **Remove 1.66× test scaling** (lines 851, 902) since PSC normalization was fixed in C
2. **Add logging for classes 7-8 neurons:**
   - Input current received
   - Voltage trajectory
   - Which synapses are active
3. **Compare background weights** between classes 0-6 vs 7-8
4. **Check receptor type distribution** for synapses targeting classes 7-8 vs others
5. **Verify initial conditions** are same across all classes

## Most Likely Culprit

Given that:
- Mapping is correct
- The issue is class-specific (only 7-8)
- It's input-independent (always active)

**Best guess: Background or recurrent weights to classes 7-8 are scaled incorrectly, causing constant excitation.**

Alternative: **Receptor type mapping inverts inhibition/excitation for those specific neuron populations.**
