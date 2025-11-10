# Autonomous Work Plan - SpiNNaker GLIF3 Debugging
**Duration**: ~1 hour unsupervised work
**Last updated**: 2025-11-10

---

## 📋 ORIGINAL USER MESSAGE (VERBATIM)

```
Pushed a bunch of logs, pull and they will be in jupyter. Own observations from picture: Not much differences, except: 1) Working are in bottom of Mean LGN Activity 0.0150 2) Working are in bottom of Expected Total Spikes 26200 3) Working have higher-ish average peak neuron activity. Meanwhile, failed are actually uniformly distributed along the y axis for Mean LGN and Expected Total Spikes. In Peak Neurons, samples <40 are clustered near the bottom, samples 40-80 in the middle and 80-128 (working) are middle-top, with middle intersecting. Honestly I cannot infer any confident conclusions and given that TensorFlow inference works with 0.8 accuracy, the network cannot just 'not work for some inputs'. Likely we have an issue either in input encoding (however since I saw digits in decoded images, it does seem valid), network topology (either Population-Projection routing or GLIF3 implementation), H5 conversion by c2.py (There was a previous incident where I was using untrained weights. However, this is not very likely now. Still, perhaps some sort of validation is in order. Heat map of weights maybe? Produce one in c2 using tensorflow weights directly to make sure we're loading the checkpoint, and produce one from the H5 as well (make sure they use the same format, either normalized or denormalized - or both side by side? Important: Make one with untrained weights as well.). A lot of things depend on specific order, such as network neurons (GLIF3 populations) and readouts (output) and are easy to mess up. Population/projection clustering code in class.py also might be causing issues? I haven't really verified it.). Output decoding might be an issue as well.

Tasks for you:
1. I am going away for an hour or so. You will have a lot of time, but during this I will NOT answer, so do not ask any questions in that time. If you have any questions, ask NOW.
2. I suggest we visualize explicitly the activities of BOTH TensorFlow and SpiNNaker networks. At runtime, that is. Do different kinds of grouping, e.g. GLIF3 population activities over time could work reasonably well. Graphs, heatmaps, spike maps across the whole network, anything - you have time. Be careful with TensorFlow code - it is lazily executed. Exact tensorflow version is in the packages document somewhere there. So it's 6 tasks - visualize Input, Output, Activity in Tensorflow and SpiNNaker.
3. Implement a version of class.py that uses NEST simulator with their GLIF3 neurons - this way we can be certain whether the issue is in our GLIF3 implementation OR population-projection routing code.
4. Static visualizations too, as mentioned before - from TensorFlow checkpoint and H5 file, trained/untrained, normalized/denormalized.
5. A) Identify potential 'problem vectors' such as buggy GLIF3, buggy population-projection routing (note: pruning is not used. ignore it.), buggy input encoding (already proven that we can see numbers from the spike probabilities!), buggy reading outputs. Honestly I am betting on the .h5 file being untrained or otherwise fucked up, or the population-projection code messing something up. To prove/disprove the former, we need weight heatmaps. To prove/disprove the latter, we need 'population-projection decomposition' and compare results with network['input'] and network['recurrent'] as they are returned to flat arrays. Some kind of array hash maybe? To prove/disprove GLIF3 bugginess, we need a NEST implementation. Identify other potential issue vectors and what must be done to prove and disprove them, and what has been done so far. Remember and remind yourself to avoid speculating and jumping to conclusions - you have plenty of time to verify things thoroughly. After doing something, always double-check.
6. Generally produce a cleaner but consistent (must run the same way) version of class.py.
```

### 🔴 CRITICAL CORRECTIONS (From Follow-up)
- **NO SAMPLES ACTUALLY WORK** - Some produce garbage output, some produce no output
- Samples 90+ are NOT "working" - they just produce different garbage than 10-80
- All SpiNNaker outputs are incorrect compared to TensorFlow's 0.8 accuracy

---

## CRITICAL REMINDERS (READ PERIODICALLY!)

### ⚠️ ANTI-SPECULATION RULES
- **DO NOT SPECULATE** - Only state facts with evidence
- **JUSTIFY EVERYTHING** - Show data, code, or logs supporting claims
- **VERIFY THOROUGHLY** - Double-check after completing each task
- **NO PREMATURE CONCLUSIONS** - Test hypotheses systematically

### 🎯 CORE FINDINGS (ESTABLISHED)
1. ✅ **LGN encoding is CORRECT** - Visual decoding shows proper digits
2. ✅ **No activity level differences** - Failed/working samples have similar LGN activity (p>0.05)
3. ✅ **Input pipeline is correct** - 1.3 scaling properly handled
4. ✅ **TensorFlow achieves 0.8 accuracy** - Network architecture is sound
5. ❌ **SpiNNaker COMPLETELY FAILS** - No samples work correctly (some garbage, some silence)

### 🔍 HYPOTHESIS STATUS

| Hypothesis | Status | Evidence | Next Action |
|------------|--------|----------|-------------|
| LGN encoding broken | ❌ DISPROVEN | Visual decoding shows digits | None |
| Activity level threshold | ❌ DISPROVEN | analyze.log: p=0.09, no separation | None |
| H5 weights wrong/untrained | 🔶 PENDING | Need weight heatmaps | **Priority 1** |
| Population-projection routing bug | 🔶 PENDING | Need decomposition analysis | **Priority 2** |
| GLIF3 implementation bug | 🔶 PENDING | Need NEST comparison | **Priority 3** |
| Output decoding bug | 🔶 PENDING | Need validation | Priority 4 |
| Synapse/ASC scaling | 🔶 PENDING | Recently fixed, but verify | Priority 5 |

---

## ENVIRONMENT SETUP

### Available Resources
- ✅ Python 3.11.14
- ✅ 14GB storage available
- ✅ wget/curl for downloads
- ⚠️ TensorFlow 2.10.0 NOT in current environment (need venv/install)
- ✅ HuggingFace token provided by user (stored in /tmp/hf_token)
- ⚠️ NEST simulator NOT installed (will attempt install or work from docs)

### Installation Strategy
1. Try: `python3 -m venv /tmp/debug_env && source /tmp/debug_env/bin/activate`
2. Then: `pip install tensorflow==2.10.0 h5py numpy scipy matplotlib`
3. Try: `pip install nest-simulator` (may fail - ok to proceed without)
4. If all fails: Work with code analysis only

### File Locations
```
/home/user/upload/
├── training_code/
│   ├── c2.py              # H5 conversion script ✅
│   ├── pip_packages.txt   # TF 2.10.0 ✅
│   ├── models.py          # Network definition
│   ├── multi_training.py  # Training script
│   ├── stim_dataset.py    # Input generation
│   └── lgn_model/lgn.py   # LGN encoding
├── jupyter/
│   ├── class.py           # SpiNNaker simulation ✅
│   ├── *.log              # Analysis logs ✅
│   └── python_models8/    # GLIF3 implementation
└── .git/
```

### External Resources (Download with HF Token)
- TensorFlow checkpoint (trained): https://huggingface.co/rmri/v1cortex-mnist_51978-ckpts-50-56
- H5 file: https://huggingface.co/rmri/v1cortex-mnist_51978-ckpts-50-56/resolve/main/ckpt_51978-153.h5
- Spikes H5: https://huggingface.co/rmri/v1cortex-mnist_51978-ckpts-50-56/resolve/main/spikes-128.h5

### Critical Determinism Requirements
- **MNIST dataset shuffling MUST be deterministic**
- `multi_training.py` and `mnist.py` both generate LGN input independently
- They MUST produce same samples for same indices
- Common code in `stim_dataset.py`
- Seeds: mnist.py uses `seed=3000`, class.py uses `np.random.seed(1)`
- **MUST verify same input when comparing TF vs SpiNNaker**

---

## TASK LIST

### Priority Ordering (Per User Instructions)
**CODING TASKS (Highest Priority):**
1. NEST implementation
2. TensorFlow runtime visualization (Input/Output/Activity)
3. SpiNNaker runtime visualization (Input/Output/Activity)

**REVIEW TASKS (High Priority):**
4. H5 weights validation (trained vs untrained)
5. Population-Projection (PP) routing decomposition
6. ASC scaling review
7. Input/output decoding review

**OTHER TASKS (If Time):**
8. Code cleanup
9. Additional problem vectors

### ✅ COMPLETED
- [x] Pull logs from repository
- [x] Read analyze.log - confirmed no LGN activity differences
- [x] Check environment capabilities
- [x] Create work plan document

### 🔴 PHASE 1: NEST IMPLEMENTATION (HIGHEST PRIORITY - 30 min)

**Goal**: Independent validation using NEST's native GLIF3 to isolate GLIF3 vs routing bugs

#### Why Priority 1:
- Definitively proves/disproves GLIF3 implementation bug
- If NEST matches TF → SpiNNaker GLIF3 is buggy
- If NEST matches SpiNNaker → routing or other issue
- Most diagnostic value per time invested

#### Tasks:
- [ ] Check NEST availability: `python3 -c "import nest"`
- [ ] If not available, attempt install: `pip install --user nest-simulator`
- [ ] If install fails, work from online NEST documentation (use WebFetch)

- [ ] Research NEST GLIF models:
  - Latest NEST version uses: `glif_psc` or `gif_psc_exp_multisynapse`
  - Check NEST docs for exact parameter mapping
  - Find BMTK examples in training_code for reference

- [ ] Create nest_simulation.py:
  ```python
  import nest
  import h5py
  import numpy as np

  # Use same seeds as class.py for determinism
  np.random.seed(1)
  nest.SetKernelStatus({'rng_seed': 1})

  # Test samples: 10, 50, 90, 100
  # Load same input from spikes-128.h5
  # Load same weights from ckpt_51978-153.h5
  # Load same connectivity structure

  # Create NEST GLIF neurons with parameters from H5
  # Connect with same topology as class.py
  # Run simulation
  # Compare outputs with TensorFlow and SpiNNaker
  ```

- [ ] Verify input determinism:
  - NEST must use EXACT same input spikes as SpiNNaker
  - Not just same digit, but same spike train realization
  - Use same Poisson sampling seed

- [ ] Generate comparison report:
  - Output spikes: NEST vs SpiNNaker vs TensorFlow
  - Activity levels: Do NEST V1 neurons fire?
  - Classification accuracy: Does NEST match TF?

#### Success Criteria:
- ✅ NEST simulation runs (or documented why it can't)
- ✅ Clear verdict: GLIF3 bug or not
- ✅ If GLIF3 is OK, focus shifts to routing

---

### 🔴 PHASE 2: TENSORFLOW RUNTIME VISUALIZATION (PRIORITY 2 - 25 min)

**Goal**: Capture TensorFlow network activity during inference for exact comparison with SpiNNaker

#### Tasks:
- [ ] Setup TensorFlow environment:
  - Create venv: `python3 -m venv /tmp/debug_env`
  - Install: `pip install tensorflow==2.10.0 h5py numpy scipy matplotlib`
  - If fails: work with code analysis only

- [ ] Download required files using HF token:
  ```bash
  export HF_TOKEN=$(cat /tmp/hf_token)
  wget --header="Authorization: Bearer $HF_TOKEN" URL
  # Or use huggingface_hub library
  ```

- [ ] Locate TensorFlow inference code:
  - How are weights extracted from TF checkpoint?
  - How are they written to H5?
  - Any transformations applied?
  - Verify layer ordering/naming

- [ ] Create weight_comparison.py script:
  ```python
  # Inputs:
  # - TensorFlow checkpoint (trained) - ckpt_51978-153
  # - TensorFlow checkpoint (untrained) - need to find or generate
  # - H5 file - ckpt_51978-153.h5

  # Outputs:
  # 1. Heatmaps (normalized): trained_tf vs untrained_tf vs h5
  # 2. Heatmaps (denormalized): same comparison
  # 3. Statistical comparison (mean, std, min, max, hash)
  # 4. Per-layer weight distribution plots
  # 5. Difference maps (trained_tf - h5)
  ```

- [ ] Generate untrained checkpoint comparison
  - Initialize network with same architecture
  - Compare against trained weights
  - Check if H5 looks more like trained or untrained

#### Success Criteria:
- ✅ Heatmaps generated showing visual similarity/difference
- ✅ Statistical metrics confirm weights match or differ
- ✅ Clear conclusion: H5 is trained/untrained/corrupted

---

### 🔶 PHASE 2: TENSORFLOW RUNTIME VISUALIZATION (25 min)

**Goal**: Capture TensorFlow network activity during inference

#### Tasks:
- [ ] Locate TensorFlow inference code:
  - Read multi_training.py for inference logic
  - Read models.py for network architecture
  - Understand how checkpoint is loaded

- [ ] CRITICAL: Verify deterministic dataset loading:
  - multi_training.py and mnist.py BOTH generate LGN independently
  - Common code in stim_dataset.py
  - Seeds: mnist.py=3000, multi_training.py=?
  - **MUST ensure sample 10 in TF == sample 10 in SpiNNaker**
  - Check MNIST shuffling implementation

- [ ] Create tf_runtime_visualizer.py:
  ```python
  # Test samples: 10, 50, 90, 100

  # For each sample, capture:
  # 1. Input: LGN spike probabilities
  #    - Temporal plot (mean activity over time)
  #    - Heatmap (time x neurons)
  #    - Spike raster (sample of neurons)

  # 2. Layer Activities:
  #    - GLIF3 population outputs (V1 neurons)
  #    - Group by population type (4 types in V1)
  #    - Plot activity over time
  #    - Note: TF is LAZY - use .numpy() to force evaluation

  # 3. Output Layer:
  #    - 10-class softmax or spike counts
  #    - Predicted class vs true label
  #    - Confidence scores

  # 4. Side-by-side comparison:
  #    - Sample 10 (failed) vs Sample 100 (working)
  #    - Where does activity differ?
  ```

- [ ] Handle TensorFlow lazy execution carefully
  - Force evaluation with .numpy() or .eval()
  - Check TF version (2.10.0) - uses eager mode by default
  - Add explicit tf.function decorators if needed

#### Success Criteria:
- ✅ Captured intermediate activations for all test samples
- ✅ Clear visualization showing network processing
- ✅ Identified where failed/working samples diverge in TF

---

### 🔶 PHASE 3: SPINNAKER RUNTIME VISUALIZATION (20 min)

**Goal**: Capture SpiNNaker network activity and compare with TensorFlow

#### Tasks:
- [ ] Modify class.py to record ALL population activities
  - Currently only records output (readout neurons)
  - Add recording for V1 populations (4 types)
  - Add recording for LGN input spikes
  - Consider voltage traces if feasible (may be large)

- [ ] Create spinnaker_runtime_visualizer.py:
  ```python
  # Use modified class.py output

  # For samples 10, 50, 90, 100:
  # 1. Input verification:
  #    - LGN spike rasters (compare with H5 probabilities)
  #    - Verify Poisson sampling is working

  # 2. V1 population activities:
  #    - Group by population type
  #    - Activity rates over time
  #    - Compare with TensorFlow layer activities

  # 3. Output spikes:
  #    - Readout neuron spikes
  #    - Classification result
  #    - Compare with TensorFlow output

  # 4. Divergence analysis:
  #    - At what point do failed samples diverge?
  #    - Do they get ANY V1 activity?
  #    - Is output layer receiving input?
  ```

- [ ] Generate side-by-side comparison plots
  - TensorFlow vs SpiNNaker for each sample
  - Highlight where they diverge
  - Quantify activity differences

#### Success Criteria:
- ✅ Full activity traces from SpiNNaker simulation
- ✅ Clear comparison with TensorFlow
- ✅ Identified divergence point (input/V1/output)

---

### 🔶 PHASE 4: H5 WEIGHTS VALIDATION (HIGH PRIORITY REVIEW - 20 min)

**Goal**: Verify H5 weights match TensorFlow checkpoint (trained vs untrained)

#### Why High Priority:
- User suspects H5 file might be untrained or corrupted
- Previous incident with untrained weights
- Weight heatmaps will definitively prove/disprove

#### Tasks:
- [ ] Read c2.py thoroughly:
  - How are weights extracted from TF checkpoint?
  - How are they normalized/denormalized?
  - How are they written to H5?
  - Any layer reordering?

- [ ] Create weight_validation.py:
  ```python
  import nest

  # NEST GLIF3 model: 'glif_psc' or 'gif_psc_exp_multisynapse'
  # Check latest NEST docs for exact model name

  # Network structure (from class.py):
  # - LGN: 17400 neurons (input)
  # - V1: 4 populations of GLIF3 neurons
  # - Output: 300 readout neurons (30 per class)

  # Load weights from H5 file (same as SpiNNaker)
  # Load connectivity (population-projection structure)
  # Load input from spikes-128.h5

  # Test samples: 10, 90, 100

  # Compare outputs:
  # - NEST vs SpiNNaker vs TensorFlow
  # - If NEST matches TF → SpiNNaker GLIF3 is buggy
  # - If NEST matches SpiNNaker → TF uses different dynamics
  ```

- [ ] Reference BMTK code in training_code
  - BMTK uses NEST backend
  - Check how they set up GLIF neurons
  - Use same parameter mapping

#### Success Criteria:
- ✅ NEST simulation runs (or detailed script ready to run)
- ✅ Output comparison: NEST vs SpiNNaker vs TensorFlow
- ✅ Clear conclusion on GLIF3 implementation validity

---

### 🔶 PHASE 5: POPULATION-PROJECTION ROUTING ANALYSIS (HIGH PRIORITY REVIEW - 20 min)

**Goal**: Verify connectivity matches expected topology (user suspects this is a major issue)

#### Why High Priority:
- User mentioned population/projection code hasn't been verified
- "A lot of things depend on specific order"
- Network neurons (populations) and readouts (output) easy to mess up
- User betting on PP code or H5 weights being the issue

#### Tasks:
- [ ] Read class.py routing code carefully
  - Lines that create populations
  - Lines that create projections
  - How weights are assigned from H5
  - Any index transformations

- [ ] Compare with TensorFlow network definition
  - Read models.py to understand layer structure
  - Verify same connectivity pattern
  - Check for any pruning (user says none used)

- [ ] Create routing_validation.py:
  ```python
  # Extract connectivity from class.py logic:
  # - Which populations connect to which
  # - Weight matrices for each projection
  # - Receptor types (excitatory/inhibitory)

  # Compare with H5 file:
  # - network['input'] - input connectivity
  # - network['recurrent'] - recurrent connectivity
  # - Are they being applied correctly?

  # Generate:
  # - Connectivity matrix visualization
  # - Weight distribution per projection
  # - Hash comparison: computed vs loaded
  # - Identify any mismatches
  ```

- [ ] Check population-projection decomposition
  - User mentioned this needs verification
  - Compare flat arrays after decomposition
  - Use hash or element-wise comparison

#### Success Criteria:
- ✅ Connectivity structure verified correct
- ✅ Weights correctly assigned to projections
- ✅ No routing bugs found (or bugs identified)

---

### 🔶 PHASE 6: OTHER REVIEW TASKS (15 min)

**Goal**: ASC scaling, input/output validation, other verifications

#### Tasks:
- [ ] ASC scaling review:
  - Read recent fixes in git history
  - Verify denormalization by source voltage scale
  - Compare with TensorFlow implementation

- [ ] Output decoding validation:
  - Read output counting logic in class.py
  - Verify readout neurons (30 per class = 300 total)
  - Check response window timing
  - Ensure spike counting is correct

- [ ] Input encoding final check:
  - Already proven LGN works (visual decoding)
  - But verify Poisson sampling in class.py
  - Confirm 1.3 scaling removal

#### Success Criteria:
- ✅ All secondary hypotheses checked
- ✅ Documentation of what's verified correct

---

### 🔶 PHASE 7: PROBLEM VECTOR MATRIX UPDATE (10 min)

**Goal**: Systematic hypothesis testing matrix

Create comprehensive markdown table:

| Problem Vector | Evidence For | Evidence Against | Test Method | Status | Next Action |
|----------------|--------------|------------------|-------------|--------|-------------|
| H5 weights untrained | - | Visual decoding works | Weight heatmaps | PENDING | Phase 1 |
| H5 weights corrupted | - | - | Hash comparison | PENDING | Phase 1 |
| GLIF3 implementation | - | TF works | NEST comparison | PENDING | Phase 4 |
| Routing/connectivity | - | - | Topology validation | PENDING | Phase 5 |
| Input encoding | ✅ Visual decode works | ✅ analyze.log shows no diff | - | DISPROVEN | - |
| Output decoding | - | - | Manual verification | PENDING | Phase 7 |
| ASC scaling | Recently fixed | - | Runtime comparison | PENDING | - |
| Voltage normalization | Recently fixed | - | Runtime comparison | PENDING | - |
| Synapse implementation | Recently fixed | - | Runtime comparison | PENDING | - |

---

### 🔶 PHASE 8: CODE CLEANUP (10 min if time permits)

**Goal**: Cleaner but functionally identical class.py

- [ ] Add clear section headers
- [ ] Extract functions for:
  - Dataset loading
  - Population creation
  - Projection creation
  - Simulation running
  - Output analysis
- [ ] Better variable names
- [ ] Remove dead code / commented sections
- [ ] Add docstrings
- [ ] **CRITICAL**: Maintain exact same behavior

---

## FINAL DELIVERABLES

### Required Outputs:
1. **weight_comparison_report.md** - H5 vs TF checkpoint analysis
2. **tf_runtime_activity/** - TensorFlow visualization plots
3. **spinnaker_runtime_activity/** - SpiNNaker visualization plots
4. **nest_simulation.py** - NEST implementation (runnable or template)
5. **routing_validation_report.md** - Connectivity analysis
6. **problem_vector_matrix.md** - Updated hypothesis matrix
7. **class_clean.py** - Cleaned version (if time permits)
8. **SUMMARY_REPORT.md** - Overall findings and next steps

### Summary Report Structure:
```markdown
# SpiNNaker GLIF3 Debugging - Autonomous Work Session

## Findings Summary
- What was proven
- What was disproven
- What remains uncertain

## Key Discoveries
- Most important finding
- Unexpected results
- Blockers encountered

## Hypothesis Status
- Updated problem vector matrix
- Confidence levels for each

## Evidence
- Links to generated plots
- Code snippets supporting claims
- Log excerpts

## Recommended Next Steps
1. Priority 1: [most urgent action]
2. Priority 2: [second priority]
3. etc.

## Questions for User
- Anything requiring clarification
- Decisions needed
- Additional information needed
```

---

## EXECUTION STRATEGY

### Time Management (Reordered by Priority):
- Phase 1 (NEST): 30 min - **HIGHEST PRIORITY CODING**
- Phase 2 (TF viz): 25 min - **PRIORITY CODING**
- Phase 3 (SpiNNaker viz): 20 min - **PRIORITY CODING**
- Phase 4 (H5 weights): 20 min - **HIGH PRIORITY REVIEW**
- Phase 5 (PP routing): 20 min - **HIGH PRIORITY REVIEW**
- Phase 6 (Other reviews): 15 min
- Phase 7 (Matrix): 10 min
- Phase 8 (Cleanup): 10 min if time
- Documentation: Ongoing throughout

### Commit Strategy:
- Commit after each major phase completion
- Commit when significant findings discovered
- Commit before attempting risky operations
- Each commit message should clearly state what was accomplished
- Push regularly to maintain progress log

### Parallel Work:
- Download files in background while coding
- Generate plots while running other analyses
- Write documentation while waiting for outputs

### Failure Handling:
- If downloads fail: work with code analysis
- If NEST unavailable: create template + doc research
- If TF can't run: analyze code structure only
- Always document what was attempted and why it failed

---

## PERIODIC CHECKS (Every 15 minutes)

1. **Re-read this document** - Stay on track
2. **Check anti-speculation rules** - Am I guessing or proving?
3. **Update TodoWrite** - Track progress
4. **Commit progress** - Git commit + push
5. **Time check** - Am I on schedule?

---

## NOTES SECTION (Update as I work)

### Insights Discovered:


### Blockers Encountered:


### Questions for User:


### Confidence Levels:
- H5 weights status: ?/10
- GLIF3 implementation: ?/10
- Routing correctness: ?/10

---

END OF WORK PLAN
