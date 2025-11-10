# Autonomous Work Plan - SpiNNaker GLIF3 Debugging
**Duration**: ~1 hour unsupervised work
**Last updated**: 2025-11-10

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
5. ❌ **SpiNNaker shows bimodal failure** - Some samples work, most fail

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
- ✅ TensorFlow 2.10.0 installed (see pip_packages.txt)
- ❓ HuggingFace downloads (may need auth - 403 error seen)
- ❓ NEST simulator (unknown if installed)

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

### External Resources
- TensorFlow checkpoint (trained): https://huggingface.co/rmri/v1cortex-mnist_51978-ckpts-50-56
- H5 file: https://huggingface.co/rmri/v1cortex-mnist_51978-ckpts-50-56/resolve/main/ckpt_51978-153.h5
- Spikes H5: https://huggingface.co/rmri/v1cortex-mnist_51978-ckpts-50-56/resolve/main/spikes-128.h5

---

## TASK LIST

### ✅ COMPLETED
- [x] Pull logs from repository
- [x] Read analyze.log - confirmed no LGN activity differences
- [x] Check environment capabilities

### 🔶 PHASE 1: WEIGHT VALIDATION (Highest Priority - 20 min)

**Goal**: Verify H5 weights match TensorFlow checkpoint (trained vs untrained)

#### Tasks:
- [ ] Attempt to download files from HuggingFace
  - Try wget with different URL formats
  - Try WebFetch tool
  - If fails: work with code analysis only

- [ ] Read and understand c2.py conversion logic
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
- [ ] Locate TensorFlow inference code
  - Check multi_training.py for inference mode
  - Check if there's a separate test script
  - Understand how model is loaded from checkpoint

- [ ] Understand dataset loading and shuffling
  - User mentioned shuffling must be deterministic
  - Find seed used for MNIST shuffling
  - Verify samples 10, 50, 90, 100 map to same digits

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

### 🔶 PHASE 4: NEST SIMULATOR IMPLEMENTATION (30 min)

**Goal**: Independent validation using NEST's native GLIF3

#### Tasks:
- [ ] Check if NEST is installed
  ```bash
  python3 -c "import nest; print(nest.__version__)"
  ```

- [ ] If not installed, attempt installation:
  ```bash
  pip3 install nest-simulator --user
  # or
  apt-get install nest  # if root access
  ```

- [ ] If installation fails:
  - Work from NEST documentation online
  - Use WebFetch to read NEST GLIF3 docs
  - Create script template without running

- [ ] Create nest_simulation.py:
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

### 🔶 PHASE 5: POPULATION-PROJECTION ROUTING ANALYSIS (15 min)

**Goal**: Verify connectivity matches expected topology

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

### 🔶 PHASE 6: PROBLEM VECTOR MATRIX (10 min)

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

### 🔶 PHASE 7: CODE CLEANUP (10 min if time permits)

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

### Time Management:
- Phase 1 (Weights): 20 min - HIGHEST PRIORITY
- Phase 2 (TF viz): 25 min
- Phase 3 (SpiNNaker viz): 20 min
- Phase 4 (NEST): 30 min
- Phase 5 (Routing): 15 min
- Phase 6 (Matrix): 10 min
- Phase 7 (Cleanup): 10 min if time
- Documentation: Ongoing throughout

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
