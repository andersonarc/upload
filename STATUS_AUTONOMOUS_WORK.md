# 🚀 AUTONOMOUS WORK SESSION - READY TO BEGIN

**Status**: ✅ All preparation complete
**Start Time**: 2025-11-10 08:33 UTC
**Expected Duration**: ~1 hour
**User Status**: Away from computer (no questions during work)

---

## ✅ SETUP COMPLETE

### Configuration:
- [x] Work plan document created and committed
- [x] Todo list initialized with 9 phases
- [x] HuggingFace token stored securely in /tmp/hf_token
- [x] Logs pulled and reviewed (analyze.log confirms no LGN activity differences)
- [x] Environment checked (Python 3.11.14, 14GB storage)
- [x] Git repository ready for commits

### Critical Understanding:
- ❌ **NO samples work on SpiNNaker** (some garbage, some silence)
- ✅ **TensorFlow achieves 0.8 accuracy** (network is sound)
- ✅ **LGN encoding is correct** (visual decoding shows proper digits)
- ✅ **No activity level differences** between "failed" and "working" samples

### User's Main Suspects:
1. **H5 weights wrong/untrained** (previous incident with this)
2. **Population-Projection routing bug** (order-dependent, not verified)
3. **GLIF3 implementation bug** (needs NEST validation)

---

## 📋 WORK PLAN PRIORITY ORDER

### CODING TASKS (Highest Priority):
1. **Phase 1: NEST Implementation** (30 min)
   - Validate GLIF3 with NEST's native implementation
   - Definitive test: GLIF3 bug vs routing bug

2. **Phase 2: TensorFlow Runtime Viz** (25 min)
   - Capture Input/Output/Activity during inference
   - Ensure deterministic sample matching

3. **Phase 3: SpiNNaker Runtime Viz** (20 min)
   - Capture Input/Output/Activity from class.py
   - Compare with TensorFlow side-by-side

### REVIEW TASKS (High Priority):
4. **Phase 4: H5 Weights Validation** (20 min)
   - Heatmaps: trained vs untrained vs H5
   - Prove/disprove weights are correct

5. **Phase 5: PP Routing Analysis** (20 min)
   - Verify population-projection decomposition
   - Check against network['input'] and network['recurrent']

6. **Phase 6: Other Reviews** (15 min)
   - ASC scaling, input/output validation

7. **Phase 7: Problem Vector Matrix** (10 min)
8. **Phase 8: Code Cleanup** (10 min if time)

---

## 🎯 SUCCESS CRITERIA

### Must Complete:
- [ ] NEST simulation created (or documented why impossible)
- [ ] TensorFlow runtime captured (or code analysis if can't run)
- [ ] SpiNNaker runtime analysis
- [ ] Weight validation report
- [ ] PP routing validation report
- [ ] Problem vector matrix updated
- [ ] Final summary report with findings

### Ideal Outcome:
- Identify THE bug causing complete SpiNNaker failure
- Clear next steps for user
- Evidence-based conclusions (no speculation)

---

## ⚠️ REMINDERS (READ PERIODICALLY)

1. **NO SPECULATION** - Only facts with evidence
2. **VERIFY DETERMINISM** - Same sample = same input spike train
3. **JUSTIFY EVERYTHING** - Show code/data supporting claims
4. **COMMIT REGULARLY** - After each phase or finding
5. **READ WORK PLAN** - Every 15 minutes to stay on track
6. **CHECK TODO LIST** - Mark progress as I go

---

## 📝 NOTES SPACE (Fill during work)

### Attempt 1: [TIME]
Status:
Findings:
Blockers:

### Attempt 2: [TIME]
Status:
Findings:
Blockers:

---

**READY TO BEGIN AUTONOMOUS WORK**
