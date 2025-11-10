#!/bin/bash
for i in {0..4}; do
    echo "========== SAMPLE $i =========="
    TARGET_INDEX=$i /opt/miniforge3/bin/python nest_glif.py 2>&1 | grep -E "Label:|Votes:|Prediction:|Correct:"
    echo ""
done
