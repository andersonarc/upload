#!/bin/bash
# Test diverse classes from mnist24.h5:
# Index 6 (label 0), 7 (label 1), 18 (label 2), 10 (label 3), 1 (label 4)
# Index 17 (label 5), 0 (label 6), 11 (label 7), 8 (label 8), 4 (label 9)

echo "Testing 10 diverse classes from mnist24.h5"
echo "==========================================="

for i in 6 7 18 10 1 17 0 11 8 4; do
    echo ""
    echo "========== INDEX $i ==========="
    TARGET_INDEX=$i /opt/miniforge3/bin/python nest_glif.py 2>&1 | grep -E "Label:|Votes:|Prediction:|Correct:"
done
