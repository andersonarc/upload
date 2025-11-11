#!/bin/bash
# Test diverse classes from mnist24.h5

echo "Testing 10 diverse digit classes"
echo "================================="
echo ""

results_200=""
results_150=""
results_100=""
correct_200=0
correct_150=0
correct_100=0

for idx in 6 7 18 10 1 17 0 11 8 4; do
    output=$(TARGET_INDEX=$idx /opt/miniforge3/bin/python nest_glif.py 2>&1)

    label=$(echo "$output" | grep "Label:" | awk '{print $2}')

    # Extract predictions for each window
    pred_200=$(echo "$output" | grep "50-200ms:" -A 2 | grep "Expected:" | sed 's/.*Prediction: \([0-9]\).*/\1/')
    pred_150=$(echo "$output" | grep "50-150ms" -A 2 | grep "Expected:" | sed 's/.*Prediction: \([0-9]\).*/\1/')
    pred_100=$(echo "$output" | grep "50-100ms" -A 2 | grep "Expected:" | sed 's/.*Prediction: \([0-9]\).*/\1/')

    # Check correctness
    if [ "$pred_200" == "$label" ]; then ((correct_200++)); fi
    if [ "$pred_150" == "$label" ]; then ((correct_150++)); fi
    if [ "$pred_100" == "$label" ]; then ((correct_100++)); fi

    results_200="$results_200\n  Idx $idx (label $label): pred=$pred_200"
    results_150="$results_150\n  Idx $idx (label $label): pred=$pred_150"
    results_100="$results_100\n  Idx $idx (label $label): pred=$pred_100"
done

echo "Results by time window:"
echo ""
echo "50-200ms: $correct_200/10 correct"
echo -e "$results_200"
echo ""
echo "50-150ms: $correct_150/10 correct"
echo -e "$results_150"
echo ""
echo "50-100ms (target): $correct_100/10 correct"
echo -e "$results_100"
