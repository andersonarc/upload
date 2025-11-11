#!/bin/bash
correct_50_200=0
correct_50_100=0
total=24

echo "Testing NEST GLIF3 on mnist24.h5 (24 samples)"
echo "==============================================="

for i in {0..23}; do
    echo "Sample $i..."
    output=$(TARGET_INDEX=$i /opt/miniforge3/bin/python nest_glif.py 2>&1)

    # Extract label
    label=$(echo "$output" | grep "Label:" | awk '{print $2}')

    # Extract 50-200ms prediction and correctness
    pred_200=$(echo "$output" | grep "50-200ms:" -A 2 | grep "Prediction:" | awk '{print $2}' | tr -d ',')
    correct_200=$(echo "$output" | grep "50-200ms:" -A 3 | grep "Correct:" | awk '{print $2}')

    # Extract 50-100ms prediction and correctness
    pred_100=$(echo "$output" | grep "50-100ms" -A 2 | grep "Prediction:" | awk '{print $2}' | tr -d ',')
    correct_100=$(echo "$output" | grep "50-100ms" -A 3 | grep "Correct:" | awk '{print $2}')

    # Count correct predictions
    if [ "$correct_200" = "True" ]; then
        ((correct_50_200++))
        marker_200="✓"
    else
        marker_200="✗"
    fi

    if [ "$correct_100" = "True" ]; then
        ((correct_50_100++))
        marker_100="✓"
    else
        marker_100="✗"
    fi

    echo "  Sample $i (label $label): 50-200ms: $pred_200 $marker_200 | 50-100ms: $pred_100 $marker_100"
done

echo ""
echo "==============================================="
echo "Final Results:"
echo "  50-200ms window: $correct_50_200/$total correct ($(echo "scale=1; $correct_50_200*100/$total" | bc)%)"
echo "  50-100ms window: $correct_50_100/$total correct ($(echo "scale=1; $correct_50_100*100/$total" | bc)%)"
echo "==============================================="
