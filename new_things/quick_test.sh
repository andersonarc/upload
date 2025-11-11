#!/bin/bash
echo "Quick test on 10 diverse samples"
echo "================================"
correct_200=0
correct_150=0
correct_100=0

for idx in 6 7 18 10 1 17 0 11 8 4; do
    output=$(TARGET_INDEX=$idx /opt/miniforge3/bin/python nest_glif.py 2>&1)
    label=$(echo "$output" | grep "Label:" | awk '{print $2}')
    
    # Extract just the "Correct: True/False" lines
    results=$(echo "$output" | grep "Correct:" | awk '{print $2}')
    
    # Convert to array
    IFS=$'\n' read -r -d '' -a arr <<< "$results"
    
    if [ "${arr[0]}" = "True" ]; then ((correct_200++)); fi
    if [ "${arr[1]}" = "True" ]; then ((correct_150++)); fi  
    if [ "${arr[2]}" = "True" ]; then ((correct_100++)); fi
    
    echo "  Sample $idx (label $label): 200=${arr[0]} 150=${arr[1]} 100=${arr[2]}"
done

echo ""
echo "Results: 50-200ms: $correct_200/10, 50-150ms: $correct_150/10, 50-100ms: $correct_100/10"
