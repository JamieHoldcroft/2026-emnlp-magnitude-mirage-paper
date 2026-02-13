#!/bin/bash
#
# Phase 15: Semantic Stability
#
# This script runs the semantic stability evaluation (Phase 15) for a given
# set of retrieval models across all BRIGHT tasks.
#
# It evaluates how retrieval effectiveness degrades when queries are transformed
# in controlled ways that preserve, neutralize, or break semantics.
#
# Usage:
# ./scripts/phase15_semantic_stability.sh <model_1> <model_2> ...
# e.g.,
# ./scripts/phase15_semantic_stability.sh google bge openai

# --- Configuration ---
set -e  # Exit immediately if a command exits with a non-zero status.

# A list of all BRIGHT tasks to be evaluated.
# These are derived from the sub-task config files.
TASKS=(
    "earth_science"
    "biology"
    "aops"
    "economics"
    "leetcode"
    "pony"
    "psychology"
    "robotics"
    "stackoverflow"
    "sustainable_living"
    "theoremqa_questions"
    "theoremqa_theorems"
)

# The models to evaluate are passed as command-line arguments.
MODELS=($@)

# Check if models were provided
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Usage: $0 <model_1> <model_2> ..."
    echo "Example: $0 google bge"
    exit 1
fi

# --- Main Loop ---
echo "Starting Phase 15: Semantic Stability Analysis"
echo "=============================================="

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo "--- Running Model: $model, Task: $task ---"
        
        # Define the output path for the results
        output_file="sigir_results/phase15_semantic_stability/$task/$model/results.json"
        
        # Check if the result file already exists
        if [ -f "$output_file" ]; then
            echo "Results for $model on $task already exist. Skipping."
            continue
        fi

        # Run the core Python script
        python src/semantic_stability.py \
            --model "$model" \
            --task "$task" \
            --output_dir "sigir_results" \
            --cache_dir "cache"
            
        echo "--- Finished Model: $model, Task: $task ---"
    done
done

echo ""
echo "=============================================="
echo "Phase 15: Semantic Stability Analysis Complete"