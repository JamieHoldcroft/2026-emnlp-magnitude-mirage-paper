#!/bin/bash
#
# Phase 17: Confidence-AUROC Analysis
#
# This script runs the confidence-accuracy alignment analysis using AUROC.
# It evaluates how well a retriever's top-1 score predicts retrieval success
# at different cutoffs (k).
#
# It uses the cached scores from Phase 1.
#
# Usage:
# ./scripts/phase17_confidence_auroc.sh <model_1> <model_2> ...
# e.g.,
# ./scripts/phase17_confidence_auroc.sh google bge openai

# --- Configuration ---
set -e  # Exit immediately if a command exits with a non-zero status.

# A list of all BRIGHT tasks to be evaluated.
TASKS=(
    "biology"
    "aops"
    "earth_science"
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
# Scores are loaded from the phase1 cache build directory
# Note: The path format is /<MODEL>/<TASK>/<TASK>_<MODEL>_long_False/score.json
# Example: /google/aops/aops_google_long_False/score.json
RESULTS_DIR="sigir_results/phase1_cache_build" 
OUTPUT_DIR="sigir_results/phase17_confidence_auroc"

# Check if models were provided
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Usage: $0 <model_1> <model_2> ..."
    echo "Example: $0 google bge"
    exit 1
fi

# --- Main Loop ---
echo "Starting Phase 17: Confidence-AUROC Analysis"
echo "============================================"

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo "--- Running Model: $model, Task: $task ---"

        # Define the input path for scores and output path for results
        scores_file="$RESULTS_DIR/$model/$task/${task}_${model}_long_False/score.json"
        output_path="$OUTPUT_DIR/$task/$model"
        results_file="$output_path/auroc_results.json"

        # Check if the score file exists
        if [ ! -f "$scores_file" ]; then
            echo "Scores file not found at $scores_file. Skipping."
            echo "Please run Phase 1 first for this model and task."
            continue
        fi

        # Check if the result file already exists
        if [ -f "$results_file" ]; then
            echo "Results for $model on $task already exist. Skipping."
            continue
        fi

        # Run the core Python script
        python src/confidence_auroc.py \
            --model "$model" \
            --task "$task" \
            --scores_file "$scores_file" \
            --output_dir "$output_path" \
            --dataset_source "xlangai/BRIGHT"

        echo "--- Finished Model: $model, Task: $task ---"
    done
done

echo ""
echo "============================================"
echo "Phase 17: Confidence-AUROC Analysis Complete"

# --- Summarize Results Across Tasks ---
echo ""
echo "Starting Summary of Confidence-AUROC Results"
echo "============================================="

for model in "${MODELS[@]}"; do
    echo ""
    echo "--- Summarizing results for Model: $model ---"
    python src/summarize_auroc.py \
        --model "$model" \
        --output_base_dir "$OUTPUT_DIR"
    echo "--- Finished summarizing for Model: $model ---"
done

echo ""
echo "============================================="
echo "Summary of Confidence-AUROC Results Complete"

