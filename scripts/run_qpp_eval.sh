#!/bin/bash
#
# QPP Evaluation Script
#
# This script runs the Query Performance Prediction (QPP) analysis
# across multiple dataset families (BEIR, BRIGHT, TEMPO), models, and tasks.
# It calculates 7 QPP metrics, NDCG, AUROC, and their correlations.
#
# Usage:
# ./scripts/run_qpp_eval.sh <model_1> <model_2> ...
# e.g.,
# ./scripts/run_qpp_eval.sh google bge openai

# --- Configuration ---
set -ex  # Exit immediately if a command exits with a non-zero status, and print commands and their arguments as they are executed.

# Define dataset families and their respective tasks
BEIR_TASKS=(
    "arguana" 
    "fiqa" 
    "nfcorpus" 
    "scidocs"
    "scifact" 
    "webis-touche2020"
    )

BRIGHT_TASKS=(
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

TEMPO_TASKS=(
    "cardano"
    "economics"
    "genealogy"
    "history"
    "hsm"
    "iota"
    "law"
    "monero"
    "politics"
    "quant"
    "travel"
    "workplace"
)

# The models to evaluate are passed as command-line arguments.
MODELS=("$@")

# Base directory for score files
# BEIR: results_beir/<model_name>/<dataset>/score.json
# BRIGHT: results_bright/<model_name>/<task>/score.json
# TEMPO: results_tempo/<model_name>/<task>/scores.json

# Output directory for QPP results
OUTPUT_BASE_DIR="qpp_results"

# Check if models were provided
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Usage: $0 <model_1> <model_2> ..."
    echo "Example: $0 google bge"
    exit 1
fi

echo "Starting QPP Evaluation"
echo "========================"

# --- Main Loop ---
for model in "${MODELS[@]}"; do
    echo ""
    echo "--- Processing Model: $model ---"

    # Process BEIR datasets
    echo "--- BEIR Datasets ---"
    for task_name in "${BEIR_TASKS[@]}"; do
        echo ""
        echo "  - Running Dataset: $task_name"
        dataset_family="BEIR"
        scores_file="results/scores_results/results_beir/$model/$task_name/score.json"
        output_dir="$OUTPUT_BASE_DIR/$dataset_family/$task_name"
        results_file="$output_dir/${model}_qpp_results.json"

        mkdir -p "$output_dir"

        if [ ! -f "$scores_file" ]; then
            echo "    Scores file not found at $scores_file. Skipping $task_name."
            continue
        fi

        if [ -f "$results_file" ]; then
            echo "    Results for $model on $task_name (BEIR) already exist. Skipping."
            continue
        fi

        echo "Executing: python src/evaluate_qpp.py --model \"$model\" --task \"$task_name\" --dataset_family \"$dataset_family\" --scores_file \"$scores_file\" --output_dir \"$output_dir\""
        python src/evaluate_qpp.py \
            --model "$model" \
            --task "$task_name" \
            --dataset_family "$dataset_family" \
            --scores_file "$scores_file" \
            --output_dir "$output_dir"
        echo "  - Finished Task: $task_name"
    done

    # Process BRIGHT datasets
    echo ""
    echo "--- BRIGHT Datasets ---"
    for task_name in "${BRIGHT_TASKS[@]}"; do
        echo ""
        echo "  - Running Task: $task_name"
        dataset_family="BRIGHT"
        scores_file="results/scores_results/results_bright/$model/$task_name/${task_name}_${model}_long_False/score.json" 
        output_dir="$OUTPUT_BASE_DIR/$dataset_family/$task_name"
        results_file="$output_dir/${model}_qpp_results.json"

        mkdir -p "$output_dir"

        if [ ! -f "$scores_file" ]; then
            echo "    Scores file not found at $scores_file. Skipping $task_name."
            continue
        fi

        if [ -f "$results_file" ]; then
            echo "    Results for $model on $task_name (BRIGHT) already exist. Skipping."
            continue
        fi

        echo "Executing: python src/evaluate_qpp.py --model \"$model\" --task \"$task_name\" --dataset_family \"$dataset_family\" --scores_file \"$scores_file\" --output_dir \"$output_dir\""
        python src/evaluate_qpp.py \
            --model "$model" \
            --task "$task_name" \
            --dataset_family "$dataset_family" \
            --scores_file "$scores_file" \
            --output_dir "$output_dir"
        echo "  - Finished Task: $task_name"
    done

    # Process TEMPO datasets
    echo ""
    echo "--- TEMPO Datasets ---"
    for task_name in "${TEMPO_TASKS[@]}"; do
        echo ""
        echo "  - Running Task: $task_name"
        dataset_family="TEMPO"
        scores_file="results/scores_results/results_tempo/$model/$task_name/scores.json" 
        output_dir="$OUTPUT_BASE_DIR/$dataset_family/$task_name"
        results_file="$output_dir/${model}_qpp_results.json"

        mkdir -p "$output_dir"

        if [ ! -f "$scores_file" ]; then
            echo "    Scores file not found at $scores_file. Skipping $task_name."
            continue
        fi

        if [ -f "$results_file" ]; then
            echo "    Results for $model on $task_name (TEMPO) already exist. Skipping."
            continue
        fi

        echo "Executing: python src/evaluate_qpp.py --model \"$model\" --task \"$task_name\" --dataset_family \"$dataset_family\" --scores_file \"$scores_file\" --output_dir \"$output_dir\""
        python src/evaluate_qpp.py \
            --model "$model" \
            --task "$task_name" \
            --dataset_family "$dataset_family" \
            --scores_file "$scores_file" \
            --output_dir "$output_dir"
        echo "  - Finished Task: $task_name"
    done

done

python src/summary_qpp.py

echo ""
echo "========================"
echo "QPP Evaluation Complete"