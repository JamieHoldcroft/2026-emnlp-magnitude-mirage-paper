#!/bin/bash
#===============================================================================
# PHASE 1: Build Document Embedding Cache
# 
# PURPOSE: 
#   1. Create document embeddings for ALL (model, task) pairs
#   2. Measure TRUE indexing time (no prior cache)
#   3. Run baseline effectiveness with ORIGINAL queries
#
# IMPORTANT: Delete cache folder before running this!
#            rm -rf cache/
#
# OUTPUT:
#   - cache/doc_emb/{model}/{task}/... (embeddings)
#   - sigir_results/phase1/{task}_{model}/efficiency.json (metrics)
#   - sigir_results/phase1/{task}_{model}/results.json (effectiveness)
#
# ESTIMATED TIME: 24-48 hours on A100-80GB
#===============================================================================

set -e

export HF_HOME=/leonardo_scratch/fast/L-AUT_024/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export XDG_CACHE_HOME=/leonardo_scratch/fast/L-AUT_024/.cache

# important: datasets uses temp space while processing
export TMPDIR=/leonardo_scratch/fast/L-AUT_024/.tmp
export TEMP=/leonardo_scratch/fast/L-AUT_024/.tmp
export TMP=/leonardo_scratch/fast/L-AUT_024/.tmp

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase1_cache_build}"
CACHE_DIR="${CACHE_DIR:-cache}"
DATA_DIR="xlangai/BRIGHT"  # HuggingFace dataset

# ALL 12 BRIGHT tasks
TASKS=(
    "biology"
    "earth_science" 
    "economics"
    "psychology"
    "robotics"
    "stackoverflow"
    "sustainable_living"
    "leetcode"
    "pony"
    "aops"
    "theoremqa_questions"
    "theoremqa_theorems"
)

# Models to build cache for
# ALL models from retrievers.py RETRIEVAL_FUNCS
# Grouped by size for better resource management

# Small/Medium models (can run with less GPU memory)
MODELS_SMALL="diver-retriever" #contriever  inst-xl diver-retriever rader nomic

# Large models (require more GPU memory)
# MODELS_LARGE="" #sf e5 qwen
#conda activate swift_1

# Cross-encoder models (different architecture, slower)
# MODELS_CE="bge_ce"

# API-based models (require API keys, set via --key argument)
# MODELS_API="cohere voyage openai google"
# Note: API models are disabled by default. Uncomment and provide API keys to use them.

# GPU settings
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

# Logging
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "PHASE 1: Building Document Embedding Cache"
echo "=============================================="
echo "Output: ${OUTPUT_DIR}"
echo "Cache: ${CACHE_DIR}"
echo "Start time: $(date)"
echo ""

# Check if cache exists - warn user
# if [ -d "${CACHE_DIR}/doc_emb" ]; then
#     echo "WARNING: Cache directory exists!"
#     echo "For TRUE indexing time measurement, delete it first:"
#     # echo "  rm -rf ${CACHE_DIR}"
#     echo ""
#     read -p "Continue anyway? (y/n) " -n 1 -r
#     echo
#     if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#         exit 1
#     fi
# fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

#-------------------------------------------------------------------------------
# Run experiments
#-------------------------------------------------------------------------------

run_experiment() {
    local task=$1
    local model=$2
    local output_path="${OUTPUT_DIR}/${model}/${task}"
    local log_file="${LOG_DIR}/phase1_${model}_${task}_${TIMESTAMP}.log"

    echo ""
    echo ">>> Model: ${model}, Task: ${task}"
    echo "    Output: ${output_path}"
    echo "    Log: ${log_file}"

    # Skip if already complete
    if [ -f "${output_path}/results.json" ]; then
        echo "    [SKIP] Already complete"
        return 0
    fi

    # Create output directory
    mkdir -p ${output_path}

    python src/run.py \
        --task ${task} \
        --model ${model} \
        --output_dir ${output_path} \
        --cache_dir ${CACHE_DIR} \
        --dataset_source ${DATA_DIR} \
        2>&1 | tee ${log_file}

    # Check if successful
    if [ -f "${output_path}/results.json" ]; then
        echo "    [SUCCESS]"
    else
        echo "    [FAILED] Check log: ${log_file}"
    fi
}

#-------------------------------------------------------------------------------
# Small models first (fast, CPU-friendly)
#-------------------------------------------------------------------------------

echo ""
echo "=== Building cache for small/medium models ==="

for model in ${MODELS_SMALL}; do
    for task in "${TASKS[@]}"; do
        run_experiment ${task} ${model}
    done
done

#-------------------------------------------------------------------------------
# Large models (require GPU)
#-------------------------------------------------------------------------------

# echo ""
# echo "=== Building cache for large models ==="

# for model in ${MODELS_LARGE}; do
#     for task in "${TASKS[@]}"; do
#         run_experiment ${task} ${model}
#     done
# done

#-------------------------------------------------------------------------------
# Cross-encoder models
#-------------------------------------------------------------------------------

echo ""
echo "=== Building cache for cross-encoder models ==="

# for model in ${MODELS_CE}; do
#     for task in "${TASKS[@]}"; do
#         run_experiment ${task} ${model}
#     done
# done

#-------------------------------------------------------------------------------
# Create summary of indexing times
#-------------------------------------------------------------------------------

echo ""
echo "=== Creating indexing time summary ==="

python << 'EOF'
import os
import json
from pathlib import Path

output_dir = os.environ.get('OUTPUT_DIR', 'sigir_results/phase1_cache_build')
summary = {
    "description": "TRUE indexing times measured with NO cache",
    "by_model": {},
    "by_task": {}
}

# Iterate through model directories
for model_dir in Path(output_dir).iterdir():
    if not model_dir.is_dir() or model_dir.name == "logs":
        continue

    model = model_dir.name

    # Iterate through task directories within each model
    for task_dir in model_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task = task_dir.name

        eff_file = task_dir / "efficiency.json"
        if not eff_file.exists():
            continue

        try:
            with open(eff_file) as f:
                data = json.load(f)

            indexing_time = data.get("indexing", {}).get("total_time_seconds", 0)
            used_cache = data.get("indexing", {}).get("used_cache", None)

            if model not in summary["by_model"]:
                summary["by_model"][model] = {}
            summary["by_model"][model][task] = {
                "indexing_seconds": indexing_time,
                "used_cache": used_cache
            }

            if task not in summary["by_task"]:
                summary["by_task"][task] = {}
            summary["by_task"][task][model] = indexing_time

        except Exception as e:
            print(f"Error processing {task_dir}: {e}")

# Save summary
with open(os.path.join(output_dir, "master_indexing_times.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to {output_dir}/master_indexing_times.json")
print(f"Processed {len(summary['by_model'])} models across {len(summary['by_task'])} tasks")
EOF

echo ""
echo "=============================================="
echo "PHASE 1 COMPLETE"
echo "End time: $(date)"
echo "=============================================="
