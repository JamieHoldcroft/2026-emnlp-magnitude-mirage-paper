#!/bin/bash
#===============================================================================
# PHASE 7: Document Expansion Analysis
#
# PURPOSE:
#   - Compare original vs expanded documents
#   - Test: gold expansion, rechunking
#
# CACHE: SEPARATE cache per expansion strategy (different documents!)
#===============================================================================

set -e

OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase7_expansion}"
BASE_CACHE_DIR="${BASE_CACHE_DIR:-cache_expansion}"
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

MODEL="e5"

EXPANSIONS="None gold rechunk"

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "PHASE 7: Document Expansion"
echo "=============================================="

for task in "${TASKS[@]}"; do
    echo ""
    echo "=== Task: ${task} ==="

    for expansion in ${EXPANSIONS}; do
        echo ""
        echo ">>> Expansion: ${expansion}"

        if [ "${expansion}" == "None" ]; then
            exp_arg=""
            cache_dir="${BASE_CACHE_DIR}/original"
            output_path="${OUTPUT_DIR}/${task}_original"
        else
            exp_arg="--document_expansion ${expansion}"
            cache_dir="${BASE_CACHE_DIR}/${expansion}"
            output_path="${OUTPUT_DIR}/${task}_${expansion}"
        fi

        if [ -f "${output_path}/efficiency.json" ]; then
            echo "    [SKIP] Already complete"
            continue
        fi

        python src/run.py \
            --task ${task} \
            --model ${MODEL} \
            ${exp_arg} \
            --cache_dir ${cache_dir} \
            --output_dir ${output_path} \
            --dataset_source ${DATA_DIR}
    done
done

echo ""
echo "PHASE 7 COMPLETE"
