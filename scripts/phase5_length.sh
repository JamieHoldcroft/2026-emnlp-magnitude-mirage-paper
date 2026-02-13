#!/bin/bash
#===============================================================================
# PHASE 5: Context Length Sensitivity
#
# PURPOSE:
#   - Test document truncation lengths
#   - Find optimal length (effectiveness vs efficiency)
#
# CACHE: SEPARATE cache per length (embeddings differ with truncation!)
#===============================================================================

set -e

OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase5_length}"
BASE_CACHE_DIR="${BASE_CACHE_DIR:-cache_length}"
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

MODEL="diver-retriever"

DOC_LENGTHS="50 100 150 200"

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "PHASE 5: Context Length Sensitivity"
echo "=============================================="

for task in "${TASKS[@]}"; do
    echo ""
    echo "=== Task: ${task} ==="

    for len in ${DOC_LENGTHS}; do
        echo ""
        echo ">>> Document max length: ${len}"

        cache_dir="${BASE_CACHE_DIR}/len_${len}"
        output_path="${OUTPUT_DIR}/${task}_doclen_${len}"

        if [ -f "${output_path}/efficiency.json" ]; then
            echo "    [SKIP] Already complete"
            continue
        fi

        python src/run.py \
            --task ${task} \
            --model ${MODEL} \
            --doc_max_length ${len} \
            --cache_dir ${cache_dir} \
            --output_dir ${output_path} \
            --dataset_source ${DATA_DIR}
    done
done

echo ""
echo "PHASE 5 COMPLETE"
