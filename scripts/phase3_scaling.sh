#!/bin/bash
#===============================================================================
# PHASE 3: Corpus Scaling Analysis
#
# PURPOSE:
#   - Measure how indexing/search scales with corpus size
#   - Test sizes: 1K, 5K, 10K, 25K, 50K, 100K, full
#
# CACHE: SEPARATE cache per size (must measure TRUE indexing for each)
#===============================================================================

set -e

OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase3_scaling}"
BASE_CACHE_DIR="${BASE_CACHE_DIR:-cache_scaling}"
DATA_DIR="xlangai/BRIGHT"  # HuggingFace dataset

# ALL 12 BRIGHT tasks
TASKS=(
    # "biology"
    # "earth_science"
    # "economics"
    # "psychology"
    # "robotics"
    # "stackoverflow"
    # "sustainable_living"
    "leetcode"
    # "pony"
    # "aops"
    # "theoremqa_questions"
    # "theoremqa_theorems"
)

MODEL="e5"

SIZES="100000" #1000 5000 10000 25000 50000

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "PHASE 3: Corpus Scaling Analysis"
echo "=============================================="

for task in "${TASKS[@]}"; do
    echo ""
    echo "=== Task: ${task} ==="

    for size in ${SIZES}; do
        echo ""
        echo ">>> Corpus size: ${size}"

        # SEPARATE cache for each size (retriever handles task organization)
        cache_dir="${BASE_CACHE_DIR}/size_${size}"
        output_path="${OUTPUT_DIR}/${task}_size_${size}"

        if [ -f "${output_path}/efficiency.json" ]; then
            echo "    [SKIP] Already complete"
            continue
        fi

        python src/run.py \
            --task ${task} \
            --model ${MODEL} \
            --corpus_size ${size} \
            --cache_dir ${cache_dir} \
            --output_dir ${output_path} \
            --dataset_source ${DATA_DIR}
    done

    # Full corpus
    echo ""
    echo ">>> Corpus size: FULL"
    python src/run.py \
        --task ${task} \
        --model ${MODEL} \
        --cache_dir ${BASE_CACHE_DIR}/size_full \
        --output_dir ${OUTPUT_DIR}/${task}_size_full \
        --dataset_source ${DATA_DIR}
done

echo ""
echo "PHASE 3 COMPLETE"
