#!/bin/bash
#===============================================================================
# PHASE 12: Two-Stage Retrieval Pipelines (Retrieve + Rerank)
#
# PURPOSE:
#   - Test industry-standard retrieve-then-rerank pipelines
#   - Compare single-stage vs two-stage effectiveness
#   - Measure latency overhead of reranking
#
# PIPELINE: Stage 1 (Retrieve top-100) → Stage 2 (Rerank to top-10)
#
# CACHE: REUSES Phase 1 cache for retrievers
#
# OUTPUT:
#   - sigir_results/phase12_two_stage/{task}/{retriever}_{reranker}/results.json
#===============================================================================

set -e

OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase12_two_stage}"
CACHE_DIR="${CACHE_DIR:-cache}"
DATA_DIR="xlangai/BRIGHT"

TASKS=(
    "biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow"
    "sustainable_living" "leetcode" "pony" "aops" "theoremqa_questions" "theoremqa_theorems"
)

# Stage 1: First-stage retrievers
RETRIEVERS="bm25 e5 qwen2"

# Stage 2: Rerankers
RERANKERS="bge_ce"

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

mkdir -p ${OUTPUT_DIR}/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "====================================================="
echo "PHASE 12: Two-Stage Retrieval Pipelines"
echo "====================================================="

for task in "${TASKS[@]}"; do
    for retriever in ${RETRIEVERS}; do
        for reranker in ${RERANKERS}; do

            output_path="${OUTPUT_DIR}/${task}/${retriever}_${reranker}"

            if [ -f "${output_path}/results.json" ]; then
                echo "[SKIP] ${task}/${retriever}_${reranker}"
                continue
            fi

            echo ">>> ${task} / Stage1:${retriever} + Stage2:${reranker}"
            mkdir -p ${output_path}

            # Stage 1: Retrieve top-100
            python src/run.py \
                --task ${task} \
                --model ${retriever} \
                --cache_dir ${CACHE_DIR} \
                --output_dir ${output_path}/stage1 \
                --dataset_source ${DATA_DIR}

            # Stage 2: Rerank top-100 to top-10
            python src/run.py \
                --task ${task} \
                --model ${reranker} \
                --cache_dir ${CACHE_DIR} \
                --output_dir ${output_path} \
                --dataset_source ${DATA_DIR} \
                --input_file ${output_path}/stage1/score.json

        done
    done
done

echo "PHASE 12 COMPLETE"
