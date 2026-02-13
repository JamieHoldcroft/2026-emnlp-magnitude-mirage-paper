#!/bin/bash
#===============================================================================
# PHASE 8: Long Context Retrieval
#
# PURPOSE:
#   - Test on long_documents subset of BRIGHT
#   - Compare short vs long context performance
#
# CACHE: SEPARATE cache for long documents
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


OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase8_long_context}"
CACHE_DIR_SHORT="${CACHE_DIR_SHORT:-cache}"
CACHE_DIR_LONG="${CACHE_DIR_LONG:-cache_long}"
DATA_DIR="xlangai/BRIGHT"  # HuggingFace dataset

TASKS=(
    "biology"
    "earth_science"
    "economics"
    "psychology"
    "robotics"
    "stackoverflow"
    "sustainable_living"
    "pony"
)
#qwen bm25 sbert bge contriever nomic inst-l inst-xl sf e5 qwen2 diver-retriever qwen qwen2 
MODELS="inst-l inst-xl reasonir rader"

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "PHASE 8: Long Context Retrieval"
echo "=============================================="

for model in ${MODELS}; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo ">>> Model: ${model}, Task: ${task} (LONG CONTEXT)"

        output_path="${OUTPUT_DIR}/${model}/${task}"

        if [ -f "${output_path}/results.json" ]; then
            echo "    [SKIP] Already complete"
            continue
        fi

        # Create output directory
        mkdir -p ${output_path}

        python src/run.py \
            --task ${task} \
            --model ${model} \
            --long_context \
            --cache_dir ${CACHE_DIR_LONG} \
            --output_dir ${output_path} \
            --dataset_source ${DATA_DIR}
    done
done

echo ""
echo "PHASE 8 COMPLETE"
