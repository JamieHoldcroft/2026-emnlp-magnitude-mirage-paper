#!/bin/bash
#===============================================================================
# PHASE 4: Quantization Impact Analysis
#
# PURPOSE:
#   - Compare FP32, FP16, BF16, INT8, INT4 precision
#   - Measure quality degradation vs memory/speed gain
#
# CACHE: SEPARATE cache per precision (embeddings differ!)
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


OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase4_quantization}"
BASE_CACHE_DIR="${BASE_CACHE_DIR:-cache_quantization}"
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

MODEL="contriever"

PRECISIONS="fp32 fp16 bf16 int8 int4"

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "PHASE 4: Quantization Impact"
echo "=============================================="

for task in "${TASKS[@]}"; do
    echo ""
    echo "=== Task: ${task} ==="

    for prec in ${PRECISIONS}; do
        echo ""
        echo ">>> Precision: ${prec}"

        cache_dir="${BASE_CACHE_DIR}/prec_${prec}"
        output_path="${OUTPUT_DIR}/${task}_${prec}"

        if [ -f "${output_path}/efficiency.json" ]; then
            echo "    [SKIP] Already complete"
            continue
        fi

        python src/run.py \
            --task ${task} \
            --model ${MODEL} \
            --quantization ${prec} \
            --cache_dir ${cache_dir} \
            --output_dir ${output_path} \
            --dataset_source ${DATA_DIR} \
            --encode_batch_size 1
    done
done

echo ""
echo "PHASE 4 COMPLETE"
