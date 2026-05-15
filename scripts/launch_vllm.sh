#!/usr/bin/env bash
set -euo pipefail

source /data/fs201059/aa17626/FinCodeAgent/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/data/fs201059/aa17626/.cache/huggingface

vllm serve Qwen/Qwen3.6-27B \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser qwen3 \
    --download-dir "$HF_HOME/hub"
