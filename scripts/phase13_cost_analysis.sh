#!/bin/bash
#===============================================================================
# PHASE 13: Cost-Effectiveness Analysis
#
# PURPOSE:
#   - Measure compute costs (GPU hours, memory, FLOPs)
#   - Track Azure OpenAI API costs
#   - Generate cost/performance Pareto frontiers
#
# METRICS:
#   - $/1K queries
#   - Cost per nDCG point
#   - ROI vs fine-tuning baseline ($50K-$200K)
#
# CACHE: REUSES Phase 1 cache
#===============================================================================

set -e

OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase13_cost}"
CACHE_DIR="${CACHE_DIR:-cache}"
DATA_DIR="xlangai/BRIGHT"

TASKS=("biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow"
       "sustainable_living" "leetcode" "pony" "aops" "theoremqa_questions" "theoremqa_theorems")

# All models including Azure OpenAI
MODELS="bm25 sbert bge e5 qwen2 grit reasonir openai"

# Azure OpenAI pricing (example rates - adjust to actual)
# text-embedding-3-large: $0.13 per 1M tokens
AZURE_PRICE_PER_1M_TOKENS=0.13

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
mkdir -p ${OUTPUT_DIR}/logs

echo "====================================================="
echo "PHASE 13: Cost-Effectiveness Analysis"
echo "====================================================="

# Enable Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://openaireceiptwestus.openai.azure.com/"
export AZURE_OPENAI_API_KEY="FQDQXtUOBMX0hoP1a62RJCBQphC0rtB8U0F7Ljkbk50JVxcZxd14JQQJ99BFAC4f1cMXJ3w3AAABACOGvW62"
export AZURE_DEPLOYMENT_NAME="text-embedding-3-large"
export AZURE_API_VERSION="2024-05-01-preview"

for model in ${MODELS}; do
    for task in "${TASKS[@]}"; do

        output_path="${OUTPUT_DIR}/${model}/${task}"

        if [ -f "${output_path}/cost_metrics.json" ]; then
            echo "[SKIP] ${model}/${task}"
            continue
        fi

        echo ">>> Model: ${model}, Task: ${task}"
        mkdir -p ${output_path}

        # Track GPU time
        start_time=$(date +%s)
        start_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo "0")

        python src/run.py \
            --task ${task} \
            --model ${model} \
            --cache_dir ${CACHE_DIR} \
            --output_dir ${output_path} \
            --dataset_source ${DATA_DIR}

        end_time=$(date +%s)
        end_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo "0")

        gpu_seconds=$((end_time - start_time))
        mem_usage=$((end_mem - start_mem))

        # Calculate costs
        python << EOF
import json
import os

gpu_hours = ${gpu_seconds} / 3600.0
# A100-80GB cost: ~$4/hour on cloud
gpu_cost = gpu_hours * 4.0

# For Azure OpenAI, estimate from results
results_file = "${output_path}/results.json"
if os.path.exists(results_file):
    with open(results_file) as f:
        results = json.load(f)

    ndcg = results.get("ndcg@10", 0)

    # Estimate API cost (if using OpenAI model)
    api_cost = 0
    if "${model}" == "openai":
        # Rough estimate: 500 queries * 100 tokens/query * ${AZURE_PRICE_PER_1M_TOKENS}/1M
        api_cost = 500 * 100 * ${AZURE_PRICE_PER_1M_TOKENS} / 1000000

    total_cost = gpu_cost + api_cost
    cost_per_1k_queries = (total_cost / 500) * 1000  # Assuming 500 queries

    cost_metrics = {
        "model": "${model}",
        "task": "${task}",
        "gpu_hours": gpu_hours,
        "gpu_cost_usd": gpu_cost,
        "api_cost_usd": api_cost,
        "total_cost_usd": total_cost,
        "cost_per_1k_queries_usd": cost_per_1k_queries,
        "ndcg@10": ndcg,
        "cost_per_ndcg_point": total_cost / ndcg if ndcg > 0 else float('inf'),
        "memory_mb": ${mem_usage}
    }

    with open("${output_path}/cost_metrics.json", "w") as f:
        json.dump(cost_metrics, f, indent=2)

    print(f"Cost: \${total_cost:.2f}, Cost/1K queries: \${cost_per_1k_queries:.3f}")
EOF

    done
done

# Generate Pareto frontier
python << 'EOF'
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

output_dir = "${OUTPUT_DIR}"
costs = []
ndcgs = []
labels = []

# Iterate through model directories
for model_dir in Path(output_dir).iterdir():
    if not model_dir.is_dir() or model_dir.name == "logs":
        continue

    # Iterate through task directories within each model
    for task_dir in model_dir.iterdir():
        if not task_dir.is_dir():
            continue

        cost_file = task_dir / "cost_metrics.json"
        if not cost_file.exists():
            continue

        try:
            with open(cost_file) as f:
                data = json.load(f)
            costs.append(data["cost_per_1k_queries_usd"])
            ndcgs.append(data["ndcg@10"])
            labels.append(f"{data['model']}")
        except Exception as e:
            print(f"Error processing {cost_file}: {e}")

# Plot Pareto frontier
plt.figure(figsize=(10, 6))
plt.scatter(costs, ndcgs)
for i, label in enumerate(labels):
    plt.annotate(label, (costs[i], ndcgs[i]))
plt.xlabel("Cost per 1K queries (USD)")
plt.ylabel("nDCG@10")
plt.title("Cost-Effectiveness Pareto Frontier")
plt.savefig(f"{output_dir}/pareto_frontier.png", dpi=300, bbox_inches='tight')
print(f"Pareto frontier saved to {output_dir}/pareto_frontier.png")
EOF

echo "PHASE 13 COMPLETE"
