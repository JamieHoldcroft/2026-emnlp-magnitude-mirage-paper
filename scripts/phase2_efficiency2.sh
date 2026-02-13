#!/bin/bash
#===============================================================================
# PHASE 2: Efficiency Profiling (Query Latency)
#
# PURPOSE:
#   - Measure query latency distribution (p50, p95, p99)
#   - Measure QPS (queries per second)
#   - Compare all models on same task
#
# CACHE: REUSES Phase 1 cache (document embeddings)
#        Do NOT report indexing time - use Phase 1 values instead
#
# OUTPUT:
#   - sigir_results/phase2/{model}/efficiency.json
#   - sigir_results/phase2/{model}/latency_raw.json (all individual latencies)
#===============================================================================

set -e
export HF_DATASETS_OFFLINE=1 
export HF_HOME=/leonardo_scratch/fast/L-AUT_024/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export XDG_CACHE_HOME=/leonardo_scratch/fast/L-AUT_024/.cache

# important: datasets uses temp space while processing
export TMPDIR=/leonardo_scratch/fast/L-AUT_024/.tmp
export TEMP=/leonardo_scratch/fast/L-AUT_024/.tmp
export TMP=/leonardo_scratch/fast/L-AUT_024/.tmp

OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase2_efficiency}"
CACHE_DIR="${CACHE_DIR:-cache}"  # Same as Phase 1!
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
    # "leetcode"
    # "pony"
    # "aops"
    "theoremqa_questions"
    "theoremqa_theorems"
)

# All models from retrievers.py (excluding API-based models by default)
MODELS="bge" #sbert

# API-based models (require API keys)
# MODELS_API="cohere voyage openai google"

# export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "PHASE 2: Efficiency Profiling"
echo "=============================================="
echo "Testing all ${#TASKS[@]} tasks"
echo "Cache: ${CACHE_DIR} (reusing Phase 1)"
echo ""

# Verify Phase 1 cache exists
if [ ! -d "${CACHE_DIR}/doc_emb" ]; then
    echo "ERROR: Phase 1 cache not found at ${CACHE_DIR}/doc_emb"
    echo "Run phase1_build_cache.sh first!"
    exit 1
fi

mkdir -p ${OUTPUT_DIR}

for model in ${MODELS}; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo ">>> Model: ${model}, Task: ${task}"

        output_path="${OUTPUT_DIR}/${model}/${task}"
        log_file="${LOG_DIR}/phase2_${model}_${task}_${TIMESTAMP}.log"

        if [ -f "${output_path}/efficiency.json" ]; then
            echo "    [SKIP] Already complete"
            continue
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
    done
done

echo ""
echo "=== Creating efficiency comparison table ==="

python << 'EOF'
import os
import json
from pathlib import Path

output_dir = os.environ.get('OUTPUT_DIR', 'sigir_results/phase2_efficiency')
results_by_model = {}

# Iterate through model directories
for model_dir in Path(output_dir).iterdir():
    if not model_dir.is_dir() or model_dir.name == "logs":
        continue

    model = model_dir.name
    model_results = []

    # Iterate through task directories within each model
    for task_dir in model_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task = task_dir.name
        eff_file = task_dir / "efficiency.json"
        res_file = task_dir / "results.json"

        if not eff_file.exists():
            continue

        try:
            with open(eff_file) as f:
                eff = json.load(f)

            ndcg = 0
            if res_file.exists():
                with open(res_file) as f:
                    res = json.load(f)
                    ndcg = res.get("ndcg@10", 0)

            model_results.append({
                "task": task,
                "qps": eff.get("query_latency", {}).get("total", {}).get("qps", 0),
                "latency_mean_ms": eff.get("query_latency", {}).get("total", {}).get("mean_ms", 0),
                "latency_p50_ms": eff.get("query_latency", {}).get("total", {}).get("p50_ms", 0),
                "latency_p95_ms": eff.get("query_latency", {}).get("total", {}).get("p95_ms", 0),
                "latency_p99_ms": eff.get("query_latency", {}).get("total", {}).get("p99_ms", 0),
                "ndcg@10": ndcg
            })
        except Exception as e:
            print(f"Error processing {task_dir}: {e}")

    if model_results:
        results_by_model[model] = model_results

# Print summary for each model
for model, tasks_results in sorted(results_by_model.items()):
    print(f"\n{'='*70}")
    print(f"Model: {model}")
    print("="*70)
    print(f"{'Task':<20} {'QPS':>10} {'Mean':>10} {'p50':>10} {'p95':>10} {'p99':>10} {'nDCG@10':>10}")
    print("-"*70)
    for r in tasks_results:
        print(f"{r['task']:<20} {r['qps']:>10.2f} {r['latency_mean_ms']:>10.2f} {r['latency_p50_ms']:>10.2f} {r['latency_p95_ms']:>10.2f} {r['latency_p99_ms']:>10.2f} {r['ndcg@10']:>10.4f}")

# Save
with open(os.path.join(output_dir, "efficiency_comparison.json"), "w") as f:
    json.dump(results_by_model, f, indent=2)

print(f"\n{'='*70}")
print(f"Summary saved to {output_dir}/efficiency_comparison.json")
print(f"Processed {len(results_by_model)} models")
EOF

echo ""
echo "PHASE 2 COMPLETE"
