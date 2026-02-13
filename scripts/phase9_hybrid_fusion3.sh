#!/bin/bash
#===============================================================================
# PHASE 9: Hybrid Retrieval Fusion
#
# PURPOSE:
#   - Combine sparse (BM25) and dense models with different fusion methods
#   - Test RRF, Linear, and Dynamic weighting strategies
#   - Measure effectiveness gain over individual models
#
# CACHE: REUSES Phase 1 cache (document embeddings already exist)
#        Very efficient - only fusion computation is new!
#
# OUTPUT:
#   - sigir_results/phase9_hybrid/{task}/{dense_model}_{fusion}/results.json
#   - sigir_results/phase9_hybrid/fusion_comparison.json
#
# KEY METRICS:
#   - nDCG@10 gain over best individual model
#   - Optimal fusion method per task
#   - Effectiveness vs latency trade-off
#===============================================================================

set -e

export JAVA_HOME=/leonardo_scratch/fast/L-AUT_024/jdk-21-temurin
export PATH=$JAVA_HOME/bin:$PATH

export HF_HOME=/leonardo_scratch/fast/L-AUT_024/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export XDG_CACHE_HOME=/leonardo_scratch/fast/L-AUT_024/.cache

# important: datasets uses temp space while processing
export TMPDIR=/leonardo_scratch/fast/L-AUT_024/.tmp
export TEMP=/leonardo_scratch/fast/L-AUT_024/.tmp
export TMP=/leonardo_scratch/fast/L-AUT_024/.tmp





OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase9_hybrid}"
CACHE_DIR="${CACHE_DIR:-cache}"  # REUSES Phase 1 cache!
DATA_DIR="xlangai/BRIGHT"

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

# Dense models to combine with BM25
DENSE_MODELS="inst-xl" #qwen sf reasonir contriever inst-l" #e5 bge reasonir

# Fusion methods
FUSION_METHODS="rrf linear dynamic"

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "PHASE 9: Hybrid Retrieval Fusion"
echo "=============================================="
echo "Cache: ${CACHE_DIR} (reusing Phase 1)"
echo "Dense models: ${DENSE_MODELS}"
echo "Fusion methods: ${FUSION_METHODS}"
echo ""

# Verify Phase 1 cache exists
if [ ! -d "${CACHE_DIR}/doc_emb" ]; then
    echo "ERROR: Phase 1 cache not found at ${CACHE_DIR}/doc_emb"
    echo "Run phase1_build_cache.sh first!"
    exit 1
fi

mkdir -p ${OUTPUT_DIR}

#-------------------------------------------------------------------------------
# Run all combinations: task × dense_model × fusion_method
#-------------------------------------------------------------------------------

for task in "${TASKS[@]}"; do
    echo ""
    echo "=== Task: ${task} ==="

    for dense_model in ${DENSE_MODELS}; do
        for fusion in ${FUSION_METHODS}; do

            echo ""
            echo ">>> ${task} / BM25+${dense_model} / ${fusion}"

            output_path="${OUTPUT_DIR}/${task}/${dense_model}_${fusion}"
            log_file="${LOG_DIR}/phase9_${task}_${dense_model}_${fusion}_${TIMESTAMP}.log"

            if [ -f "${output_path}/results.json" ]; then
                echo "    [SKIP] Already complete"
                continue
            fi

            mkdir -p ${output_path}

            # Use model_id to identify this as hybrid variant in outputs
            model_id="hybrid_bm25_${dense_model}_${fusion}"

            python src/run.py \
                --task ${task} \
                --model hybrid \
                --model_id ${model_id} \
                --fusion_method ${fusion} \
                --dense_model ${dense_model} \
                --cache_dir ${CACHE_DIR} \
                --output_dir ${output_path} \
                --dataset_source ${DATA_DIR} \
                2>&1 | tee ${log_file}

        done
    done
done

#-------------------------------------------------------------------------------
# Create fusion comparison summary
#-------------------------------------------------------------------------------

echo ""
echo "=== Creating fusion comparison summary ==="

python << 'EOF'
import os
import json
from pathlib import Path
from collections import defaultdict

output_dir = os.environ.get('OUTPUT_DIR', 'sigir_results/phase9_hybrid')

# Collect all results
results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

for task_dir in Path(output_dir).iterdir():
    if not task_dir.is_dir() or task_dir.name == "logs":
        continue

    task = task_dir.name

    for config_dir in task_dir.iterdir():
        if not config_dir.is_dir():
            continue

        # Parse: {dense_model}_{fusion}
        parts = config_dir.name.rsplit("_", 1)
        if len(parts) != 2:
            continue

        dense_model, fusion = parts

        res_file = config_dir / "results.json"
        eff_file = config_dir / "efficiency.json"

        if not res_file.exists():
            continue

        try:
            with open(res_file) as f:
                res = json.load(f)

            ndcg = res.get("ndcg@10", 0)

            latency = 0
            if eff_file.exists():
                with open(eff_file) as f:
                    eff = json.load(f)
                    latency = eff.get("query_latency", {}).get("total", {}).get("mean_ms", 0)

            results[task][dense_model][fusion] = {
                "ndcg@10": ndcg,
                "latency_ms": latency,
                "recall@10": res.get("recall@10", 0),
                "recall@100": res.get("recall@100", 0)
            }

        except Exception as e:
            print(f"Error processing {config_dir}: {e}")

# Find best fusion method per task
summary = {
    "by_task": {},
    "by_dense_model": defaultdict(list),
    "best_configurations": []
}

for task, dense_models in results.items():
    summary["by_task"][task] = {}

    for dense_model, fusions in dense_models.items():
        summary["by_task"][task][dense_model] = fusions

        # Track average nDCG per fusion method across tasks
        for fusion, metrics in fusions.items():
            summary["by_dense_model"][f"{dense_model}_{fusion}"].append(metrics["ndcg@10"])

# Average performance across tasks
print("\n" + "="*70)
print("HYBRID FUSION PERFORMANCE (averaged across tasks)")
print("="*70)

fusion_avg = defaultdict(list)
for config, ndcgs in summary["by_dense_model"].items():
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0
    dense_model, fusion = config.rsplit("_", 1)
    fusion_avg[fusion].append(avg_ndcg)

    print(f"{config:<30}: {avg_ndcg:.4f} nDCG@10 (avg across {len(ndcgs)} tasks)")

    summary["best_configurations"].append({
        "config": config,
        "dense_model": dense_model,
        "fusion": fusion,
        "avg_ndcg@10": avg_ndcg
    })

print("\n" + "="*70)
print("FUSION METHOD COMPARISON")
print("="*70)

for fusion, ndcgs in fusion_avg.items():
    avg = sum(ndcgs) / len(ndcgs) if ndcgs else 0
    print(f"{fusion.upper():<15}: {avg:.4f} nDCG@10 (avg across all configs)")

# Sort best configurations
summary["best_configurations"].sort(key=lambda x: x["avg_ndcg@10"], reverse=True)

print("\n" + "="*70)
print("TOP 5 HYBRID CONFIGURATIONS")
print("="*70)

for i, cfg in enumerate(summary["best_configurations"][:5], 1):
    print(f"{i}. {cfg['config']:<30} nDCG@10={cfg['avg_ndcg@10']:.4f}")

# Save summary
with open(os.path.join(output_dir, "fusion_comparison.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFull results saved to {output_dir}/fusion_comparison.json")
EOF

echo ""
echo "=============================================="
echo "PHASE 9 COMPLETE"
echo "=============================================="
echo ""
echo "Key output: ${OUTPUT_DIR}/fusion_comparison.json"
