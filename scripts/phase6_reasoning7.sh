#!/bin/bash
#===============================================================================
# PHASE 6: Reasoning Query Analysis (GPT-4, Llama-3, etc.)
#
# PURPOSE:
#   - Compare original queries vs LLM-augmented reasoning queries
#   - Test: gpt4_reason, llama3-70b_reason, claude-3-opus_reason, etc.
#
# CACHE: REUSES Phase 1 cache!!! 
#        Document embeddings are SAME regardless of query type
#        This is very efficient - no re-indexing needed!
#
# KEY METRIC: Efficiency Ratio = nDCG gain / latency penalty
#===============================================================================

set -e

OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase6_reasoning}"
CACHE_DIR="${CACHE_DIR:-cache}"  # SAME as Phase 1!
DATA_DIR="xlangai/BRIGHT"  # HuggingFace dataset

# All 12 tasks
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

# Models to test (all non-API models)
#        diver-retriever 
MODELS="sf e5  reasonir"
#bge sbert contriever
# Reasoning query types from BRIGHT

#['Gemini-1.0_reason', 'claude-3-opus_reason', 'documents', 'examples', 'gpt4_reason', 'grit_reason', 'llama3-70b_reason', 'long_documents']
REASONING_TYPES=(
    # ""                      # Empty = original queries (examples)
    "gpt4_reason"           # GPT-4 reasoning
    "llama3-70b_reason"     # Llama-3-70B reasoning  
    "claude-3-opus_reason"  # Claude reasoning
    "grit_reason"           # GritLM reasoning
    "Gemini-1.0_reason"     # Gemini reasoning
)

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "PHASE 6: Reasoning Query Analysis"
echo "=============================================="
echo "Cache: ${CACHE_DIR} (reusing Phase 1 - no re-indexing!)"
echo ""

# Verify Phase 1 cache exists
if [ ! -d "${CACHE_DIR}/doc_emb" ]; then
    echo "ERROR: Phase 1 cache not found!"
    echo "Run phase1_build_cache.sh first!"
    exit 1
fi

#-------------------------------------------------------------------------------
# Run ALL combinations: task × model × reasoning_type
#-------------------------------------------------------------------------------

for task in "${TASKS[@]}"; do
    echo ""
    echo "=== Task: ${task} ==="
    
    for model in ${MODELS}; do
        for reasoning in "${REASONING_TYPES[@]}"; do
            
            # Determine output name
            if [ -z "${reasoning}" ]; then
                reason_name="original"
                reasoning_arg=""
            else
                reason_name="${reasoning}"
                reasoning_arg="--reasoning ${reasoning}"
            fi
            
            output_path="${OUTPUT_DIR}/${task}/${model}_${reason_name}"
            
            echo ""
            echo ">>> ${task} / ${model} / ${reason_name}"
            
            if [ -f "${output_path}/results.json" ]; then
                echo "    [SKIP] Already complete"
                continue
            fi
            
            mkdir -p ${output_path}
            
            python src/run.py \
                --task ${task} \
                --model ${model} \
                ${reasoning_arg} \
                --cache_dir ${CACHE_DIR} \
                --output_dir ${output_path} \
                --dataset_source ${DATA_DIR} \
                2>&1 | tee ${output_path}/run.log
            
        done
    done
done

#-------------------------------------------------------------------------------
# Create comparison summary
#-------------------------------------------------------------------------------

echo ""
echo "=== Creating reasoning comparison summary ==="

python << 'EOF'
import os
import json
from pathlib import Path
from collections import defaultdict

output_dir = os.environ.get('OUTPUT_DIR', 'sigir_results/phase6_reasoning')

# Collect all results
results = defaultdict(lambda: defaultdict(dict))

for task_dir in Path(output_dir).iterdir():
    if not task_dir.is_dir():
        continue
    task = task_dir.name
    
    for exp_dir in task_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        parts = exp_dir.name.split("_", 1)
        if len(parts) != 2:
            continue
        model, reasoning = parts
        
        res_file = exp_dir / "results.json"
        eff_file = exp_dir / "efficiency.json"
        
        if not res_file.exists():
            continue
        
        try:
            with open(res_file) as f:
                res = json.load(f)
            
            latency = 0
            if eff_file.exists():
                with open(eff_file) as f:
                    eff = json.load(f)
                    latency = eff.get("query_latency", {}).get("total", {}).get("mean_ms", 0)
            
            results[task][model][reasoning] = {
                "ndcg@10": res.get("ndcg@10", 0),
                "latency_ms": latency
            }
        except Exception as e:
            print(f"Error: {e}")

# Calculate gains relative to original
summary = {"by_task": {}, "aggregated": defaultdict(list)}

for task, models in results.items():
    summary["by_task"].setdefault(task, {})
    
    for model, reasonings in models.items():
        if "original" not in reasonings:
            continue
        
        baseline = reasonings["original"]
        summary["by_task"][task][model] = {"original": baseline}
        
        for reasoning, metrics in reasonings.items():
            if reasoning == "original":
                continue
            
            ndcg_gain = metrics["ndcg@10"] - baseline["ndcg@10"]
            latency_penalty = metrics["latency_ms"] - baseline["latency_ms"]
            
            if latency_penalty > 0:
                efficiency_ratio = ndcg_gain / latency_penalty
            else:
                efficiency_ratio = float('inf') if ndcg_gain > 0 else 0
            
            summary["by_task"][task][model][reasoning] = {
                "ndcg@10": metrics["ndcg@10"],
                "ndcg_gain": ndcg_gain,
                "latency_ms": metrics["latency_ms"],
                "latency_penalty_ms": latency_penalty,
                "efficiency_ratio": efficiency_ratio
            }
            
            summary["aggregated"].setdefault(reasoning, []).append(ndcg_gain)

# Average gains across all tasks
print("\n" + "="*60)
print("REASONING QUERY EFFECTIVENESS GAINS (averaged across tasks)")
print("="*60)

for reasoning, gains in summary["aggregated"].items():
    avg_gain = sum(gains) / len(gains) if gains else 0
    print(f"{reasoning:<25}: +{avg_gain:.4f} nDCG@10 (avg)")

# Save
with open(os.path.join(output_dir, "reasoning_comparison.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFull results saved to {output_dir}/reasoning_comparison.json")
EOF

echo ""
echo "PHASE 6 COMPLETE"
