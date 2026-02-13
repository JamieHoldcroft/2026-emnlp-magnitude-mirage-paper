#!/bin/bash
#===============================================================================
# PHASE 10: Robustness Evaluation
#
# PURPOSE:
#   - Test retrieval model robustness to query perturbations
#   - Measure performance degradation under adversarial conditions
#   - Critical for reproducibility claims
#
# PERTURBATION TYPES:
#   1. Query paraphrasing (5 variants)
#   2. Synonym replacement (3 levels)
#   3. Adversarial token insertion (3 levels)
#   4. Query length perturbation (expand/contract)
#
# CACHE: REUSES Phase 1 cache (efficient!)
#
# OUTPUT:
#   - sigir_results/phase10_robustness/{task}/{model}_{perturbation}/results.json
#   - sigir_results/phase10_robustness/robustness_summary.json
#===============================================================================

set -e

OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase10_robustness}"
CACHE_DIR="${CACHE_DIR:-cache}"  # REUSES Phase 1 cache
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

# Top 5 models to test (from Phase 1 results)
MODELS="e5 bge qwen2 reasonir"

# Perturbation types
PERTURBATIONS=("paraphrase_0" "paraphrase_1" "paraphrase_2" "synonym_0" "synonym_1" "adversarial_0" "adversarial_1" "length_expand" "length_contract")

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "PHASE 10: Robustness Evaluation"
echo "=============================================="
echo "Cache: ${CACHE_DIR} (reusing Phase 1)"
echo "Models: ${MODELS}"
echo "Perturbations: ${#PERTURBATIONS[@]} types"
echo ""

# Verify Phase 1 cache exists
if [ ! -d "${CACHE_DIR}/doc_emb" ]; then
    echo "ERROR: Phase 1 cache not found at ${CACHE_DIR}/doc_emb"
    echo "Run phase1_build_cache.sh first!"
    exit 1
fi

mkdir -p ${OUTPUT_DIR}

# Install NLTK if not already available
echo "Checking NLTK installation..."
python -c "import nltk" 2>/dev/null || pip install nltk

#-------------------------------------------------------------------------------
# Helper script to generate perturbed queries
#-------------------------------------------------------------------------------

cat > ${OUTPUT_DIR}/generate_perturbed_queries.py << 'PYTHON_SCRIPT'
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
from utils.perturbations import apply_perturbation

def generate_perturbed_dataset(task, perturbation_type, data_dir, output_file):
    """Generate perturbed query dataset."""
    from datasets import load_dataset

    # Load original queries
    dataset = load_dataset(data_dir, 'examples')[task]

    output_examples = []
    for example in dataset:
        query = example['query']
        
        # Parse perturbation type and parameters
        if perturbation_type.startswith('paraphrase_'):
            variant_id = int(perturbation_type.split('_')[1])
            perturbed = apply_perturbation(query, 'paraphrase', variant_id=variant_id)

        elif perturbation_type.startswith('synonym_'):
            num_replacements = int(perturbation_type.split('_')[1]) + 1
            perturbed = apply_perturbation(query, 'synonym', num_replacements=num_replacements)

        elif perturbation_type.startswith('adversarial_'):
            num_tokens = int(perturbation_type.split('_')[1]) + 1
            perturbed = apply_perturbation(query, 'adversarial', num_tokens=num_tokens)

        elif perturbation_type == 'length_expand':
            perturbed = apply_perturbation(query, 'length_expand')

        elif perturbation_type == 'length_contract':
            perturbed = apply_perturbation(query, 'length_contract')

        else:
            perturbed = query

        # Create updated example
        new_example = dict(example)
        new_example['query'] = perturbed
        output_examples.append(new_example)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_examples, f, indent=2)

    print(f"Generated {len(output_examples)} perturbed queries for {task}/{perturbation_type}")
    return output_file


if __name__ == '__main__':
    task = sys.argv[1]
    perturbation = sys.argv[2]
    data_dir = sys.argv[3]
    output_file = sys.argv[4]

    generate_perturbed_dataset(task, perturbation, data_dir, output_file)
PYTHON_SCRIPT

#-------------------------------------------------------------------------------
# Run robustness experiments
#-------------------------------------------------------------------------------

for task in "${TASKS[@]}"; do
    echo ""
    echo "=== Task: ${task} ==="

    for model in ${MODELS}; do
        for perturb in "${PERTURBATIONS[@]}"; do

            echo ""
            echo ">>> ${task} / ${model} / ${perturb}"

            output_path="${OUTPUT_DIR}/${task}/${model}_${perturb}"
            log_file="${LOG_DIR}/phase10_${task}_${model}_${perturb}_${TIMESTAMP}.log"

            if [ -f "${output_path}/results.json" ]; then
                echo "    [SKIP] Already complete"
                continue
            fi

            mkdir -p ${output_path}

            # Generate perturbed queries
            perturbed_queries_file="${output_path}/perturbed_queries.json"

            python ${OUTPUT_DIR}/generate_perturbed_queries.py \
                ${task} \
                ${perturb} \
                ${DATA_DIR} \
                ${perturbed_queries_file}

            # Run retrieval with perturbed queries
            python src/run.py \
                --task ${task} \
                --model ${model} \
                --cache_dir ${CACHE_DIR} \
                --output_dir ${output_path} \
                --dataset_source ${DATA_DIR} \
                --input_file ${perturbed_queries_file} \
                2>&1 | tee ${log_file}

        done
    done
done

#-------------------------------------------------------------------------------
# Create robustness summary
#-------------------------------------------------------------------------------

echo ""
echo "=== Creating robustness summary ==="

python << 'EOF'
import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

output_dir = os.environ.get('OUTPUT_DIR', 'sigir_results/phase10_robustness')

# Collect all results
results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
baselines = defaultdict(lambda: defaultdict(float))

for task_dir in Path(output_dir).iterdir():
    if not task_dir.is_dir() or task_dir.name == "logs":
        continue

    task = task_dir.name

    for exp_dir in task_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        # Parse: {model}_{perturbation}
        parts = exp_dir.name.rsplit("_", 1)
        if len(parts) < 2:
            continue

        # Handle multi-part perturbations (e.g., paraphrase_0)
        model = parts[0]
        for perturb_type in ['paraphrase', 'synonym', 'adversarial', 'length']:
            if perturb_type in exp_dir.name:
                # Extract full perturbation name
                name_parts = exp_dir.name.split(f"{model}_", 1)
                if len(name_parts) == 2:
                    perturbation = name_parts[1]
                else:
                    perturbation = exp_dir.name
                break
        else:
            continue

        res_file = exp_dir / "results.json"

        if not res_file.exists():
            continue

        try:
            with open(res_file) as f:
                res = json.load(f)

            ndcg = res.get("ndcg@10", 0)
            results[task][model][perturbation] = ndcg

        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")

# Get baseline performance (original queries from Phase 1)
phase1_dir = Path("sigir_results/phase1_cache_build")
if phase1_dir.exists():
    for task in results:
        for model in results[task]:
            # Structure: phase1_cache_build/{model}/{task}/{task}_{model}_long_False/results.json
            baseline_file = phase1_dir / model / task / f"{task}_{model}_long_False" / "results.json"
            if baseline_file.exists():
                try:
                    with open(baseline_file) as f:
                        res = json.load(f)
                    baselines[task][model] = res.get("ndcg@10", 0)
                except:
                    pass

# Calculate robustness metrics
summary = {
    "by_model": {},
    "by_perturbation": defaultdict(list),
    "aggregate_metrics": {}
}

print("\n" + "="*80)
print("ROBUSTNESS EVALUATION SUMMARY")
print("="*80)

for model in set().union(*[set(results[task].keys()) for task in results]):
    model_drops = []
    model_data = {}

    for task in results:
        if model not in results[task]:
            continue

        baseline = baselines.get(task, {}).get(model, 0)
        if baseline == 0:
            continue

        task_data = {}

        for perturbation, ndcg in results[task][model].items():
            drop = baseline - ndcg
            drop_pct = (drop / baseline * 100) if baseline > 0 else 0

            task_data[perturbation] = {
                "ndcg@10": ndcg,
                "baseline_ndcg@10": baseline,
                "drop": drop,
                "drop_pct": drop_pct
            }

            model_drops.append(drop_pct)
            summary["by_perturbation"][perturbation].append(drop_pct)

        model_data[task] = task_data

    # Average robustness for this model
    avg_drop = np.mean(model_drops) if model_drops else 0
    std_drop = np.std(model_drops) if model_drops else 0

    summary["by_model"][model] = {
        "tasks": model_data,
        "avg_performance_drop_pct": avg_drop,
        "std_performance_drop_pct": std_drop,
        "robustness_score": 100 - avg_drop  # Higher is better
    }

    print(f"\n{model.upper()}:")
    print(f"  Avg Performance Drop: {avg_drop:.2f}% ± {std_drop:.2f}%")
    print(f"  Robustness Score: {100 - avg_drop:.2f}/100")

# Aggregate by perturbation type
print("\n" + "="*80)
print("PERFORMANCE DROP BY PERTURBATION TYPE")
print("="*80)

for perturbation, drops in summary["by_perturbation"].items():
    avg_drop = np.mean(drops) if drops else 0
    print(f"{perturbation:<25}: {avg_drop:.2f}% average drop")

# Overall summary
all_drops = []
for drops in summary["by_perturbation"].values():
    all_drops.extend(drops)

summary["aggregate_metrics"] = {
    "overall_avg_drop_pct": np.mean(all_drops) if all_drops else 0,
    "overall_std_drop_pct": np.std(all_drops) if all_drops else 0,
    "most_vulnerable_perturbation": max(summary["by_perturbation"].items(),
                                         key=lambda x: np.mean(x[1]))[0] if summary["by_perturbation"] else None,
    "most_robust_model": min(summary["by_model"].items(),
                              key=lambda x: x[1]["avg_performance_drop_pct"])[0] if summary["by_model"] else None
}

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print(f"Overall Avg Drop: {summary['aggregate_metrics']['overall_avg_drop_pct']:.2f}%")
print(f"Most Vulnerable To: {summary['aggregate_metrics']['most_vulnerable_perturbation']}")
print(f"Most Robust Model: {summary['aggregate_metrics']['most_robust_model']}")

# Save
with open(os.path.join(output_dir, "robustness_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFull results saved to {output_dir}/robustness_summary.json")
EOF

echo ""
echo "=============================================="
echo "PHASE 10 COMPLETE"
echo "=============================================="
echo ""
echo "Key output: ${OUTPUT_DIR}/robustness_summary.json"
