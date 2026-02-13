#!/bin/bash
#===============================================================================
# PHASE 11: Cross-Benchmark Generalization (BEIR)
#
# PURPOSE:
#   - Evaluate models on BEIR benchmark for generalization
#   - Compare BRIGHT (reasoning-intensive) vs BEIR (semantic) performance
#   - Validate reasoning gap hypothesis
#
# BEIR DATASETS (13 total):
#   - arguana, fiqa, nfcorpus, quora, scidocs, scifact, trec-covid
#   - dbpedia-entity, fever, hotpotqa, nq, climate-fever, touche-2020
#
# CACHE: Creates separate BEIR cache
#
# OUTPUT:
#   - sigir_results/phase11_beir/{beir_dataset}_{model}/results.json
#   - sigir_results/phase11_beir/cross_benchmark_comparison.json
#===============================================================================

set -e

OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase11_beir}"
CACHE_DIR="${CACHE_DIR:-cache_beir}"
DATA_DIR="beir"  # BEIR uses different loader

# BEIR datasets (13 standard tasks)
BEIR_DATASETS=(
    "arguana"
    "fiqa"
    "nfcorpus"
    "quora"
    "scidocs"
    "scifact"
    "trec-covid"
    "dbpedia-entity"
    "fever"
    "hotpotqa"
    "nq"
    "climate-fever"
    "touche-2020"
)

# All 17 non-API models
MODELS="bm25 sbert bge contriever nomic inst-l inst-xl sf e5 qwen qwen2 grit m2 reasonir rader diver-retriever bge_ce"

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "PHASE 11: Cross-Benchmark Generalization (BEIR)"
echo "=============================================="
echo "BEIR Datasets: ${#BEIR_DATASETS[@]}"
echo "Models: ${MODELS}"
echo "Cache: ${CACHE_DIR}"
echo ""

mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

# Install BEIR if not available
echo "Checking BEIR installation..."
python -c "import beir" 2>/dev/null || pip install beir

#-------------------------------------------------------------------------------
# Helper script to run BEIR evaluation
#-------------------------------------------------------------------------------

cat > ${OUTPUT_DIR}/run_beir_eval.py << 'PYTHON_SCRIPT'
import sys
import json
import time
from pathlib import Path

# Import BEIR
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# Add src to path for retrievers
sys.path.insert(0, 'src')
from retrievers import RETRIEVAL_FUNCS

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def run_beir_evaluation(dataset_name, model_name, cache_dir, output_dir):
    """Run BEIR evaluation for a specific dataset and model."""

    # Download and load BEIR dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, f"datasets/{dataset_name}")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # Prepare data in BRIGHT format
    documents = [corpus[doc_id]['text'] for doc_id in corpus]
    doc_ids = list(corpus.keys())

    query_texts = [queries[qid] for qid in queries]
    query_ids = list(queries.keys())

    # Create excluded_ids dict
    excluded_ids = {}

    logging.info(f"Loaded {len(documents)} documents, {len(queries)} queries for {dataset_name}")

    # Get retrieval function
    if model_name not in RETRIEVAL_FUNCS:
        logging.error(f"Unknown model: {model_name}")
        return None

    retrieval_func = RETRIEVAL_FUNCS[model_name]

    # Run retrieval
    logging.info(f"Running {model_name} on {dataset_name}...")
    start_time = time.time()

    try:
        scores = retrieval_func(
            queries=query_texts,
            query_ids=query_ids,
            documents=documents,
            doc_ids=doc_ids,
            task=dataset_name,
            cache_dir=cache_dir,
            excluded_ids=excluded_ids,
            long_context=False,
            model_id=model_name
        )
    except Exception as e:
        logging.error(f"Error running {model_name}: {e}")
        return None

    query_time = time.time() - start_time

    # Convert scores to BEIR format
    results = {}
    for qid in query_ids:
        qid_str = str(qid)
        if qid_str in scores:
            results[qid] = scores[qid_str]

    # Evaluate
    logging.info("Evaluating...")
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, [10, 100])

    # Prepare output
    output = {
        "dataset": dataset_name,
        "model": model_name,
        "ndcg@10": ndcg["NDCG@10"],
        "ndcg@100": ndcg["NDCG@100"],
        "map@10": _map["MAP@10"],
        "map@100": _map["MAP@100"],
        "recall@10": recall["Recall@10"],
        "recall@100": recall["Recall@100"],
        "precision@10": precision["P@10"],
        "num_queries": len(queries),
        "num_documents": len(documents),
        "query_time_seconds": query_time
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    logging.info(f"Results saved to {output_path / 'results.json'}")
    logging.info(f"nDCG@10: {ndcg['NDCG@10']:.4f}")

    return output


if __name__ == '__main__':
    dataset = sys.argv[1]
    model = sys.argv[2]
    cache_dir = sys.argv[3]
    output_dir = sys.argv[4]

    run_beir_evaluation(dataset, model, cache_dir, output_dir)
PYTHON_SCRIPT

#-------------------------------------------------------------------------------
# Run BEIR experiments
#-------------------------------------------------------------------------------

for model in ${MODELS}; do
    echo ""
    echo "=== Model: ${model} ==="

    for dataset in "${BEIR_DATASETS[@]}"; do

        echo ""
        echo ">>> Model: ${model}, Dataset: ${dataset}"

        output_path="${OUTPUT_DIR}/${model}/${dataset}"
        log_file="${LOG_DIR}/phase11_${model}_${dataset}_${TIMESTAMP}.log"

        if [ -f "${output_path}/results.json" ]; then
            echo "    [SKIP] Already complete"
            continue
        fi

        mkdir -p ${output_path}

        # Run BEIR evaluation
        python ${OUTPUT_DIR}/run_beir_eval.py \
            ${dataset} \
            ${model} \
            ${CACHE_DIR} \
            ${output_path} \
            2>&1 | tee ${log_file}

    done
done

#-------------------------------------------------------------------------------
# Create cross-benchmark comparison
#-------------------------------------------------------------------------------

echo ""
echo "=== Creating cross-benchmark comparison ==="

python << 'EOF'
import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

output_dir = os.environ.get('OUTPUT_DIR', 'sigir_results/phase11_beir')

# Collect BEIR results with new structure: {model}/{dataset}/
beir_results = defaultdict(lambda: defaultdict(dict))

for model_dir in Path(output_dir).iterdir():
    if not model_dir.is_dir() or model_dir.name == "logs":
        continue

    model = model_dir.name

    for dataset_dir in model_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        res_file = dataset_dir / "results.json"
        if not res_file.exists():
            continue

        try:
            with open(res_file) as f:
                res = json.load(f)

            beir_results[model][dataset] = {
                "ndcg@10": res.get("ndcg@10", 0),
                "recall@10": res.get("recall@10", 0),
                "recall@100": res.get("recall@100", 0)
            }

        except Exception as e:
            print(f"Error processing {dataset_dir}: {e}")

# Collect BRIGHT results (from Phase 1) with new structure: {model}/{task}/
bright_results = defaultdict(lambda: defaultdict(dict))
phase1_dir = Path("sigir_results/phase1_cache_build")

if phase1_dir.exists():
    for model_dir in phase1_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == "logs":
            continue

        model = model_dir.name

        for task_dir in model_dir.iterdir():
            if not task_dir.is_dir():
                continue

            task = task_dir.name
            res_file = task_dir / "results.json"
            if not res_file.exists():
                continue

            try:
                with open(res_file) as f:
                    res = json.load(f)

                bright_results[model][task] = {
                    "ndcg@10": res.get("ndcg@10", 0),
                    "recall@10": res.get("recall@10", 0),
                    "recall@100": res.get("recall@100", 0)
                }

            except:
                pass

# Compare BEIR vs BRIGHT performance
summary = {
    "by_model": {},
    "reasoning_gap": {}
}

print("\n" + "="*80)
print("CROSS-BENCHMARK COMPARISON: BRIGHT vs BEIR")
print("="*80)

for model in set(beir_results.keys()) & set(bright_results.keys()):
    # Average performance on each benchmark
    beir_ndcg = np.mean([metrics["ndcg@10"] for metrics in beir_results[model].values()])
    bright_ndcg = np.mean([metrics["ndcg@10"] for metrics in bright_results[model].values()])

    gap = beir_ndcg - bright_ndcg
    gap_pct = (gap / beir_ndcg * 100) if beir_ndcg > 0 else 0

    summary["by_model"][model] = {
        "beir_avg_ndcg@10": beir_ndcg,
        "bright_avg_ndcg@10": bright_ndcg,
        "performance_gap": gap,
        "gap_percentage": gap_pct,
        "beir_datasets": beir_results[model],
        "bright_tasks": bright_results[model]
    }

    summary["reasoning_gap"][model] = gap_pct

    print(f"\n{model.upper()}:")
    print(f"  BEIR Avg nDCG@10:   {beir_ndcg:.4f}")
    print(f"  BRIGHT Avg nDCG@10: {bright_ndcg:.4f}")
    print(f"  Reasoning Gap:      {gap:.4f} ({gap_pct:.1f}%)")

# Overall findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

avg_gap = np.mean(list(summary["reasoning_gap"].values()))
max_gap_model = max(summary["reasoning_gap"].items(), key=lambda x: x[1])
min_gap_model = min(summary["reasoning_gap"].items(), key=lambda x: x[1])

print(f"Average Reasoning Gap: {avg_gap:.1f}%")
print(f"Largest Gap: {max_gap_model[0]} ({max_gap_model[1]:.1f}%)")
print(f"Smallest Gap: {min_gap_model[0]} ({min_gap_model[1]:.1f}%)")

summary["aggregate"] = {
    "avg_reasoning_gap_pct": avg_gap,
    "largest_gap_model": max_gap_model[0],
    "largest_gap_pct": max_gap_model[1],
    "smallest_gap_model": min_gap_model[0],
    "smallest_gap_pct": min_gap_model[1]
}

# Save
with open(os.path.join(output_dir, "cross_benchmark_comparison.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFull results saved to {output_dir}/cross_benchmark_comparison.json")
EOF

echo ""
echo "=============================================="
echo "PHASE 11 COMPLETE"
echo "=============================================="
echo ""
echo "Key output: ${OUTPUT_DIR}/cross_benchmark_comparison.json"
