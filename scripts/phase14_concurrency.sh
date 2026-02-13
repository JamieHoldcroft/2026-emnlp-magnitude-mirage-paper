#!/bin/bash
#===============================================================================
# PHASE 14: Concurrency & Load Testing
#
# PURPOSE:
#   - Test throughput under concurrent query load
#   - Measure p50/p95/p99 latency at scale
#   - Identify throughput saturation points
#
# CONCURRENCY LEVELS: 1, 5, 10, 50, 100 concurrent queries
#
# CACHE: REUSES Phase 1 cache
#===============================================================================

set -e

OUTPUT_DIR="${OUTPUT_DIR:-sigir_results/phase14_concurrency}"
CACHE_DIR="${CACHE_DIR:-cache}"
DATA_DIR="xlangai/BRIGHT"

# Representative tasks
TASKS=("biology" "stackoverflow" "theoremqa_questions")

# Top 5 models
MODELS="e5 qwen2 grit reasonir bge"

# Concurrency levels
CONCURRENCY_LEVELS="1 5 10 50 100"

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
mkdir -p ${OUTPUT_DIR}/logs

echo "====================================================="
echo "PHASE 14: Concurrency & Load Testing"
echo "====================================================="

# Create load testing script
cat > ${OUTPUT_DIR}/load_test.py << 'PYTHON_SCRIPT'
import sys
import json
import time
import concurrent.futures
from pathlib import Path

sys.path.insert(0, 'src')
from retrievers import RETRIEVAL_FUNCS
from datasets import load_dataset

def run_single_query(args):
    """Run a single query."""
    retrieval_func, query, qid, documents, doc_ids, task, cache_dir, model = args

    try:
        start = time.time()
        scores = retrieval_func(
            queries=[query],
            query_ids=[qid],
            documents=documents,
            doc_ids=doc_ids,
            task=task,
            cache_dir=cache_dir,
            excluded_ids={},
            long_context=False,
            model_id=model
        )
        latency = (time.time() - start) * 1000  # ms
        return {"qid": qid, "latency_ms": latency, "success": True}
    except Exception as e:
        return {"qid": qid, "error": str(e), "success": False}

def load_test(task, model, concurrency, cache_dir, data_dir, output_file):
    """Run load test with specified concurrency."""

    print(f"Loading {task} dataset...")
    dataset = load_dataset(data_dir, task)

    queries = dataset['examples']['query'][:100]  # Limit to 100 queries
    query_ids = dataset['examples']['_id'][:100]
    documents = dataset['corpus']
    doc_ids = list(range(len(documents)))

    retrieval_func = RETRIEVAL_FUNCS[model]

    print(f"Running {concurrency} concurrent queries...")
    args_list = [(retrieval_func, q, qid, documents, doc_ids, task, cache_dir, model)
                 for q, qid in zip(queries, query_ids)]

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(run_single_query, args_list))

    total_time = time.time() - start_time

    # Calculate metrics
    latencies = [r["latency_ms"] for r in results if r["success"]]
    successes = sum(1 for r in results if r["success"])

    metrics = {
        "task": task,
        "model": model,
        "concurrency": concurrency,
        "total_queries": len(queries),
        "successful_queries": successes,
        "total_time_seconds": total_time,
        "throughput_qps": successes / total_time if total_time > 0 else 0,
        "latency_p50_ms": sorted(latencies)[len(latencies)//2] if latencies else 0,
        "latency_p95_ms": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
        "latency_p99_ms": sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0,
        "latency_mean_ms": sum(latencies)/len(latencies) if latencies else 0,
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"QPS: {metrics['throughput_qps']:.2f}, p50: {metrics['latency_p50_ms']:.2f}ms")

if __name__ == '__main__':
    task = sys.argv[1]
    model = sys.argv[2]
    concurrency = int(sys.argv[3])
    cache_dir = sys.argv[4]
    data_dir = sys.argv[5]
    output_file = sys.argv[6]

    load_test(task, model, concurrency, cache_dir, data_dir, output_file)
PYTHON_SCRIPT

# Run concurrency tests
for task in "${TASKS[@]}"; do
    for model in ${MODELS}; do
        for concurrency in ${CONCURRENCY_LEVELS}; do

            output_path="${OUTPUT_DIR}/${task}/${model}_c${concurrency}"

            if [ -f "${output_path}/metrics.json" ]; then
                echo "[SKIP] ${task}/${model}/c${concurrency}"
                continue
            fi

            echo ">>> ${task} / ${model} / concurrency=${concurrency}"
            mkdir -p ${output_path}

            python ${OUTPUT_DIR}/load_test.py \
                ${task} \
                ${model} \
                ${concurrency} \
                ${CACHE_DIR} \
                ${DATA_DIR} \
                ${output_path}/metrics.json

        done
    done
done

# Generate summary
python << 'EOF'
import json
from pathlib import Path

output_dir = "${OUTPUT_DIR}"
summary = {}

for metrics_file in Path(output_dir).glob("*/*/metrics.json"):
    with open(metrics_file) as f:
        data = json.load(f)

    key = f"{data['model']}_c{data['concurrency']}"
    if key not in summary:
        summary[key] = []
    summary[key].append(data["throughput_qps"])

print("\nCONCURRENCY SUMMARY (Avg QPS across tasks):")
print("="*60)
for key, qps_list in sorted(summary.items()):
    avg_qps = sum(qps_list) / len(qps_list)
    print(f"{key:<30}: {avg_qps:.2f} QPS")

with open(f"{output_dir}/concurrency_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
EOF

echo "PHASE 14 COMPLETE"
