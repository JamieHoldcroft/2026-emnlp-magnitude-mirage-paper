# src/confidence_auroc.py

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve

from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def plot_roc_curves(roc_data, output_path, task_name, model_name):
    """Plots ROC curves for multiple k values on a single plot."""
    plt.figure(figsize=(8, 8))
    
    for k, data in roc_data.items():
        if data['auroc'] is not None and data['fpr'] is not None:
            label = f"k={k} (AUROC = {data['auroc']:.3f})"
            plt.plot(data['fpr'], data['tpr'], label=label)

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Confidence-Success Separability ({model_name}, {task_name})')
    plt.legend(loc="lower right")
    plt.grid(False)
    
    plt.tight_layout()
    print(f"Saving ROC plot to: {output_path}")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curve_k10(roc_data, output_path, task_name, model_name):
    """
    Plot a single ROC curve for k=10 
    """
    k = 10
    data = roc_data.get(k)

    if data is None or data["auroc"] is None:
        print("No valid ROC data for k=10. Skipping plot.")
        return

    plt.figure(figsize=(8, 8))

    # Main ROC curve
    plt.plot(
        data["fpr"],
        data["tpr"],
        linewidth=2.0,
        alpha=0.85,
        label=f"k=10 (AUROC = {data['auroc']:.3f})",
    )

    # Random baseline
    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1.2,
        label="Random",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(f"Confidence-Success Separability ({model_name}, {task_name})")

    plt.legend(loc="lower right", frameon=False)
    plt.grid(False)

    plt.tight_layout()
    print(f"Saving ROC plot to: {output_path}")
    plt.savefig(output_path, dpi=300)
    plt.close()


def main(args):
    """Main execution logic for the confidence-AUROC experiment."""

    # 1. Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Qrels from BRIGHT dataset
    print(f"Loading BRIGHT dataset for task: {args.task}")
    examples_dataset = load_dataset(args.dataset_source, "examples", trust_remote_code=True)
    examples = examples_dataset[args.task]
    qrels = {str(e["id"]): {str(gid) for gid in e["gold_ids"]} for e in examples}

    # 3. Load pre-computed scores
    print(f"Loading scores from: {args.scores_file}")
    with open(args.scores_file, "r") as f:
        all_scores_by_query = json.load(f)

    # 4. For each query, extract raw confidence and determine success@k
    print("Extracting confidence and success labels for each query...")
    
    query_data = []
    raw_confidences = []

    CUTOFFS = [5, 10, 25, 50]

    for query_id, doc_scores in tqdm(all_scores_by_query.items(), desc="Processing queries"):
        query_qrels = qrels.get(query_id, set())
        
        if not doc_scores or not query_qrels:
            continue

        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Confidence is the score of the top-ranked document
        confidence_raw = sorted_docs[0][1]
        raw_confidences.append(confidence_raw)

        # Get top 50 doc IDs for success check
        retrieved_doc_ids = [doc_id for doc_id, score in sorted_docs[:max(CUTOFFS)]]
        
        success_at_k = {}
        for k in CUTOFFS:
            top_k_ids = set(retrieved_doc_ids[:k])
            success_at_k[k] = 1 if len(top_k_ids.intersection(query_qrels)) > 0 else 0
            
        query_data.append({
            "query_id": query_id,
            "confidence_raw": confidence_raw,
            "success_at_k": success_at_k
        })

    if not query_data:
        print("No valid queries found with scores and qrels. Exiting.")
        # Create an empty results file to prevent re-running
        (output_dir / "auroc_results.json").touch()
        return

    # 5. Per-task min-max normalization for confidence scores
    min_task, max_task = min(raw_confidences), max(raw_confidences)
    if max_task == min_task:
        normalized_confidences = {q['query_id']: 0.5 for q in query_data}
    else:
        normalized_confidences = {
            q['query_id']: (q['confidence_raw'] - min_task) / (max_task - min_task)
            for q in query_data
        }

    # 6. Compute AUROC and ROC curves for each k
    print("Computing AUROC and ROC curves...")
    results = {
        "model": args.model,
        "task": args.task,
        "num_queries": len(query_data),
        "normalization_stats": {"min": min_task, "max": max_task},
        "auroc_at_k": {},
    }
    roc_plot_data = {}

    for k in CUTOFFS:
        labels = [q['success_at_k'][k] for q in query_data]
        confidences = [normalized_confidences[q['query_id']] for q in query_data]
        
        # Check for single class
        if len(set(labels)) < 2:
            print(f"Warning: Only one class present for k={k}. Skipping AUROC calculation.")
            results["auroc_at_k"][f"@{k}"] = None
            roc_plot_data[k] = {"auroc": None, "fpr": None, "tpr": None}
            continue

        auroc = roc_auc_score(labels, confidences)
        fpr, tpr, _ = roc_curve(labels, confidences)
        
        results["auroc_at_k"][f"@{k}"] = auroc
        roc_plot_data[k] = {"auroc": auroc, "fpr": fpr, "tpr": tpr}
        print(f"  AUROC @{k}: {auroc:.4f}")

    # 7. Plot ROC curves
    plot_path = output_dir / "roc_curves.png"
    plot_roc_curves(roc_plot_data, plot_path, args.task, args.model)
    plot_path = output_dir / "roc_curve_k10.png"
    plot_roc_curve_k10(roc_plot_data, plot_path, args.task, args.model)

    # 8. Save results JSON
    results_path = output_dir / "auroc_results.json"
    print(f"Saving AUROC results to: {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Confidence-AUROC analysis experiment."
    )
    parser.add_argument("--model", type=str, required=True, help="Retrieval model to evaluate.")
    parser.add_argument("--task", type=str, required=True, help="BRIGHT task to evaluate.")
    parser.add_argument("--scores_file", type=str, required=True, help="Path to the score.json file from Phase 1.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results and plots.")
    parser.add_argument("--dataset_source", type=str, default="xlangai/BRIGHT", help="Source for the BRIGHT dataset.")

    args = parser.parse_args()
    main(args)