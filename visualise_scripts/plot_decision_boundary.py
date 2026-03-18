import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datasets import load_dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import math

# --- 1. Data Loading Functions (From your evaluate_qpp.py) ---

def load_qrels(dataset_family, task_name):
    if dataset_family == "BRIGHT":
        print(f"Loading BRIGHT dataset for task: {task_name}")
        examples_dataset = load_dataset("xlangai/BRIGHT", "examples", trust_remote_code=True)
        examples = examples_dataset[task_name]
        return {str(e["id"]): {str(gid) for gid in e["gold_ids"]} for e in examples}
    
    elif dataset_family == "BEIR":
        print(f"Loading BEIR dataset for task: {task_name}")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{task_name}.zip"
        beir_data_dir = Path("data") / "BEIR_datasets"
        beir_data_dir.mkdir(parents=True, exist_ok=True)
        data_path = util.download_and_unzip(url, beir_data_dir.as_posix())
        _, _, beir_qrels_raw = GenericDataLoader(data_folder=data_path).load(split="test")
        
        qrels = {}
        for query_id, doc_relevances in beir_qrels_raw.items():
            relevant_doc_ids = {doc_id for doc_id, relevance in doc_relevances.items() if relevance > 0}
            if relevant_doc_ids:
                qrels[query_id] = relevant_doc_ids
        return qrels
    
    elif dataset_family == "TEMPO":
        print(f"Loading TEMPO dataset for task: {task_name}")
        examples_dataset = load_dataset("tempo26/Tempo", "examples", trust_remote_code=True)
        examples = examples_dataset[task_name]
        return {str(e["id"]): {str(gid) for gid in e["gold_ids"]} for e in examples}
    else:
        raise ValueError(f"Unknown dataset family: {dataset_family}")

def calculate_ndcg(retrieved_docs_with_scores, qrels, k):
    if not retrieved_docs_with_scores or k == 0:
        return 0.0
    dcg = 0.0
    idcg = 0.0
    for i in range(min(k, len(retrieved_docs_with_scores))):
        doc_id, _ = retrieved_docs_with_scores[i]
        relevance = 1 if doc_id in qrels else 0
        dcg += relevance / math.log2(i + 2)
    num_relevant = len(qrels)
    for i in range(min(k, num_relevant)):
        idcg += 1 / math.log2(i + 2)
    if idcg == 0:
        return 0.0 
    return dcg / idcg

# --- 2. Main Plotting Logic ---

def main():
    parser = argparse.ArgumentParser(description="Generate a 2D Scatter Plot Decision Boundary.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., bge)")
    parser.add_argument("--dataset_family", type=str, required=True, choices=["BEIR", "BRIGHT", "TEMPO"])
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., biology)")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K for variance calculation")
    parser.add_argument("--out", type=str, default="decision_boundary.png", help="Output filename")
    args = parser.parse_args()

    # Automatically resolve the correct filepath based on your bash script's structure
    if args.dataset_family == "BEIR":
        scores_file = Path(f"results/scores_results/results_beir/{args.model}/{args.task}/score.json")
    elif args.dataset_family == "BRIGHT":
        scores_file = Path(f"results/scores_results/results_bright/{args.model}/{args.task}/{args.task}_{args.model}_long_False/score.json")
    else: # TEMPO
        scores_file = Path(f"results/scores_results/results_tempo/{args.model}/{args.task}/scores.json")

    if not scores_file.exists():
        print(f"Error: Could not find raw scores at {scores_file}")
        sys.exit(1)

    qrels = load_qrels(args.dataset_family, args.task)

    with open(scores_file, "r") as f:
        all_scores_by_query = json.load(f)

    # Extract X (MaxScore), Y (Variance), and Color (Success)
    successful_x, successful_y = [], []
    failed_x, failed_y = [], []

    for query_id, doc_scores_raw in all_scores_by_query.items():
        query_qrels = qrels.get(query_id, set())
        if not doc_scores_raw or not query_qrels:
            continue

        sorted_docs = sorted(doc_scores_raw.items(), key=lambda item: item[1], reverse=True)
        retrieved_docs_with_scores = sorted_docs[:args.k]
        scores_only = [score for _, score in retrieved_docs_with_scores]
        
        if len(scores_only) < 2:
            continue

        max_score = scores_only[0]
        variance = np.std(scores_only)
        ndcg = calculate_ndcg(retrieved_docs_with_scores, query_qrels, args.k)

        if ndcg > 0:
            successful_x.append(max_score)
            successful_y.append(variance)
        else:
            failed_x.append(max_score)
            failed_y.append(variance)

    # --- 3. Generate Academic Plot ---
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.family": "serif", "axes.labelsize": 14, "axes.titlesize": 16})

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Failures first (so they are under successes if they overlap)
    ax.scatter(failed_x, failed_y, c='#D62728', alpha=0.5, s=40, edgecolors='none', label='Failed Retrieval')
    # Plot Successes
    ax.scatter(successful_x, successful_y, c='#2CA02C', alpha=0.6, s=40, edgecolors='w', linewidth=0.5, label='Successful Retrieval')

    # Add illustrative decision boundaries
    # Vertical Line: LangChain's static threshold (estimated median of MaxScores)
    median_max = np.median(successful_x + failed_x)
    ax.axvline(x=median_max, color='black', linestyle='--', linewidth=2, label='Similarity Threshold')
    
    # Horizontal Line: Variance threshold (estimated threshold separating red from green)
    # We estimate a good variance split visually
    median_var = np.median(successful_y)
    ax.axhline(y=median_var, color='blue', linestyle=':', linewidth=2, label='Variance Threshold')

    ax.set_title(f"The Magnitude Mirage ({args.model.upper()} on TheoremQA Questions)", pad=15, fontweight="bold")
    ax.set_xlabel(r"Absolute Magnitude ($s_1$)")
    ax.set_ylabel(rf"Top-{args.k} Score Variance ($\sigma^2_{{{args.k}}}$)")
    ax.legend(loc='best', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(args.out, format="png", dpi=300, bbox_inches="tight")
    print(f"Successfully generated 2D Scatter Plot: {args.out}")

if __name__ == "__main__":
    main()