import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import math
from beir import util
from beir.datasets.data_loader import GenericDataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_qrels(dataset_family, task_name):
    """
    Loads qrels (ground truth relevance judgments) for a given dataset family and task.
    """
    if dataset_family == "BRIGHT":
        print(f"Loading BRIGHT dataset for task: {task_name}")
        examples_dataset = load_dataset("xlangai/BRIGHT", "examples", trust_remote_code=True)
        examples = examples_dataset[task_name]
        qrels = {str(e["id"]): {str(gid) for gid in e["gold_ids"]} for e in examples}
        return qrels
    
    elif dataset_family == "BEIR":
        print(f"Loading BEIR dataset for task: {task_name}")
        # task_name will be the specific BEIR dataset, e.g., "scifact"
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{task_name}.zip"
        
        # Define a base directory for BEIR datasets
        beir_data_dir = Path(__file__).resolve().parents[1] / "data" / "BEIR_datasets"
        beir_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and unzip the dataset
        # The returned data_path will be the path to the unzipped dataset, e.g., .../BEIR_datasets/scifact
        data_path = util.download_and_unzip(url, beir_data_dir.as_posix())
        
        # Load corpus, queries, and qrels
        # For BEIR datasets, relevance scores are typically 1 for relevant documents.
        # We convert the qrels to a set of relevant doc_ids for consistency with BRIGHT.
        _, _, beir_qrels_raw = GenericDataLoader(data_folder=data_path).load(split="test")
        
        qrels = {}
        for query_id, doc_relevances in beir_qrels_raw.items():
            relevant_doc_ids = {doc_id for doc_id, relevance in doc_relevances.items() if relevance > 0}
            if relevant_doc_ids: # Only add if there are actual relevant documents
                qrels[query_id] = relevant_doc_ids
        
        return qrels
    
    elif dataset_family == "TEMPO":
        print(f"Loading TEMPO dataset for task: {task_name}")
        examples_dataset = load_dataset("tempo26/Tempo", "examples", trust_remote_code=True)
        examples = examples_dataset[task_name]
        qrels = {str(e["id"]): {str(gid) for gid in e["gold_ids"]} for e in examples}
        return qrels
    
    else:
        raise ValueError(f"Unknown dataset family: {dataset_family}")


# --- QPP Metric Functions ---

def calculate_max_score(scores):
    """MaxScore: The score of the top-ranked document."""
    if not scores:
        return 0.0
    return scores[0]

def calculate_score_gap(scores, k):
    """Score Gap / Top-to-Tail Dropoff: The difference between the 1st document score and the k-th document score."""
    if len(scores) < k or k == 0: # k=0 should be handled by caller, but defensive check
        return 0.0
    return scores[0] - scores[k-1]

def calculate_top_k_std(scores, k):
    """Top-k Standard Deviation: The standard deviation of the top-k document scores."""
    if len(scores) < k or k == 0:
        return 0.0
    return np.std(scores[:k])

def calculate_nqc(scores, k):
    """Normalized Query Commitment (NQC): The standard deviation of the top-k scores divided by the absolute value of the mean of the top-k scores."""
    if len(scores) < k or k == 0:
        return 0.0
    top_k_scores = np.array(scores[:k])
    std_dev = np.std(top_k_scores)
    mean_abs = np.mean(np.abs(top_k_scores))
    if mean_abs == 0:
        return 0.0
    return std_dev / mean_abs

def calculate_max_iterative_variance(scores, k_max):
    """Maximum Iterative Variance (sigma_{max}).
    Calculates the variance for every subset from top-2 to top-k.
    The metric is the maximum variance found across these iterations.
    (Pérez-Iglesias & Araujo, 2010).
    """
    if len(scores) < 2 or k_max < 2: 
        return 0.0
    
    max_k_to_check = min(k_max, len(scores))
    
    variances = []
    # Iterate through every possible cutoff from 2 to max_k
    for i in range(2, max_k_to_check + 1):
        variances.append(np.var(scores[:i]))
        
    if not variances:
        return 0.0
        
    return max(variances)


def calculate_smv(scores, k):
    """Score Magnitude and Variance (SMV): MaxScore multiplied by the Top-k Standard Deviation."""
    if len(scores) < k or k == 0:
        return 0.0
    max_score = calculate_max_score(scores)
    top_k_std = calculate_top_k_std(scores, k)
    return max_score * top_k_std


# --- NDCG Calculation ---

def calculate_ndcg(retrieved_docs_with_scores, qrels, k):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG) at cutoff k.
    retrieved_docs_with_scores: List of (doc_id, score) tuples, sorted by score descending.
    qrels: Set of relevant document IDs for the query.
    """
    if not retrieved_docs_with_scores or k == 0:
        return 0.0

    dcg = 0.0
    idcg = 0.0
    
    # Calculate DCG
    for i in range(min(k, len(retrieved_docs_with_scores))):
        doc_id, _ = retrieved_docs_with_scores[i]
        relevance = 1 if doc_id in qrels else 0
        dcg += relevance / math.log2(i + 2) # i+1 is rank, so i+2 for log base 2

    # Calculate IDCG (ideal DCG)
    # Sort qrels by hypothetical perfect scores (highest relevance first)
    # Since all relevant documents have relevance 1, their order doesn't matter for sum,
    # just that top K relevant documents contribute.
    num_relevant = len(qrels)
    for i in range(min(k, num_relevant)):
        idcg += 1 / math.log2(i + 2)

    if idcg == 0:
        return 0.0 # No relevant documents in top k, or no relevant documents at all
    
    return dcg / idcg


def main(args):
    """Main execution logic for the QPP evaluation experiment."""

    # 1. Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Qrels
    qrels = load_qrels(args.dataset_family, args.task)

    # 3. Load pre-computed scores
    print(f"Loading scores from: {args.scores_file}")
    with open(args.scores_file, "r") as f:
        all_scores_by_query = json.load(f)

    # 4. For each query, extract all scores and determine success@k
    print("Processing queries and calculating QPP metrics...")
    
    query_data_processed = []
    
    # Define the cutoffs for QPP metrics and NDCG
    QPP_CUTOFFS = [5, 10, 25, 50]
    # Max K for any metric calculation
    MAX_K_OVERALL = max(QPP_CUTOFFS)

    for query_id, doc_scores_raw in tqdm(all_scores_by_query.items(), desc="Processing queries"):
        query_qrels = qrels.get(query_id, set())
        
        if not doc_scores_raw or not query_qrels:
            # Skip queries with no scores or no relevant documents
            continue

        # Sort documents by score in descending order
        sorted_docs = sorted(doc_scores_raw.items(), key=lambda item: item[1], reverse=True)
        
        # Ensure we have at least MAX_K_OVERALL documents to consider for all metrics
        # Pad with zeros if fewer documents, though QPP metrics should handle this.
        retrieved_docs_with_scores = sorted_docs[:MAX_K_OVERALL]
        # Just scores for QPP calculations
        retrieved_scores_only = [score for _, score in retrieved_docs_with_scores]
        # Just doc_ids for success check
        retrieved_doc_ids = [doc_id for doc_id, _ in retrieved_docs_with_scores]


        query_metrics = {
            "query_id": query_id,
            "qpp_metrics": defaultdict(dict), # metric_name -> k -> value
            "ndcg_at_k": {}, # k -> ndcg_value
            "success_at_k": {} # k -> 0/1
        }
        
        for k in QPP_CUTOFFS:
            # Calculate Success@k (binary relevance)
            top_k_ids = set(retrieved_doc_ids[:k])
            query_metrics["success_at_k"][k] = 1 if len(top_k_ids.intersection(query_qrels)) > 0 else 0

            # Calculate NDCG@k
            query_metrics["ndcg_at_k"][k] = calculate_ndcg(retrieved_docs_with_scores, query_qrels, k)

            # Calculate QPP Metrics at cutoff k
            current_scores_for_qpp = retrieved_scores_only[:k]
            
            query_metrics["qpp_metrics"]["MaxScore"][k] = calculate_max_score(current_scores_for_qpp)
            query_metrics["qpp_metrics"]["ScoreGap"][k] = calculate_score_gap(current_scores_for_qpp, k)
            query_metrics["qpp_metrics"]["TopKStd"][k] = calculate_top_k_std(current_scores_for_qpp, k)
            query_metrics["qpp_metrics"]["NQC"][k] = calculate_nqc(current_scores_for_qpp, k)
            query_metrics["qpp_metrics"]["MaxIterativeVariance"][k] = calculate_max_iterative_variance(current_scores_for_qpp, k)
            query_metrics["qpp_metrics"]["SMV"][k] = calculate_smv(current_scores_for_qpp, k)
            
        query_data_processed.append(query_metrics)

    if not query_data_processed:
        print("No valid queries found with scores and qrels after filtering. Exiting.")
        (output_dir / "qpp_results.json").touch() 
        return

    # --- Aggregation and Correlation Analysis ---
    all_results = {
        "model": args.model,
        "task": args.task,
        "dataset_family": args.dataset_family,
        "num_queries": len(query_data_processed),
        "metrics": defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) # metric -> k -> {auroc, pearson, spearman}
    }

    # List of QPP metric names
    QPP_METRIC_NAMES = [
        "MaxScore", "ScoreGap", "TopKStd", "NQC", 
        "MaxIterativeVariance", "SMV"
    ]

    for metric_name in QPP_METRIC_NAMES:
        for k in QPP_CUTOFFS:
            qpp_values = [q_data["qpp_metrics"][metric_name][k] for q_data in query_data_processed]
            ndcg_values = [q_data["ndcg_at_k"][k] for q_data in query_data_processed]
            success_values = [q_data["success_at_k"][k] for q_data in query_data_processed] # For AUROC

            # Calculate Pearson and Spearman correlations with NDCG
            # Handle cases where std dev is zero, or all values are same for correlation
            pearson_r, spearman_rho = None, None
            if len(set(qpp_values)) > 1 and len(set(ndcg_values)) > 1:
                pearson_r, _ = pearsonr(qpp_values, ndcg_values)
                spearman_rho, _ = spearmanr(qpp_values, ndcg_values)
            else:
                print(f"Warning: Not enough variance in {metric_name}@{k} or NDCG@{k} for correlation calculation. Skipping.")

            all_results["metrics"][metric_name][k]["pearson"] = pearson_r
            all_results["metrics"][metric_name][k]["spearman"] = spearman_rho

            # Calculate AUROC with binary success
            auroc_score = None
            if len(set(success_values)) > 1: # Need at least two classes for AUROC
                # Filter out None values in qpp_values if any were produced by QPP functions
                # and ensure corresponding success_values are kept.
                # roc_auc_score uses the ranking of scores, so explicit normalization
                # is not necessary, but we should handle non-numeric values.
                
                valid_pairs = [(q, s) for q, s in zip(qpp_values, success_values) if q is not None]
                if not valid_pairs:
                    auroc_score = None
                else:
                    valid_qpp_values, valid_success_values = zip(*valid_pairs)
                    if len(set(valid_success_values)) < 2: # Check again for single class after filtering
                        print(f"Warning: Only one success class present for k={k} after filtering. Skipping AUROC calculation for {metric_name}.")
                        auroc_score = None
                    else:
                        auroc_score = roc_auc_score(valid_success_values, valid_qpp_values)
            else:
                print(f"Warning: Only one success class present for k={k}. Skipping AUROC calculation for {metric_name}.")
            
            all_results["metrics"][metric_name][k]["auroc"] = auroc_score


    # 8. Save results JSON
    results_path = output_dir / f"{args.model}_qpp_results.json"
    print(f"Saving QPP results to: {results_path}")
    # Convert defaultdict to dict for JSON serialization
    def default_to_dict(d):
        if isinstance(d, defaultdict):
            return {k: default_to_dict(v) for k, v in d.items()}
        return d
        
    with open(results_path, "w") as f:
        json.dump(default_to_dict(all_results), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run QPP analysis experiment."
    )
    parser.add_argument("--model", type=str, required=True, help="Retrieval model to evaluate.")
    parser.add_argument("--task", type=str, required=True, help="Task to evaluate.")
    parser.add_argument("--dataset_family", type=str, required=True, help="Dataset family (e.g., BEIR, BRIGHT, TEMPO).")
    parser.add_argument("--scores_file", type=str, required=True, help="Path to the score.json or scores.json file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")

    args = parser.parse_args()
    main(args)