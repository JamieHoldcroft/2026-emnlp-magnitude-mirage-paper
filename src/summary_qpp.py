import json
import numpy as np
from pathlib import Path

# Define paths and lists from the provided content
BEIR_PATH = Path('results/qpp_results/BEIR')
BRIGHT_PATH = Path('results/qpp_results/BRIGHT')
TEMPO_PATH = Path('results/qpp_results/TEMPO')

BEIR_TASKS=[
    "fiqa", "nfcorpus", "scidocs", "scifact",
]

BRIGHT_TASKS=[
    "biology", "aops", "earth_science", "economics", "leetcode", "pony",
    "psychology", "robotics", "stackoverflow", "sustainable_living",
    "theoremqa_questions", "theoremqa_theorems",
]

TEMPO_TASKS=[
    "cardano", "economics", "genealogy", "history", "hsm", "iota",
    "law", "monero", "politics", "quant", "travel", "workplace",
]

MODELS=[
    "bge", "bm25", "contriever", "e5", "inst-l", "rader", "reasonir", "sbert", "sf", "qwen", "diver-retriever"
]

# Define the expected metric types and k-values based on the example JSON
METRIC_TYPES = ["MaxScore", "ScoreGap", "TopKStd", "NQC", "MaxIterativeVariance", "SMV"]
K_VALUES = ["5", "10", "25", "50"]
SCORE_TYPES = ["pearson", "spearman", "auroc"]

def calculate_average_qpp_results(dataset_family_name, tasks, output_base_path):
    """
    Calculates the average QPP results for each model across all tasks in a given dataset family.
    Outputs a JSON file for each model with the averaged results.
    """
    print(f"Processing {dataset_family_name} dataset...")

    # Initialize aggregation structure for each model
    # { 'model_name': { 'metrics': { 'metric_type': { 'k_value': { 'pearson_sum': 0.0, 'spearman_sum': 0.0, 'auroc_sum': 0.0, 'count': 0 } } } } }
    aggregated_data = {model: {
        "model": model,
        "dataset_family": dataset_family_name,
        "metrics": {mt: {kv: {"pearson_sum": 0.0, "spearman_sum": 0.0, "auroc_sum": 0.0, "count": 0}
                         for kv in K_VALUES} for mt in METRIC_TYPES}
    } for model in MODELS}

    for task in tasks:
        for model in MODELS:
            file_path = output_base_path / task / f"{model}_qpp_results.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error: Could not decode JSON from {file_path}. Skipping.")
                        continue

                    for metric_type in METRIC_TYPES:
                        if metric_type in data["metrics"]:
                            for k_value in K_VALUES:
                                if k_value in data["metrics"][metric_type]:
                                    current_metric_data = data["metrics"][metric_type][k_value]
                                    for score_type in SCORE_TYPES:
                                        score = current_metric_data.get(score_type)
                                        if score is not None:
                                            aggregated_data[model]["metrics"][metric_type][k_value][f"{score_type}_sum"] += score
                                    aggregated_data[model]["metrics"][metric_type][k_value]["count"] += 1
            # else:
            #     print(f"Warning: File not found: {file_path}. Skipping.")

    # Calculate averages and prepare final output for each model
    for model in MODELS:
        final_model_results = {
            "model": model,
            "dataset_family": dataset_family_name,
            "metrics": {}
        }
        for metric_type in METRIC_TYPES:
            final_model_results["metrics"][metric_type] = {}
            for k_value in K_VALUES:
                counts = aggregated_data[model]["metrics"][metric_type][k_value]["count"]
                if counts > 0:
                    final_model_results["metrics"][metric_type][k_value] = {}
                    for score_type in SCORE_TYPES:
                        total_sum = aggregated_data[model]["metrics"][metric_type][k_value][f"{score_type}_sum"]
                        final_model_results["metrics"][metric_type][k_value][score_type] = total_sum / counts
                else:
                    # If no data for this k_value, set scores to None
                    final_model_results["metrics"][metric_type][k_value] = {
                        "pearson": None,
                        "spearman": None,
                        "auroc": None
                    }

        # Write the averaged results to a new JSON file in the main dataset family folder
        output_file_path = output_base_path / f"{model}_qpp_results.json"
        with open(output_file_path, 'w') as f:
            json.dump(final_model_results, f, indent=4)
        print(f"Generated averaged results for model '{model}' in {output_file_path}")

def main():
    calculate_average_qpp_results("BEIR", BEIR_TASKS, BEIR_PATH)
    calculate_average_qpp_results("BRIGHT", BRIGHT_TASKS, BRIGHT_PATH)
    calculate_average_qpp_results("TEMPO", TEMPO_TASKS, TEMPO_PATH)

if __name__ == "__main__":
    main()