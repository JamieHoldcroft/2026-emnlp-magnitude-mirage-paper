# src/aggregate_auroc.py

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

def main():
    """Summarizes AUROC results for a given model across all tasks."""
    parser = argparse.ArgumentParser(description="Summarize Confidence-AUROC results for a model.")
    parser.add_argument("--model", type=str, required=True, help="The model to summarize results for.")
    parser.add_argument("--output_base_dir", type=str, required=True, help="The base output directory for Phase 17.")
    args = parser.parse_args()

    model_dir = Path(args.output_base_dir)
    
    per_task_results = {}
    all_aurocs = defaultdict(list)
    
    print(f"Summarizing results for model: {args.model}")

    # Find all task directories within the output base directory
    for task_dir in model_dir.iterdir():
        if not task_dir.is_dir():
            continue

        model_task_dir = task_dir / args.model
        results_file = model_task_dir / "auroc_results.json"

        if results_file.exists():
            print(f"Loading results from: {results_file}")
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            task_name = data.get("task")
            auroc_at_k = data.get("auroc_at_k", {})
            per_task_results[task_name] = auroc_at_k
            
            for k, auroc in auroc_at_k.items():
                if auroc is not None:
                    all_aurocs[k].append(auroc)
    
    if not per_task_results:
        print(f"No results found for model '{args.model}' in '{args.output_base_dir}'.")
        return

    # Calculate macro-average AUROC
    mean_aurocs = {
        k: np.mean(aurocs) for k, aurocs in all_aurocs.items()
    }

    summary_data = {
        "model": args.model,
        "mean_auroc_at_k": mean_aurocs,
        "per_task_auroc": per_task_results
    }

    # Save summary file
    summary_file_path = model_dir / f"{args.model}_auroc_summary.json"
    print(f"Saving summary to: {summary_file_path}")
    with open(summary_file_path, 'w') as f:
        json.dump(summary_data, f, indent=4)

if __name__ == "__main__":
    main()