"""
Aggregate Phase 9 hybrid fusion results.
Compares standalone dense vs RRF, Linear, and Dynamic (DAT) fusions.
Includes 12 tasks.
"""

import os
import json
import glob
import numpy as np

MODELS = ['bge', 'e5', 'inst-l', 'inst-xl', 'qwen2', 'reasonir', 'sf']
FUSIONS = ['rrf', 'linear', 'dynamic']
TASKS = ['biology', 'earth_science', 'economics', 'psychology', 'robotics', 
         'stackoverflow', 'sustainable_living', 'leetcode', 'pony', 
         'aops', 'theoremqa_questions', 'theoremqa_theorems']

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def main():
    p1_dir = r'E:\llm-retrieval\sigir_results\phase1_cache_build'
    p9_dir = r'E:\llm-retrieval\sigir_results\phase9_hybrid'
    
    # baseline bm25 (avg across tasks)
    # I'll just collect it task by task to be precise
    
    summary = {}
    for model in MODELS:
        summary[model] = {'dense': [], 'rrf': [], 'linear': [], 'dynamic': []}

    bm25_ndcgs = []

    for task in TASKS:
        # BM25 baseline from Phase 1
        bm25_path = os.path.join(p1_dir, 'bm25', task, f"{task}_bm25_long_False", "results.json")
        data = load_json(bm25_path)
        if data: bm25_ndcgs.append(data.get('NDCG@10', 0) * 100)

        for model in MODELS:
            # Dense baseline from Phase 1
            dense_path = os.path.join(p1_dir, model, task, f"{task}_{model}_long_False", "results.json")
            data = load_json(dense_path)
            if data: summary[model]['dense'].append(data.get('NDCG@10', 0) * 100)
            
            # Hybrid Fusions from Phase 9
            for fusion in FUSIONS:
                # Dir name pattern: {model}_{fusion}
                # Result folder pattern inside: {task}_hybrid_bm25_{model}_{fusion}_long_False
                f_dir = os.path.join(p9_dir, task, f"{model}_{fusion}")
                res_folder_pattern = os.path.join(f_dir, f"{task}_hybrid_bm25_{model}_{fusion}_long_False", "results.json")
                res_files = glob.glob(res_folder_pattern)
                if not res_files:
                    # Fallback pattern
                    res_files = glob.glob(os.path.join(f_dir, "*", "results.json"))
                
                if res_files:
                    data = load_json(res_files[0])
                    if data: summary[model][fusion].append(data.get('NDCG@10', 0) * 100)

    # Compute Averages
    aggregated = {}
    avg_bm25 = np.mean(bm25_ndcgs) if bm25_ndcgs else 0.0
    
    for model in MODELS:
        aggregated[model] = {}
        for key in ['dense', 'rrf', 'linear', 'dynamic']:
            vals = summary[model][key]
            aggregated[model][key] = round(float(np.mean(vals)), 2) if vals else 0.0

    output = {
        'bm25_avg': round(float(avg_bm25), 2),
        'results': aggregated
    }

    output_path = r'E:\llm-retrieval\sigir_results\hybrid_fusion_final.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Hybrid fusion comparison saved to {output_path}")

    # Print table snippet
    print(f"\nBM25 Average: {avg_bm25:.2f}")
    print("\nModel | Dense | RRF | Linear | Dynamic")
    print("-" * 50)
    for model in MODELS:
        d = aggregated[model]['dense']
        r = aggregated[model]['rrf']
        l = aggregated[model]['linear']
        dyn = aggregated[model]['dynamic']
        print(f"{model:<10} | {d:6.1f} | {r:5.1f} | {l:6.1f} | {dyn:7.1f}")

if __name__ == "__main__":
    main()
