"""
Aggregate Phase 8 long-context results.
Compares long_context=False (Phase 1) vs long_context=True (Phase 8).
Only uses the 8 tasks available in Phase 8.
"""

import os
import json
import glob
import numpy as np

MODELS = ['bm25', 'sbert', 'bge', 'contriever', 'nomic', 'inst-l', 'inst-xl', 
          'e5', 'sf', 'qwen', 'qwen2', 'reasonir', 'rader', 'diver-retriever']

# Common tasks in Phase 8
TASKS = ['biology', 'earth_science', 'economics', 'pony', 
         'psychology', 'robotics', 'stackoverflow', 'sustainable_living']

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
    p8_dir = r'E:\llm-retrieval\sigir_results\phase8_long_context'
    
    results = {}

    for model in MODELS:
        results[model] = {'short': [], 'long': []}
        
        for task in TASKS:
            # 1. Short context (False) from Phase 1
            # Note: Phase 1 follows pattern {p1_dir}/{model}/{task}/{task}_{model}_long_False/results.json
            p1_task_pattern = os.path.join(p1_dir, model, task, f"{task}_{model}_long_False", "results.json")
            p1_files = glob.glob(p1_task_pattern)
            if not p1_files:
                # Fallback check
                p1_task_pattern = os.path.join(p1_dir, model, task, "*", "results.json")
                p1_files = glob.glob(p1_task_pattern)
            
            if p1_files:
                data = load_json(p1_files[0])
                if data: results[model]['short'].append(data.get('NDCG@10', 0) * 100)

            # 2. Long context (True) from Phase 8
            # Note: Phase 8 pattern {p8_dir}/{model}/{task}/{task}_{model}_long_True/results.json
            p8_task_pattern = os.path.join(p8_dir, model, task, f"{task}_{model}_long_True", "results.json")
            p8_files = glob.glob(p8_task_pattern)
            
            if p8_files:
                data = load_json(p8_files[0])
                if data: results[model]['long'].append(data.get('NDCG@10', 0) * 100)
    
    # Aggregate
    final_comparison = {}
    for model in MODELS:
        short_list = results[model]['short']
        long_list = results[model]['long']
        
        final_comparison[model] = {
            'avg_ndcg_short': round(float(np.mean(short_list)), 2) if short_list else 0.0,
            'avg_ndcg_long': round(float(np.mean(long_list)), 2) if long_list else 0.0,
            'task_count_short': len(short_list),
            'task_count_long': len(long_list)
        }

    # Save
    output_path = r'E:\llm-retrieval\sigir_results\long_context_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(final_comparison, f, indent=2)
    
    print(f"Long context comparison saved to {output_path}")

    # Print table snippet
    print("\nModel | Short-Ctx | Long-Ctx | Delta")
    print("-" * 40)
    for model in MODELS:
        s = final_comparison[model]['avg_ndcg_short']
        l = final_comparison[model]['avg_ndcg_long']
        delta = l - s
        print(f"{model:<15} | {s:10.1f} | {l:8.1f} | {delta:+6.1f}")

if __name__ == "__main__":
    main()
