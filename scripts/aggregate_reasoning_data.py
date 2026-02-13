"""
Aggregate Phase 6 reasoning results (effectiveness and latency).
Collects NDCG@10 and Query Latency from reasoning augmented queries.
"""

import os
import json
import glob
import numpy as np

MODELS = ['bm25', 'sbert', 'bge', 'contriever', 'nomic', 'inst-l', 'inst-xl', 
          'e5', 'sf', 'qwen', 'qwen2', 'reasonir', 'rader', 'diver-retriever']

TASKS = ['biology', 'earth_science', 'economics', 'psychology', 'robotics', 
         'stackoverflow', 'sustainable_living', 'leetcode', 'pony', 
         'aops', 'theoremqa_questions', 'theoremqa_theorems']

REASONING_TYPES = ['gpt4', 'llama3-70b', 'claude-3-opus', 'grit', 'Gemini-1.0']

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def main():
    base_dir = r'E:\llm-retrieval\sigir_results\phase6_reasoning'
    
    # Structure: results[model][reasoning_type] = {ndcg: [], latency: []}
    # reasoning_type 'original' is stored in phase2_latency_aggregated.json (mostly)
    # But some models (reasonir, rader, diver) might have 'original' in phase6 too or phase1
    
    final_summary = {}

    for model in MODELS:
        final_summary[model] = {}
        for r_type in REASONING_TYPES + ['original']:
            final_summary[model][r_type] = {'ndcg': [], 'latency': []}

    for task in TASKS:
        task_dir = os.path.join(base_dir, task)
        if not os.path.exists(task_dir):
            continue
            
        for model in MODELS:
            # 1. Handle Original Baseline
            # Try phase6_reasoning/{task}/{model}_original/
            orig_dir_p6 = os.path.join(task_dir, f"{model}_original")
            if os.path.exists(orig_dir_p6):
                results_path = glob.glob(os.path.join(orig_dir_p6, '*', 'results.json'))
                eff_path = glob.glob(os.path.join(orig_dir_p6, '*', 'efficiency.json'))
                if results_path:
                    data = load_json(results_path[0])
                    if data: final_summary[model]['original']['ndcg'].append(data.get('NDCG@10', 0) * 100)
                if eff_path:
                    data = load_json(eff_path[0])
                    if data: final_summary[model]['original']['latency'].append(data.get('query_latency', {}).get('total', {}).get('mean_ms', 0))

            # 2. Handle Reasoning Augmented
            for r_type in REASONING_TYPES:
                # Dir name pattern: {model}_{r_type}_reason
                r_dir = os.path.join(task_dir, f"{model}_{r_type}_reason")
                if os.path.exists(r_dir):
                    results_path = glob.glob(os.path.join(r_dir, '*', 'results.json'))
                    eff_path = glob.glob(os.path.join(r_dir, '*', 'efficiency.json'))
                    if results_path:
                        data = load_json(results_path[0])
                        if data: final_summary[model][r_type]['ndcg'].append(data.get('NDCG@10', 0) * 100)
                    if eff_path:
                        data = load_json(eff_path[0])
                        if data: final_summary[model][r_type]['latency'].append(data.get('query_latency', {}).get('total', {}).get('mean_ms', 0))

    # Compute Averages
    aggregated = {}
    for model in MODELS:
        aggregated[model] = {}
        for r_type in REASONING_TYPES + ['original']:
            ndcg_list = final_summary[model][r_type]['ndcg']
            lat_list = final_summary[model][r_type]['latency']
            
            aggregated[model][r_type] = {
                'ndcg': round(float(np.mean(ndcg_list)), 2) if ndcg_list else 0.0,
                'latency': round(float(np.mean(lat_list)), 2) if lat_list else 0.0,
                'count': len(ndcg_list)
            }

    # Save results
    output_path = r'E:\llm-retrieval\sigir_results\reasoning_overhead_final.json'
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Aggregated reasoning data saved to {output_path}")

    # Output a small table snippet for reference
    print("\nModel | Original | GPT-4 | Llama-3 | Claude-3 | GritLM | Gemini")
    print("-" * 75)
    for model in MODELS:
        row = [model]
        for rt in ['original'] + REASONING_TYPES:
            ndcg = aggregated[model][rt]['ndcg']
            row.append(f"{ndcg:.1f}" if ndcg > 0 else "--")
        print(" | ".join(row))

if __name__ == "__main__":
    main()
