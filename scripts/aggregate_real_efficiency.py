"""
Aggregate REAL indexing time data from efficiency.json files.
Only uses data that exists in the JSON files - no fabricated values.
"""

import os
import json
import glob

# Models to analyze
MODELS = ['bm25', 'bge', 'inst-l', 'sbert', 'e5', 'sf', 'inst-xl', 'qwen', 'qwen2', 
          'nomic', 'contriever', 'reasonir', 'rader', 'diver-retriever']

# Tasks
TASKS = ['biology', 'earth_science', 'economics', 'psychology', 'robotics', 
         'stackoverflow', 'sustainable_living', 'leetcode', 'pony', 
         'aops', 'theoremqa_questions', 'theoremqa_theorems']

def load_efficiency_data(base_dir):
    """Load efficiency data from phase1_cache_build directory."""
    results = {}
    
    for model in MODELS:
        model_dir = os.path.join(base_dir, model)
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model}")
            continue
        
        results[model] = {
            'indexing_times': [],
            'docs_per_second': [],
            'query_latency_mean': [],
            'query_qps': [],
            'used_cache': []
        }
        
        for task in TASKS:
            task_dir = os.path.join(model_dir, task)
            if not os.path.exists(task_dir):
                continue
            
            # Find efficiency.json
            eff_files = glob.glob(os.path.join(task_dir, f'{task}_{model}_long_False', 'efficiency.json'))
            if not eff_files:
                eff_files = glob.glob(os.path.join(task_dir, '*', 'efficiency.json'))
            
            if eff_files:
                try:
                    with open(eff_files[0], 'r') as f:
                        data = json.load(f)
                        
                        # Indexing data
                        idx = data.get('indexing', {})
                        if 'total_time_seconds' in idx:
                            results[model]['indexing_times'].append(idx['total_time_seconds'])
                        if 'documents_per_second' in idx:
                            results[model]['docs_per_second'].append(idx['documents_per_second'])
                        if 'used_cache' in idx:
                            results[model]['used_cache'].append(idx['used_cache'])
                        
                        # Query latency data
                        ql = data.get('query_latency', {}).get('total', {})
                        if 'mean_ms' in ql:
                            results[model]['query_latency_mean'].append(ql['mean_ms'])
                        if 'qps' in ql:
                            results[model]['query_qps'].append(ql['qps'])
                            
                except Exception as e:
                    print(f"Error loading {eff_files[0]}: {e}")
    
    return results

def compute_averages(results):
    """Compute averages for each model."""
    averages = {}
    
    for model, data in results.items():
        averages[model] = {}
        
        if data['indexing_times']:
            averages[model]['avg_indexing_time'] = sum(data['indexing_times']) / len(data['indexing_times'])
            averages[model]['total_indexing_time'] = sum(data['indexing_times'])
        
        if data['docs_per_second']:
            averages[model]['avg_docs_per_second'] = sum(data['docs_per_second']) / len(data['docs_per_second'])
        
        if data['query_latency_mean']:
            averages[model]['avg_query_latency_ms'] = sum(data['query_latency_mean']) / len(data['query_latency_mean'])
        
        if data['query_qps']:
            averages[model]['avg_qps'] = sum(data['query_qps']) / len(data['query_qps'])
        
        # Check if cold start (used_cache = false or null)
        cold_start = all(c in [False, None] for c in data['used_cache']) if data['used_cache'] else False
        averages[model]['cold_start'] = cold_start
        averages[model]['num_tasks'] = len(data['indexing_times'])
    
    return averages

def print_latex_table(averages):
    """Print LaTeX table with real data only."""
    print("\n\n=== LaTeX Table (Real Data Only) ===\n")
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\caption{Cold-start indexing efficiency (actual measurements from experiments). Times summed across all 12 BRIGHT tasks.}")
    print(r"\small")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Total Time (s)} & \textbf{Avg. Docs/sec} & \textbf{Avg. Latency (ms)} \\")
    print(r"\midrule")
    
    # Group by category
    sparse = ['bm25']
    dense_small = ['bge', 'inst-l', 'sbert', 'contriever', 'nomic']
    dense_large = ['e5', 'sf', 'inst-xl', 'qwen', 'qwen2']
    reasoning = ['reasonir', 'rader', 'diver-retriever']
    
    for category, models, title in [
        (sparse, sparse, 'Sparse'),
        (dense_small, dense_small, 'Dense (<1B)'),
        (dense_large, dense_large, 'Dense (>1B)'),
        (reasoning, reasoning, 'Reasoning-Specialized'),
    ]:
        print(f"\\multicolumn{{4}}{{c}}{{\\textit{{{title}}}}} \\\\")
        print(r"\midrule")
        
        for model in models:
            if model in averages:
                a = averages[model]
                total_time = a.get('total_indexing_time', 0)
                docs_sec = a.get('avg_docs_per_second', 0)
                latency = a.get('avg_query_latency_ms', 0)
                
                model_name = model.upper() if model in ['bm25', 'bge', 'sbert', 'e5', 'sf'] else model.replace('-', ' ').title()
                
                print(f"{model_name} & {total_time:,.0f} & {docs_sec:,.1f} & {latency:.1f} \\\\")
        
        print(r"\midrule")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:indexing}")
    print(r"\end{table}")

def main():
    base_dir = 'E:\llm-retrieval\sigir_results\phase1_cache_build'
    
    print("Loading efficiency data from JSON files...")
    results = load_efficiency_data(base_dir)
    
    print("\nComputing averages...")
    averages = compute_averages(results)
    
    # Print summary
    print("\n=== Summary (Real Data) ===")
    for model, a in sorted(averages.items(), key=lambda x: x[1].get('total_indexing_time', 0)):
        if a.get('total_indexing_time'):
            print(f"{model}: Total={a['total_indexing_time']:,.0f}s, "
                  f"Docs/s={a.get('avg_docs_per_second', 0):,.1f}, "
                  f"Latency={a.get('avg_query_latency_ms', 0):.1f}ms, "
                  f"Tasks={a['num_tasks']}, ColdStart={a.get('cold_start', 'N/A')}")
    
    # Print LaTeX
    print_latex_table(averages)
    
    # Save to JSON
    with open('./indexing_efficiency_real.json', 'w') as f:
        json.dump(averages, f, indent=2)
    print("\nResults saved to sigir_results/indexing_efficiency_real.json")

if __name__ == '__main__':
    main()
