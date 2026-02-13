"""
Aggregate Phase 2 query latency data for Table 5.
Extracts QPS and latency percentiles from efficiency.json files.
"""

import os
import json
import glob

MODELS = ['bm25', 'sbert', 'bge', 'contriever', 'nomic', 'inst-l', 'inst-xl', 
          'e5', 'sf', 'qwen', 'qwen2', 'reasonir', 'rader', 'diver-retriever']

TASKS = ['biology', 'earth_science', 'economics', 'psychology', 'robotics', 
         'stackoverflow', 'sustainable_living', 'leetcode', 'pony', 
         'aops', 'theoremqa_questions', 'theoremqa_theorems']

# nDCG@10 averages from Phase 1 (for correlation)
NDCG_AVG = {
    'bm25': 14.5, 'sbert': 14.9, 'bge': 13.7, 'contriever': 11.1, 'nomic': 14.2,
    'inst-l': 14.2, 'inst-xl': 18.9, 'e5': 17.9, 'sf': 18.3, 'qwen': 22.5, 'qwen2': 23.3,
    'reasonir': 24.1, 'rader': 23.5, 'diver-retriever': 29.2
}

def load_phase2_latency(base_dir):
    """Load query latency data from phase2_efficiency."""
    results = {}
    
    for model in MODELS:
        model_dir = os.path.join(base_dir, model)
        if not os.path.exists(model_dir):
            print(f"Model not found: {model}")
            results[model] = None
            continue
        
        results[model] = {
            'qps': [], 'mean_ms': [], 'p50_ms': [], 'p95_ms': [], 'p99_ms': []
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
                        ql = data.get('query_latency', {}).get('total', {})
                        
                        if 'qps' in ql:
                            results[model]['qps'].append(ql['qps'])
                        if 'mean_ms' in ql:
                            results[model]['mean_ms'].append(ql['mean_ms'])
                        if 'p50_ms' in ql:
                            results[model]['p50_ms'].append(ql['p50_ms'])
                        if 'p95_ms' in ql:
                            results[model]['p95_ms'].append(ql['p95_ms'])
                        if 'p99_ms' in ql:
                            results[model]['p99_ms'].append(ql['p99_ms'])
                except Exception as e:
                    print(f"Error loading {eff_files[0]}: {e}")
    
    return results

def compute_averages(results):
    """Compute averages for each model."""
    averages = {}
    
    for model, data in results.items():
        if data is None:
            averages[model] = None
            continue
            
        averages[model] = {}
        for metric in ['qps', 'mean_ms', 'p50_ms', 'p95_ms', 'p99_ms']:
            if data[metric]:
                averages[model][metric] = sum(data[metric]) / len(data[metric])
            else:
                averages[model][metric] = None
        
        averages[model]['ndcg'] = NDCG_AVG.get(model, None)
        averages[model]['num_tasks'] = len(data['qps'])
    
    return averages

def print_latex_table(averages):
    """Print LaTeX Table 5: Query Latency and Throughput."""
    print("\n=== Table 5: Query Latency and Throughput ===\n")
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\caption{Query latency and throughput comparison across retrieval models. Latency percentiles measured over all 12 BRIGHT tasks.}")
    print(r"\small")
    print(r"\begin{tabular}{lrrrrrr}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{QPS} & \textbf{Mean (ms)} & \textbf{p50} & \textbf{p95} & \textbf{p99} & \textbf{nDCG@10} \\")
    print(r"\midrule")
    
    # Sparse
    print(r"\multicolumn{7}{c}{\textit{Sparse}} \\")
    print(r"\midrule")
    for model in ['bm25']:
        print_row(model, averages)
    
    # Dense <1B
    print(r"\midrule")
    print(r"\multicolumn{7}{c}{\textit{Dense (<1B)}} \\")
    print(r"\midrule")
    for model in ['sbert', 'bge', 'contriever', 'nomic', 'inst-l']:
        print_row(model, averages)
    
    # Dense >1B
    print(r"\midrule")
    print(r"\multicolumn{7}{c}{\textit{Dense (>1B)}} \\")
    print(r"\midrule")
    for model in ['inst-xl', 'e5', 'sf', 'qwen', 'qwen2']:
        print_row(model, averages)
    
    # Reasoning
    print(r"\midrule")
    print(r"\multicolumn{7}{c}{\textit{Reasoning-Specialized}} \\")
    print(r"\midrule")
    for model in ['reasonir', 'rader', 'diver-retriever']:
        print_row(model, averages)
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:latency}")
    print(r"\end{table}")

def print_row(model, averages):
    """Print a single row for the latency table."""
    a = averages.get(model)
    model_name = model.upper() if model in ['bm25', 'bge', 'sbert', 'e5', 'sf'] else model.replace('-', '-').title()
    
    if a is None:
        print(f"{model_name} & -- & -- & -- & -- & -- & {NDCG_AVG.get(model, '--')} \\\\")
    else:
        qps = f"{a['qps']:.1f}" if a.get('qps') else "--"
        mean = f"{a['mean_ms']:.1f}" if a.get('mean_ms') else "--"
        p50 = f"{a['p50_ms']:.1f}" if a.get('p50_ms') else "--"
        p95 = f"{a['p95_ms']:.1f}" if a.get('p95_ms') else "--"
        p99 = f"{a['p99_ms']:.1f}" if a.get('p99_ms') else "--"
        ndcg = f"{a['ndcg']:.1f}" if a.get('ndcg') else "--"
        print(f"{model_name} & {qps} & {mean} & {p50} & {p95} & {p99} & {ndcg} \\\\")

def main():
    base_dir = r'E:\llm-retrieval\sigir_results\phase2_efficiency'
    
    print("Loading Phase 2 query latency data...")
    results = load_phase2_latency(base_dir)
    
    print("\nComputing averages...")
    averages = compute_averages(results)
    
    # Print summary
    print("\n=== Summary ===")
    for model in MODELS:
        a = averages.get(model)
        if a and a.get('qps'):
            print(f"{model}: QPS={a['qps']:.1f}, Mean={a['mean_ms']:.1f}ms, "
                  f"p95={a['p95_ms']:.1f}ms, nDCG={a.get('ndcg', 'N/A')}")
        else:
            print(f"{model}: Data not available")
    
    # Print LaTeX
    print_latex_table(averages)
    
    # Save to JSON
    os.makedirs('sigir_results', exist_ok=True)
    with open('sigir_results/phase2_latency_aggregated.json', 'w') as f:
        json.dump(averages, f, indent=2, default=str)
    print("\nResults saved to sigir_results/phase2_latency_aggregated.json")

if __name__ == '__main__':
    main()
