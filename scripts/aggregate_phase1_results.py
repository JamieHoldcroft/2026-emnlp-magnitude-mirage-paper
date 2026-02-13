"""
Aggregate Phase 1 results from BRIGHT benchmark reproduction.
Generates LaTeX tables for Section 5 of the SIGIR paper.
"""

import os
import json
import glob

# Models to include in tables
MODELS = ['bm25', 'bge', 'inst-l', 'sbert', 'e5', 'sf', 'inst-xl', 'qwen', 'qwen2', 
          'nomic', 'contriever', 'reasonir', 'rader', 'diver-retriever']

# Tasks in display order (matching BRIGHT paper)
TASKS = ['biology', 'earth_science', 'economics', 'psychology', 'robotics', 
         'stackoverflow', 'sustainable_living', 'leetcode', 'pony', 
         'aops', 'theoremqa_questions', 'theoremqa_theorems']

# Short task names for table header
TASK_SHORT = {
    'biology': 'Bio.', 'earth_science': 'Earth.', 'economics': 'Econ.', 
    'psychology': 'Psy.', 'robotics': 'Rob.', 'stackoverflow': 'Stack.', 
    'sustainable_living': 'Sus.', 'leetcode': 'Leet.', 'pony': 'Pony',
    'aops': 'AoPS', 'theoremqa_questions': 'TheoQ.', 'theoremqa_theorems': 'TheoT.'
}

# Original BRIGHT paper results (nDCG@10) - from the paper's tables
BRIGHT_ORIGINAL = {
    'bm25': {'biology': 18.9, 'earth_science': 27.2, 'economics': 14.9, 'psychology': 12.5, 
             'robotics': 13.6, 'stackoverflow': 18.4, 'sustainable_living': 15.0, 
             'leetcode': 24.4, 'pony': 7.9, 'aops': 6.2, 'theoremqa_questions': 10.4, 
             'theoremqa_theorems': 4.9},
    'bge': {'biology': 11.7, 'earth_science': 24.6, 'economics': 16.6, 'psychology': 17.5, 
            'robotics': 11.7, 'stackoverflow': 10.8, 'sustainable_living': 13.3, 
            'leetcode': 26.7, 'pony': 5.7, 'aops': 6.0, 'theoremqa_questions': 13.0, 
            'theoremqa_theorems': 6.9},
    'inst-l': {'biology': 15.2, 'earth_science': 21.2, 'economics': 14.7, 'psychology': 22.3, 
               'robotics': 11.4, 'stackoverflow': 13.3, 'sustainable_living': 13.5, 
               'leetcode': 19.5, 'pony': 1.3, 'aops': 8.1, 'theoremqa_questions': 20.9, 
               'theoremqa_theorems': 9.1},
    'sbert': {'biology': 15.1, 'earth_science': 20.4, 'economics': 16.6, 'psychology': 22.7, 
              'robotics': 8.2, 'stackoverflow': 11.0, 'sustainable_living': 15.3, 
              'leetcode': 26.4, 'pony': 7.0, 'aops': 5.3, 'theoremqa_questions': 20.0, 
              'theoremqa_theorems': 10.8},
    'e5': {'biology': 18.6, 'earth_science': 26.0, 'economics': 15.5, 'psychology': 15.8, 
           'robotics': 16.3, 'stackoverflow': 11.2, 'sustainable_living': 18.1, 
           'leetcode': 28.7, 'pony': 4.9, 'aops': 7.1, 'theoremqa_questions': 26.1, 
           'theoremqa_theorems': 26.8},
    'sf': {'biology': 19.1, 'earth_science': 26.7, 'economics': 17.8, 'psychology': 19.0, 
           'robotics': 16.3, 'stackoverflow': 14.4, 'sustainable_living': 19.2, 
           'leetcode': 27.4, 'pony': 2.0, 'aops': 7.4, 'theoremqa_questions': 24.3, 
           'theoremqa_theorems': 26.0},
    'inst-xl': {'biology': 21.6, 'earth_science': 34.3, 'economics': 22.4, 'psychology': 27.4, 
                'robotics': 18.2, 'stackoverflow': 21.2, 'sustainable_living': 19.1, 
                'leetcode': 27.5, 'pony': 5.0, 'aops': 8.5, 'theoremqa_questions': 15.6, 
                'theoremqa_theorems': 5.9},
    'qwen': {'biology': 24.8, 'earth_science': 32.3, 'economics': 18.9, 'psychology': 19.8, 
             'robotics': 17.1, 'stackoverflow': 13.6, 'sustainable_living': 17.8, 
             'leetcode': 29.9, 'pony': 22.0, 'aops': 8.8, 'theoremqa_questions': 25.2, 
             'theoremqa_theorems': 21.2},
    'qwen2': {'biology': 30.6, 'earth_science': 36.4, 'economics': 17.8, 'psychology': 24.6, 
              'robotics': 13.2, 'stackoverflow': 22.2, 'sustainable_living': 14.8, 
              'leetcode': 25.5, 'pony': 9.9, 'aops': 14.4, 'theoremqa_questions': 27.8, 
              'theoremqa_theorems': 32.9},
}

def load_results(base_dir):
    """Load all Phase 1 results from the sigir_results directory."""
    results = {}
    
    for model in MODELS:
        model_dir = os.path.join(base_dir, model)
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model}")
            continue
            
        results[model] = {}
        for task in TASKS:
            task_dir = os.path.join(model_dir, task)
            if not os.path.exists(task_dir):
                print(f"Task directory not found: {model}/{task}")
                continue
            
            # Find results.json file
            result_files = glob.glob(os.path.join(task_dir, f'{task}_{model}_long_False', 'results.json'))
            if not result_files:
                result_files = glob.glob(os.path.join(task_dir, '*', 'results.json'))
            
            if result_files:
                try:
                    with open(result_files[0], 'r') as f:
                        data = json.load(f)
                        results[model][task] = data.get('NDCG@10', 0) * 100  # Convert to percentage
                except Exception as e:
                    print(f"Error loading {result_files[0]}: {e}")
                    results[model][task] = 0
            else:
                print(f"No results file found for {model}/{task}")
                results[model][task] = 0
    
    return results

def load_efficiency(base_dir):
    """Load efficiency metrics from Phase 1 results."""
    efficiency = {}
    
    for model in MODELS:
        model_dir = os.path.join(base_dir, model)
        if not os.path.exists(model_dir):
            continue
            
        efficiency[model] = {'index_times': [], 'memory': []}
        for task in TASKS:
            task_dir = os.path.join(model_dir, task)
            if not os.path.exists(task_dir):
                continue
            
            # Find efficiency.json file
            eff_files = glob.glob(os.path.join(task_dir, f'{task}_{model}_long_False', 'efficiency.json'))
            if not eff_files:
                eff_files = glob.glob(os.path.join(task_dir, '*', 'efficiency.json'))
            
            if eff_files:
                try:
                    with open(eff_files[0], 'r') as f:
                        data = json.load(f)
                        idx_info = data.get('indexing', {})
                        idx_time = idx_info.get('total_time_seconds', 0)
                        if idx_time > 0:
                            efficiency[model]['index_times'].append(idx_time)
                except Exception as e:
                    pass
    
    return efficiency

def print_effectiveness_table(reproduced):
    """Print LaTeX table comparing original BRIGHT vs reproduced results."""
    print("\n\n=== Table 3: Effectiveness Reproduction ===\n")
    print(r"\begin{table*}")
    print(r"\centering")
    print(r"\caption{Reproducibility analysis: comparison of original BRIGHT results (nDCG@10) with our reproduction. " +
          r"Values show percentage points; \textcolor{ForestGreen}{green} indicates exact match, \textcolor{red}{red} indicates discrepancy.}")
    print(r"\small")
    print(r"\setlength\tabcolsep{2.5pt}")
    print(r"\resizebox{\textwidth}{!}{")
    
    # Header
    print(r"\begin{tabular}{l|ccccccc|cc|ccc|c}")
    print(r"\toprule")
    print(r"& \multicolumn{7}{c|}{StackExchange} & \multicolumn{2}{c|}{Coding} & \multicolumn{3}{c|}{Theorems} & \\")
    header = "Model & " + " & ".join([TASK_SHORT[t] for t in TASKS]) + " & Avg. \\\\"
    print(header)
    print(r"\midrule")
    
    # Sparse model
    print(r"\multicolumn{14}{c}{\textit{Sparse Model}} \\")
    print(r"\midrule")
    for model in ['bm25']:
        if model in reproduced and model in BRIGHT_ORIGINAL:
            row_data = []
            for task in TASKS:
                orig = BRIGHT_ORIGINAL[model].get(task, 0)
                repro = reproduced[model].get(task, 0)
                diff = abs(repro - orig)
                if diff < 0.5:
                    row_data.append(f"{repro:.1f}")
                else:
                    row_data.append(f"\\textcolor{{red}}{{{repro:.1f}}}")
            avg = sum(reproduced[model].values()) / len(reproduced[model]) if reproduced[model] else 0
            orig_avg = sum(BRIGHT_ORIGINAL[model].values()) / len(BRIGHT_ORIGINAL[model])
            print(f"BM25 & " + " & ".join(row_data) + f" & {avg:.1f} \\\\")
    
    # Dense models (<1B)
    print(r"\midrule")
    print(r"\multicolumn{14}{c}{\textit{Open-sourced Models (<1B)}} \\")
    print(r"\midrule")
    for model in ['bge', 'inst-l', 'sbert', 'contriever', 'nomic']:
        if model in reproduced:
            row_data = []
            for task in TASKS:
                orig = BRIGHT_ORIGINAL.get(model, {}).get(task, 0)
                repro = reproduced[model].get(task, 0)
                diff = abs(repro - orig)
                if diff < 0.5 or orig == 0:
                    row_data.append(f"{repro:.1f}")
                else:
                    row_data.append(f"\\textcolor{{red}}{{{repro:.1f}}}")
            avg = sum(reproduced[model].values()) / len(reproduced[model]) if reproduced[model] else 0
            model_name = model.upper() if model in ['bge', 'sbert'] else model.replace('-', '-').title()
            print(f"{model_name} & " + " & ".join(row_data) + f" & {avg:.1f} \\\\")
    
    # LLM-based models (>1B)
    print(r"\midrule")
    print(r"\multicolumn{14}{c}{\textit{Open-sourced Models (>1B)}} \\")
    print(r"\midrule")
    for model in ['e5', 'sf', 'inst-xl', 'qwen', 'qwen2']:
        if model in reproduced:
            row_data = []
            for task in TASKS:
                orig = BRIGHT_ORIGINAL.get(model, {}).get(task, 0)
                repro = reproduced[model].get(task, 0)
                diff = abs(repro - orig)
                if diff < 0.5 or orig == 0:
                    row_data.append(f"{repro:.1f}")
                else:
                    row_data.append(f"\\textcolor{{red}}{{{repro:.1f}}}")
            avg = sum(reproduced[model].values()) / len(reproduced[model]) if reproduced[model] else 0
            model_name = model.upper() if model in ['e5', 'sf'] else model.replace('-', '-').title()
            print(f"{model_name} & " + " & ".join(row_data) + f" & {avg:.1f} \\\\")
    
    # Reasoning models
    print(r"\midrule")
    print(r"\multicolumn{14}{c}{\textit{Reasoning-Specialized Models}} \\")
    print(r"\midrule")
    for model in ['reasonir', 'rader', 'diver-retriever']:
        if model in reproduced:
            row_data = []
            for task in TASKS:
                repro = reproduced[model].get(task, 0)
                row_data.append(f"{repro:.1f}")
            avg = sum(reproduced[model].values()) / len(reproduced[model]) if reproduced[model] else 0
            model_name = model.title().replace('-', ' ')
            print(f"{model_name} & " + " & ".join(row_data) + f" & {avg:.1f} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\label{tab:reproduction}")
    print(r"\end{table*}")

def main():
    base_dir = 'sigir_results/phase1_cache_build'
    
    print("Loading Phase 1 results...")
    reproduced = load_results(base_dir)
    efficiency = load_efficiency(base_dir)
    
    # Print summary
    print("\n=== Summary ===")
    for model, tasks in reproduced.items():
        if tasks:
            avg = sum(tasks.values()) / len(tasks)
            print(f"{model}: {avg:.1f} nDCG@10 (avg over {len(tasks)} tasks)")
    
    # Print LaTeX table
    print_effectiveness_table(reproduced)
    
    # Save to JSON for reference
    with open('./phase1_aggregated.json', 'w') as f:
        json.dump({'reproduced': reproduced, 'efficiency': efficiency}, f, indent=2)
    print("\nResults saved to sigir_results/phase1_aggregated.json")

if __name__ == '__main__':
    main()
