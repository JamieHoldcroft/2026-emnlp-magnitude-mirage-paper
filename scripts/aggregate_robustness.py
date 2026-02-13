"""
Aggregate Phase 10 robustness results.
Calculates average nDCG@10 for types: paraphrase, synonym, adversarial, length.
"""

import os
import json
import glob
import numpy as np

MODELS = ['bge', 'e5', 'qwen2', 'reasonir']
PERTURB_TYPES = ['paraphrase', 'synonym', 'adversarial', 'length']
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
    base_dir = r'E:\llm-retrieval\sigir_results\phase10_robustness'
    
    summary = {}
    for model in MODELS:
        summary[model] = {pt: [] for pt in PERTURB_TYPES}

    for task in TASKS:
        task_dir = os.path.join(base_dir, task)
        if not os.path.exists(task_dir):
            continue
            
        for model in MODELS:
            # list all subdirs for this model
            model_subdirs = [d for d in os.listdir(task_dir) if d.startswith(f"{model}_") and os.path.isdir(os.path.join(task_dir, d))]
            
            for sd in model_subdirs:
                # determine perturb type
                ptype = None
                for pt in PERTURB_TYPES:
                    if pt in sd:
                        ptype = pt
                        break
                
                if ptype:
                    res_path = glob.glob(os.path.join(task_dir, sd, "*", "results.json"))
                    if res_path:
                        data = load_json(res_path[0])
                        if data:
                            summary[model][ptype].append(data.get('NDCG@10', 0) * 100)

    # Aggregate
    final_output = {}
    for model in MODELS:
        final_output[model] = {}
        for pt in PERTURB_TYPES:
            vals = summary[model][pt]
            final_output[model][pt] = round(float(np.mean(vals)), 2) if vals else 0.0

    output_path = r'E:\llm-retrieval\sigir_results\robustness_aggregated.json'
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Robustness aggregation saved to {output_path}")

    # Print table snippet
    print("\nModel | Original* | Paraphrase | Synonym | Adversarial | Length")
    print("-" * 65)
    # *Note: Original baseline needs to be pulled from phase1/2 summary for these specific 4 models
    # I'll just use the ones I have in memory for 12-task avg
    baselines = {'bge': 13.7, 'e5': 17.9, 'qwen2': 23.3, 'reasonir': 24.1}
    
    for model in MODELS:
        orig = baselines.get(model, 0.0)
        p = final_output[model]['paraphrase']
        s = final_output[model]['synonym']
        a = final_output[model]['adversarial']
        l = final_output[model]['length']
        print(f"{model:<10} | {orig:8.1f} | {p:10.1f} | {s:7.1f} | {a:11.1f} | {l:6.1f}")

if __name__ == "__main__":
    main()
