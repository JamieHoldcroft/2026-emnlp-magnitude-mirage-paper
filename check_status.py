import os
import json
from tabulate import tabulate

BASE_DIR = "/leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results"
TASKS = [
    "biology", "earth_science", "economics", "psychology", "robotics", 
    "stackoverflow", "sustainable_living", "leetcode", "pony", "aops", 
    "theoremqa_questions", "theoremqa_theorems"
]

MODELS_PHASE1 = [
    "bm25", "sbert", "bge", "contriever", "nomic", "inst-l", "inst-xl", 
    "sf", "e5", "qwen", "qwen2", "reasonir", "rader", "diver-retriever"
]

SIZES_PHASE3 = ["1000", "5000", "10000", "25000", "50000", "100000", "full"]
MODELS_PHASE3 = ["e5"] # Phase 3 seems to focus on e5 based on folder structure

QUANTS_PHASE4 = ["fp32", "fp16", "bf16", "int8", "int4"]
MODELS_PHASE4 = ["e5", "rader"] # Phase 4 seems to focus on e5 and rader

REASONING_TYPES = ["gpt4_reason", "llama3-70b_reason", "claude-3-opus_reason", "grit_reason", "Gemini-1.0_reason"]
MODELS_PHASE6 = [
    "bm25", "sbert", "bge", "contriever", "nomic", "inst-l", "inst-xl", 
    "sf", "e5", "qwen", "qwen2", "reasonir", "rader", "diver-retriever"
]

MODELS_PHASE8 = [
    "bm25", "sbert", "bge", "contriever", "nomic", "inst-l", "inst-xl", 
    "sf", "e5", "qwen", "qwen2", "grit"
]

DENSE_MODELS_PHASE9 = ["e5", "bge", "qwen2", "reasonir"]
FUSION_METHODS_PHASE9 = ["rrf", "linear", "dynamic"]

def check_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def check_phase1():
    print("\n[PHASE 1: Cache Build / Original Queries]")
    results = []
    for task in TASKS:
        row = [task]
        for model in MODELS_PHASE1:
            # Structure: phase1_cache_build/{model}/{task}/{task}_{model}_long_False/results.json
            path = os.path.join(BASE_DIR, "phase1_cache_build", model, task, f"{task}_{model}_long_False", "results.json")
            row.append("OK" if check_file(path) else "-")
        results.append(row)
    print(tabulate(results, headers=["Task"] + MODELS_PHASE1, tablefmt="grid"))

def check_phase3():
    print("\n[PHASE 3: Scaling]")
    results = []
    for task in TASKS:
        row = [task]
        for size in SIZES_PHASE3:
            # Structure: phase3_scaling/{task}_size_{size}/{task}_e5_long_False/results.json
            path = os.path.join(BASE_DIR, "phase3_scaling", f"{task}_size_{size}", f"{task}_e5_long_False", "results.json")
            row.append("OK" if check_file(path) else "-")
        results.append(row)
    print(tabulate(results, headers=["Task"] + SIZES_PHASE3, tablefmt="grid"))

def check_phase4():
    print("\n[PHASE 4: Quantization]")
    results = []
    headers = ["Task/Model"]
    for q in QUANTS_PHASE4:
        headers.append(f"e5_{q}")
        headers.append(f"rader_{q}")
        
    for task in TASKS:
        row = [task]
        for q in QUANTS_PHASE4:
            # e5
            path_e5 = os.path.join(BASE_DIR, "phase4_quantization", f"{task}_{q}", f"{task}_e5_long_False", "results.json")
            row.append("OK" if check_file(path_e5) else "-")
            # rader
            path_rader = os.path.join(BASE_DIR, "phase4_quantization", f"{task}_{q}", f"{task}_rader_long_False", "results.json")
            row.append("OK" if check_file(path_rader) else "-")
        results.append(row)
    print(tabulate(results, headers=headers, tablefmt="grid"))

def check_phase6():
    print("\n[PHASE 6: Reasoning]")
    for model in MODELS_PHASE6:
        print(f"\nModel: {model}")
        results = []
        for task in TASKS:
            row = [task]
            for rtype in REASONING_TYPES:
                # Structure: phase6_reasoning/{task}/{model}_{rtype}/{task}_{model}_long_False/results.json
                path = os.path.join(BASE_DIR, "phase6_reasoning", task, f"{model}_{rtype}", f"{task}_{model}_long_False", "results.json")
                row.append("OK" if check_file(path) else "-")
            results.append(row)
        print(tabulate(results, headers=["Task"] + REASONING_TYPES, tablefmt="grid"))

def check_phase8():
    print("\n[PHASE 8: Long Context]")
    results = []
    for task in TASKS:
        row = [task]
        for model in MODELS_PHASE8:
            # Structure: phase8_long_context/{model}/{task}/{task}_{model}_long_True/results.json
            path = os.path.join(BASE_DIR, "phase8_long_context", model, task, f"{task}_{model}_long_True", "results.json")
            row.append("OK" if check_file(path) else "-")
        results.append(row)
    print(tabulate(results, headers=["Task"] + MODELS_PHASE8, tablefmt="grid"))

def check_phase9():
    print("\n[PHASE 9: Hybrid Retrieval]")
    for task in TASKS:
        print(f"\nTask: {task}")
        results = []
        for model in DENSE_MODELS_PHASE9:
            row = [model]
            for method in FUSION_METHODS_PHASE9:
                # Structure: phase9_hybrid/{task}/{model}_{method}/{task}_hybrid_bm25_{model}_{method}_long_False/results.json
                path = os.path.join(BASE_DIR, "phase9_hybrid", task, f"{model}_{method}", f"{task}_hybrid_bm25_{model}_{method}_long_False", "results.json")
                row.append("OK" if check_file(path) else "-")
            results.append(row)
        print(tabulate(results, headers=["Dense Model"] + FUSION_METHODS_PHASE9, tablefmt="grid"))

if __name__ == "__main__":
    check_phase1()
    check_phase3()
    check_phase4()
    check_phase6()
    check_phase8()
    check_phase9()
