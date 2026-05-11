import os
import json
import time
import random
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

from sklearn.metrics import roc_auc_score


# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
MODEL_RETRIEVER = "diver-retriever"
TOP_K = 5                   # Aligns the retrieval confidence window and the LLM context window

LLM_GENERATOR = "gpt-4o-mini" 
LLM_JUDGE = "gpt-4o"

BRIGHT_TASKS = [
    "biology", "aops", "earth_science", "economics", "leetcode", "pony",
    "psychology", "robotics", "stackoverflow", "sustainable_living",
    "theoremqa_questions", "theoremqa_theorems"
]

SCORES_BASE_PATH = Path("results/scores_results/results_bright")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def call_llm(prompt, model, system_msg="You are a helpful assistant.", max_tokens=300):
    """Wrapper for robust OpenAI API calls with basic rate-limit protection."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=max_tokens
        )
        time.sleep(0.05) 
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        time.sleep(2) # Backoff on error
        return "ERROR"

def compute_selective_prediction_curve(df, sort_column, ascending=False):
    """
    Computes the full Coverage vs Accuracy curve.
    This calculates the running accuracy across all possible thresholds.
    """
    sorted_df = df.sort_values(by=sort_column, ascending=ascending).reset_index(drop=True)
    
    # Convert CORRECT/INCORRECT to 1/0
    is_correct = (sorted_df["judgment"] == "CORRECT").astype(int).values
    
    # Cumulative sum of correct answers
    cumulative_correct = np.cumsum(is_correct)
    
    # Array of total queries processed at each step (1 to N)
    k_values = np.arange(1, len(sorted_df) + 1)
    
    # Calculate running arrays
    accuracies = (cumulative_correct / k_values) * 100
    coverages = (k_values / len(sorted_df)) * 100
    
    return coverages, accuracies

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def main():
    print("PHASE 1: Data Gathering (No queries dropped)")
    
    all_queries = []
    for task in BRIGHT_TASKS:
        score_file = SCORES_BASE_PATH / MODEL_RETRIEVER / task / f"{task}_{MODEL_RETRIEVER}_long_False" / "score.json"
        if not score_file.exists(): continue
            
        with open(score_file, "r") as f:
            task_scores = json.load(f)
            
        for query_id, doc_scores in task_scores.items():
            if not doc_scores: continue
            sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
            
            # Robustness: Handle queries with fewer docs than TOP_K without dropping them
            actual_k = min(len(sorted_docs), TOP_K)
            s_1 = sorted_docs[0][1]
            s_k = sorted_docs[actual_k - 1][1]
            
            all_queries.append({
                "query_id": query_id,
                "task": task,
                "max_score": s_1,
                "score_gap": s_1 - s_k,
                "top_docs": [doc_id for doc_id, score in sorted_docs[:actual_k]]
            })

    print(f"Loaded {len(all_queries)} total queries across all BRIGHT tasks.")

    print("\nLoading BRIGHT corpus (Memory-Safe Mode)...")
    try:
        hf_examples = load_dataset("xlangai/BRIGHT", "examples", trust_remote_code=True)
        corpus = load_dataset("xlangai/BRIGHT", "corpus", split="train", trust_remote_code=True)
    except Exception as e:
         print("Dataset load failed. Check internet.")
         return

    needed_doc_ids = set()
    for q in all_queries: needed_doc_ids.update(q["top_docs"])
        
    corpus_dict = {}
    for doc in tqdm(corpus, desc="Extracting required documents"):
        doc_id = str(doc["id"])
        if doc_id in needed_doc_ids:
            corpus_dict[doc_id] = doc["content"]

    print("\nPHASE 2: Forced Generation & Judging (This will take a while...)")
    results_log = []
    
    for q_data in tqdm(all_queries, desc="Evaluating Queries"):
        q_id, task = q_data["query_id"], q_data["task"]
        
        hf_task_data = {str(e["id"]): e for e in hf_examples[task]}
        if q_id not in hf_task_data: continue
            
        query_text = hf_task_data[q_id]["query"]
        gold_ids = [str(g_id) for g_id in hf_task_data[q_id]["gold_ids"]]
        
        context_texts = [corpus_dict.get(str(d_id), "") for d_id in q_data["top_docs"]]
        context_block = "\n\n".join([f"Document {i+1}:\n{text}" for i, text in enumerate(context_texts)])
        
        gold_texts = []
        for g_id in gold_ids:
            g_id = str(g_id)
            if g_id in corpus_dict:
                gold_texts.append(corpus_dict[g_id])

        # fallback safety (VERY important for evaluation integrity)
        if len(gold_texts) == 0:
            gold_texts = ["NO_GOLD_DOCUMENT_FOUND"]

        gold_block = "\n\n".join(gold_texts)
        
        # --- GENERATOR (Forced to Answer) ---
        sys_gen = "You are an expert QA assistant. You must answer the question based on the provided context. Provide the best possible answer. Do not apologize or refuse."
        prompt_gen = f"Context:\n{context_block}\n\nQuestion: {query_text}\nAnswer:"
        prediction = call_llm(prompt_gen, LLM_GENERATOR, system_msg=sys_gen)
        
        # --- JUDGE (Strict Evaluation using Reference Evidence framing) ---
        sys_judge = """
        You are an expert evaluator grading whether an AI system correctly answered a question.

        A reference evidence section is provided to help determine the expected answer, but the AI prediction does not need to exactly match the wording or structure of the reference.

        Mark CORRECT if the prediction:
        - correctly answers the question,
        - is semantically consistent with the reference evidence,
        - and would reasonably be accepted by a human expert.

        Mark INCORRECT if the answer is wrong, unsupported, incomplete for the task, or contradicts the reference evidence.

        Output ONLY:
        CORRECT
        or
        INCORRECT
        """
        prompt_judge = f"""
        Question:
        {query_text}

        Reference Evidence:
        {gold_block}

        AI Prediction:
        {prediction}

        Is the AI prediction correct?
        """
        raw_judgment = call_llm(prompt_judge, LLM_JUDGE, system_msg=sys_judge, max_tokens=10)
        
        # Safe Parsing
        judgment_clean = raw_judgment.strip().upper()
        if judgment_clean == "CORRECT":
            judgment = "CORRECT"
        else:
            judgment = "INCORRECT"
            
        results_log.append({
            "query_id": q_id,
            "task": task,
            "max_score": q_data["max_score"],
            "score_gap": q_data["score_gap"],
            "judgment": judgment
        })

    # Save the Master CSV
    df = pd.DataFrame(results_log)
    df.to_csv("rag_master_results.csv", index=False)
    print("\nSaved master table to 'rag_master_results.csv'!")

    # ==========================================
    # 4. FULL SELECTIVE PREDICTION CURVE EXPORT
    # ==========================================
    print("\n" + "="*50)
    print("PHASE 3: Computing Selective Prediction Curves")
    print("="*50)
    
    # Calculate Continuous Curves
    cov_gap, acc_gap = compute_selective_prediction_curve(df, "score_gap", ascending=False)
    cov_max, acc_max = compute_selective_prediction_curve(df, "max_score", ascending=False)
    
    # Calculate Random Baseline (Shuffle the dataframe)
    df_random = df.sample(frac=1, random_state=42)
    cov_rand, acc_rand = compute_selective_prediction_curve(df_random, "score_gap", ascending=False) # sort column doesn't matter since it's shuffled

    # Save the curve data to a new CSV for easy plotting
    curve_df = pd.DataFrame({
        "Coverage_%": cov_gap, # Coverage is the same (1/N to N/N) for all
        "Accuracy_ScoreGap": acc_gap,
        "Accuracy_MaxScore": acc_max,
        "Accuracy_Random": acc_rand
    })
    
    curve_df.to_csv("rag_curve_plot_data.csv", index=False)
    
    # Print a few summary points for terminal feedback
    baseline_acc = (df["judgment"] == "CORRECT").sum() / len(df) * 100
    print(f"Overall Baseline Accuracy: {baseline_acc:.1f}%")
    print("Saved 'rag_curve_plot_data.csv'. You can plug this directly into matplotlib or Excel to plot the full curve!")

if __name__ == "__main__":
    main()