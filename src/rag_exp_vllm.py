"""
End-to-end RAG abstention experiment for The Magnitude Mirage.

Generator: local vLLM (Qwen3.6-27B by default), accessed via OpenAI-compatible API.
Judge: Azure OpenAI GPT-4o (different family from generator -> avoids self-judge bias).

Per query, we capture:
- max_score = s1
- gap_ctx   = s1 - s_{top-k context}    (signal at RAG operating point)
- gap_25    = s1 - s_25                 (paper-headline signal)
- retrieval_success_5 / _25 (NDCG@k > 0 ground-truth from gold_ids)
- judgment (CORRECT / INCORRECT) on the generated answer

Supports concurrency and resume-from-checkpoint.
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORES_BASE_PATH = REPO_ROOT / "results" / "scores_results" / "results_bright"
OUTPUT_DIR = REPO_ROOT / "results" / "rag_exp"

BRIGHT_TASKS = [
    "biology", "aops", "earth_science", "economics", "leetcode", "pony",
    "psychology", "robotics", "stackoverflow", "sustainable_living",
    "theoremqa_questions", "theoremqa_theorems",
]


SYS_GEN = (
    "You are an expert QA assistant. You must answer the question based on the "
    "provided context. Provide the best possible answer. Do not apologize or refuse."
)

SYS_JUDGE = (
    "You are an expert evaluator grading whether an AI system correctly answered a question.\n\n"
    "A reference evidence section is provided to help determine the expected answer, but the "
    "AI prediction does not need to exactly match the wording or structure of the reference.\n\n"
    "Mark CORRECT if the prediction:\n"
    "- correctly answers the question,\n"
    "- is semantically consistent with the reference evidence,\n"
    "- and would reasonably be accepted by a human expert.\n\n"
    "Mark INCORRECT if the answer is wrong, unsupported, incomplete for the task, "
    "or contradicts the reference evidence.\n\n"
    "Output ONLY:\nCORRECT\nor\nINCORRECT"
)


def build_generator_client(base_url):
    return OpenAI(api_key="EMPTY", base_url=base_url)


def build_judge_client():
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


def call_chat(client, model, prompt, system_msg, max_tokens=300, temperature=0.0,
              retries=6, base_delay=1.5, extra_body=None):
    last = None
    for attempt in range(retries):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            r = client.chat.completions.create(**kwargs)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last = e
            time.sleep(base_delay * (2 ** attempt))
    return f"ERROR: {last}"


def truncate(text, limit):
    if not text or limit <= 0:
        return text or ""
    return text if len(text) <= limit else (text[:limit] + " ...[truncated]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--gen-model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--judge-model",
                        default=os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o"))
    parser.add_argument("--top-k-ctx", type=int, default=5,
                        help="Context window for generation (operating point).")
    parser.add_argument("--top-k-gap25", type=int, default=25,
                        help="Cutoff for the paper-headline Gap@25 signal.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-queries-per-task", type=int, default=0)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any existing CSV and start fresh.")
    parser.add_argument("--max-doc-chars", type=int, default=4000,
                        help="Truncate each context/gold doc to this many chars.")
    parser.add_argument("--max-gold-docs", type=int, default=4,
                        help="Cap on number of gold docs included in judge prompt.")
    args = parser.parse_args()

    tag = args.output_tag or f"{args.retriever}_qwen36_gpt4o"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master_csv = OUTPUT_DIR / f"rag_master_{tag}.csv"

    # ---------- resume ----------
    prior_rows = []
    done = set()
    if (not args.no_resume) and master_csv.exists():
        prior_df = pd.read_csv(master_csv)
        prior_rows = prior_df.to_dict("records")
        done = {(str(r["task"]), str(r["query_id"])) for r in prior_rows}
        print(f"[resume] {len(done)} queries already in {master_csv.name}")

    # ---------- PHASE 1: query list + signals ----------
    all_queries = []
    for task in BRIGHT_TASKS:
        score_file = (SCORES_BASE_PATH / args.retriever / task /
                      f"{task}_{args.retriever}_long_False" / "score.json")
        if not score_file.exists():
            print(f"  [skip missing scores] {score_file}")
            continue
        with open(score_file, "r") as f:
            task_scores = json.load(f)
        cnt = 0
        for query_id, doc_scores in task_scores.items():
            if not doc_scores:
                continue
            if (task, str(query_id)) in done:
                continue
            sorted_docs = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
            n = len(sorted_docs)
            k_ctx = min(args.top_k_ctx, n)
            k_25 = min(args.top_k_gap25, n)
            s1 = sorted_docs[0][1]
            sk_ctx = sorted_docs[k_ctx - 1][1]
            sk_25 = sorted_docs[k_25 - 1][1]
            all_queries.append({
                "query_id": str(query_id),
                "task": task,
                "max_score": s1,
                "gap_ctx": s1 - sk_ctx,
                "gap_25": s1 - sk_25,
                "top_docs": [d for d, _ in sorted_docs[:k_ctx]],
                "top_25_ids": [d for d, _ in sorted_docs[:k_25]],
            })
            cnt += 1
            if args.max_queries_per_task and cnt >= args.max_queries_per_task:
                break
    print(f"[{args.retriever}] new queries to evaluate: {len(all_queries)}")
    if not all_queries:
        print("Nothing to do.")
        return

    # ---------- PHASE 2: load BRIGHT data ----------
    hf_examples = {}
    corpus_dict = {}
    needed_per_task = {}
    for q in all_queries:
        needed_per_task.setdefault(q["task"], set()).update(q["top_docs"])
        needed_per_task[q["task"]].update(q["top_25_ids"])
    for task, needed in needed_per_task.items():
        examples_split = load_dataset("xlangai/BRIGHT", "examples", split=task)
        hf_examples[task] = {str(e["id"]): e for e in examples_split}
        for ex in examples_split:
            needed.update(str(g) for g in ex.get("gold_ids", []))
        documents_split = load_dataset("xlangai/BRIGHT", "documents", split=task)
        for doc in tqdm(documents_split, desc=f"docs[{task}]"):
            doc_id = str(doc["id"])
            if doc_id in needed:
                corpus_dict[doc_id] = doc["content"]

    # ---------- PHASE 3: parallel gen + judge ----------
    gen_client = build_generator_client(args.vllm_url)
    judge_client = build_judge_client()

    results_log = list(prior_rows)
    write_lock = threading.Lock()
    flush_every = 25

    def flush():
        pd.DataFrame(results_log).to_csv(master_csv, index=False)

    def work(q_data):
        q_id, task = q_data["query_id"], q_data["task"]
        ex = hf_examples[task].get(q_id)
        if ex is None:
            return None
        query_text = ex["query"]
        gold_ids = [str(g) for g in ex.get("gold_ids", [])]
        retrieval_success_5 = int(any(g in q_data["top_docs"] for g in gold_ids))
        retrieval_success_25 = int(any(g in q_data["top_25_ids"] for g in gold_ids))

        # context for generation (top-k_ctx retrieved docs, truncated)
        context_texts = [truncate(corpus_dict.get(str(d), ""), args.max_doc_chars)
                         for d in q_data["top_docs"]]
        context_block = "\n\n".join(
            f"Document {i+1}:\n{t}" for i, t in enumerate(context_texts)
        )

        # reference for judge (truncated gold docs, capped count)
        kept = [g for g in gold_ids if g in corpus_dict][:args.max_gold_docs]
        gold_texts = [truncate(corpus_dict[g], args.max_doc_chars) for g in kept]
        gold_block = "\n\n".join(gold_texts) if gold_texts else "NO_GOLD_DOCUMENT_FOUND"

        prompt_gen = f"Context:\n{context_block}\n\nQuestion: {query_text}\nAnswer:"
        prediction = call_chat(
            gen_client, args.gen_model, prompt_gen, SYS_GEN, max_tokens=300,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        prompt_judge = (
            f"Question:\n{query_text}\n\n"
            f"Reference Evidence:\n{gold_block}\n\n"
            f"AI Prediction:\n{prediction}\n\n"
            "Is the AI prediction correct?"
        )
        raw = call_chat(judge_client, args.judge_model, prompt_judge, SYS_JUDGE,
                        max_tokens=10)
        judgment = "CORRECT" if raw.strip().upper().startswith("CORRECT") else "INCORRECT"

        return {
            "retriever": args.retriever,
            "task": task,
            "query_id": q_id,
            "max_score": q_data["max_score"],
            "gap_ctx": q_data["gap_ctx"],
            "gap_25": q_data["gap_25"],
            "retrieval_success_5": retrieval_success_5,
            "retrieval_success_25": retrieval_success_25,
            "judgment": judgment,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(work, q) for q in all_queries]
        with tqdm(total=len(futures), desc=args.retriever) as pbar:
            for fut in as_completed(futures):
                pbar.update(1)
                row = fut.result()
                if row is None:
                    continue
                with write_lock:
                    results_log.append(row)
                    if len(results_log) % flush_every == 0:
                        flush()

    flush()
    print(f"\nWrote {master_csv} ({len(results_log)} rows)")


if __name__ == "__main__":
    main()
