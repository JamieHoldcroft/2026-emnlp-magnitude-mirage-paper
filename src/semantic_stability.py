# src/semantic_stability.py

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List

import warnings
warnings.simplefilter("ignore", FutureWarning)

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrievers import RETRIEVAL_FUNCS
from utils.eval_util import calculate_retrieval_metrics
from src.query_transformations import (
    TRANSFORMATION_NAMES,
    SEMANTICS_PRESERVING_PARAPHRASE_PROMPT,
    SEMANTICS_BREAKING_INCORRECT_PREMISE_PROMPT,
    SEMANTICS_NEUTRAL_NOISE_PROMPT,
)

# --- LLM Configuration & Call ---
def initialize_llm(args):
    """Initializes the generative model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. Please provide the API key."
        )
    client = OpenAI(api_key=api_key)
    return client


def llm_call(model: OpenAI, prompt: str, query: str) -> str:
    """
    Calls the specified LLM to transform the query.
    Includes basic retry logic.
    """
    full_prompt = prompt.format(QUERY=query)
    for attempt in range(3):  # Retry up to 3 times
        try:
            response = model.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=256,
            )
            response_text = response.choices[0].message.content
            if not response_text or response_text.isspace():
                raise ValueError("LLM returned an empty response.")
            return response_text.strip()
        except Exception as e:
            print(f"LLM call failed on attempt {attempt + 1}: {e}")
            time.sleep(2**attempt)  # Exponential backoff
    raise RuntimeError(f"Failed to transform query after multiple attempts: {query}")


# ------------------------------------


def get_transformed_queries(
    queries: List[Dict],
    transformation: str,
    cache_path: Path,
    llm_model: Any,
) -> List[str]:
    """
    Applies a transformation to a list of queries, using a cache if available.
    """
    if cache_path.exists():
        print(f"Loading transformed queries from cache: {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)["queries"]

    print(f"Generating transformed queries for '{transformation}'...")
    transformed_queries = []

    # LLM-based transformations
    llm_prompts = {
        "semantics_preserving_paraphrase": SEMANTICS_PRESERVING_PARAPHRASE_PROMPT,
        "semantics_neutral_noise": SEMANTICS_NEUTRAL_NOISE_PROMPT,
        "semantics_breaking_incorrect_premise": SEMANTICS_BREAKING_INCORRECT_PREMISE_PROMPT,
    }

    prompt = llm_prompts[transformation]

    for i, q in tqdm(enumerate(queries), total=len(queries)):
        query_text = q["query"]

        try:
            transformed_text = llm_call(llm_model, prompt, query_text)
            transformed_queries.append(transformed_text)
            print(f"\n\n\nOriginal: {query_text}")
            print(f"\n\n\nTransformed: {transformed_text}\n\n\n")
        except RuntimeError as e:
            print(f"Warning: {e}. Skipping this query.")
            transformed_queries.append(None)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            {"transformation": transformation, "queries": transformed_queries},
            f,
            indent=2,
        )

    return transformed_queries


def run_retrieval(
    args: argparse.Namespace,
    queries: List[str],
    query_ids: List[str],
    documents: List[str],
    doc_ids: List[str],
    excluded_ids: Dict[str, List[str]],
    retriever_func: callable,
) -> Dict[str, Dict[str, float]]:
    """A wrapper to run the retriever and get scores."""


    cfg_dir = Path("configs") / args.model.split('_ckpt')[0].split('_bilevel')[0]
    cfg_path = cfg_dir / f"{args.task}.json"

    if cfg_path.exists():
        with open(cfg_path) as f:
            config = json.load(f)
        instructions = config.get("instructions", None)
    else:
        instructions = None

    # IMPORTANT: Reuse Phase 1 document embeddings by default.
    retrieval_result = retriever_func(
        queries=queries,
        query_ids=query_ids,
        documents=documents,
        doc_ids=doc_ids,
        excluded_ids=excluded_ids,
        instructions=instructions,  # Use default instructions
        task=args.task,
        cache_dir=args.cache_dir,
        model_id=args.model,
        long_context=False,  # Phase 15 uses short context
        skip_doc_emb=True,  # Explicitly skip re-embedding docs
    )

    if isinstance(retrieval_result, dict) and "scores" in retrieval_result:
        return retrieval_result["scores"]
    return retrieval_result


def main(args):
    """Main execution logic for Phase 15."""

    output_dir = (
        Path(args.output_dir) / "phase15_semantic_stability" / args.task / args.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    if results_path.exists() and not args.force_rerun:
        print(
            f"Results already exist for {args.task}/{args.model}. Skipping. Use --force_rerun to override."
        )
        return

    # 1. Initialize LLM for transformations
    print(f"Initializing generative model: {args.llm_model}")
    llm_model = initialize_llm(args)

    # 2. Load dataset (queries, documents, qrels)
    print(f"Loading BRIGHT dataset for task: {args.task}")
    examples_dataset = load_dataset(
        args.dataset_source, "examples", cache_dir=args.hf_cache_dir, trust_remote_code=True
    )
    docs_dataset = load_dataset(
        args.dataset_source, "documents", cache_dir=args.hf_cache_dir, trust_remote_code=True
    )
    
    examples = examples_dataset[args.task]
    doc_pairs = docs_dataset[args.task]

    documents = [dp["content"] for dp in doc_pairs]
    doc_ids = [dp["id"] for dp in doc_pairs]

    query_ids = [e["id"] for e in examples]
    excluded_ids = {e["id"]: e["excluded_ids"] for e in examples}

    qrels = {}      
    for e in examples:
        qrels[str(e["id"])] = {str(gid): 1 for gid in e["gold_ids"]}

    # 3. Get retriever function
    if args.model not in RETRIEVAL_FUNCS:
        raise ValueError(f"Model '{args.model}' not found in RETRIEVAL_FUNCS.")
    retriever_func = RETRIEVAL_FUNCS[args.model]

    # 4. Get nDCG for clean queries
    print("Loading cached retrieval scores for clean queries from Phase 1...")
    phase1_results_path = (
        Path("sigir_results")
        / "phase1_cache_build"
        / args.model
        / args.task
        / f"{args.task}_{args.model}_long_False"
        / "results.json"
    )

    if not phase1_results_path.exists():
        raise FileNotFoundError(
            f"Phase 1 results not found at {phase1_results_path}. "
            f"Please run 'scripts/phase1_build_cache.sh {args.model}' for task '{args.task}' first."
        )

    with open(phase1_results_path, "r") as f:
        phase1_data = json.load(f)

    clean_metrics = phase1_data 

    # 5. Loop through transformations
    all_results = []

    for transformation in TRANSFORMATION_NAMES:
        print("-" * 50)
        print(f"Processing transformation: {transformation}")

        # a. Get transformed queries
        transform_cache_path = (
            Path(args.cache_dir)
            / "query_transformations"
            / args.llm_model
            / args.task
            / f"{transformation}.json"
        )
        transformed_queries = get_transformed_queries(
            examples, transformation, transform_cache_path, llm_model
        )

        # b. Run retrieval
        transformed_scores = run_retrieval(
            args,
            transformed_queries,
            query_ids,
            documents,
            doc_ids,
            excluded_ids,
            retriever_func,
        )

        # c. Compute full metrics
        transformed_metrics = calculate_retrieval_metrics(
            results=transformed_scores,
            qrels=qrels
        )

        # d. Compute deltas for all overlapping metrics
        delta_metrics = {}
        for metric, clean_value in clean_metrics.items():
            if metric in transformed_metrics:
                delta_metrics[metric] = clean_value - transformed_metrics[metric]

        print(f"nDCG@10 (Transformed): {transformed_metrics['NDCG@10']:.4f}")
        print(f"Delta nDCG@10: {delta_metrics['NDCG@10']:.4f}")

        all_results.append({
            "transformation": transformation,
            "clean_metrics": clean_metrics,
            "transformed_metrics": transformed_metrics,
            "delta_metrics": delta_metrics
        })

    # 6. Save results
    print("-" * 50)
    print(f"Saving results to {results_path}")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Phase 15: Semantic Stability Analysis."
    )
    parser.add_argument("--task", type=str, required=True, help="BRIGHT task to evaluate.")
    parser.add_argument(
        "--model", type=str, required=True, help="Retrieval model to evaluate."
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-4o-mini",
        help="Generative model for query transformations.",
    )
    parser.add_argument(
        "--dataset_source",
        type=str,
        default="xlangai/BRIGHT",
        help="Source for the BRIGHT dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sigir_results",
        help="Directory to save final results.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory for caching embeddings and transformed queries.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Directory for HuggingFace datasets cache.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force re-running the experiment even if results exist.",
    )

    args = parser.parse_args()
    main(args)