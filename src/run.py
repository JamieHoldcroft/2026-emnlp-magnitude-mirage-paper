import os
import sys

# Workaround for broken torch 2.8.0 nightly / inductor interaction in vLLM
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# try:
#     import torch._inductor.autotune_process
#     if not hasattr(torch._inductor.autotune_process, 'CuteDSLBenchmarkRequest'):
#         print("Monkeypatching missing symbols in torch._inductor.autotune_process")
#         class CuteDSLBenchmarkRequest: pass
#         torch._inductor.autotune_process.CuteDSLBenchmarkRequest = CuteDSLBenchmarkRequest
#     if not hasattr(torch._inductor.autotune_process, 'TensorMeta'):
#         class TensorMeta: pass
#         torch._inductor.autotune_process.TensorMeta = TensorMeta
# except (ImportError, ModuleNotFoundError):
#     pass

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # project dir

import argparse
import json
import time
import numpy as np
from tqdm import tqdm
from retrievers import RETRIEVAL_FUNCS
from utils.eval_util import calculate_retrieval_metrics
from datasets import load_dataset, Dataset


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics',
                                 'stackoverflow','sustainable_living','aops','leetcode','theoremqa_theorems',
                                 'theoremqa_questions'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['bm25','cohere','e5','google','grit','inst-l','inst-xl',
                                 'openai','qwen','qwen2','sbert','sf','voyage','bge',
                                 'bge_ce', 'nomic', 'm2', 'contriever', 'reasonir', 'rader', 'diver-retriever', 'hybrid'])
    parser.add_argument('--model_id', type=str, default=None, help='(Optional) Pass a different model ID for cache and output path naming.')
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--dataset_source', type=str, default='xlangai/BRIGHT')
    parser.add_argument('--document_expansion', default=None, type=str, choices=[None, 'gold', 'full', 'rechunk'],
                        help="Set to None to use original documents provided by BRIGHT; Set to `oracle` to use documents with oracle ones expanded'; Set to `full` to use all expanded documents.")
    parser.add_argument('--global_summary', default=None, choices=[None, 'concat'])
    parser.add_argument('--query_max_length', type=int, default=-1)
    parser.add_argument('--doc_max_length', type=int, default=-1)
    parser.add_argument('--corpus_size', type=int, default=0)
    parser.add_argument('--quantization', type=str, default=None, choices=['fp32', 'fp16', 'bf16', 'int8', 'int4'])
    parser.add_argument('--encode_batch_size', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Directory for embedding cache')
    parser.add_argument('--hf_cache_dir', type=str, default=None, help='Directory for HuggingFace datasets cache. Default: None (uses HF_HOME env variable)')
    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--reasoning', type=str, default=None)
    parser.add_argument('--reasoning_id', type=str, default=None)
    parser.add_argument('--reasoning_length_limit', type=int, default=None)
    parser.add_argument('--separate_reasoning', action='store_true', help='Append reasoning after the original query, separate by <REASON>.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')
    parser.add_argument('--no_log', action='store_true', help="Disable logging to Google Sheets.")
    parser.add_argument('--sweep_output_dir', type=str, default=None)
    parser.add_argument('--skip_doc_emb', action='store_true', help="Skip document embedding.")
    parser.add_argument('--store_all_scores', action='store_true', help="The default is to store the top 1000 scores. This option will store all scores.")
    # Hybrid retrieval arguments
    parser.add_argument('--fusion_method', type=str, default='rrf', choices=['rrf', 'linear', 'dynamic'],
                        help="Fusion method for hybrid retrieval: rrf, linear, or dynamic")
    parser.add_argument('--dense_model', type=str, default='e5',
                        choices=['e5', 'qwen2', 'bge', 'reasonir', 'grit', 'qwen', 'sf', 'inst-l', 'inst-xl', 'rader'],
                        help="Dense model to combine with BM25 in hybrid retrieval")
    parser.add_argument('--linear_weights', type=str, default='0.5,0.5',
                        help="Comma-separated weights for linear fusion: sparse_weight,dense_weight")
    args = parser.parse_args()
    if args.model_id is None:
        args.output_dir = os.path.join(args.output_dir,f"{args.task}_{args.model}_long_{args.long_context}")
    else:
        args.output_dir = os.path.join(args.output_dir,f"{args.task}_{args.model_id}_long_{args.long_context}")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.reasoning is not None:
        score_file_path = os.path.join(args.output_dir,f'{args.reasoning}_score.json')
    else:
        score_file_path = os.path.join(args.output_dir,f'score.json')
    
    efficiency_file_path = os.path.join(args.output_dir, 'efficiency.json')

    assert args.document_expansion is None or args.global_summary is None, "Cannot use expansion and summary together!"
    if args.global_summary:
        assert not args.long_context, "Global summary is supposed to enhance short-context retrieval!"

    dataset_source = args.dataset_source
    document_postfix = ''
    if args.document_expansion == 'gold':
        document_postfix = '_expanded_gold_only'
        dataset_source = 'rulins/bright-expanded'
    elif args.document_expansion == 'rechunk':
        document_postfix = 'rechunk'
    
    print(f"Dataset source: {dataset_source}")
    # print(f"Reasoning source: {reasoning_source}")

    if args.input_file is not None:
        with open(args.input_file) as f:
            examples = json.load(f)
    elif args.reasoning is not None and args.separate_reasoning:
        examples = load_dataset(dataset_source, 'examples', cache_dir=args.hf_cache_dir)[args.task]
        reasoning_examples = load_dataset(dataset_source, f"{args.reasoning}_reason", cache_dir=args.hf_cache_dir)[args.task]
    elif args.reasoning is not None and args.reasoning_length_limit is None:
        if args.reasoning in ['xrr2']:
            json_path = os.path.join(dataset_source, f"{args.reasoning}_reason", f"{args.task}_query.json")
            examples = load_dataset("json", data_files=json_path)["train"]
        elif args.reasoning in ['thinkqe', 'diver-qexpand']:
            # examples = load_dataset(dataset_source, 'examples',cache_dir=args.hf_cache_dir)[args.task]  # 包含字典里所有data
            examples = load_dataset("parquet", data_files=os.path.join(dataset_source, f"examples/{args.task}-00000-of-00001.parquet"))["train"]
            # replacing original query with expanded query
            json_path = f"{dataset_source}/output/{args.reasoning}/{args.task}/diver_aug2_result_retrieval.expand_query.json"
            id_exp_query = load_dataset("json", data_files=json_path)["train"]
            
            id_query_dict = {str(item["id"]): item["query"] for item in id_exp_query}
            def replace_query(example):
                example_id = example["id"]
                if example_id in id_query_dict:
                    example["query"] = id_query_dict[example_id]
                return example

            examples = examples.map(replace_query, num_proc=4)
        else:
            examples = load_dataset(dataset_source, f"{args.reasoning}", cache_dir=args.hf_cache_dir)[args.task]
            # examples = load_dataset("parquet", data_files=os.path.join(dataset_source, f"{args.reasoning}_reason", f"{args.task}-00000-of-00001.parquet"))["train"]

    elif args.reasoning is not None and args.reasoning_length_limit:
        reasoning_file = f"cache/reasoning/{args.task}_{args.reasoning}_{args.reasoning_length_limit}"
        with open(reasoning_file, 'r') as f:
            examples = json.load(f)
    else:
        examples = load_dataset(dataset_source, 'examples',cache_dir=args.hf_cache_dir)[args.task]
        # examples = load_dataset("parquet", data_files=os.path.join(dataset_source, f"examples/{args.task}-00000-of-00001.parquet"))["train"]
    
    if args.long_context:
        # doc_pairs = load_dataset(dataset_source, 'long_documents'+document_postfix, cache_dir=args.hf_cache_dir)[args.task]
        doc_pairs = load_dataset(dataset_source, "long_documents"+document_postfix, cache_dir=args.hf_cache_dir)[args.task]

    else:
        if args.document_expansion == 'summary':
            data_path = os.path.join(dataset_source, 'summary_documents', f'{args.task}_summary_docs.json')
            doc_pairs = load_dataset('json', data_files=data_path, cache_dir=os.path.join(args.cache_dir, 'summary_docs'))['train']
            print("Load summary documents...")
        elif args.document_expansion == 'rechunk':
            data_path = os.path.join(dataset_source, 'rechunk_documents', f'{args.task}_refine_docs.json')
            doc_pairs = load_dataset('json', data_files=data_path, cache_dir=os.path.join(args.cache_dir, 'refine_docs'))['train']
            print("Load rechunk documents...")
        else:
            # data_path = os.path.join(dataset_source, 'documents', document_postfix, f'{args.task}-00000-of-00001.parquet')
            # data_path = os.path.join(dataset_source, 'documents', f'{args.task}-00000-of-00001.parquet')
            # doc_pairs = load_dataset("parquet", data_files=data_path, cache_dir=args.hf_cache_dir)["train"]
            doc_pairs = load_dataset(dataset_source, "documents", cache_dir=args.hf_cache_dir)[args.task]


    doc_ids = []
    documents = []
    for dp in doc_pairs:
        doc_ids.append(dp['id'])
        documents.append(dp['content'])

    if args.corpus_size > 0:
        doc_ids = doc_ids[:args.corpus_size]
        documents = documents[:args.corpus_size]

    if not os.path.isfile(score_file_path):
        print("The scores file does not exist, start retrieving...")
        config = {"instructions": None, "instructions_long": None}

        cfg_dir = os.path.join(args.config_dir, args.model.split('_ckpt')[0].split('_bilevel')[0])
        cfg_path = os.path.join(cfg_dir, f"{args.task}.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                config = json.load(f)
        else:
            config = {}
            config['instructions'] = None  # default instructions
            
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        queries = []
        query_ids = []
        excluded_ids = {}
        for qid, e in enumerate(examples):
            if args.separate_reasoning:
                new_query = f"{e['query']}\n<REASON>\n{reasoning_examples[qid]['query']}"
                queries.append(new_query)
            else:
                queries.append(e["query"])
            query_ids.append(e['id'])
            excluded_ids[e['id']] = e['excluded_ids']
            overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
            assert len(overlap)==0
        assert len(queries)==len(query_ids), f"{len(queries)}, {len(query_ids)}"
        if not os.path.isdir(os.path.join(args.cache_dir, 'doc_ids')):
            os.makedirs(os.path.join(args.cache_dir, 'doc_ids'))
        if os.path.isfile(os.path.join(args.cache_dir, 'doc_ids',f"{args.task}_{args.long_context}.json")):
            try:
                with open(os.path.join(args.cache_dir, 'doc_ids',f"{args.task}_{args.long_context}.json")) as f:
                    cached_doc_ids = json.load(f)
                # Note: This check is disabled for corpus scaling experiments
                if args.corpus_size == 0:
                    for id1,id2 in zip(cached_doc_ids,doc_ids):
                        assert id2 in cached_doc_ids
            except:
                print("Document IDs matched failed with the cached version!")
        else:
            with open(os.path.join(args.cache_dir, 'doc_ids',f"{args.task}_{args.long_context}.json"),'w') as f:
                json.dump(doc_ids,f,indent=2)
        assert len(doc_ids)==len(documents), f"{len(doc_ids)}, {len(documents)}"

        print(f"{len(queries)} queries")
        print(f"{len(documents)} documents")
        if args.debug:
            documents = documents[:30]
            doc_paths = doc_ids[:30]
        kwargs = {}
        if args.query_max_length>0:
            kwargs = {'query_max_length': args.query_max_length}
        if args.doc_max_length>0:
            kwargs.update({'doc_max_length': args.doc_max_length})
        if args.encode_batch_size>0:
            kwargs.update({'batch_size': args.encode_batch_size})
        if args.key is not None:
            kwargs.update({'key': args.key})
        if args.ignore_cache:
            kwargs.update({'ignore_cache': args.ignore_cache})
        if args.skip_doc_emb:
            kwargs.update({'skip_doc_emb': args.skip_doc_emb})
        if args.store_all_scores:
            kwargs.update({'store_all_scores': args.store_all_scores})
        if args.quantization:
            kwargs.update({'quantization': args.quantization})
        kwargs.update({'document_postfix': document_postfix})
        kwargs.update({'model_name': args.model})

        # Hybrid retrieval parameters
        if args.model == 'hybrid':
            kwargs.update({
                'fusion_method': args.fusion_method,
                'dense_model': args.dense_model,
                'linear_weights': [float(w) for w in args.linear_weights.split(',')]
            })

        model_id = args.model_id if args.model_id is not None else args.model

        # Call retrieval function (use instructions_long for long_context if available)
        selected_instructions = config.get('instructions_long', config['instructions']) if args.long_context else config['instructions']
        retrieval_result = RETRIEVAL_FUNCS[args.model](queries=queries,query_ids=query_ids,documents=documents,excluded_ids=excluded_ids,
                                             instructions=selected_instructions,
                                             doc_ids=doc_ids,task=args.task,cache_dir=args.cache_dir,long_context=args.long_context,
                                             model_id=model_id,checkpoint=args.checkpoint,**kwargs)

        # Handle both old and new return formats for backward compatibility
        if isinstance(retrieval_result, dict) and 'scores' in retrieval_result and 'timing' in retrieval_result:
            # New format with timing info
            scores = retrieval_result['scores']
            timing = retrieval_result['timing']

            # Extract timing metrics
            index_time = timing.get('index_time_seconds', 0)
            query_time_total = timing.get('query_time_total_seconds', 0)
            query_times = timing.get('query_times_seconds', [])
            num_queries = timing.get('num_queries', len(queries))
            used_cache = timing.get('used_cache', None)

            # Build efficiency metrics
            efficiency_metrics = {
                'indexing': {
                    'total_time_seconds': index_time,
                    'documents_per_second': len(documents) / index_time if index_time > 0 else 0,
                    'used_cache': used_cache
                }
            }

            # Calculate query latency statistics
            if len(query_times) > 0:
                # We have per-query times
                query_times_ms = [t * 1000 for t in query_times]
                efficiency_metrics['query_latency'] = {
                    'total': {
                        'total_time_seconds': query_time_total,
                        'num_queries': num_queries,
                        'qps': num_queries / query_time_total if query_time_total > 0 else 0,
                        'mean_ms': np.mean(query_times_ms) if query_times_ms else 0,
                        'p50_ms': np.percentile(query_times_ms, 50) if query_times_ms else 0,
                        'p95_ms': np.percentile(query_times_ms, 95) if query_times_ms else 0,
                        'p99_ms': np.percentile(query_times_ms, 99) if query_times_ms else 0,
                        'min_ms': np.min(query_times_ms) if query_times_ms else 0,
                        'max_ms': np.max(query_times_ms) if query_times_ms else 0
                    }
                }
            else:
                # Only have total query time
                avg_latency_ms = (query_time_total / num_queries) * 1000 if num_queries > 0 else 0
                efficiency_metrics['query_latency'] = {
                    'total': {
                        'total_time_seconds': query_time_total,
                        'num_queries': num_queries,
                        'qps': num_queries / query_time_total if query_time_total > 0 else 0,
                        'mean_ms': avg_latency_ms
                    }
                }
        else:
            # Old format - just scores, no timing (backward compatibility)
            scores = retrieval_result
            efficiency_metrics = {
                'error': 'Timing information not available - retriever function needs to be updated'
            }

        with open(efficiency_file_path, 'w') as f:
            json.dump(efficiency_metrics, f, indent=2)

        with open(score_file_path,'w') as f:
            json.dump(scores,f,indent=2)
    else:
        with open(score_file_path) as f:
            scores = json.load(f)
        print(score_file_path,'exists')

    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in tqdm(examples):
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for did in e['excluded_ids']:
            assert not did in scores[e['id']]
            assert not did in ground_truth[e['id']]

    print(args.output_dir)
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    

    # track successful completion of the run
    if args.sweep_output_dir:
        with open(os.path.join(args.sweep_output_dir, 'done'), 'w') as f:
            f.write('done')