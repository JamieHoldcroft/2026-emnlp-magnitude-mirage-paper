import os.path
import time
import torch
import json
import numpy as np
import tiktoken
from tqdm import tqdm, trange
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from collections import defaultdict
from vllm import LLM


def cut_text(text,tokenizer,threshold):
    text_ids = tokenizer(text)['input_ids']
    if len(text_ids) > threshold:
        text = tokenizer.decode(text_ids[:threshold])
    return text

def cut_text_openai(text,tokenizer,threshold=6000):
    token_ids = tokenizer.encode(text)
    if len(token_ids) > threshold:
        text = tokenizer.decode(token_ids[:threshold])
    return text

def get_embedding_google(texts,task,model,dimensionality=768):
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    success = False
    while not success:
        try:
            new_texts = []
            for t in texts:
                if t.strip()=='':
                    print('empty content')
                    new_texts.append('empty')
                else:
                    new_texts.append(t)
            texts = new_texts
            inputs = [TextEmbeddingInput(text, task) for text in texts]
            kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
            embeddings = model.get_embeddings(inputs, **kwargs)
            success = True
        except Exception as e:
            print(e)
    return [embedding.values for embedding in embeddings]

def get_embedding_openai(texts, openai_client,tokenizer,model="text-embedding-3-large"):
    texts =[json.dumps(text.replace("\n", " ")) for text in texts]
    success = False
    threshold = 6000
    count = 0
    cur_emb = None
    exec_count = 0
    while not success:
        exec_count += 1
        if exec_count>5:
            print('execute too many times')
            exit(0)
        try:
            emb_obj = openai_client.embeddings.create(input=texts, model=model).data
            cur_emb = [e.embedding for e in emb_obj]
            success = True
        except Exception as e:
            print(e)
            count += 1
            threshold -= 500
            if count>4:
                print('openai cut',count)
                exit(0)
            new_texts = []
            for t in texts:
                new_texts.append(cut_text_openai(text=t, tokenizer=tokenizer,threshold=threshold))
            texts = new_texts
    if cur_emb is None:
        raise ValueError("Fail to embed, openai")
    return cur_emb

TASK_MAP = {
    'biology': 'Biology',
    'earth_science': 'Earth Science',
    'economics': 'Economics',
    'psychology': 'Psychology',
    'robotics': 'Robotics',
    'stackoverflow': 'Stack Overflow',
    'sustainable_living': 'Sustainable Living',
}

def add_instruct_concatenate(texts,task,instruction):
    return [instruction.format(task=task)+t for t in texts]

def add_instruct_list(texts,task,instruction):
    return [[instruction.format(task=task),t] for t in texts]

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def safe_load_cache(cur_cache_file, expected_len):
    """
    Safely load a numpy cache file and verify its length.
    Returns the loaded data if valid, otherwise None.
    """
    if os.path.isfile(cur_cache_file):
        try:
            # np.load with allow_pickle=True is needed for some objects.
            data = np.load(cur_cache_file, allow_pickle=True)
            if len(data) != expected_len:
                print(f"Cache size mismatch for {cur_cache_file}: expected {expected_len}, got {len(data)}. Ignoring cache.")
                return None
            return data
        except Exception as e:
            print(f"Error loading cache {cur_cache_file}: {e}. Ignoring cache.")
            return None
    return None

def get_scores(query_ids,doc_ids,scores,excluded_ids, return_full_scores=False, num_hits=1000):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0])==len(doc_ids),f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id,doc_scores in zip(query_ids,scores):
        cur_scores = {}
        assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did,s in zip(doc_ids,doc_scores):
            cur_scores[str(did)] = float(s)
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A" and did in cur_scores:
                cur_scores.pop(did)
        if return_full_scores:
            cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)
        else:
            cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:num_hits]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores

def get_bilevel_scores(query_ids,doc_ids,scores,excluded_ids):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    emb_scores = {}
    for query_id,doc_scores,query_doc_ids in zip(query_ids,scores,doc_ids):
        assert len(doc_scores)==len(query_doc_ids),f"{len(doc_scores)}, {len(query_doc_ids)}"
        cur_scores = {}
        assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did,s in zip(query_doc_ids,doc_scores):
            cur_scores[str(did)] = float(s)
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                cur_scores.pop(did)
        cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:1000]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores

def retrieval_sf_qwen_e5(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    quantization = kwargs.get('quantization', None)
    model_kwargs = {"device_map": "auto"}
    if quantization == 'fp16':
        model_kwargs['torch_dtype'] = torch.float16
        model_kwargs['dtype'] = torch.float16
    elif quantization == 'bf16':
        model_kwargs['torch_dtype'] = torch.bfloat16
        model_kwargs['dtype'] = torch.bfloat16
    elif quantization == 'int8':
        model_kwargs['load_in_8bit'] = True
    elif quantization == 'int4':
        model_kwargs['load_in_4bit'] = True
    model_kwargs['trust_remote_code'] = True
    if model_id=='sf':
        tokenizer = AutoTokenizer.from_pretrained('/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Salesforce--SFR-Embedding-Mistral/snapshots/33a60fe7b5e00b9414981280c571e8a4f9f42f1b', trust_remote_code=True)
        model = AutoModel.from_pretrained('/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Salesforce--SFR-Embedding-Mistral/snapshots/33a60fe7b5e00b9414981280c571e8a4f9f42f1b', **model_kwargs).eval()
        max_length = kwargs.get('doc_max_length',4096)
    elif model_id=='qwen':
        tokenizer = AutoTokenizer.from_pretrained('/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Alibaba-NLP--gte-Qwen1.5-7B-instruct/snapshots/e2f771ba1aba0666210c7fa446fdcf467fb49108', trust_remote_code=True)
        model = AutoModel.from_pretrained('/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Alibaba-NLP--gte-Qwen1.5-7B-instruct/snapshots/e2f771ba1aba0666210c7fa446fdcf467fb49108', **model_kwargs).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='qwen2':
        tokenizer = AutoTokenizer.from_pretrained('/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Alibaba-NLP--gte-Qwen2-7B-instruct/snapshots/a8d08b36ada9cacfe34c4d6f80957772a025daf2', trust_remote_code=True)
        model = AutoModel.from_pretrained('/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Alibaba-NLP--gte-Qwen2-7B-instruct/snapshots/a8d08b36ada9cacfe34c4d6f80957772a025daf2', **model_kwargs).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='e5':
        tokenizer = AutoTokenizer.from_pretrained('/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--intfloat--e5-mistral-7b-instruct/snapshots/07163b72af1488142a360786df853f237b1a3ca1', trust_remote_code=True)
        model = AutoModel.from_pretrained('/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--intfloat--e5-mistral-7b-instruct/snapshots/07163b72af1488142a360786df853f237b1a3ca1', **model_kwargs).eval()
        max_length = kwargs.get('doc_max_length',4096)
    elif model_id=='rader':
        model_name = '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Raderspace--RaDeR_Qwen_25_7B_instruct_MATH_LLMq_CoT_lexical/snapshots/f3f15af7632da5d5e52f6089debf12c54160b147'  # rader检索
        tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True,)
        # rader uses float16 by default, but we should respect model_kwargs if they override it
        rader_kwargs = model_kwargs.copy()
        if 'torch_dtype' not in rader_kwargs and 'dtype' not in rader_kwargs:
            rader_kwargs['torch_dtype'] = torch.float16
        rader_kwargs['attn_implementation'] = "eager"
        model = AutoModel.from_pretrained(model_name,local_files_only=True, **rader_kwargs).eval()
        max_length = kwargs.get('doc_max_length',4096)
    # Handle missing instructions
    if instructions is None:
        instructions = {
            'query': "Instruct: Given a {task} query, retrieve relevant passages that help answer the query\nQuery: ",
            'document': "Represent this text: "
        }
    
    display_task = TASK_MAP.get(task, task.replace('_', ' ').title())
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions.get('query', "Instruct: Given a {task} query, retrieve relevant passages that help answer the query\nQuery: ").format(task=display_task))
    batch_size = kwargs.get('batch_size', kwargs.get('encode_batch_size', 1))

    # Include quantization in cache path to prevent mixing different precision embeddings
    quant_suffix = f"_quant_{quantization}" if quantization else ""
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}{quant_suffix}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False
    
    doc_emb = safe_load_cache(cur_cache_file, len(documents))
    if doc_emb is not None:
        used_cache = True
    else:
        # Generate embeddings for all documents
        doc_emb = []
        with torch.inference_mode():
            for start_idx in trange(0, len(documents), batch_size, desc="Encoding documents"):
                batch_dict = tokenizer(documents[start_idx:start_idx+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
                batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
                outputs = model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().tolist()
                doc_emb.extend(embeddings)
        
        # Convert to NumPy (float32) and save
        doc_emb = np.array(doc_emb, dtype=np.float32)
        np.save(cur_cache_file, doc_emb)
    
    index_time = time.time() - index_start

    # Measure query time with per-query tracking
    query_start = time.time()
    query_emb = []
    query_times = []
    
    with torch.inference_mode():
        for start_idx in trange(0, len(queries), batch_size, desc="Encoding queries"):
            batch_start = time.time()
            batch_dict = tokenizer(queries[start_idx:start_idx + batch_size], max_length=max_length, padding=True,
                                truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().tolist()
            query_emb.extend(embeddings)
            
            # Track per-query time (distribute batch time across queries in batch)
            batch_time = time.time() - batch_start
            num_queries_in_batch = len(embeddings)
            per_query_time = batch_time / num_queries_in_batch if num_queries_in_batch > 0 else 0
            query_times.extend([per_query_time] * num_queries_in_batch)
    
    query_emb = torch.tensor(query_emb, dtype=torch.float32)
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)

    # Convert doc_emb to tensor and normalize (ensure float32 for consistency)
    doc_emb = torch.tensor(doc_emb, dtype=torch.float32)
    print("doc_emb shape:", doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)

    # Compute scores (process in chunks to avoid memory issues)
    # Track similarity computation time per query
    all_scores = []
    chunk_size = 5000
    similarity_start = time.time()
    for i in trange(0, len(doc_emb), chunk_size, desc="Computing scores"):
        doc_chunk = doc_emb[i:min(i+chunk_size, len(doc_emb))]
        scores = (query_emb @ doc_chunk.T) * 100
        all_scores.append(scores)
    
    all_scores = torch.cat(all_scores, dim=1).tolist()
    similarity_time = time.time() - similarity_start
    per_query_similarity = similarity_time / len(queries) if len(queries) > 0 else 0
    
    # Add similarity time to each query's total time
    query_times = [qt + per_query_similarity for qt in query_times]
    
    query_time_total = time.time() - query_start

    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'query_times_seconds': query_times,
        'num_queries': len(queries),
        'used_cache': used_cache
    }

    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=all_scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}

def retrieval_bm25(queries,query_ids,documents,doc_ids,task,cache_dir,excluded_ids,long_context,**kwargs):
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    store_all_score = kwargs.get('store_all_scores', False)

    # BM25 Cache logic
    cache_base = os.path.join(cache_dir, 'bm25_cache', task)
    os.makedirs(cache_base, exist_ok=True)
    corpus_cache_file = os.path.join(cache_base, 'corpus.json')
    
    # Measure indexing time
    index_start = time.time()
    used_cache = False
    
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    
    if os.path.exists(corpus_cache_file):
        with open(corpus_cache_file, 'r') as f:
            corpus = json.load(f)
        used_cache = True
    else:
        corpus = [analyzer.analyze(x) for x in documents]
        with open(corpus_cache_file, 'w') as f:
            json.dump(corpus, f)
            
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)
    index_time = time.time() - index_start

    # Measure query time
    query_start = time.time()
    query_times = []
    all_scores = {}
    repeat_scores = {}
    bar = tqdm(queries, desc="BM25 retrieval")
    for query_id, query in zip(query_ids, queries):
        query_individual_start = time.time()
        bar.update(1)
        query = analyzer.analyze(query)
        bm25_query = model[dictionary.doc2bow(query)]
        similarities = bm25_index[bm25_query].tolist()
        all_scores[str(query_id)] = {}
        repeat_scores[str(query_id)] = defaultdict(list)
        for did, s in zip(doc_ids, similarities):
            # all_scores[str(query_id)][did] = s
            # repeat_scores[str(query_id)][did].append(s)
            if did not in all_scores[str(query_id)] or s > all_scores[str(query_id)][did]:
                    all_scores[str(query_id)][did] = s  # refine docs, save the best score for each query-doc pair

        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                all_scores[str(query_id)].pop(did)
        if store_all_score:
            cur_scores = sorted(all_scores[str(query_id)].items(),key=lambda x:x[1],reverse=True)
        else:
            cur_scores = sorted(all_scores[str(query_id)].items(),key=lambda x:x[1],reverse=True)[:1000]
        all_scores[str(query_id)] = {}
        for pair in cur_scores:
            all_scores[str(query_id)][pair[0]] = pair[1]
        query_times.append(time.time() - query_individual_start)

    query_time_total = time.time() - query_start
    print("Dedup Scores shape", len(repeat_scores))

    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'query_times_seconds': query_times,
        'num_queries': len(queries),
        'used_cache': used_cache
    }

    return {'scores': all_scores, 'timing': timing}

def retrieval_sbert_bge(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    quantization = kwargs.get('quantization', None)
    model_kwargs = {}
    if quantization == 'fp16':
        model_kwargs['torch_dtype'] = torch.float16
        model_kwargs['dtype'] = torch.float16
    elif quantization == 'bf16':
        model_kwargs['torch_dtype'] = torch.bfloat16
        model_kwargs['dtype'] = torch.bfloat16
    elif quantization == 'int8':
        model_kwargs['load_in_8bit'] = True
        model_kwargs['device_map'] = 'auto'
    elif quantization == 'int4':
        model_kwargs['load_in_4bit'] = True
        model_kwargs['device_map'] = 'auto'

    if model_id=='bge':
        model = SentenceTransformer('BAAI/bge-large-en-v1.5', model_kwargs=model_kwargs)
        queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    elif model_id=='sbert':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', model_kwargs=model_kwargs)
    elif model_id=='qwen2':
        model_kwargs['trust_remote_code'] = True
        model = SentenceTransformer("/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Alibaba-NLP--gte-Qwen2-7B-instruct/snapshots/a8d08b36ada9cacfe34c4d6f80957772a025daf2", model_kwargs=model_kwargs)
    elif model_id=='contriever_st':
        model = SentenceTransformer('nishimoto/contriever-sentencetransformer', model_kwargs=model_kwargs)
    else:
        raise ValueError(f"The model {model_id} is not supported")
    batch_size = kwargs.get('batch_size',1)

    # Include quantization in cache path to prevent mixing different precision embeddings
    quant_suffix = f"_quant_{quantization}" if quantization else ""
    os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}{quant_suffix}"), exist_ok=True)
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}{quant_suffix}", f'0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False
    doc_emb = safe_load_cache(cur_cache_file, len(documents))
    if doc_emb is not None:
        used_cache = True
    else:
        doc_emb = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    index_time = time.time() - index_start

    # Measure query time with per-query tracking
    query_start = time.time()
    query_emb = model.encode(queries, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)

    # For SentenceTransformer, we can't track individual query times during encoding
    # So we'll measure similarity computation and estimate encoding time per query
    encoding_time = time.time() - query_start
    
    # Compute similarity scores with per-query timing
    similarity_start = time.time()
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    similarity_time = time.time() - similarity_start
    
    query_time_total = time.time() - query_start
    
    # Estimate per-query times (distribute total time evenly)
    per_query_time = query_time_total / len(queries) if len(queries) > 0 else 0
    query_times = [per_query_time] * len(queries)

    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'query_times_seconds': query_times,
        'num_queries': len(queries),
        'used_cache': used_cache
    }

    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}

def retrieval_sbert_bge_ce(queries, query_ids, documents, doc_ids, task, instructions, model_id, cache_dir, excluded_ids, long_context, **kwargs):
    quantization = kwargs.get('quantization', None)
    model_kwargs = {"device_map": "auto"}
    if quantization == 'fp16':
        model_kwargs['torch_dtype'] = torch.float16
        model_kwargs['dtype'] = torch.float16
    elif quantization == 'bf16':
        model_kwargs['torch_dtype'] = torch.bfloat16
        model_kwargs['dtype'] = torch.bfloat16
    elif quantization == 'int8':
        model_kwargs['load_in_8bit'] = True
    elif quantization == 'int4':
        model_kwargs['load_in_4bit'] = True
        
    if model_id == 'bge_ce':
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
        model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large', **model_kwargs).eval()
        max_length = kwargs.get('doc_max_length', 512)
        queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    else:
        raise ValueError(f"The model {model_id} is not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_start = time.time()
    scores = []
    for query in tqdm(queries):
        pairs_per_query = []
        for document in documents:
            pairs_per_query.append([query, document])

        batch_size = kwargs.get('batch_size',1)
        total_batches = len(pairs_per_query) // batch_size + (0 if len(pairs_per_query) % batch_size == 0 else 1)

        scores_per_query = []
        with torch.no_grad():
            for i in tqdm(range(total_batches)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(pairs_per_query))
                batch = pairs_per_query[start_idx:end_idx]

                inputs = tokenizer(batch, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                batch_scores = model(**inputs, return_dict=True).logits.view(-1).float()

                scores_per_query.extend(batch_scores.tolist())

        scores.append(scores_per_query)
    
    query_time_total = time.time() - query_start

    timing = {
        'index_time_seconds': 0, # Cross-encoders don't usually pre-index
        'query_time_total_seconds': query_time_total,
        'num_queries': len(queries),
        'used_cache': False
    }

    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}
    
@torch.no_grad()
def retrieval_instructor(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    quantization = kwargs.get('quantization', None)
    model_kwargs = {}
    if quantization == 'fp16':
        model_kwargs['torch_dtype'] = torch.float16
    elif quantization == 'bf16':
        model_kwargs['torch_dtype'] = torch.bfloat16
    elif quantization == 'int8':
        model_kwargs['load_in_8bit'] = True
        model_kwargs['device_map'] = 'auto'
    elif quantization == 'int4':
        model_kwargs['load_in_4bit'] = True
        model_kwargs['device_map'] = 'auto'

    if model_id=='inst-l':
        model = SentenceTransformer("/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--hkunlp--instructor-large/snapshots/54e5ffb8d484de506e59443b07dc819fb15c7233", model_kwargs=model_kwargs)
    elif model_id=='inst-xl':
        model = SentenceTransformer("/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--hkunlp--instructor-xl/snapshots/ce48b213095e647a6c3536364b9fa00daf57f436", model_kwargs=model_kwargs)
    else:
        raise ValueError(f"The model {model_id} is not supported")
    model.set_pooling_include_prompt(False)

    batch_size = kwargs.get('batch_size',4)
    model.max_seq_length = kwargs.get('doc_max_length',2048)

    # Handle missing instructions
    if instructions is None:
        instructions = {
            'query': "Represent the {task} query for retrieval: ",
            'document': "Represent the {task} document for retrieval: "
        }

    # Measure query encoding time
    query_start = time.time()
    query_embs = model.encode(queries, batch_size=batch_size, show_progress_bar=True, prompt=instructions['query'].format(task=task), normalize_embeddings=True)
    query_encoding_time = time.time() - query_start

    # Create cache directory
    quant_suffix = f"_quant_{quantization}" if quantization else ""
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}{quant_suffix}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False
    if os.path.isfile(cur_cache_file):
        doc_embs = np.load(cur_cache_file, allow_pickle=True)
        used_cache = True
    else:
        doc_embs = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True, prompt=instructions['document'].format(task=task))
        np.save(cur_cache_file, doc_embs)
    index_time = time.time() - index_start

    # Compute similarity scores with timing
    similarity_start = time.time()
    scores = cosine_similarity(query_embs, doc_embs)
    scores = scores.tolist()
    similarity_time = time.time() - similarity_start

    query_time_total = query_encoding_time + similarity_time

    # Estimate per-query times
    per_query_time = query_time_total / len(queries) if len(queries) > 0 else 0
    query_times = [per_query_time] * len(queries)

    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'query_times_seconds': query_times,
        'num_queries': len(queries),
        'used_cache': used_cache
    }

    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}

def retrieval_grit(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    from gritlm import GritLM
    quantization = kwargs.get('quantization', None)
    model_kwargs = {"mode": "embedding"}
    if quantization == 'fp16':
        model_kwargs['torch_dtype'] = torch.float16
        model_kwargs['dtype'] = torch.float16
    elif quantization == 'bf16':
        model_kwargs['torch_dtype'] = torch.bfloat16
        model_kwargs['dtype'] = torch.bfloat16
    elif quantization == 'int8':
        model_kwargs['load_in_8bit'] = True
        model_kwargs['device_map'] = 'auto'
    elif quantization == 'int4':
        model_kwargs['load_in_4bit'] = True
        model_kwargs['device_map'] = 'auto'
    else:
        model_kwargs['torch_dtype'] = 'auto'
        model_kwargs['dtype'] = 'auto'
        
    customized_checkpoint = kwargs.get('checkpoint',None)
    if customized_checkpoint is None:
        customized_checkpoint = 'GritLM/GritLM-7B'
    else:
        print('use',customized_checkpoint)
    model = GritLM(customized_checkpoint, **model_kwargs)
    
    # Handle missing instructions
    if instructions is None:
        instructions = {
            'query': "Instruct: Given a {task} query, retrieve relevant passages that help answer the query\nQuery: ",
            'document': "Represent this text: "
        }
    
    display_task = TASK_MAP.get(task, task.replace('_', ' ').title())
    
    query_instruction = instructions.get('query', "Instruct: Given a {task} query, retrieve relevant passages that help answer the query\nQuery: ").format(task=display_task)
    doc_instruction = instructions.get('document', "Represent this text: ")
    query_max_length = kwargs.get('query_max_length',32768)
    doc_max_length = kwargs.get('doc_max_length',32768)
    print("doc max length:",doc_max_length)
    print("query max length:", query_max_length)
    batch_size = kwargs.get('batch_size',1)
    # model_id = 'grit'

    # Include quantization in cache path to prevent mixing different precision embeddings
    quant_suffix = f"_quant_{quantization}" if quantization else ""
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}{quant_suffix}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False
    
    ignore_cache = kwargs.pop('ignore_cache',False)
    skip_doc_emb = kwargs.pop('skip_doc_emb',False)
    if not skip_doc_emb:
        doc_emb = safe_load_cache(cur_cache_file, len(documents))
        if doc_emb is not None:
            used_cache = True
        else:
            doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
            np.save(cur_cache_file, doc_emb)
    
    index_time = time.time() - index_start

    # Measure query time with per-query tracking
    query_start = time.time()
    query_emb = model.encode(queries, instruction=query_instruction, batch_size=1, max_length=query_max_length)
    encoding_time = time.time() - query_start
    
    if skip_doc_emb:
        exit()
    
    # Compute similarity scores with timing    
    similarity_start = time.time()
    scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(doc_emb))
    scores = scores.tolist()
    similarity_time = time.time() - similarity_start
    
    query_time_total = time.time() - query_start
    
    # Estimate per-query times
    per_query_time = query_time_total / len(queries) if len(queries) > 0 else 0
    query_times = [per_query_time] * len(queries)

    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'query_times_seconds': query_times,
        'num_queries': len(queries),
        'used_cache': used_cache
    }

    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}

def retrieval_openai(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    from openai import OpenAI
    tokenizer = tiktoken.get_encoding("cl100k_base")
    new_queries = []
    for q in queries:
        new_queries.append(cut_text_openai(text=q,tokenizer=tokenizer))
    queries = new_queries
    new_documents = []
    for d in documents:
        new_documents.append(cut_text_openai(text=d,tokenizer=tokenizer))
    documents = new_documents
    batch_size = kwargs.get('batch_size',1)
    openai_client = OpenAI(api_key=kwargs['key'])

    # Create cache directory WITHOUT batch_size in path (NumPy format)
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False
    
    doc_emb = safe_load_cache(cur_cache_file, len(documents))
    if doc_emb is not None:
        doc_emb = doc_emb.tolist()
        used_cache = True
    else:
        # Generate embeddings via API (process in batches for rate limiting)
        doc_emb = []
        for idx in trange(0, len(documents), batch_size, desc="Encoding documents via API"):
            cur_emb = get_embedding_openai(texts=documents[idx:idx + batch_size], openai_client=openai_client, tokenizer=tokenizer)
            doc_emb.extend(cur_emb)
        
        # Convert to NumPy and save
        np.save(cur_cache_file, np.array(doc_emb))
    
    index_time = time.time() - index_start
    
    # Measure query time
    query_start = time.time()
    query_emb = []
    for idx in trange(0, len(queries), batch_size, desc="Encoding queries via API"):
        cur_emb = get_embedding_openai(texts=queries[idx:idx + batch_size], openai_client=openai_client, tokenizer=tokenizer)
        query_emb.extend(cur_emb)
    
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    query_time_total = time.time() - query_start
    
    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'num_queries': len(queries),
        'used_cache': used_cache
    }
    
    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}

def retrieval_cohere(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    import cohere
    batch_size = kwargs.get('batch_size',1)
    cohere_client = cohere.Client(kwargs['key'])

    # Create cache directory WITHOUT batch_size in path (NumPy format)
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False
    
    doc_emb = safe_load_cache(cur_cache_file, len(documents))
    if doc_emb is not None:
        doc_emb = doc_emb.tolist()
        used_cache = True
    else:
        # Generate embeddings via API (process in batches for rate limiting)
        doc_emb = []
        for idx in trange(0, len(documents), batch_size, desc="Encoding documents via API"):
            success = False
            exec_count = 0
            cur_emb = []
            while not success:
                exec_count += 1
                if exec_count>5:
                    print('cohere execute too many times')
                    exit(0)
                try:
                    cur_emb = cohere_client.embed(documents[idx:idx+batch_size], input_type="search_document",
                                                  model="embed-english-v3.0").embeddings
                    success = True
                except Exception as e:
                    print(e)
                    time.sleep(60)
            doc_emb.extend(cur_emb)
        
        # Convert to NumPy and save
        np.save(cur_cache_file, np.array(doc_emb))
    
    index_time = time.time() - index_start
    
    # Measure query time
    query_start = time.time()
    query_emb = []
    for idx in trange(0, len(queries), batch_size, desc="Encoding queries via API"):
        success = False
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('cohere query execute too many times')
                exit(0)
            try:
                cur_emb = cohere_client.embed(queries[idx:idx+batch_size], input_type="search_query",
                                              model="embed-english-v3.0").embeddings
                query_emb.extend(cur_emb)
                success = True
            except Exception as e:
                print(e)
                time.sleep(60)
    
    scores = (torch.tensor(query_emb) @ torch.tensor(doc_emb).T) * 100
    scores = scores.tolist()
    query_time_total = time.time() - query_start
    
    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'num_queries': len(queries),
        'used_cache': used_cache
    }
    
    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}

def retrieval_voyage(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    import voyageai
    tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage')
    new_queries = []
    for q in queries:
        new_queries.append(cut_text(text=q,tokenizer=tokenizer,threshold=16000))
    queries = new_queries
    new_documents = []
    for d in tqdm(documents,desc='preprocess documents'):
        new_documents.append(cut_text(text=d,tokenizer=tokenizer,threshold=16000))
    documents = new_documents

    query_emb = []
    doc_emb = []
    batch_size = kwargs.get('batch_size',1)
    voyage_client = voyageai.Client(api_key=kwargs['key'])

    # Create cache directory WITHOUT batch_size in path (NumPy format)
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False
    
    doc_emb = safe_load_cache(cur_cache_file, len(documents))
    if doc_emb is not None:
        doc_emb = doc_emb.tolist()
        used_cache = True
    else:
        # Generate embeddings via API (process in batches for rate limiting)
        doc_emb = []
        for i in trange(0, len(documents), batch_size, desc="Encoding documents via API"):
            success = False
            threshold = 16000
            cur_texts = documents[i:i+batch_size]
            count_over = 0
            exec_count = 0
            while not success:
                exec_count += 1
                if exec_count > 5:
                    print('voyage document too many times')
                    exit(0)
                try:
                    cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="document").embeddings
                    success = True
                except Exception as e:
                    print(e)
                    count_over += 1
                    threshold = threshold-500
                    if count_over>4:
                        print('voyage:',count_over)
                    new_texts = []
                    for t in cur_texts:
                        new_texts.append(cut_text(text=t,tokenizer=tokenizer,threshold=threshold))
                    cur_texts = new_texts
                    time.sleep(5)
            doc_emb.extend(cur_emb)
        
        # Convert to NumPy and save
        np.save(cur_cache_file, np.array(doc_emb))
    
    index_time = time.time() - index_start
    
    # Measure query time
    query_start = time.time()
    query_emb = []
    for i in trange(0, len(queries), batch_size, desc="Encoding queries via API"):
        success = False
        threshold = 16000
        cur_texts = queries[i:i+batch_size]
        count_over = 0
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('voyage query execute too many times')
                exit(0)
            try:
                cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="query").embeddings
                query_emb.extend(cur_emb)
                success = True
            except Exception as e:
                print(e)
                count_over += 1
                threshold = threshold-500
                if count_over>4:
                    print('voyage:',count_over)
                new_texts = []
                for t in cur_texts:
                    new_texts.append(cut_text(text=t,tokenizer=tokenizer,threshold=threshold))
                cur_texts = new_texts
                time.sleep(60)
    
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    query_time_total = time.time() - query_start
    
    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'num_queries': len(queries),
        'used_cache': used_cache
    }
    
    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}

def retrieval_google(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    model = TextEmbeddingModel.from_pretrained("text-embedding-preview-0409")
    batch_size = kwargs.get('batch_size',1)

    # Create cache directory WITHOUT batch_size in path (NumPy format)
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False
    
    doc_emb = safe_load_cache(cur_cache_file, len(documents))
    if doc_emb is not None:
        doc_emb = doc_emb.tolist()
        used_cache = True
    else:
        # Generate embeddings via API (process in batches for rate limiting)
        doc_emb = []
        for start_idx in tqdm(range(0, len(documents), batch_size), desc='Encoding documents via API'):
            cur_emb = get_embedding_google(texts=documents[start_idx:start_idx + batch_size], task='RETRIEVAL_DOCUMENT', model=model)
            doc_emb.extend(cur_emb)
        
        # Convert to NumPy and save
        np.save(cur_cache_file, np.array(doc_emb))
    
    index_time = time.time() - index_start
    
    # Measure query time
    query_start = time.time()
    query_emb = []
    for start_idx in tqdm(range(0, len(queries), batch_size), desc='Encoding queries via API'):
        query_emb.extend(get_embedding_google(texts=queries[start_idx:start_idx+ batch_size], task='RETRIEVAL_QUERY', model=model))
    
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    query_time_total = time.time() - query_start
    
    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'num_queries': len(queries),
        'used_cache': used_cache
    }
    
    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}


def retrieval_nomic(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    quantization = kwargs.get('quantization', None)
    customized_checkpoint = kwargs.get('checkpoint', None)
    model_kwargs = {}
    if quantization == 'fp16':
        model_kwargs['torch_dtype'] = torch.float16
    elif quantization == 'bf16':
        model_kwargs['torch_dtype'] = torch.bfloat16
    elif quantization == 'int8':
        model_kwargs['load_in_8bit'] = True
        model_kwargs['device_map'] = 'auto'
    elif quantization == 'int4':
        model_kwargs['load_in_4bit'] = True
        model_kwargs['device_map'] = 'auto'

    model_kwargs['trust_remote_code'] = True
    if customized_checkpoint is not None:
        model = SentenceTransformer(customized_checkpoint, **model_kwargs)
        model_id = customized_checkpoint  # Use custom checkpoint for cache
    else:
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", **model_kwargs)
        # Keep the model_id that was passed to the function

    batch_size = kwargs.get('batch_size', 1)

    # Include quantization in cache path to prevent mixing different precision embeddings
    quant_suffix = f"_quant_{quantization}" if quantization else ""
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}{quant_suffix}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Handle missing instructions
    if instructions is None:
        instructions = {
            'query': "search_query: ",
            'document': "search_document: "
        }
    
    # Measure indexing time
    index_start = time.time()
    used_cache = False
    
    doc_emb = safe_load_cache(cur_cache_file, len(documents))
    if doc_emb is not None:
        used_cache = True
    else:
        # Prefix documents for Nomic
        documents_with_instr = [instructions.get('document', "search_document: ") + d for d in documents]
        doc_emb = model.encode(documents_with_instr, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    index_time = time.time() - index_start

    # Prefix queries
    queries_with_instr = [instructions.get('query', "search_query: ") + q for q in queries]

    # Measure query time with per-query tracking
    query_start = time.time()
    query_emb = model.encode(queries_with_instr, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
    encoding_time = time.time() - query_start
    
    # Compute similarity scores with timing
    similarity_start = time.time()
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    similarity_time = time.time() - similarity_start
    
    query_time_total = time.time() - query_start
    
    # Estimate per-query times
    per_query_time = query_time_total / len(queries) if len(queries) > 0 else 0
    query_times = [per_query_time] * len(queries)

    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'query_times_seconds': query_times,
        'num_queries': len(queries),
        'used_cache': used_cache
    }

    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}


def retrieval_m2(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    quantization = kwargs.get('quantization', None)
    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if quantization == 'fp16':
        model_kwargs['torch_dtype'] = torch.float16
        model_kwargs['dtype'] = torch.float16
    elif quantization == 'bf16':
        model_kwargs['torch_dtype'] = torch.bfloat16
        model_kwargs['dtype'] = torch.bfloat16
    elif quantization == 'int8':
        model_kwargs['load_in_8bit'] = True
    elif quantization == 'int4':
        model_kwargs['load_in_4bit'] = True

    model = AutoModelForSequenceClassification.from_pretrained(
        "togethercomputer/m2-bert-80M-32k-retrieval",
        **model_kwargs
    )
    max_length = 32768

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        model_max_length=max_length
    )

    # Handle missing instructions
    if instructions is None:
        instructions = {
            'query': "Instruct: Given a {task} query, retrieve relevant passages that help answer the query\nQuery: ",
            'document': "Represent this text: "
        }
    
    display_task = TASK_MAP.get(task, task.replace('_', ' ').title())
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions.get('query', "Instruct: Given a {task} query, retrieve relevant passages that help answer the query\nQuery: ").format(task=display_task))
    batch_size = kwargs.get('batch_size', kwargs.get('encode_batch_size', 1))

    # Include quantization in cache path to prevent mixing different precision embeddings
    quant_suffix = f"_quant_{quantization}" if quantization else ""
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}{quant_suffix}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False
    
    doc_emb = safe_load_cache(cur_cache_file, len(documents))
    if doc_emb is not None:
        used_cache = True
    else:
        # Generate embeddings for all documents
        doc_emb = []
        for start_idx in trange(0, len(documents), batch_size, desc="Encoding documents"):
            batch_dict = tokenizer(documents[start_idx:start_idx+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**batch_dict)
            embeddings = outputs['sentence_embedding'].cpu().tolist()
            doc_emb.extend(embeddings)
        
        # Convert to NumPy and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)
    
    index_time = time.time() - index_start
    
    # Convert to tensor and normalize (ensure float32 for consistency)
    doc_emb = torch.tensor(doc_emb, dtype=torch.float32)
    print("doc_emb shape:",doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    
    # Measure query time with per-query tracking
    query_start = time.time()
    query_emb = []
    query_times = []
    
    for start_idx in trange(0, len(queries), batch_size, desc="Encoding queries"):
        batch_start = time.time()
        batch_dict = tokenizer(queries[start_idx:start_idx + batch_size], max_length=max_length, padding=True,
                               truncation=True, return_tensors='pt')
        outputs = model(**batch_dict)
        embeddings = outputs['sentence_embedding'].cpu().tolist()
        query_emb.extend(embeddings)
        
        # Track per-query time
        batch_time = time.time() - batch_start
        per_query_time = batch_time / len(embeddings) if len(embeddings) > 0 else 0
        query_times.extend([per_query_time] * len(embeddings))
        
    query_emb = torch.tensor(query_emb, dtype=torch.float32)
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    query_time_total = time.time() - query_start
    
    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'query_times_seconds': query_times,
        'num_queries': len(queries),
        'used_cache': used_cache
    }
    
    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}


def retrieval_contriever(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    # tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    # model = AutoModel.from_pretrained('facebook/contriever')
    quantization = kwargs.get('quantization', None)
    model_kwargs = {}
    if quantization == 'fp16':
        model_kwargs['torch_dtype'] = torch.float16
        model_kwargs['dtype'] = torch.float16
    elif quantization == 'bf16':
        model_kwargs['torch_dtype'] = torch.bfloat16
        model_kwargs['dtype'] = torch.bfloat16
    elif quantization == 'int8':
        model_kwargs['load_in_8bit'] = True
        model_kwargs['device_map'] = 'auto'
    elif quantization == 'int4':
        model_kwargs['load_in_4bit'] = True
        model_kwargs['device_map'] = 'auto'
    
    model_kwargs['trust_remote_code'] = True
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco', trust_remote_code=True)
    model = AutoModel.from_pretrained('facebook/contriever-msmarco', **model_kwargs)
    if 'device_map' not in model_kwargs:
        model = model.to('cuda')
    model.eval()

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode(model, texts, show_progress_bar=True,batch_size=1, normalize_embeddings=True): # encode a batch of documents into the embeddings
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            # move inputs to cuda
            for k, v in inputs.items():
                inputs[k] = v.to('cuda')
            with torch.inference_mode():
                outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if normalize_embeddings:
            all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
        all_embeddings = all_embeddings.float().cpu().numpy()
        return all_embeddings

    batch_size = kwargs.get('batch_size', 1)

    # Include quantization in cache path
    quant_suffix = f"_quant_{quantization}" if quantization else ""
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}{quant_suffix}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False
    doc_emb = safe_load_cache(cur_cache_file, len(documents))
    if doc_emb is not None:
        used_cache = True
    else:
        doc_emb = encode(model, documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    index_time = time.time() - index_start

    # Measure query time with per-query tracking
    query_start = time.time()
    query_emb = encode(model, queries, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
    encoding_time = time.time() - query_start
    
    # Compute similarity scores with timing
    similarity_start = time.time()
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    similarity_time = time.time() - similarity_start
    
    query_time_total = time.time() - query_start
    
    # Estimate per-query times
    per_query_time = query_time_total / len(queries) if len(queries) > 0 else 0
    query_times = [per_query_time] * len(queries)

    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'query_times_seconds': query_times,
        'num_queries': len(queries),
        'used_cache': used_cache
    }

    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}


def retrieval_reasonir(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    # NOTE: HF version does not come with pooling function, need to add it manually.
    # Must use AutoModel (not SentenceTransformer) to call the model's built-in encode() method.
    customized_checkpoint = "/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--reasonir--ReasonIR-8B/snapshots/c3d0690370ff4a8c3d3882d8dfa85c43650034fa" #kwargs.get('checkpoint',None)
    # if customized_checkpoint is None:
    #     customized_checkpoint = 'reasonir/ReasonIR-8B'
    # else:
    #     print('use',customized_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(customized_checkpoint, torch_dtype="auto", trust_remote_code=True)
    model = AutoModel.from_pretrained(customized_checkpoint, torch_dtype="auto", trust_remote_code=True)
    model.eval()
    model.to(device)

    # Handle missing instructions
    if instructions is None:
        print("WARNING: No config instructions found for ReasonIR. Using default instructions.")
        instructions = {
            'query': "Given a {task} question, retrieve relevant documents that help answer the question",
            'document': ""
        }

    query_instruction = instructions['query'].format(task=task)
    doc_instruction = instructions['document']
    query_max_length = kwargs.get('query_max_length',32768)
    doc_max_length = kwargs.get('doc_max_length',32768)
    print("doc max length:",doc_max_length)
    print("query max length:", query_max_length)
    batch_size = kwargs.get('batch_size',1)

    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False

    ignore_cache = kwargs.pop('ignore_cache',False)
    skip_doc_emb = kwargs.pop('skip_doc_emb',False)
    if not skip_doc_emb:
        if os.path.isfile(cur_cache_file):
            doc_emb = np.load(cur_cache_file, allow_pickle=True)
            used_cache = True
        else:
            doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=batch_size, max_length=doc_max_length)
            np.save(cur_cache_file, doc_emb)

    index_time = time.time() - index_start

    # Measure query time
    query_start = time.time()
    query_emb = model.encode(queries, instruction=query_instruction, batch_size=batch_size, max_length=query_max_length)
    encoding_time = time.time() - query_start

    if skip_doc_emb:
        exit()

    # Compute similarity scores
    similarity_start = time.time()
    scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(doc_emb))
    scores = scores.tolist()
    similarity_time = time.time() - similarity_start

    query_time_total = time.time() - query_start

    # Estimate per-query times
    per_query_time = query_time_total / len(queries) if len(queries) > 0 else 0
    query_times = [per_query_time] * len(queries)

    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'query_times_seconds': query_times,
        'num_queries': len(queries),
        'used_cache': used_cache
    }

    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(documents), f"{len(scores[0])}, {len(documents)}"
    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}


@torch.no_grad()
def retrieval_rader(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    model_name = '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Raderspace--RaDeR_Qwen_25_7B_instruct_MATH_LLMq_CoT_lexical/snapshots/f3f15af7632da5d5e52f6089debf12c54160b147'  # rader检索
    batch_size = kwargs.get('batch_size', kwargs.get('encode_batch_size', 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16).eval()

    # Append instructions before queries (matching original code behavior)
    if instructions is not None and 'query' in instructions:
        queries = add_instruct_concatenate(texts=queries, task=task, instruction=instructions['query'])

    # Check if documents are already encoded
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False

    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file, allow_pickle=True)
        used_cache = True
    else:
        doc_emb = []

        for i in tqdm(range(0, len(documents))):
            text = documents[i]
            inputs = tokenizer(f"document: {text[:8192]}{tokenizer.eos_token}", return_tensors='pt', padding=True, truncation=True)

            inputs = {key: val.to(device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

                if i==0:
                    print("Doc outputs shape", outputs.last_hidden_state.shape)

                embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
                embeddings = embeddings.cpu().numpy()
                doc_emb.extend(embeddings)
            torch.cuda.empty_cache()

        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)

    index_time = time.time() - index_start

    print("Shape of doc emb", doc_emb.shape)

    # Measure query time with per-query tracking
    query_start = time.time()
    query_emb = []
    query_times = []

    for i in tqdm(range(0, len(queries))):
        batch_start = time.time()
        text = queries[i]
        inputs = tokenizer(f"query: {text}{tokenizer.eos_token}", return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
            embeddings = embeddings.cpu().numpy()
            query_emb.extend(embeddings)

        query_times.append(time.time() - batch_start)

    # Convert to numpy array
    query_emb = np.array(query_emb)
    query_time_total = time.time() - query_start
    print("Shape of query emb", query_emb.shape)

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()

    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'query_times_seconds': query_times,
        'num_queries': len(queries),
        'used_cache': used_cache
    }

    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}



'''vllm version'''
from vllm.transformers_utils.tokenizer import get_tokenizer as get_vllm_tokenizer
class Qwen3EmbeddingModel:
    def __init__(self, model_path, max_length=16384, device="auto", quantization=None):
        self.model = LLM(model=model_path, task="embed", gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count(), quantization=quantization)
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'
        self.max_length = max_length 
        self.tokenizer = get_vllm_tokenizer(model_path, trust_remote_code=False)

    def truncate_text(self, text):
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(text_ids) > self.max_length:
            text_ids = text_ids[:self.max_length]
            text = self.tokenizer.decode(text_ids)
        return text

    def embed_query(self, query):
        outputs = self.model.embed(query)
        return outputs[0].outputs.embedding

    def embed_queries(self, queries, batch_size=1):
        input_queries = ['Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{}'.format(x) for x in queries]
        all_embeddings = []
        for i in range(0, len(input_queries), batch_size):
            batch = input_queries[i : i + batch_size]
            outputs = self.model.embed(batch)
            all_embeddings.extend([x.outputs.embedding for x in outputs])
        return all_embeddings

    def embed_doc(self, doc):
        outputs = self.model.embed("Represent this text:{}".format(doc))
        return outputs[0].outputs.embedding

    def embed_docs(self, docs, batch_size=1):
        input_docs = ["Represent this text:{}".format(doc) for doc in docs]
        all_embeddings = []
        for i in range(0, len(input_docs), batch_size):
            batch = input_docs[i : i + batch_size]
            # Batch truncation
            batch = [self.truncate_text(doc) for doc in batch]
            outputs = self.model.embed(batch)
            all_embeddings.extend([x.outputs.embedding for x in outputs])
        return all_embeddings


@torch.no_grad()
def retrieval_qwen3_ft_diver(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    cache_model_name = kwargs.get('model_name', 'diver')
    batch_size = kwargs.get('batch_size', kwargs.get('encode_batch_size', 1))
    quantization = kwargs.get('quantization', None)

    model_path = '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--AQ-MedAI--Diver-Retriever-4B/snapshots/ff7f8bc8d1827734dcf2bbfab99f065be618069a'
    model = Qwen3EmbeddingModel(model_path, max_length=16384, quantization=quantization)

    # Include quantization in cache path
    document_postfix = '_' + kwargs['document_postfix'] if 'document_postfix' in kwargs and len(kwargs['document_postfix']) > 0 else ''
    quant_suffix = f"_quant_{quantization}" if quantization else ""
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb' + document_postfix, cache_model_name, task, f"long_{long_context}{quant_suffix}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    # Measure indexing time
    index_start = time.time()
    used_cache = False
    
    doc_emb = safe_load_cache(cur_cache_file, len(documents))
    if doc_emb is not None:
        used_cache = True
    else:
        doc_emb = []
        with torch.inference_mode():
            doc_emb = model.embed_docs(documents, batch_size=batch_size)
        torch.cuda.empty_cache() 
        
        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)
    index_time = time.time() - index_start

    # Measure query time with per-query tracking
    query_start = time.time()
    query_emb = []
    with torch.inference_mode():
        query_emb = model.embed_queries(queries, batch_size=batch_size)
    query_emb = np.array(query_emb)
    encoding_time = time.time() - query_start
    
    # Find cosine similarity between doc_emb and query_emb with timing
    similarity_start = time.time()
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    similarity_time = time.time() - similarity_start
    
    query_time_total = time.time() - query_start
    
    # Estimate per-query times
    per_query_time = query_time_total / len(queries) if len(queries) > 0 else 0
    query_times = [per_query_time] * len(queries)

    if len(kwargs['document_postfix']) > 0:  # rechunk setting
        dedup_doc_ids = set(doc_ids)
        dedup_scores = []  # shape:[len(scores), len(dedup_doc_ids)], save only the best score for each query-doc pair
        for query_idx in range(len(query_emb)):
            best_scores = {}  # for each query, save the best score for each doc_id
            for idx, score in enumerate(scores[query_idx]):
                doc_id = doc_ids[idx]
                if doc_id not in best_scores or score > best_scores[doc_id]:
                    best_scores[doc_id] = score
            q_doc_scores = []
            for doc_id in dedup_doc_ids:
                q_doc_scores.append(best_scores.get(doc_id))
            dedup_scores.append(q_doc_scores)

        doc_ids, scores = dedup_doc_ids, dedup_scores
        print("Dedup Scores shape:", len(scores[0]))
        
    timing = {
        'index_time_seconds': index_time,
        'query_time_total_seconds': query_time_total,
        'query_times_seconds': query_times,
        'num_queries': len(queries),
        'used_cache': used_cache
    }

    result_scores = get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    return {'scores': result_scores, 'timing': timing}



def reciprocal_rank_fusion(score_dict_list, k=60):
    """
    Reciprocal Rank Fusion (RRF) for combining multiple retrieval results.

    Args:
        score_dict_list: List of dicts, each mapping doc_id -> score
        k: RRF parameter (default 60 as recommended in literature)

    Returns:
        Combined scores dict
    """
    fused_scores = defaultdict(float)

    for scores in score_dict_list:
        # Sort by score descending to get ranks
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
            # RRF formula: 1 / (k + rank)
            fused_scores[doc_id] += 1.0 / (k + rank)

    return dict(fused_scores)


def linear_fusion(score_dict_list, weights=None):
    """
    Linear weighted fusion of retrieval scores.

    Args:
        score_dict_list: List of dicts, each mapping doc_id -> score
        weights: List of weights for each score dict (default: equal weights)

    Returns:
        Combined scores dict
    """
    if weights is None:
        weights = [1.0 / len(score_dict_list)] * len(score_dict_list)

    # Normalize scores within each retriever to [0, 1]
    normalized_scores = []
    for scores in score_dict_list:
        if not scores:
            normalized_scores.append({})
            continue

        min_score = min(scores.values())
        max_score = max(scores.values())
        score_range = max_score - min_score

        if score_range == 0:
            norm_scores = {doc_id: 0.5 for doc_id in scores}
        else:
            norm_scores = {doc_id: (score - min_score) / score_range
                          for doc_id, score in scores.items()}
        normalized_scores.append(norm_scores)

    # Weighted combination
    fused_scores = defaultdict(float)
    for scores, weight in zip(normalized_scores, weights):
        for doc_id, score in scores.items():
            fused_scores[doc_id] += weight * score

    return dict(fused_scores)


def retrieval_hybrid(queries, query_ids, documents, doc_ids, task, cache_dir, excluded_ids, long_context, **kwargs):
    """
    Hybrid retrieval combining sparse (BM25) and dense models with fusion.

    Supported fusion methods:
    - 'rrf': Reciprocal Rank Fusion (k=60)
    - 'linear': Linear weighted combination (default weights: 0.5, 0.5)
    - 'dynamic': Query-dependent weighting (simplified DAT)

    Args:
        fusion_method: One of ['rrf', 'linear', 'dynamic']
        dense_model: Which dense model to use ['e5', 'qwen2', 'bge', 'reasonir']
        linear_weights: [sparse_weight, dense_weight] for linear fusion (default: [0.5, 0.5])
    """
    fusion_method = kwargs.get('fusion_method', 'rrf')
    dense_model = kwargs.get('dense_model', 'e5')
    linear_weights = kwargs.get('linear_weights', [0.5, 0.5])

    print(f"\n=== Hybrid Retrieval: BM25 + {dense_model} (fusion={fusion_method}) ===")

    # Get BM25 scores
    print("Running BM25...")
    bm25_result = retrieval_bm25(
        queries=queries,
        query_ids=query_ids,
        documents=documents,
        doc_ids=doc_ids,
        task=task,
        cache_dir=cache_dir,
        excluded_ids=excluded_ids,
        long_context=long_context,
        **kwargs
    )
    bm25_scores = bm25_result['scores'] if isinstance(bm25_result, dict) and 'scores' in bm25_result else bm25_result
    bm25_timing = bm25_result.get('timing', {}) if isinstance(bm25_result, dict) else {}

    # Get dense model scores
    print(f"Running {dense_model}...")
    dense_func = RETRIEVAL_FUNCS.get(dense_model)
    if dense_func is None:
        raise ValueError(f"Unknown dense model: {dense_model}")

    # Prepare kwargs for dense model - exclude keys that are passed explicitly
    exclude_keys = {
        'queries', 'query_ids', 'documents', 'doc_ids', 'task', 
        'cache_dir', 'excluded_ids', 'long_context', 'instructions', 'model_id'
    }
    dense_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_keys}
    dense_kwargs['model_name'] = dense_model

    # Ensure instructions is a dictionary with valid keys and no {text} placeholders
    instructions = kwargs.get('instructions', {})
    if instructions is None: 
        instructions = {}
    
    # Create a copy to avoid modifying the original if shared
    instructions = instructions.copy()
    
    if 'query' not in instructions or instructions['query'] is None:
        instructions['query'] = ""
    else:
        # Some configs use {text} which causes KeyError in add_instruct_concatenate
        instructions['query'] = instructions['query'].replace('{text}', '')
        
    if 'document' not in instructions or instructions['document'] is None:
        instructions['document'] = ""
    else:
        instructions['document'] = instructions['document'].replace('{text}', '')

    dense_result = dense_func(
        queries=queries,
        query_ids=query_ids,
        documents=documents,
        doc_ids=doc_ids,
        task=task,
        cache_dir=cache_dir,
        excluded_ids=excluded_ids,
        long_context=long_context,
        instructions=instructions,
        model_id=dense_model,
        **dense_kwargs
    )
    dense_scores = dense_result['scores'] if isinstance(dense_result, dict) and 'scores' in dense_result else dense_result
    dense_timing = dense_result.get('timing', {}) if isinstance(dense_result, dict) else {}

    # Fuse scores
    fusion_start = time.time()
    print(f"Fusing with {fusion_method}...")
    fused_results = {}

    for qid in query_ids:
        qid_str = str(qid)

        bm25_dict = bm25_scores.get(qid_str, {})
        dense_dict = dense_scores.get(qid_str, {})

        if fusion_method == 'rrf':
            # Reciprocal Rank Fusion
            fused_dict = reciprocal_rank_fusion([bm25_dict, dense_dict], k=60)

        elif fusion_method == 'linear':
            # Linear weighted fusion
            fused_dict = linear_fusion([bm25_dict, dense_dict], weights=linear_weights)

        elif fusion_method == 'dynamic':
            # Simplified dynamic weighting: higher weight to better-performing retriever
            # Based on top-1 score magnitude
            bm25_top1 = max(bm25_dict.values()) if bm25_dict else 0
            dense_top1 = max(dense_dict.values()) if dense_dict else 0

            total = bm25_top1 + dense_top1
            if total > 0:
                weights = [bm25_top1 / total, dense_top1 / total]
            else:
                weights = [0.5, 0.5]

            fused_dict = linear_fusion([bm25_dict, dense_dict], weights=weights)

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Sort and keep top 1000
        sorted_docs = sorted(fused_dict.items(), key=lambda x: x[1], reverse=True)[:1000]
        fused_results[qid_str] = {doc_id: score for doc_id, score in sorted_docs}

    fusion_time = time.time() - fusion_start

    # Combine timing information
    timing = {
        'index_time_seconds': bm25_timing.get('index_time_seconds', 0) + dense_timing.get('index_time_seconds', 0),
        'query_time_total_seconds': bm25_timing.get('query_time_total_seconds', 0) + dense_timing.get('query_time_total_seconds', 0) + fusion_time,
        'num_queries': len(queries),
        'fusion_time_seconds': fusion_time,
        'bm25_timing': bm25_timing,
        'dense_timing': dense_timing
    }

    return {'scores': fused_results, 'timing': timing}


RETRIEVAL_FUNCS = {
    'sf': retrieval_sf_qwen_e5,
    'qwen': retrieval_sf_qwen_e5,
    'qwen2': retrieval_sf_qwen_e5,
    'e5': retrieval_sf_qwen_e5,
    'bm25': retrieval_bm25,
    'sbert': retrieval_sbert_bge,
    'bge': retrieval_sbert_bge,
    'inst-l': retrieval_instructor,
    'inst-xl': retrieval_instructor,
    'grit': retrieval_grit,
    'cohere': retrieval_cohere,
    'voyage': retrieval_voyage,
    'openai': retrieval_openai,
    'google': retrieval_google,
    'bge_ce': retrieval_sbert_bge_ce,
    'nomic': retrieval_nomic,
    'm2': retrieval_m2,
    'contriever': retrieval_contriever,
    'reasonir': retrieval_reasonir,
    'rader': retrieval_rader,
    'diver-retriever': retrieval_qwen3_ft_diver,
    'hybrid': retrieval_hybrid,
}