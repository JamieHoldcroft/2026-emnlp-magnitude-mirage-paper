"""
Measure GPU memory usage for each retrieval model.
Loads each model exactly as done in retrievers.py and records peak GPU memory.

Based on model loading patterns from src/retrievers.py
"""

import os
import sys
import json
import torch
import gc

# Model configurations based on retrievers.py
MODEL_CONFIGS = {
    # BM25 - no GPU
    'bm25': {
        'type': 'bm25',
        'model_path': None,
        'uses_gpu': False
    },
    
    # SentenceTransformer models
    'sbert': {
        'type': 'sentence_transformer',
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130',
        'model_kwargs': {}
    },
    'bge': {
        'type': 'sentence_transformer',
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09',
        'model_kwargs': {}
    },
    'nomic': {
        'type': 'auto_model',
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--nomic-ai--nomic-embed-text-v1.5/snapshots/e5cf08aadaa33385f5990def41f7a23405aec398',
        'model_kwargs': {'trust_remote_code': True}
    },
    'inst-l': {
        'type': 'sentence_transformer',
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--hkunlp--instructor-large/snapshots/54e5ffb8d484de506e59443b07dc819fb15c7233',
        'model_kwargs': {}
    },
    'inst-xl': {
        'type': 'sentence_transformer',
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--hkunlp--instructor-xl/snapshots/ce48b213095e647a6c3536364b9fa00daf57f436',
        'model_kwargs': {}
    },
    
    # AutoModel models (large LLM-based)
    'contriever': {
        'type': 'auto_model',
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--facebook--contriever-msmarco/snapshots/abe8c1493371369031bcb1e02acb754cf4e162fa',
        'model_kwargs': {'trust_remote_code': True}
    },
    'e5': {
        'type': 'auto_model',
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--intfloat--e5-mistral-7b-instruct/snapshots/07163b72af1488142a360786df853f237b1a3ca1',
        'model_kwargs': {'device_map': 'auto', 'trust_remote_code': True, 'torch_dtype': torch.float16}
    },
    'sf': {
        'type': 'auto_model',
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Salesforce--SFR-Embedding-Mistral/snapshots/33a60fe7b5e00b9414981280c571e8a4f9f42f1b',
        'model_kwargs': {'device_map': 'auto', 'trust_remote_code': True, 'torch_dtype': torch.float16}
    },
    'qwen': {
        'type': 'auto_model',
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Alibaba-NLP--gte-Qwen1.5-7B-instruct/snapshots/e2f771ba1aba0666210c7fa446fdcf467fb49108',
        'model_kwargs': {'device_map': 'auto', 'trust_remote_code': True, 'torch_dtype': torch.float16}
    },
    'qwen2': {
        'type': 'auto_model',
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Alibaba-NLP--gte-Qwen2-7B-instruct/snapshots/a8d08b36ada9cacfe34c4d6f80957772a025daf2',
        'model_kwargs': {'device_map': 'auto', 'trust_remote_code': True, 'torch_dtype': torch.float16}
    },
    
    # Reasoning-specialized models
    'reasonir': {
        'type': 'auto_model',
        'model_path': 'reasonir/ReasonIR-8B',
        'model_kwargs': {'torch_dtype': 'auto', 'trust_remote_code': True}
    },
    'rader': {
        'type': 'auto_model',
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--Raderspace--RaDeR_Qwen_25_7B_instruct_MATH_LLMq_CoT_lexical/snapshots/f3f15af7632da5d5e52f6089debf12c54160b147',
        'model_kwargs': {'torch_dtype': torch.float16, 'attn_implementation': 'eager'}
    },
    'diver-retriever': {
        'type': 'auto_model',  # Uses vLLM, different loading
        'model_path': '/leonardo_scratch/fast/L-AUT_024/.cache/huggingface/models--AQ-MedAI--Diver-Retriever-4B/snapshots/ff7f8bc8d1827734dcf2bbfab99f065be618069a',
        'note': 'Uses vLLM, requires special handling'
    }
}


def measure_model_memory(model_name):
    """Load a model exactly as in retrievers.py and measure GPU memory usage."""
    
    if model_name not in MODEL_CONFIGS:
        return {'gpu_memory_gb': None, 'error': f'Unknown model: {model_name}'}
    
    config = MODEL_CONFIGS[model_name]
    
    # BM25 doesn't use GPU
    if config.get('uses_gpu') == False:
        return {'gpu_memory_gb': 0, 'uses_gpu': False, 'model_path': None}
    
    # Clear GPU memory first
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    initial_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
    
    try:
        if config['type'] == 'sentence_transformer':
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config['model_path'], model_kwargs=config.get('model_kwargs', {}))
            
        elif config['type'] == 'auto_model':
            from transformers import AutoModel, AutoTokenizer
            
            model_kwargs = config.get('model_kwargs', {}).copy()
            tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=model_kwargs.get('trust_remote_code', False))
            model = AutoModel.from_pretrained(config['model_path'], **model_kwargs)
            
            # Move to GPU if not using device_map
            if 'device_map' not in model_kwargs:
                model = model.to('cuda')
            model.eval()
            
        elif config['type'] == 'vllm':
            # vLLM uses a different memory model
            return {
                'gpu_memory_gb': None, 
                'note': 'Uses vLLM - memory allocation is dynamic',
                'model_path': config['model_path'],
                'uses_gpu': True
            }
        else:
            return {'gpu_memory_gb': None, 'error': f'Unknown model type: {config["type"]}'}
        
        # Measure peak memory after loading
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        current_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        # Clean up
        del model
        if 'tokenizer' in dir():
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            'gpu_memory_gb': round(peak_memory, 2),
            'current_memory_gb': round(current_memory, 2),
            'model_path': config['model_path'],
            'model_type': config['type'],
            'uses_gpu': True
        }
        
    except Exception as e:
        return {
            'gpu_memory_gb': None, 
            'error': str(e),
            'model_path': config.get('model_path')
        }


def main():
    # Order models by expected size (smaller first)
    models_ordered = [
        # 'bm25', 'sbert', 'bge', 'contriever',  'inst-l', 'inst-xl',
        # 'e5', 'sf', 'qwen', 'qwen2', 'reasonir', 'rader',
        'nomic',  'diver-retriever'
    ]
    
    results = {}
    
    print("=" * 60)
    print("GPU Memory Measurement Script")
    print("Based on model loading from src/retrievers.py")
    print("=" * 60)
    
    for model_name in models_ordered:
        print(f"\n[{models_ordered.index(model_name)+1}/{len(models_ordered)}] Measuring: {model_name}")
        print(f"    Model path: {MODEL_CONFIGS[model_name].get('model_path', 'N/A')}")
        
        result = measure_model_memory(model_name)
        results[model_name] = result
        
        if result.get('gpu_memory_gb') is not None:
            print(f"    GPU Memory: {result['gpu_memory_gb']:.2f} GB")
        elif result.get('error'):
            print(f"    Error: {result['error']}")
        elif result.get('note'):
            print(f"    Note: {result['note']}")
        else:
            print(f"    Uses GPU: {result.get('uses_gpu', 'Unknown')}")
    
    # Save results
    # os.makedirs('sigir_results', exist_ok=True)
    output_path = './model_memory_usage.json'
    with open(output_path, 'w') as f:
        # Convert torch.dtype to string for JSON serialization
        serializable_results = {}
        for k, v in results.items():
            serializable_results[k] = {
                key: (str(val) if 'torch' in str(type(val)) else val) 
                for key, val in v.items()
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n\nResults saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Summary: GPU Memory Usage per Model")
    print("=" * 60)
    print(f"{'Model':<20} {'Memory (GB)':<15} {'Type':<20}")
    print("-" * 55)
    
    for model_name in models_ordered:
        r = results[model_name]
        mem = r.get('gpu_memory_gb')
        if mem is not None:
            mem_str = f"{mem:.2f}" if mem > 0 else "--"
        else:
            mem_str = "N/A"
        model_type = r.get('model_type', r.get('note', 'N/A'))
        print(f"{model_name:<20} {mem_str:<15} {str(model_type)[:20]:<20}")


if __name__ == '__main__':
    main()
