"""
Calculate average query lengths for BRIGHT benchmark datasets.
This script computes the average query length for each CoT reasoning model 
and generates a LaTeX table for the SIGIR paper.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import tiktoken
import json

# Initialize GPT-2 tokenizer for consistent length measurement
tokenizer = tiktoken.get_encoding("gpt2")

def count_tokens(text):
    """Count tokens using GPT-2 tokenizer."""
    return len(tokenizer.encode(text))

def main():
    tasks = [
        'biology', 'earth_science', 'economics', 'psychology', 
        'robotics', 'stackoverflow', 'sustainable_living',
        'leetcode', 'pony', 'aops', 'theoremqa_questions', 'theoremqa_theorems'
    ]
    
    # CoT reasoning models to analyze
    cot_models = ['gpt4_reason', 'llama3-70b_reason', 'claude-3-opus_reason', 
                  'grit_reason', 'Gemini-1.0_reason']
    
    # Also include original queries
    query_types = ['examples'] + cot_models
    
    # Results storage
    results = {}
    
    for task in tasks:
        results[task] = {
            'num_queries': 0,
            'num_docs': 0,
            'avg_pos_docs': 0,
            'avg_query_len': {},
            'avg_doc_len': 0
        }
        
        # Load original examples
        try:
            examples = load_dataset('xlangai/BRIGHT', 'examples')[task]
            results[task]['num_queries'] = len(examples)
            results[task]['avg_pos_docs'] = sum(len(e['gold_ids']) for e in examples) / len(examples) if examples else 0
            
            # Original query lengths
            query_lens = [count_tokens(e['query']) for e in examples]
            results[task]['avg_query_len']['original'] = sum(query_lens) / len(query_lens) if query_lens else 0
        except Exception as e:
            print(f"Error loading examples for {task}: {e}")
            continue
        
        # Load documents
        try:
            docs = load_dataset('xlangai/BRIGHT', 'documents')[task]
            results[task]['num_docs'] = len(docs)
            doc_lens = [count_tokens(d['content']) for d in docs]
            results[task]['avg_doc_len'] = sum(doc_lens) / len(doc_lens) if doc_lens else 0
        except Exception as e:
            print(f"Error loading documents for {task}: {e}")
        
        # Load CoT reasoning variants
        for cot_model in cot_models:
            model_name = cot_model.replace('_reason', '')
            try:
                cot_examples = load_dataset('xlangai/BRIGHT', cot_model)[task]
                cot_lens = [count_tokens(e['query']) for e in cot_examples]
                results[task]['avg_query_len'][model_name] = sum(cot_lens) / len(cot_lens) if cot_lens else 0
            except Exception as e:
                print(f"Error loading {cot_model} for {task}: {e}")
                results[task]['avg_query_len'][model_name] = 0
    
    # Save results to JSON
    with open('./bright_statistics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print LaTeX table
    print("\n\n=== LaTeX Table ===\n")
    print(r"\begin{table*}")
    print(r"\centering")
    print(r"\caption{BRIGHT benchmark statistics. Q = number of queries, D = number of documents, " + 
          r"D$^+$ = avg positive documents per query. Q Len columns show average query length " +
          r"(GPT-2 tokens) for original queries and CoT-augmented variants.}")
    print(r"\small")
    print(r"\setlength\tabcolsep{3pt}")
    print(r"\begin{tabular}{l|rrr|r|rrrrr}")
    print(r"\toprule")
    print(r"& \multicolumn{3}{c|}{Counts} & D Len & \multicolumn{5}{c}{Avg. Query Length (tokens)} \\")
    print(r"Dataset & Q & D & D$^+$ & Avg. & Orig. & GPT-4 & Llama-3 & Claude-3 & Gemini \\")
    print(r"\midrule")
    
    # Group by domain
    stackexchange = ['biology', 'earth_science', 'economics', 'psychology', 
                     'robotics', 'stackoverflow', 'sustainable_living']
    coding = ['leetcode', 'pony']
    theorems = ['aops', 'theoremqa_questions', 'theoremqa_theorems']
    
    # Print StackExchange
    print(r"\multicolumn{10}{c}{\textit{StackExchange}} \\")
    print(r"\midrule")
    for task in stackexchange:
        if task in results:
            r = results[task]
            task_name = task.replace('_', ' ').title()
            orig = r['avg_query_len'].get('original', 0)
            gpt4 = r['avg_query_len'].get('gpt4', 0)
            llama = r['avg_query_len'].get('llama3-70b', 0)
            claude = r['avg_query_len'].get('claude-3-opus', 0)
            gemini = r['avg_query_len'].get('Gemini-1.0', 0)
            print(f"{task_name} & {r['num_queries']} & {r['num_docs']:,} & {r['avg_pos_docs']:.1f} & "
                  f"{r['avg_doc_len']:.1f} & "
                  f"{orig:.1f} & {gpt4:.1f} & {llama:.1f} & {claude:.1f} & {gemini:.1f} \\\\")
    
    print(r"\midrule")
    print(r"\multicolumn{10}{c}{\textit{Coding}} \\")
    print(r"\midrule")
    for task in coding:
        if task in results:
            r = results[task]
            task_name = task.replace('_', ' ').title()
            orig = r['avg_query_len'].get('original', 0)
            gpt4 = r['avg_query_len'].get('gpt4', 0)
            llama = r['avg_query_len'].get('llama3-70b', 0)
            claude = r['avg_query_len'].get('claude-3-opus', 0)
            gemini = r['avg_query_len'].get('Gemini-1.0', 0)
            print(f"{task_name} & {r['num_queries']} & {r['num_docs']:,} & {r['avg_pos_docs']:.1f} & "
                  f"{r['avg_doc_len']:.1f} & "
                  f"{orig:.1f} & {gpt4:.1f} & {llama:.1f} & {claude:.1f} & {gemini:.1f} \\\\")
    
    print(r"\midrule")
    print(r"\multicolumn{10}{c}{\textit{Theorems}} \\")
    print(r"\midrule")
    for task in theorems:
        if task in results:
            r = results[task]
            task_name = task.replace('_', ' ').title()
            orig = r['avg_query_len'].get('original', 0)
            gpt4 = r['avg_query_len'].get('gpt4', 0)
            llama = r['avg_query_len'].get('llama3-70b', 0)
            claude = r['avg_query_len'].get('claude-3-opus', 0)
            gemini = r['avg_query_len'].get('Gemini-1.0', 0)
            print(f"{task_name} & {r['num_queries']} & {r['num_docs']:,} & {r['avg_pos_docs']:.1f} & "
                  f"{r['avg_doc_len']:.1f} & "
                  f"{orig:.1f} & {gpt4:.1f} & {llama:.1f} & {claude:.1f} & {gemini:.1f} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:bright_stats}")
    print(r"\end{table*}")

if __name__ == '__main__':
    main()
