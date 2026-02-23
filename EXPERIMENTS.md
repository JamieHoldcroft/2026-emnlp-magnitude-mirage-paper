# Experiment Documentation

## Overview

This document describes the **experiments and evaluation framework** for evaluating retrieval confidence signals on the BRIGHT benchmark. Each experiment focuses on a different retrieval confidence signal that is currently practiced in modern RAG systems: ___.

**Total Experiments:** ~X unique experiments across all ```exp``` files.

**Datasets**:
- **BRIGHT**: All 12 tasks (biology, earth_science, economics, psychology, robotics, stackoverflow, sustainable_living, leetcode, pony, aops, theoremqa_questions, theoremqa_theorems)
- **BEIR**: 13 datasets (arguana, fiqa, nfcorpus, quora, scidocs, scifact, trec-covid, dbpedia-entity, fever, hotpotqa, nq, climate-fever, touche-2020)

**Models**: 17+ models tested across both BRIGHT and BEIR.

---

## Experiment (Exp) Dependencies

```
Exp 1 & Exp 2 (Required First)
   ↓
   ├──→ Exp 3 (reuses cache)
   └──→ Exp 4 (reuses cache)
   ...
```

**Key Points**:
- **Exp 1 and Exp 2 MUST run first** - they build the document embedding cache
- **All proceeding experiments reuse the cached results**

---

## Quick Reference Table

| Exp | Name | Purpose | # Experiments | Runtime | Cache | Depends On |
|-------|------|---------|-------------|---------|-------|------------|
| 1 | Cache Build for BRIGHT | Cache Retrieval Scores on BRIGHT | X | 48-72h | Creates `cache/` | None |
| 2 | Cache Build for BEIR | Cache Retrieval Scores on BEIR | X | 48-72h | Creates `cache/` | None |
| 3 | Efficiency | Measure query latency & QPS | 204 | 24-36h | Reuses Cache | Exp 1 and Exp 2 |
| 4 | Scaling | Test corpus size impact | 84 | 12-24h | Reuses Cache | Exp 1 and Exp 2 |
| 5 | Quantization | Test precision trade-offs | 60 | 10-20h | Reuses Cache | Exp 1 and Exp 2 |
| 6 | Length | Test document length impact | 72 | 12-24h | Reuses Cache | Exp 1 and Exp 2 |

---

## Detailed Experiment Descriptions

### Exp 1: Build Cache

**Script**: `scripts/phase1_build_cache.sh`

#### Purpose
Compute and cache retrieval scores for all models on all tasks in BRIGHT. This is the foundation for all subsequent experiments on BRIGHT.

#### Configuration
- **Models**: All 16 non-API models
  - Sparse Retrievers: `bm25`
  - Small Dense Bi-Encoders: `sbert`, `bge`, `contriever`, `nomic`, `inst-l`
  - Large Dense Bi-Encoders: `sf`, `e5`, `qwen`, `qwen2`, `grit`,  `inst-l`, `m2`
  - Reasoning-Augmented Bi-Encoders: `reasonir`, `rader`, `diver-retriever`
- **Tasks**: All 12 BRIGHT tasks
- **Total Experiments**: 16 models × 12 tasks = **192 experiments**

#### Cache Strategy
- **Creates**: `cache/` directory
- **Structure**: `cache/doc_emb/{model}/{task}/long_False_{batch_size}/{chunk_id}.json`

#### Outputs
```
sigir_results/phase1_cache_build/
├── {task}_{model}/
│   ├── efficiency.json          # Indexing time, docs/sec
│   ├── results.json              # nDCG@10, Recall@100, etc.
│   └── score.json                # Full retrieval scores
└── master_indexing_times.json    # Summary of all indexing times
```

#### Key Metrics
- **Indexing Time**: Total time to embed all documents (seconds)
- **Documents/Second**: Indexing throughput
- **Retrieval Effectiveness**: nDCG@10, Recall@10/100, Precision@10

#### Estimated Runtime
- **48-72 hours** on A100-80GB GPU
- Varies by model size (BM25 is fast, Reasonir/Rader are slow)

#### Dependencies
None (must run first!)

#### Example Command
```bash
bash scripts/phase1_build_cache.sh
```

#### When to Run
- **Always first** before any other phase
- When adding new models or tasks
- When cache is corrupted or deleted

---

### Phase 2: Efficiency Profiling (Query Latency)

**Script**: `scripts/phase2_efficiency.sh`

#### Purpose
Measure **query latency distribution** and **queries per second (QPS)** for all models. This phase focuses on online serving performance.

#### Configuration
- **Models**: All 17 non-API models
- **Tasks**: All 12 BRIGHT tasks
- **Total Experiments**: 17 models × 12 tasks = **204 experiments**

#### Cache Strategy
- **Reuses**: Phase 1 cache (`cache/`)
- **No re-indexing**: Documents already embedded, only query encoding is measured
- **Very efficient**: Much faster than Phase 1

#### Outputs
```
sigir_results/phase2_efficiency/
├── {task}_{model}/
│   └── efficiency.json           # QPS, latency percentiles
└── efficiency_comparison.json     # Aggregated comparison across models
```

#### Key Metrics
- **QPS**: Queries per second
- **Latency**: Mean, p50, p95, p99 (milliseconds)
- **Effectiveness**: nDCG@10 (for context)

#### Estimated Runtime
- **24-36 hours** on A100-80GB GPU
- Faster than Phase 1 (no document re-indexing)

#### Dependencies
- **Requires**: Phase 1 cache must exist

#### Example Command
```bash
bash scripts/phase2_efficiency.sh
```

#### When to Run
- After Phase 1 completes
- To compare query latency across models
- For production deployment planning

---

### Phase 3: Corpus Scaling Analysis

**Script**: `scripts/phase3_scaling.sh`

#### Purpose
Analyze how indexing time and search performance **scale with corpus size**. Answers: "How does performance degrade as the corpus grows?"

#### Configuration
- **Model**: `e5` (representative 7B model)
- **Tasks**: All 12 BRIGHT tasks
- **Corpus Sizes**: 1K, 5K, 10K, 25K, 50K, 100K, full
- **Total Experiments**: 1 model × 12 tasks × 7 sizes = **84 experiments**

#### Cache Strategy
- **Creates**: Separate `cache_scaling/{task}/size_{N}/` for each size
- **Why separate**: Different corpus sizes = different embeddings needed

#### Outputs
```
sigir_results/phase3_scaling/
├── {task}_size_1000/
│   └── efficiency.json
├── {task}_size_5000/
│   └── efficiency.json
...
└── {task}_size_full/
    └── efficiency.json
```

#### Key Metrics
- **Indexing Time vs Size**: How indexing scales (linear? sublinear?)
- **Search Time vs Size**: How query latency grows with corpus
- **Effectiveness vs Size**: Does performance improve with more documents?

#### Estimated Runtime
- **12-24 hours** on A100-80GB GPU

#### Dependencies
None (independent)

#### Example Command
```bash
bash scripts/phase3_scaling.sh
```

#### When to Run
- To understand scalability characteristics
- For capacity planning
- To identify corpus size sweet spots

---

### Phase 4: Quantization Impact Analysis

**Script**: `scripts/phase4_quantization.sh`

#### Purpose
Measure the **trade-off between model precision and performance**. Tests fp32, fp16, bf16, int8, and int4 quantization.

#### Configuration
- **Model**: `e5` (supports all quantization levels)
- **Tasks**: All 12 BRIGHT tasks
- **Precisions**: fp32, fp16, bf16, int8, int4
- **Total Experiments**: 1 model × 12 tasks × 5 precisions = **60 experiments**

#### Cache Strategy
- **Creates**: Separate `cache_quantization/{task}/prec_{precision}/` for each precision
- **Why separate**: Different precisions = different embeddings (especially int8/int4)

#### Outputs
```
sigir_results/phase4_quantization/
├── {task}_fp32/
│   ├── efficiency.json
│   └── results.json
├── {task}_fp16/
│   ├── efficiency.json
│   └── results.json
...
└── {task}_int4/
    ├── efficiency.json
    └── results.json
```

#### Key Metrics
- **Memory Footprint**: GB per precision level
- **Speed Improvement**: Indexing & query time speedup
- **Quality Degradation**: nDCG@10 drop from fp32 baseline
- **Pareto Frontier**: Best speed/quality trade-offs

#### Estimated Runtime
- **10-20 hours** on A100-80GB GPU

#### Dependencies
None (independent)

#### Example Command
```bash
bash scripts/phase4_quantization.sh
```

#### When to Run
- For deployment on resource-constrained devices
- To find optimal precision for production
- To compare memory vs quality trade-offs

---

### Phase 5: Context Length Sensitivity

**Script**: `scripts/phase5_length.sh`

#### Purpose
Find the **optimal document truncation length** that balances effectiveness and efficiency. Tests max lengths from 512 to 16384 tokens.

#### Configuration
- **Model**: `qwen2` (supports up to 8192 tokens natively)
- **Tasks**: All 12 BRIGHT tasks
- **Document Max Lengths**: 512, 1024, 2048, 4096, 8192, 16384
- **Total Experiments**: 1 model × 12 tasks × 6 lengths = **72 experiments**

#### Cache Strategy
- **Creates**: Separate `cache_length/{task}/len_{length}/` for each length
- **Why separate**: Different truncation = different embeddings

#### Outputs
```
sigir_results/phase5_length/
├── {task}_doclen_512/
│   ├── efficiency.json
│   └── results.json
├── {task}_doclen_1024/
│   ├── efficiency.json
│   └── results.json
...
└── {task}_doclen_16384/
    ├── efficiency.json
    └── results.json
```

#### Key Metrics
- **Effectiveness vs Length**: Does longer context improve nDCG?
- **Speed vs Length**: How much slower are long contexts?
- **Optimal Length**: Best effectiveness/speed trade-off
- **Diminishing Returns**: At what length do gains plateau?

#### Estimated Runtime
- **12-24 hours** on A100-80GB GPU

#### Dependencies
None (independent)

#### Example Command
```bash
bash scripts/phase5_length.sh
```

#### When to Run
- To optimize document chunking strategies
- For cost/performance optimization
- To validate long-context claims

---

### Phase 6: Reasoning Queries (⭐ Main Contribution)

**Script**: `scripts/phase6_reasoning.sh`

#### Purpose
Compare **original queries vs LLM-augmented reasoning queries**. This is the **main contribution** of the reproducibility study, analyzing the effectiveness vs efficiency trade-off of reasoning-enhanced retrieval.

#### Configuration
- **Models**: All 17 non-API models
- **Tasks**: All 12 BRIGHT tasks
- **Reasoning Types**: 6 types
  - `original` - Baseline (no reasoning)
  - `gpt4_reason` - GPT-4 reasoning
  - `llama3-70b_reason` - Llama-3-70B reasoning
  - `claude-3-opus_reason` - Claude reasoning
  - `grit_reason` - GritLM reasoning
  - `Gemini-1.0_reason` - Gemini reasoning
- **Total Experiments**: 17 models × 12 tasks × 6 types = **1,224 experiments**

#### Cache Strategy
- **Reuses**: Phase 1 cache (`cache/`)
- **Why efficient**: Document embeddings are identical regardless of query type
- **No re-indexing**: Only query embedding changes

#### Outputs
```
sigir_results/phase6_reasoning/
├── {task}/
│   ├── {model}_original/
│   │   ├── results.json
│   │   └── efficiency.json
│   ├── {model}_gpt4_reason/
│   │   ├── results.json
│   │   └── efficiency.json
│   ...
└── reasoning_comparison.json      # Aggregated comparison
```

#### Key Metrics
- **nDCG Gain**: Improvement over original queries
- **Latency Penalty**: Additional query processing time
- **Efficiency Ratio**: nDCG gain / latency penalty (higher is better)
- **Cost-Benefit**: Is reasoning worth the extra time?

#### Estimated Runtime
- **36-48 hours** on A100-80GB GPU
- Longest phase due to 1,224 experiments

#### Dependencies
- **Requires**: Phase 1 cache must exist

#### Example Command
```bash
bash scripts/phase6_reasoning.sh
```

#### When to Run
- To evaluate reasoning query effectiveness
- For main paper results
- To compare different reasoning LLMs

---

### Phase 7: Document Expansion Analysis

**Script**: `scripts/phase7_expansion.sh`

#### Purpose
Compare **original documents vs expanded/rechunked documents**. Tests whether document preprocessing improves retrieval.

#### Configuration
- **Model**: `e5`
- **Tasks**: All 12 BRIGHT tasks
- **Expansion Strategies**: 3 types
  - `None` - Original documents
  - `gold` - Oracle expansion (gold answers)
  - `rechunk` - Smart rechunking
- **Total Experiments**: 1 model × 12 tasks × 3 strategies = **36 experiments**

#### Cache Strategy
- **Creates**: Separate `cache_expansion/{task}/{strategy}/` for each strategy
- **Why separate**: Different documents = different embeddings

#### Outputs
```
sigir_results/phase7_expansion/
├── {task}_original/
│   ├── efficiency.json
│   └── results.json
├── {task}_gold/
│   ├── efficiency.json
│   └── results.json
└── {task}_rechunk/
    ├── efficiency.json
    └── results.json
```

#### Key Metrics
- **Effectiveness Gain**: nDCG improvement from expansion
- **Index Size**: Increase in corpus size
- **Indexing Overhead**: Additional indexing time
- **Optimal Strategy**: Best expansion approach

#### Estimated Runtime
- **8-16 hours** on A100-80GB GPU

#### Dependencies
None (independent)

#### Example Command
```bash
bash scripts/phase7_expansion.sh
```

#### When to Run
- To evaluate document preprocessing strategies
- For corpus augmentation experiments
- To compare expansion methods

---

### Phase 8: Long Context Retrieval

**Script**: `scripts/phase8_long_context.sh`

#### Purpose
Test retrieval on the **long_documents subset** of BRIGHT. Evaluates how models handle documents with 10K+ tokens.

#### Configuration
- **Models**: All 17 non-API models
- **Tasks**: All 12 BRIGHT tasks
- **Context**: `--long_context` flag enabled
- **Total Experiments**: 17 models × 12 tasks = **204 experiments**

#### Cache Strategy
- **Creates**: Separate `cache_long/` directory
- **Why separate**: Long documents are different from regular documents

#### Outputs
```
sigir_results/phase8_long_context/
└── {task}_{model}/
    ├── efficiency.json
    └── results.json
```

#### Key Metrics
- **Long-Context Effectiveness**: nDCG@10 on long documents
- **Comparison**: Short vs long context performance delta
- **Model Ranking**: Which models excel at long context?

#### Estimated Runtime
- **24-36 hours** on A100-80GB GPU

#### Dependencies
None (independent)

#### Example Command
```bash
bash scripts/phase8_long_context.sh
```

#### When to Run
- To evaluate long-context capabilities
- For long-document retrieval scenarios
- To validate max_length claims

---

### Phase 9: Hybrid Retrieval Fusion ⭐ (NEW)

**Script**: `scripts/phase9_hybrid_fusion.sh`

#### Purpose
Combine sparse (BM25) and dense models using different fusion strategies (RRF, Linear, Dynamic). **Hot topic at SIGIR 2025** - addresses major gap in current evaluation.

#### Configuration
- **Fusion Methods**: RRF (k=60), Linear weighted, Dynamic weighting
- **Dense Models**: e5, qwen2, bge, reasonir (combined with BM25)
- **Tasks**: All 12 BRIGHT tasks
- **Total Experiments**: 4 dense models × 3 fusion methods × 12 tasks = **144 experiments**

#### Cache Strategy
- **Reuses**: Phase 1 cache (very efficient!)
- **Structure**: Uses existing BM25 and dense model embeddings

#### Outputs
```
sigir_results/phase9_hybrid/
├── {task}/
│   ├── e5_rrf/results.json
│   ├── e5_linear/results.json
│   ├── e5_dynamic/results.json
│   └── ...
└── fusion_comparison.json
```

#### Key Metrics
- nDCG@10 gain over best individual model
- Optimal fusion method per task/model
- Fusion method effectiveness comparison

#### Estimated Runtime
- **18-24 hours** on A100-80GB GPU

#### Dependencies
- **Requires**: Phase 1 cache

#### Example Command
```bash
bash scripts/phase9_hybrid_fusion.sh
```

#### When to Run
- To evaluate hybrid retrieval strategies (industry standard)
- For production deployment insights
- To compare fusion methods

---

### Phase 10: Robustness Evaluation ⭐ (NEW)

**Script**: `scripts/phase10_robustness.sh`

#### Purpose
**Critical for reproducibility paper!** Test model robustness to query perturbations, adversarial attacks, and variations. Validates reproducibility claims.

#### Configuration
- **Perturbation Types**:
  - Query paraphrasing (5 variants)
  - Synonym replacement (3 levels)
  - Adversarial token insertion (3 levels)
  - Query length perturbation (expand/contract)
- **Models**: Top 5 models (e5, qwen2, grit, reasonir, bge)
- **Tasks**: All 12 BRIGHT tasks
- **Total Experiments**: 5 models × 9 perturbations × 12 tasks = **540 experiments**

#### Cache Strategy
- **Reuses**: Phase 1 cache
- **Perturbations**: Generated on-the-fly using NLTK

#### Outputs
```
sigir_results/phase10_robustness/
├── {task}/
│   ├── e5_paraphrase_0/results.json
│   ├── e5_synonym_0/results.json
│   ├── e5_adversarial_0/results.json
│   └── ...
└── robustness_summary.json
```

#### Key Metrics
- Performance drop under perturbation (%)
- Robustness score (100 - avg_drop)
- Most vulnerable perturbation types
- Model ranking by robustness

#### Estimated Runtime
- **30-40 hours** on A100-80GB GPU

#### Dependencies
- **Requires**: Phase 1 cache, NLTK installed

#### Example Command
```bash
bash scripts/phase10_robustness.sh
```

#### When to Run
- Essential for reproducibility studies
- To validate model stability
- For adversarial robustness analysis

---

### Phase 11: Cross-Benchmark Generalization (BEIR) (NEW)

**Script**: `scripts/phase11_cross_benchmark.sh`

#### Purpose
Evaluate models on BEIR benchmark to test generalization beyond BRIGHT. Validates **reasoning gap hypothesis** (59.0 BEIR → 18.3 BRIGHT performance drop).

#### Configuration
- **Benchmarks**: BEIR (13 datasets) + BRIGHT (12 tasks)
- **BEIR Datasets**: arguana, fiqa, nfcorpus, quora, scidocs, scifact, trec-covid, dbpedia-entity, fever, hotpotqa, nq, climate-fever, touche-2020
- **Models**: All 17 non-API models
- **Total Experiments**: 17 models × 13 BEIR datasets = **221 experiments**

#### Cache Strategy
- **Creates**: Separate `cache_beir/` directory
- **Why separate**: Different dataset format and documents

#### Outputs
```
sigir_results/phase11_beir/
├── arguana_e5/results.json
├── fiqa_e5/results.json
├── ...
└── cross_benchmark_comparison.json
```

#### Key Metrics
- BEIR avg nDCG@10 vs BRIGHT avg nDCG@10
- Reasoning gap percentage
- Cross-domain transfer scores
- Generalization capability ranking

#### Estimated Runtime
- **36-48 hours** on A100-80GB GPU

#### Dependencies
None (independent, but compares with Phase 1 results)

#### Example Command
```bash
bash scripts/phase11_cross_benchmark.sh
```

#### When to Run
- To validate generalization claims
- To compare reasoning vs semantic retrieval
- For comprehensive benchmark coverage

---

### Phase 12: Two-Stage Retrieval Pipelines (NEW)

**Script**: `scripts/phase12_two_stage.sh`

#### Purpose
Test **industry-standard retrieve-then-rerank pipelines**. Currently bge_ce is only tested in isolation - this evaluates systematic two-stage approaches.

#### Configuration
- **Stage 1 (Retrieve)**: BM25, e5, qwen2 (top-100)
- **Stage 2 (Rerank)**: bge_ce cross-encoder (top-100 → top-10)
- **Tasks**: All 12 BRIGHT tasks
- **Total Experiments**: 3 retrievers × 1 reranker × 12 tasks = **36 experiments**

#### Cache Strategy
- **Reuses**: Phase 1 cache for retrievers
- **New**: Reranker scores for top-100 candidates

#### Outputs
```
sigir_results/phase12_two_stage/
└── {task}/
    ├── bm25_bge_ce/results.json
    ├── e5_bge_ce/results.json
    └── qwen2_bge_ce/results.json
```

#### Key Metrics
- Stage 1 Recall@100
- Stage 2 nDCG@10 (final)
- Two-stage latency overhead
- Effectiveness gain vs single-stage

#### Estimated Runtime
- **10-16 hours** on A100-80GB GPU

#### Dependencies
- **Requires**: Phase 1 cache

#### Example Command
```bash
bash scripts/phase12_two_stage.sh
```

#### When to Run
- For production pipeline design
- To compare retrieve vs rerank trade-offs
- When optimizing for precision

---

### Phase 13: Cost-Effectiveness Analysis (NEW)

**Script**: `scripts/phase13_cost_analysis.sh`

#### Purpose
**Critical for deployment decisions!** Measure compute costs, API costs, and generate cost/performance Pareto frontiers. Enables ROI analysis.

#### Configuration
- **Cost Dimensions**:
  - Compute: GPU hours, memory usage
  - API: Azure OpenAI costs ($0.13/1M tokens)
  - Storage: Index size
- **Models**: All models including Azure OpenAI
- **Tasks**: All 12 BRIGHT tasks
- **Total Experiments**: 18 models × 12 tasks = **216 experiments**

#### Cache Strategy
- **Reuses**: Phase 1 cache

#### Outputs
```
sigir_results/phase13_cost/
├── {task}_{model}/cost_metrics.json
└── pareto_frontier.png
```

#### Key Metrics
- $/1K queries
- Cost per nDCG point
- Pareto frontier: cost vs effectiveness
- ROI vs fine-tuning baseline ($50K-$200K)

#### Estimated Runtime
- **24-30 hours** on A100-80GB GPU

#### Dependencies
- **Requires**: Phase 1 cache, Azure OpenAI API key

#### Example Command
```bash
# Set Azure credentials
export AZURE_OPENAI_ENDPOINT="https://..."
export AZURE_OPENAI_API_KEY="..."
bash scripts/phase13_cost_analysis.sh
```

#### When to Run
- For production deployment planning
- To justify model selection decisions
- For cost optimization

---

### Phase 14: Concurrency & Load Testing (NEW)

**Script**: `scripts/phase14_concurrency.sh`

#### Purpose
Test **real-world deployment constraints**: throughput under concurrent load, p99 latency at scale, saturation points.

#### Configuration
- **Concurrency Levels**: 1, 5, 10, 50, 100 concurrent queries
- **Models**: Top 5 (e5, qwen2, grit, reasonir, bge)
- **Tasks**: 3 representative tasks (biology, stackoverflow, theoremqa_questions)
- **Total Experiments**: 5 models × 5 concurrency × 3 tasks = **75 experiments**

#### Cache Strategy
- **Reuses**: Phase 1 cache

#### Outputs
```
sigir_results/phase14_concurrency/
├── {task}/
│   ├── e5_c1/metrics.json
│   ├── e5_c5/metrics.json
│   └── ...
└── concurrency_summary.json
```

#### Key Metrics
- Throughput (QPS under load)
- p50/p95/p99 latency at each concurrency
- Memory scaling
- Throughput saturation point

#### Estimated Runtime
- **8-12 hours** on A100-80GB GPU

#### Dependencies
- **Requires**: Phase 1 cache

#### Example Command
```bash
bash scripts/phase14_concurrency.sh
```

#### When to Run
- For production scalability testing
- To determine infrastructure requirements
- For SLA planning (<3s latency target)

---

## Recommended Execution Order

### Option 1: Full Pipeline (All Phases)
```bash
# Run all phases in recommended order
bash scripts/run_all.sh
```
Total time: ~160-240 hours

### Option 2: Core Phases Only
```bash
# Phase 1: Build cache (required)
bash scripts/phase1_build_cache.sh

# Phase 2: Efficiency profiling
bash scripts/phase2_efficiency.sh

# Phase 6: Reasoning queries (main contribution)
bash scripts/phase6_reasoning.sh
```
Total time: ~100-140 hours

### Option 3: Specific Phases
```bash
# Run only phases 1, 2, and 6
bash scripts/run_all.sh 1 2 6
```

---

## Checking Phase Progress

### Check if a phase is complete
```bash
# Phase 1: Check for master summary
ls sigir_results/phase1_cache_build/master_indexing_times.json

# Phase 2: Check for comparison file
ls sigir_results/phase2_efficiency/efficiency_comparison.json

# Phase 6: Check for reasoning comparison
ls sigir_results/phase6_reasoning/reasoning_comparison.json

# Any phase: Count completed experiments
find sigir_results/phase1_cache_build -name "results.json" | wc -l
```

### Monitor running phase
```bash
# Watch log files
tail -f sigir_results/phase1_cache_build/logs/*.log

# Check GPU usage
nvidia-smi

# Count cache files created
find cache/doc_emb -name "*.json" | wc -l
```

---

## Resuming Interrupted Phases

All scripts are **resume-friendly**:
- If `results.json` exists for an experiment, it's skipped
- Simply re-run the same script to continue

Example:
```bash
# Phase interrupted? Just run again
bash scripts/phase1_build_cache.sh
# Output: "[SKIP] Already complete" for finished experiments
```

---

## Common Issues & Solutions

### Issue 1: "Phase 1 cache not found"
**Solution**: Run Phase 1 first
```bash
bash scripts/phase1_build_cache.sh
```

### Issue 2: Out of GPU memory
**Solution**:
- Use quantization: Add `--quantization fp16` to run.py
- Reduce batch size in scripts
- Run smaller models first

### Issue 3: Disk space full
**Check cache size**:
```bash
du -sh cache/
du -sh sigir_results/
```
**Solution**: Delete unused phase caches (but keep Phase 1!)

### Issue 4: Permission denied
**Solution**:
```bash
chmod +x scripts/*.sh
```

---

## Output File Formats

### efficiency.json
```json
{
  "indexing": {
    "total_time_seconds": 3600.5,
    "documents_per_second": 27.3
  },
  "query_latency": {
    "total": {
      "total_time_seconds": 120.4,
      "num_queries": 500,
      "qps": 4.15,
      "mean_ms": 240.8
    }
  }
}
```

### results.json
```json
{
  "ndcg@10": 0.4523,
  "ndcg@100": 0.5234,
  "recall@10": 0.3421,
  "recall@100": 0.7832,
  "precision@10": 0.4100
}
```

### master_indexing_times.json (Phase 1)
```json
{
  "description": "TRUE indexing times measured with NO cache",
  "by_model": {
    "e5": {
      "biology": {"indexing_seconds": 1234.5},
      "economics": {"indexing_seconds": 987.2}
    }
  }
}
```

---

## Storage Requirements

### Disk Space
- **Phase 1 cache**: ~50-100 GB
- **Phase 3-8 caches**: ~200-300 GB total
- **Results**: ~5-10 GB

### Total: ~300-500 GB recommended

---

## Questions?

- **Model details**: See [MODELS.md](MODELS.md)
- **All fixes**: See [FIXES_SUMMARY.md](FIXES_SUMMARY.md)
- **Source code**: Check `src/run.py` and `src/retrievers.py`
