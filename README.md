<div align="center">

# 🔍 The Magnitude Mirage

### Rethinking Confidence for Reasoning-Intensive Retrieval

[![Paper](https://img.shields.io/badge/📄_Paper-EMNLP_2026-blue)](https://aclanthology.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Findings](#-key-findings)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Metrics](#-metrics)
- [Results](#-results)
- [Citation](#-citation)

---

## 🧠 Overview

Reliable Retrieval-Augmented Generation (RAG) requires detecting when retrieval has failed. Many production RAG systems use simple similarity thresholding, treating raw retrieval scores as calibrated confidence signals.

We expose the **Magnitude Mirage**: as queries transition from simple semantic matching to complex reasoning, neural retrievers often assign high similarity scores to semantically related but constraint-violating documents — causing magnitude thresholding to degrade to **near-random abstention performance**.

We conduct a large-scale empirical study evaluating **6 zero-cost QPP metrics** across **11 retrieval architectures** and **28 datasets**, benchmarked across three cognitive tiers:

| Tier | Description |
|------|-------------|
| 🟢 **Semantic Matching** | Standard similarity-based retrieval |
| 🟡 **Logical Reasoning** | Constraint-aware retrieval |
| 🔴 **Temporal Reasoning** | Time-sensitive retrieval |

---

## 🎯 Key Findings

1. 🔬 **Magnitude Mirage** — Raw retrieval score thresholds are unreliable indicators of success, especially for reasoning-intensive queries.
2. 📊 **Large-Scale QPP Evaluation** — Six zero-cost QPP metrics evaluated across 11 architectures and 28 datasets spanning semantic, logical, and temporal reasoning.
3. ⚡ **Zero-Cost Alternative** — Score Gap (s₁−sₖ) and LSMV improve abstention AUROC by up to **+0.16** with no additional inference, retraining, or latency.

---

## 📁 Repository Structure

```
📦 magnitude-mirage
├── 📂 src/                          # Core source code
│   ├── run.py                       # Main experiment runner
│   ├── retrievers.py                # Retrieval model implementations
│   ├── evaluate_qpp.py              # QPP metric evaluation
│   ├── compute_auroc.py             # AUROC computation
│   ├── aggregate_auroc.py           # Aggregate AUROC results
│   ├── coverage_accuracy.py         # Coverage-accuracy analysis
│   ├── summarize_qpp.py             # QPP result summarization
│   ├── merge_scores.py              # Score merging utilities
│   ├── prompts.py                   # Prompt templates
│   ├── rag_exp.py                   # RAG experiment pipeline
│   ├── rag_exp_vllm.py              # RAG experiments with vLLM
│   ├── summarize_rag_vllm.py        # Summarize vLLM RAG results
│   └── utils/
│       ├── eval_util.py             # Evaluation utilities
│       └── perturbations.py         # Query perturbation methods
├── 📂 configs/                      # Model configuration files
├── 📂 scripts/                      # Shell scripts for running experiments
│   ├── run_qpp_eval.sh              # Run full QPP evaluation pipeline
│   ├── run_all_retrievers.sh        # Run all retrieval models
│   ├── build_bright_cache.sh        # Build BRIGHT benchmark cache
│   └── launch_vllm.sh              # Launch vLLM server
├── 📂 visualization/                # Plotting and figure generation
│   ├── plot_qpp_metrics.py          # QPP metric visualizations
│   ├── plot_coverage_accuracy.py    # Coverage-accuracy plots
│   ├── plot_performance_line.py     # Performance line plots
│   ├── plot_decision_boundary.py    # Decision boundary plots
│   └── generate_table.py           # LaTeX table generation
├── 📂 results/                      # Experiment outputs
│   ├── qpp_results/                 # QPP evaluation results
│   ├── scores_results/              # Cached similarity scores
│   └── rag_exp/                     # RAG experiment results
├── 📂 baselines/                    # Baseline implementations
└── 📂 docs/                         # Documentation & supplementary material
    └── reviews/                     # Peer review materials
```

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.9
- PyTorch
- Transformers

### Installation

```bash
pip install -r requirements.txt
```

### Running the Full Pipeline

The easiest way to run the entire evaluation pipeline:

```bash
bash scripts/run_qpp_eval.sh
```

This computes all QPP metrics across all models and all datasets.

### 💾 Caching Strategy

Similarity scores are pre-computed and cached at:

```
results/scores_results/results_{benchmark}/{model}/{dataset}
```

This avoids re-computing document similarity scores on subsequent runs.

---

## 📏 Metrics

We evaluate QPP signal quality using three complementary measures:

| Metric | Description |
|--------|-------------|
| **AUROC** 📈 | Primary metric — measures ability to distinguish successful vs. failed retrieval across all thresholds |
| **Pearson _r_** 📉 | Linear correlation between predicted QPP confidence and observed NDCG@k |
| **Spearman _ρ_** 📊 | Monotonic rank correlation — robust to non-linear scaling effects |

All metrics are computed per dataset and reported as macro-averages within each benchmark.

---

## 📊 Results

Experiment outputs are saved to:

```
results/qpp_results/{benchmark}/{model}_qpp_results.json
```

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{holdcroft2026magnitude,
  title     = {The Magnitude Mirage: Rethinking Confidence for Reasoning-Intensive Retrieval},
  author    = {Holdcroft, Jamie and Abdallah, Abdelrahman and Jatowt, Adam},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026}
}
```

---

<div align="center">

⭐ If you find this repository helpful, please consider giving it a star!

</div>
