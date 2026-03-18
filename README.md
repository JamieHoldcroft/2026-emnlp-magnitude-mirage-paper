# Systematic Audit of Confidence Signals in RAG

This repository contains the evaluation framework and codebase for the paper: **'The Magnitude Mirage: Rethinking Confidence for Reasoning-Intensive Retrieval'**

## Project Overview

Reliable Retrieval-Augmented Generation (RAG) requires detecting when retrieval has failed. Many production RAG systems implement simple similarity thresholding, treating raw retrieval scores as calibrated confidence signals. We expose the *Magnitude Mirage*: as queries transition from simple semantic matching to complex reasoning, neural retrievers often assign high similarity scores to semantically related but constraint-violating documents, causing magnitude thresholding to degrade to near-random abstention performance. To address this without relying on computationally expensive LLM-based evaluators, we conduct a large-scale empirical study evaluating six zero-cost Query Performance Prediction (QPP) metrics across 11 retrieval architectures and 28 datasets. Benchmarking across three cognitive tiers -- semantic matching, logical reasoning, and temporal reasoning -- we show that score-distribution signals consistently outperform raw magnitude thresholding. In particular, Score Gap ($s_1-s_k$) and Linearized Score Magnitude and Variance (LSMV) improve abstention AUROC by up to 0.16 across models and reasoning tiers. Because these signals operate solely on the score distribution already produced during retrieval, they do not require additional model inference, retraining, or latency, providing a practical zero-cost replacement for magnitude thresholding in deployed RAG systems.

The core objectives of this framework are to determine:
1. We identify and systematically characterize the *Magnitude Mirage*, showing that raw retrieval score thresholds are unreliable indicators of retrieval success, particularly for reasoning-intensive queries.
2. We present a large-scale evaluation of six zero-cost QPP metrics across eleven retrieval architectures and twenty-eight datasets spanning semantic, logical, and temporal reasoning retrieval tasks.
3. We demonstrate that variance-based QPP metrics provide a simple and computationally free alternative to magnitude thresholding, offering more reliable abstention signals for modern RAG pipelines.


## Core Metrics

We evaluate the predictive quality of QPP signals using three complementary statistical measures:

**AUROC (Area Under the Receiver Operating Characteristic Curve).**
This serves as our primary evaluation metric and simulates the RAG abstention decision. AUROC measures the ability of a QPP signal to distinguish between successful and failed retrieval instances across all possible decision thresholds, rather than relying on a fixed confidence threshold.

**Pearson Correlation Coefficient (r).**
Measures the linear correlation between the predicted QPP confidence value and the observed retrieval effectiveness ($NDCG@k$).

**Spearman Rank Correlation ($\rho$).**
Measures the monotonic relationship between predicted confidence and retrieval effectiveness, providing robustness to non-linear scaling effects.

All metrics are computed per dataset and reported as macro-averages across datasets within each benchmark.


## Directory Structure

```
.
├───results/                # Raw results from experiments
│   ├───qpp_results/        # Directory containing all qpp experiment results
│   ├───scores_results/     # Directory containing the similarity scores information for all models on all datasets
├───scripts/                # Bash scripts to run experiments
│   ├───run_qpp_eval.sh
│   ├───...
├───src/                    # Python source code
│   ├───run.py              # Main script to run experiments
│   ├───retrievers.py       # Retrieval model implementations
│   └───utils/
│       └───eval_util.py
├───visualise_scripts/      # Code that produces the visualisations as displayed in the paper
└───original_code           # Original implementation code for baselines
```

## Running the Experiments
The easiest and most efficient way to run the entire evaluation pipeline is to use the script:
``` 
bash scripts/run_qpp_eval.sh
```
This script will compute all evaluated QPP metrics across all models and all datasets.

## Caching Strategy

The scripts use a caching strategy to avoid re-computing document similarity scores unnecessarily.

Within ```results/scores_results/results_{benchmark}/{model}/{dataset}```, the cached similarity scores are pre-computed and stored for fast experiment evaluation.

## Expected Output

The results of the experiments will be saved in the `results/qpp_results/{benchmark}/{model}_qpp_results.json` directory. 
