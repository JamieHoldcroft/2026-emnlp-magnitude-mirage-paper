# Systematic Audit of Confidence Signals in RAG

This repository contains the evaluation framework and codebase for the paper: **'The Magnitude Mirage: Rethinking Confidence for Reasoning-Intensive Retrieval'**

## Project Overview

Modern RAG systems rely on retriever similarity scores for routing, abstention, and filtering. However, these scores are rarely validated as reliable confidence signals. This project systematically evaluates multiple retriever families (Sparse, Dense, LLM-based, Reasoning-Intensive) across a spectrum of cognitive complexities, from standard factoid search **(BEIR benchmark)** to reasoning-intensive tasks **(BRIGHT benchmark)**. 

The core objectives of this framework are to determine:
1. Which signals (Raw Similarity, Score Gap, Top-*k* Entropy etc) best discriminate query-level success from failure.
2. How the complexity of tasks affects retriever calibration.
3. The "Reliability-Effectiveness Gap" in state-of-the-art retrievers.

## Core Metrics
We evaluate confidence reliability using **AUROC** (Area Under the Receiver Operating Characteristic), measuring how well a confidence signal $S$ discriminates between successful retrieval (gold document in top-$k$) and unsuccessful retrieval.

$$AUROC = \int_{0}^{1} TPR(FPR^{-1}(u)) du$$

## Directory Structure

```
.
├───results/                # Raw results from experiments
├───scripts/                # Bash scripts to run experiments
│   ├───exp1.sh
│   ├───...
│   ├───run_all.sh
├───src/                    # Python source code
│   ├───run.py              # Main script to run experiments
│   ├───retrievers.py       # Retrieval model implementations
│   └───utils/
│       └───eval_util.py
└───original_code           # Original implementation code for baselines
```

## Running the Experiments
The easiest and most efficient way to run the entire evaluation pipeline is to use the master script:
``` 
bash scripts/run_all.sh
```

### Running Specific Phases

You can also run specific experiments by providing the experiment numbers as arguments to the script. For example, to run only experiment 1, 2, and 6:

```
bash scripts/run_all.sh 1 2 6
```

## Caching Strategy

The scripts use a caching strategy to avoid re-computing document embeddings unnecessarily.

*   **Phase 1** builds the main cache in the `cache/` directory for BRIGHT.
*   **Phase 2** builds the main cache in the `cache/` directory for BEIR cross-benchmark evaluation.

Subsequent evaluation phases reuse these caches to calculate confidence metrics without requiring re-indexing. For a detailed breakdown of dependencies and execution order, please refer to `EXPERIMENTS.md`.

## Expected Output

The results of the experiments will be saved in the `results/` directory, with a subdirectory for each confidence signal experiment. Key summary files will be generated in each experiment's output directory, as described in `EXPERIMENTS.md`.
