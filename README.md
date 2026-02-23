# Systematic Audit of Confidence Signals in RAG

This repository contains the evaluation framework and codebase for ___ paper, "Systematic Audit of Confidence Signals in RAG".

## Project Overview

Modern RAG systems rely on retriever similarity scores for routing, abstention, and filtering. However, these scores are rarely validated as reliable confidence signals. This project systematically evaluates multiple retriever families (Sparse, Dense, LLM-based, Reasoning-Intensive) on the **BRIGHT benchmark** to determine:
1. Which signals (Raw Similarity, Score Gap, Entropy) best discriminate success from failure.
2. How reasoning-intensive tasks affect retriever calibration.
3. The "Reliability-Effectiveness Gap" in state-of-the-art retrievers.

## Core Metrics
We evaluate confidence reliability using **AUROC** (Area Under the Receiver Operating Characteristic), measuring how well a signal $S$ discriminates between successful retrieval (gold document in top-$k$) and unsuccessful retrieval.

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
└───original_code           # Original implementation code for retrievers
```

## Running the Experiments

The easiest way to run the experiments is to use the master script `scripts/run_all.sh`.

### Running Specific Phases

You can also run specific phases by providing the experiment numbers as arguments to the script. For example, to run only experiment 1, 2, and 6:

```bash
bash scripts/run_all.sh 1 2 6
```

## Caching Strategy

The scripts use a caching strategy to avoid re-computing document embeddings unnecessarily.

*   **Phase 1** builds the main cache in the `cache/` directory.

Please see `EXPERIMENTS.md` for a detailed explanation of the caching strategy.

## Expected Output

The results of the experiments will be saved in the `results/` directory, with a subdirectory for each phase. Key summary files will be generated in each experiment's output directory, as described in `EXPERIMENTS.md`.
