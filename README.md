# RAG Retrieval Confidence Evaluation Comparisons

This repository contains the code and scripts for ___ paper, "___".

## Project Overview

This study ___.

## Directory Structure

```
.
├───results/                # Raw results from experiments
├───scripts/                # Bash scripts to run experiments
│   ├───exp1.sh
│   ├───...
|   └───run_all.sh
├───src/                    # Python source code
│   ├───run.py              # Main script to run experiments
│   ├───retrievers.py       # Retrieval model implementations
│   └───utils/
│       └───eval_util.py
└─── 
```

## Running the Experiments

The easiest way to run the experiments is to use the master script `scripts/run_all.sh`.

### Running All Phases

To run all experiments in the recommended order, simply execute the `run_all.sh` script:

```bash
bash scripts/run_all.sh
```

### Running Specific Phases

You can also run specific phases by providing the experiment numbers as arguments to the script. For example, to run only experiment 1, 2, and 6:

```bash
bash scripts/run_all.sh 1 2 6
```

## Caching Strategy

The scripts use a caching strategy to avoid re-computing document embeddings unnecessarily.

*   **Phase 1** builds the main cache in the `cache/` directory.

Please see `___.md` for a detailed explanation of the caching strategy.

## Expected Output

The results of the experiments will be saved in the `results/` directory, with a subdirectory for each phase. Key summary files will be generated in each experiment's output directory, as described in `___`.
