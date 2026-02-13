# SIGIR 2026 Reproducibility: Efficiency, Effectiveness, and Reasoning Overhead in LLM-Based Retrieval

This repository contains the code and scripts for the SIGIR 2026 Reproducibility Track paper, "Beyond the Score: Reproducibility Study of Efficiency, Effectiveness, and Reasoning Overhead in LLM-Based Retrieval".

## Project Overview

This study provides a comprehensive analysis of the trade-offs between effectiveness and efficiency in LLM-based retrieval. We reproduce and extend the findings of the BRIGHT benchmark by systematically evaluating various retrieval models across 12 tasks, focusing on indexing time, query latency, and performance under different operational constraints.

## Directory Structure

```
.
├───data/                   # Placeholder for data
├───docs/                   # Detailed documentation
│   └───SIGIR2026_Bash_Scripts.md
├───results/                # Raw results from experiments
├───scripts/                # Bash scripts to run experiments
│   ├───phase1_build_cache.sh
│   ├───...
│   └───run_all.sh
├───src/                    # Python source code
│   ├───run.py              # Main script to run experiments
│   ├───retrievers.py       # Retrieval model implementations
│   └───utils/
│       └───eval_util.py
└───SIGIR2026.md           # Experimental design document
```

## Running the Experiments

The easiest way to run the experiments is to use the master script `scripts/run_all.sh`.

### Running All Phases

To run all 8 experimental phases in the recommended order, simply execute the `run_all.sh` script:

```bash
bash scripts/run_all.sh
```

**Warning:** Phase 1 will build the cache from scratch and can take a significant amount of time (24-48 hours on an A100-80GB GPU).

### Running Specific Phases

You can also run specific phases by providing the phase numbers as arguments to the script. For example, to run only Phase 1, 2, and 6:

```bash
bash scripts/run_all.sh 1 2 6
```

### Listing Available Phases

To see a list of all available phases and their descriptions, use the `--list` or `-l` flag:

```bash
bash scripts/run_all.sh --list
```

## Caching Strategy

The scripts use a caching strategy to avoid re-computing document embeddings unnecessarily.

*   **Phase 1** builds the main cache in the `cache/` directory.
*   **Phase 2 and 6** reuse the cache from Phase 1.
*   Other phases (3, 4, 5, 7, 8) create separate caches (e.g., `cache_scaling/`, `cache_quantization/`) because the document embeddings are different for these experiments.

Please see `docs/SIGIR2026_Bash_Scripts.md` for a detailed explanation of the caching strategy.

## Expected Output

The results of the experiments will be saved in the `sigir_results/` directory, with a subdirectory for each phase. Key summary files will be generated in each phase's output directory, as described in `SIGIR2026.md`.
