#!/bin/bash
#===============================================================================
# MASTER SCRIPT: Run All SIGIR 2026 Reproducibility Experiments
#
# USAGE:
#   ./run_all.sh              # Run all phases in order
#   ./run_all.sh 1            # Run only phase 1
#   ./run_all.sh 1 2 6        # Run phases 1, 2, and 6
#   ./run_all.sh --list       # List all phases
#
# RECOMMENDED ORDER:
#   1. Phase 1 first (builds cache, measures TRUE indexing time)
#   2. Then any other phase (they reuse or create separate caches)
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Phase descriptions
declare -A PHASE_DESC
PHASE_DESC[1]="Build Cache + TRUE Indexing Time (MUST RUN FIRST)"
PHASE_DESC[2]="Efficiency Profiling (Query Latency)"
PHASE_DESC[3]="Corpus Scaling Analysis"
PHASE_DESC[4]="Quantization Impact"
PHASE_DESC[5]="Context Length Sensitivity"
PHASE_DESC[6]="Reasoning Queries (GPT-4, Llama-3, etc.) - MAIN CONTRIBUTION"
PHASE_DESC[7]="Document Expansion"
PHASE_DESC[8]="Long Context Retrieval"
PHASE_DESC[9]="Hybrid Retrieval Fusion (BM25 + Dense) - NEW"
PHASE_DESC[10]="Robustness Evaluation (Query Perturbations) - NEW"
PHASE_DESC[11]="Cross-Benchmark Generalization (BEIR) - NEW"
PHASE_DESC[12]="Two-Stage Retrieval Pipelines - NEW"
PHASE_DESC[13]="Cost-Effectiveness Analysis - NEW"
PHASE_DESC[14]="Concurrency & Load Testing - NEW"

# Cache usage
declare -A PHASE_CACHE
PHASE_CACHE[1]="Creates NEW cache (cache/)"
PHASE_CACHE[2]="REUSES Phase 1 cache"
PHASE_CACHE[3]="Creates SEPARATE caches (cache_scaling/)"
PHASE_CACHE[4]="Creates SEPARATE caches (cache_quantization/)"
PHASE_CACHE[5]="Creates SEPARATE caches (cache_length/)"
PHASE_CACHE[6]="REUSES Phase 1 cache - NO re-indexing!"
PHASE_CACHE[7]="Creates SEPARATE caches (cache_expansion/)"
PHASE_CACHE[8]="Creates SEPARATE cache (cache_long/)"
PHASE_CACHE[9]="REUSES Phase 1 cache - EFFICIENT!"
PHASE_CACHE[10]="REUSES Phase 1 cache"
PHASE_CACHE[11]="Creates SEPARATE cache (cache_beir/)"
PHASE_CACHE[12]="REUSES Phase 1 cache"
PHASE_CACHE[13]="REUSES Phase 1 cache + API tracking"
PHASE_CACHE[14]="REUSES Phase 1 cache"

list_phases() {
    echo ""
    echo "Available Phases:"
    echo "================="
    for i in {1..14}; do
        echo "  Phase ${i}: ${PHASE_DESC[$i]}"
        echo "           Cache: ${PHASE_CACHE[$i]}"
        echo ""
    done
    echo "Phases 9-14 are NEW additions for comprehensive evaluation"
}

run_phase() {
    local phase=$1
    
    echo ""
    echo "###################################################################"
    echo "# PHASE ${phase}: ${PHASE_DESC[$phase]}"
    echo "# Cache: ${PHASE_CACHE[$phase]}"
    echo "###################################################################"
    echo ""
    
    local script="${SCRIPT_DIR}/phase${phase}_"
    
    case ${phase} in
        1) script="${SCRIPT_DIR}/phase1_build_cache.sh" ;;
        2) script="${SCRIPT_DIR}/phase2_efficiency.sh" ;;
        3) script="${SCRIPT_DIR}/phase3_scaling.sh" ;;
        4) script="${SCRIPT_DIR}/phase4_quantization.sh" ;;
        5) script="${SCRIPT_DIR}/phase5_length.sh" ;;
        6) script="${SCRIPT_DIR}/phase6_reasoning.sh" ;;
        7) script="${SCRIPT_DIR}/phase7_expansion.sh" ;;
        8) script="${SCRIPT_DIR}/phase8_long_context.sh" ;;
        9) script="${SCRIPT_DIR}/phase9_hybrid_fusion.sh" ;;
        10) script="${SCRIPT_DIR}/phase10_robustness.sh" ;;
        11) script="${SCRIPT_DIR}/phase11_cross_benchmark.sh" ;;
        12) script="${SCRIPT_DIR}/phase12_two_stage.sh" ;;
        13) script="${SCRIPT_DIR}/phase13_cost_analysis.sh" ;;
        14) script="${SCRIPT_DIR}/phase14_concurrency.sh" ;;
        *)
            echo "Unknown phase: ${phase}"
            return 1
            ;;
    esac
    
    if [ ! -f "${script}" ]; then
        echo "ERROR: Script not found: ${script}"
        return 1
    fi
    
    bash "${script}"
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

echo "=============================================="
echo "SIGIR 2026 REPRODUCIBILITY EXPERIMENTS"
echo "=============================================="
echo "Script directory: ${SCRIPT_DIR}"
echo "Start time: $(date)"
echo ""

# Handle --list argument
if [ "$1" == "--list" ] || [ "$1" == "-l" ]; then
    list_phases
    exit 0
fi

# Determine which phases to run
if [ $# -eq 0 ]; then
    PHASES="1 2 3 4 5 6 7 8 9 10 11 12 13 14"
    echo "Running ALL phases (1-14)"
else
    PHASES="$@"
    echo "Running phases: ${PHASES}"
fi

# Warn if Phase 1 not included but other phases need cache
if [[ ! " ${PHASES} " =~ " 1 " ]]; then
    DEPENDENT_PHASES="2 6 9 10 12 13 14"
    for dep_phase in ${DEPENDENT_PHASES}; do
        if [[ " ${PHASES} " =~ " ${dep_phase} " ]]; then
            echo ""
            echo "WARNING: Phase ${dep_phase} requires Phase 1 cache to exist!"
            echo "Make sure Phase 1 has been run previously."
            echo ""
            break
        fi
    done
fi

# Run requested phases
for phase in ${PHASES}; do
    run_phase ${phase}
done

echo ""
echo "=============================================="
echo "ALL REQUESTED PHASES COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results saved in: sigir_results/"
echo ""
echo "Key outputs:"
echo "  - sigir_results/phase1_cache_build/master_indexing_times.json"
echo "  - sigir_results/phase2_efficiency/efficiency_comparison.json"
echo "  - sigir_results/phase6_reasoning/reasoning_comparison.json"
echo "  - sigir_results/phase9_hybrid/fusion_comparison.json"
echo "  - sigir_results/phase10_robustness/robustness_summary.json"
echo "  - sigir_results/phase11_beir/cross_benchmark_comparison.json"
echo "  - sigir_results/phase13_cost/pareto_frontier.png"