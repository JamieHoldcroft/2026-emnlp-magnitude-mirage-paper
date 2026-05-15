#!/usr/bin/env bash
# Sweep RAG abstention experiment across all retrievers with cached BRIGHT scores.
# Generator: local vLLM. Judge: Azure GPT-4o.
set -euo pipefail

REPO=/data/fs201059/aa17626/2026-emnlp-magnitude-mirage-paper
source /data/fs201059/aa17626/FinCodeAgent/.venv/bin/activate
source "$REPO/scripts/.azure.env"

cd "$REPO"

# Order: cheap baselines first, reasoning encoders last.
RETRIEVERS=(
  bm25
  sbert
  contriever
  e5
  bge
  sf
  inst-l
  inst-xl
  qwen
  qwen2
  nomic
  rader
  reasonir
  diver-retriever
)

CONCURRENCY="${CONCURRENCY:-16}"
TOP_K_CTX="${TOP_K_CTX:-5}"
TOP_K_GAP25="${TOP_K_GAP25:-25}"

mkdir -p results/rag_exp/per_retriever_logs

for r in "${RETRIEVERS[@]}"; do
  TAG="${r}_qwen36_gpt4o"
  CSV="results/rag_exp/rag_master_${TAG}.csv"
  LOG="results/rag_exp/per_retriever_logs/${TAG}.log"
  echo "==================== ${r} ===================="
  python -u src/rag_exp_vllm.py \
      --retriever "$r" \
      --output-tag "$TAG" \
      --top-k-ctx "$TOP_K_CTX" \
      --top-k-gap25 "$TOP_K_GAP25" \
      --concurrency "$CONCURRENCY" \
      2>&1 | tee "$LOG"
  echo "saved -> $CSV"
done

echo "==== DONE ===="
