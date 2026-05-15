"""
Aggregate per-retriever RAG abstention CSVs into paper-style tables
with bootstrap 95% confidence intervals.

For each retriever × signal we report:
  AUROC predicting *answer correctness* (downstream RAG, judge=GPT-4o)
  AUROC predicting *retrieval success* (NDCG@k > 0, paper-style)

Signals: max_score (s1), gap_ctx (s1 - s_k_ctx), gap_25 (s1 - s_25)

Tables generated:
  T1: retriever x {Max, Gap_ctx, Gap_25}  AUROC for ANSWER correctness (with CIs)
  T2: retriever x {Max, Gap_ctx, Gap_25}  AUROC for RETRIEVAL_SUCCESS_5
  T3: retriever x {Max, Gap_ctx, Gap_25}  AUROC for RETRIEVAL_SUCCESS_25
  T4: per-task breakdown (one block per retriever)
  T5: selective-prediction operating points (cov 10/25/50/75% accuracies)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = REPO_ROOT / "results" / "rag_exp"

RETRIEVER_ORDER = [
    "bm25", "sbert", "contriever", "e5", "bge", "sf",
    "inst-l", "inst-xl", "qwen", "qwen2", "nomic",
    "rader", "reasonir", "diver-retriever",
]
SIGNALS = ["max_score", "gap_ctx", "gap_25"]
SIGNAL_DISPLAY = {"max_score": "MaxScore", "gap_ctx": "Gap@k_ctx", "gap_25": "Gap@25"}


def safe_auroc(y, scores):
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return roc_auc_score(y, scores)


def bootstrap_auroc(y, scores, n_boot=1000, seed=0):
    y = np.asarray(y)
    scores = np.asarray(scores)
    point = safe_auroc(y, scores)
    if np.isnan(point) or len(y) < 5:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        out.append(roc_auc_score(yb, scores[idx]))
    if not out:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(out, [2.5, 97.5])
    return point, lo, hi


def selective_curve_at(df, signal, coverage_pcts):
    """Sort by signal desc, return accuracy among the top-coverage% answered queries."""
    sorted_df = df.sort_values(by=signal, ascending=False).reset_index(drop=True)
    n = len(sorted_df)
    correct = (sorted_df["judgment"] == "CORRECT").astype(int).cumsum().values
    out = {}
    for cov in coverage_pcts:
        k = max(1, round(n * cov / 100))
        out[cov] = correct[k - 1] / k * 100
    return out


def random_curve_at(df, coverage_pcts, seed=0):
    rng = np.random.default_rng(seed)
    shuffled = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    n = len(shuffled)
    correct = (shuffled["judgment"] == "CORRECT").astype(int).cumsum().values
    return {cov: correct[max(1, round(n * cov / 100)) - 1] /
                  max(1, round(n * cov / 100)) * 100 for cov in coverage_pcts}


def load_all(tag_suffix):
    rows = []
    found = []
    for r in RETRIEVER_ORDER:
        p = RAG_DIR / f"rag_master_{r}_{tag_suffix}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["retriever"] = r
            rows.append(df)
            found.append((r, len(df)))
    if not rows:
        raise SystemExit(f"No CSVs found matching tag suffix '{tag_suffix}'")
    return pd.concat(rows, ignore_index=True), found


def fmt_ci(point, lo, hi):
    if any(np.isnan([point, lo, hi])):
        return "  --  "
    return f"{point:.3f} [{lo:.3f},{hi:.3f}]"


def table_auroc(df, target_col, label, n_boot=1000):
    print(f"\n========== {label}  (AUROC, 95% bootstrap CI, N={n_boot}) ==========")
    header = f"{'retriever':<18s}" + "".join(f"  {SIGNAL_DISPLAY[s]:^22s}" for s in SIGNALS)
    print(header)
    print("-" * len(header))
    macro_rows = []
    for r in RETRIEVER_ORDER:
        sub = df[df["retriever"] == r]
        if sub.empty:
            continue
        cells = []
        macro_entry = {"retriever": r, "n": len(sub)}
        for s in SIGNALS:
            point, lo, hi = bootstrap_auroc(sub[target_col].values, sub[s].values, n_boot=n_boot)
            cells.append(fmt_ci(point, lo, hi))
            macro_entry[s] = point
        print(f"{r:<18s}" + "".join(f"  {c:^22s}" for c in cells))
        macro_rows.append(macro_entry)
    if macro_rows:
        macro_df = pd.DataFrame(macro_rows)
        print("-" * len(header))
        means = {s: macro_df[s].mean() for s in SIGNALS}
        print(f"{'MACRO_MEAN':<18s}" + "".join(f"  {means[s]:^22.3f}" for s in SIGNALS))
    return macro_rows


def per_task_breakdown(df, target_col, label):
    print(f"\n========== Per-task AUROC ({label}) ==========")
    tasks = sorted(df["task"].unique())
    rows = []
    for r in RETRIEVER_ORDER:
        for task in tasks:
            sub = df[(df["retriever"] == r) & (df["task"] == task)]
            if sub.empty:
                continue
            entry = {"retriever": r, "task": task, "n": len(sub)}
            for s in SIGNALS:
                entry[s] = safe_auroc(sub[target_col].values, sub[s].values)
            rows.append(entry)
    pt = pd.DataFrame(rows)
    return pt


def operating_points_table(df, coverages=(10, 25, 50, 75, 100)):
    print(f"\n========== Selective Prediction — accuracy at coverage% (answered queries) ==========")
    cols = ["retriever", "n", "baseline"]
    for cov in coverages:
        for sig in SIGNALS + ["random"]:
            cols.append(f"{sig}@{cov}")
    rows = []
    for r in RETRIEVER_ORDER:
        sub = df[df["retriever"] == r]
        if sub.empty:
            continue
        baseline = (sub["judgment"] == "CORRECT").mean() * 100
        entry = {"retriever": r, "n": len(sub), "baseline": round(baseline, 2)}
        rand_curve = random_curve_at(sub, coverages, seed=0)
        for sig in SIGNALS:
            curve = selective_curve_at(sub, sig, coverages)
            for cov in coverages:
                entry[f"{sig}@{cov}"] = round(curve[cov], 2)
        for cov in coverages:
            entry[f"random@{cov}"] = round(rand_curve[cov], 2)
        rows.append(entry)
    op_df = pd.DataFrame(rows)
    # Pretty-print a compact view: per-cov columns side by side for headline signals
    compact_cols = ["retriever", "n", "baseline"]
    for cov in coverages:
        compact_cols += [f"max_score@{cov}", f"gap_ctx@{cov}", f"gap_25@{cov}", f"random@{cov}"]
    print(op_df[compact_cols].to_string(index=False))
    return op_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-suffix", default="qwen36_gpt4o",
                        help="Files matched: rag_master_<retriever>_<tag-suffix>.csv")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out-prefix", default="rag_summary")
    args = parser.parse_args()

    df, found = load_all(args.tag_suffix)
    print("Retrievers found:")
    for r, n in found:
        print(f"  {r}: {n} rows")

    # Coerce judgment to string just in case
    df["judgment"] = df["judgment"].astype(str)
    df["judgment_int"] = (df["judgment"] == "CORRECT").astype(int)

    # ---- Headline tables ----
    ans = table_auroc(df, "judgment_int", "ANSWER correctness", n_boot=args.n_boot)
    rs5 = table_auroc(df, "retrieval_success_5", "RETRIEVAL_SUCCESS@5 (NDCG@5>0)", n_boot=args.n_boot)
    rs25 = table_auroc(df, "retrieval_success_25", "RETRIEVAL_SUCCESS@25 (NDCG@25>0)", n_boot=args.n_boot)

    # ---- Operating points ----
    op_df = operating_points_table(df)

    # ---- Per-task ----
    pt_ans = per_task_breakdown(df, "judgment_int", "ANSWER correctness")

    # ---- Save artifacts ----
    out_dir = RAG_DIR
    pd.DataFrame(ans).to_csv(out_dir / f"{args.out_prefix}_answer_auroc.csv", index=False)
    pd.DataFrame(rs5).to_csv(out_dir / f"{args.out_prefix}_retrieval5_auroc.csv", index=False)
    pd.DataFrame(rs25).to_csv(out_dir / f"{args.out_prefix}_retrieval25_auroc.csv", index=False)
    op_df.to_csv(out_dir / f"{args.out_prefix}_operating_points.csv", index=False)
    pt_ans.to_csv(out_dir / f"{args.out_prefix}_per_task_answer_auroc.csv", index=False)

    summary = {
        "n_retrievers": len(found),
        "retrievers": [r for r, _ in found],
        "n_queries_total": int(len(df)),
        "macro_answer_auroc": {
            s: float(np.nanmean([row[s] for row in ans])) for s in SIGNALS
        },
        "macro_retrieval5_auroc": {
            s: float(np.nanmean([row[s] for row in rs5])) for s in SIGNALS
        },
        "macro_retrieval25_auroc": {
            s: float(np.nanmean([row[s] for row in rs25])) for s in SIGNALS
        },
    }
    with open(out_dir / f"{args.out_prefix}_macro.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMacro summary -> {out_dir / f'{args.out_prefix}_macro.json'}")


if __name__ == "__main__":
    main()
