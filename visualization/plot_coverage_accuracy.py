"""
Plot risk-coverage / coverage-accuracy curves for all retrievers.

Generates:
    results/rag_exp/plots/

For each retriever:
    - max_score abstention curve
    - gap_ctx abstention curve
    - gap_25 abstention curve

Coverage:
    fraction of questions answered (not abstained)

Accuracy:
    accuracy on answered questions only

Interpretation:
    Better uncertainty signals maintain higher accuracy
    as coverage increases.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

RAG_EXP_DIR = REPO_ROOT / "results" / "rag_exp"
PLOT_DIR = RAG_EXP_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS = [
    ("max_score", "Max Score"),
    ("gap_25", "Gap@25"),
]

COLORS = {
    "max_score": "#D55E00",   # muted vermillion
    "gap_25": "#009E73",      # bluish green
}


def compute_curve(df, signal, num_points=200):
    """
    Computes coverage-accuracy curve.

    Sort descending by confidence signal.
    Keep top-X% as answered.
    """

    d = df[[signal, "correct"]].dropna().copy()

    d = d.sort_values(signal, ascending=False).reset_index(drop=True)

    n = len(d)

    coverages = []
    accuracies = []

    # evaluate many operating points
    for frac in np.linspace(0.01, 1.0, num_points):

        keep = max(1, int(frac * n))

        subset = d.iloc[:keep]

        coverage = keep / n
        accuracy = subset["correct"].mean()

        coverages.append(coverage)
        accuracies.append(accuracy)

    return np.array(coverages), np.array(accuracies)


def main():

    csv_files = sorted(
        p for p in RAG_EXP_DIR.glob("*_gpt4o.csv")
    )

    if not csv_files:
        print("No CSV files found.")
        return

    all_rows = []

    for csv_path in csv_files:

        print(f"\nProcessing {csv_path.name}")

        df = pd.read_csv(csv_path)

        required_cols = [
            "judgment",
            "max_score",
            "gap_ctx",
            "gap_25",
        ]

        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            print(f"  [skip] missing columns: {missing}")
            continue

        if "retriever" not in df.columns:
            inferred = csv_path.stem.replace("rag_master_", "")
            df["retriever"] = inferred

        df["correct"] = (
            df["judgment"]
            .astype(str)
            .str.upper()
            .eq("CORRECT")
            .astype(int)
        )

        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)

    retrievers = sorted(full_df["retriever"].unique())

    # ---------------------------------------------------------
    # individual retriever plots
    # ---------------------------------------------------------

    for retriever in retrievers:

        sub = full_df[full_df["retriever"] == retriever].copy()

        plt.figure(figsize=(7, 5))

        for signal, label in SIGNALS:

            cov, acc = compute_curve(sub, signal)

            plt.plot(
                cov,
                acc,
                label=label,
                color=COLORS[signal],
            )

        baseline_acc = sub["correct"].mean()

        plt.axhline(
            baseline_acc,
            linestyle="--",
            alpha=0.5,
            label=f"Baseline ({baseline_acc:.3f})"
        )

        plt.xlabel("Coverage")
        plt.ylabel("Accuracy")
        plt.title(f"Coverage vs Accuracy — {retriever}")
        plt.legend()
        plt.grid(alpha=0.3)

        out_path = PLOT_DIR / f"{retriever}_coverage_accuracy.png"

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"  saved: {out_path.name}")

    # ---------------------------------------------------------
    # aggregate average plot
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 6))

    for signal, label in SIGNALS:

        curves = []

        for retriever in retrievers:

            sub = full_df[full_df["retriever"] == retriever]

            cov, acc = compute_curve(sub, signal)

            curves.append(acc)

        mean_acc = np.mean(curves, axis=0)

        plt.plot(
            cov,
            mean_acc,
            linewidth=2,
            label=label,
            color=COLORS[signal],
        )

    overall_baseline = full_df["correct"].mean()

    plt.axhline(
        overall_baseline,
        linestyle="--",
        alpha=0.5,
        label=f"Baseline ({overall_baseline:.3f})"
    )

    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.title("Average Coverage vs Accuracy Across Retrievers")
    plt.legend()
    plt.grid(alpha=0.3)

    agg_path = PLOT_DIR / "ALL_average_coverage_accuracy.png"

    plt.tight_layout()
    plt.savefig(agg_path, dpi=300)
    plt.close()

    print(f"\nSaved aggregate plot:\n{agg_path}")


if __name__ == "__main__":
    main()