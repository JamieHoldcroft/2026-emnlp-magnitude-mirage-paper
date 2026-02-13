#!/usr/bin/env python3
"""
Figure 4: Corpus Scaling Curves (E5 model only)

Two subplots stacked vertically:
  (a) Indexing Time (s) vs corpus size
  (b) nDCG@10 vs corpus size

Supports *_size_full and plots it AFTER 100K as an extra point labeled "FULL".
We do this by mapping "full" to a numeric x-position (FULL_X).

Structure:
  phase3_scaling/{task}_size_{N}/{task}_size_{N}_e5_long_False/
    - efficiency.json -> indexing time, query latency
    - results.json    -> nDCG@10

Usage:
  python figure4_scaling.py --base_dir /leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

# Put FULL after 100K on the (log) x-axis
FULL_X = 200000  # you can change to 150000 if you want it closer to 100K

CORPUS_SIZES = [1000, 5000, 10000, 25000, 50000, 100000, FULL_X]
SIZE_LABELS = ["1K", "5K", "10K", "25K", "50K", "100K", "FULL"]

TASKS = [
    "aops", "biology", "earth_science", "economics",
    "leetcode", "pony", "psychology", "robotics",
    "stackoverflow", "sustainable_living",
    "theoremqa_questions", "theoremqa_theorems",
]

TASK_DISPLAY = {
    "aops": "AoPS", "biology": "Biology", "earth_science": "Earth Sci.",
    "economics": "Economics", "leetcode": "LeetCode", "pony": "Pony",
    "psychology": "Psychology", "robotics": "Robotics",
    "stackoverflow": "StackOF", "sustainable_living": "Sust. Living",
    "theoremqa_questions": "TheoremQA-Q", "theoremqa_theorems": "TheoremQA-T",
}

TASK_COLORS = {
    "aops": "#D64045", "biology": "#1B6B93", "earth_science": "#E8963E",
    "economics": "#5B8C5A", "leetcode": "#7B68EE", "pony": "#FF6B9D",
    "psychology": "#20B2AA", "robotics": "#DAA520", "stackoverflow": "#CD853F",
    "sustainable_living": "#4682B4", "theoremqa_questions": "#9370DB",
    "theoremqa_theorems": "#708090",
}

TASK_MARKERS = {
    "aops": "o", "biology": "s", "earth_science": "^", "economics": "D",
    "leetcode": "v", "pony": "P", "psychology": "X", "robotics": "p",
    "stackoverflow": "h", "sustainable_living": "<", "theoremqa_questions": ">",
    "theoremqa_theorems": "d",
}

# ============================================================
# DATA
# ============================================================

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def discover_subfolders(path):
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def extract_scaling_data(base_dir):
    """
    Extract scaling data for E5 across all tasks and corpus sizes.
    Returns:
      {task: {size: {indexing_time, query_latency_mean, qps, ndcg10}}}
    Where size is an int x-position. "full" is mapped to FULL_X.
    """
    phase_dir = os.path.join(base_dir, "phase3_scaling")
    if not os.path.isdir(phase_dir):
        print(f"[ERROR] Missing: {phase_dir}")
        return {}

    data = defaultdict(dict)

    for size_folder in discover_subfolders(phase_dir):
        size_path = os.path.join(phase_dir, size_folder)
        parts = size_folder.rsplit("_size_", 1)
        if len(parts) != 2:
            continue
        task, size_str = parts

        # Map "full" to numeric x-position so it plots after 100K
        if size_str == "full":
            numeric_size = FULL_X
        else:
            try:
                numeric_size = int(size_str)
            except ValueError:
                continue

        # Look for e5 results inside subfolders
        for run_folder in discover_subfolders(size_path):
            if "e5" not in run_folder.lower():
                continue

            run_lower = run_folder.lower()
            if not (
                "_e5_" in run_lower
                or run_lower.endswith("_e5")
                or run_folder.startswith(f"{task}_size_{size_str}_e5")
                or run_folder.startswith(f"{task}_e5")
            ):
                continue

            run_path = os.path.join(size_path, run_folder)

            entry = {}

            # efficiency.json
            edata = load_json(os.path.join(run_path, "efficiency.json"))
            if edata:
                if "indexing" in edata:
                    entry["indexing_time"] = edata["indexing"].get("total_time_seconds", None)
                    entry["docs_per_sec"] = edata["indexing"].get("documents_per_second", None)
                if "query_latency" in edata:
                    ql = edata["query_latency"]
                    if "total" in ql:
                        entry["query_latency_mean"] = ql["total"].get("mean_ms", None)
                        entry["qps"] = ql["total"].get("qps", None)
                    elif "mean_ms" in ql:
                        entry["query_latency_mean"] = ql.get("mean_ms", None)
                        entry["qps"] = ql.get("qps", None)

            # results.json
            rdata = load_json(os.path.join(run_path, "results.json"))
            if rdata:
                entry["ndcg10"] = rdata.get("NDCG@10", None)

            if entry:
                data[task][numeric_size] = entry
                break  # found e5 for this task/size

    return dict(data)

# ============================================================
# PLOT
# ============================================================

def plot_scaling(data, output_path):
    """
    Two subplots stacked vertically:
      Top: Indexing time vs corpus size (per task)
      Bottom: nDCG@10 vs corpus size (per task)
    Includes FULL as an extra x point.
    """
    available_tasks = [t for t in TASKS if t in data]
    numeric_sizes = CORPUS_SIZES[:]  # already ordered with FULL at the end

    print(f"  Tasks with data: {available_tasks}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 5.0), sharex=True)
    fig.patch.set_facecolor("white")

    # ---- TOP: Indexing Time ----
    ax1.set_facecolor("#FAFAFA")
    for task in available_tasks:
        sizes, idx_times = [], []
        for s in numeric_sizes:
            if s in data[task] and data[task][s].get("indexing_time") is not None:
                sizes.append(s)
                idx_times.append(data[task][s]["indexing_time"])
        if sizes:
            ax1.plot(
                sizes, idx_times,
                color=TASK_COLORS.get(task, "#333"),
                marker=TASK_MARKERS.get(task, "o"),
                markersize=4,
                linewidth=1.2,
                label=TASK_DISPLAY.get(task, task),
                markeredgecolor="white",
                markeredgewidth=0.3,
            )

    ax1.set_ylabel("Indexing Time (s)", fontsize=9, fontweight="bold", color="#1a1a1a")
    ax1.tick_params(axis="both", labelsize=7, colors="#333")
    ax1.grid(True, alpha=0.3, linewidth=0.4, color="#CCC")
    ax1.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax1.spines[spine].set_color("#CCC")
        ax1.spines[spine].set_linewidth(0.5)

    # ---- BOTTOM: nDCG@10 ----
    ax2.set_facecolor("#FAFAFA")
    for task in available_tasks:
        sizes, ndcg_vals = [], []
        for s in numeric_sizes:
            if s in data[task] and data[task][s].get("ndcg10") is not None:
                sizes.append(s)
                ndcg_vals.append(data[task][s]["ndcg10"])
        if sizes:
            ax2.plot(
                sizes, ndcg_vals,
                color=TASK_COLORS.get(task, "#333"),
                marker=TASK_MARKERS.get(task, "o"),
                markersize=4,
                linewidth=1.2,
                label=TASK_DISPLAY.get(task, task),
                markeredgecolor="white",
                markeredgewidth=0.3,
            )

    ax2.set_ylabel("nDCG@10", fontsize=9, fontweight="bold", color="#1a1a1a")
    ax2.set_xlabel("Corpus Size", fontsize=9, fontweight="bold", color="#1a1a1a")
    ax2.tick_params(axis="both", labelsize=7, colors="#333")
    ax2.grid(True, alpha=0.3, linewidth=0.4, color="#CCC")
    ax2.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax2.spines[spine].set_color("#CCC")
        ax2.spines[spine].set_linewidth(0.5)

    
    # X-axis formatting (apply to BOTH since sharex=True)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(CORPUS_SIZES))
        ax.xaxis.set_major_formatter(FixedFormatter(SIZE_LABELS))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax1.set_yscale('log')

    # Shared legend below both plots
    handles, labels = ax2.get_legend_handles_labels()
    ncol = 4 if len(handles) <= 12 else 3
    legend = fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=ncol,
        fontsize=6,
        frameon=True,
        fancybox=False,
        edgecolor="#CCC",
        borderpad=0.5,
        columnspacing=0.6,
        handletextpad=0.3,
        handlelength=1.5,
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_facecolor("white")

    # Subplot labels
    ax1.text(-0.02, 1.05, "(a)", transform=ax1.transAxes,
             fontsize=9, fontweight="bold", va="bottom")
    ax2.text(-0.02, 1.05, "(b)", transform=ax2.transAxes,
             fontsize=9, fontweight="bold", va="bottom")

    plt.tight_layout()
    pdf_path = output_path if output_path.endswith(".pdf") else output_path + ".pdf"
    png_path = pdf_path.replace(".pdf", ".png")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Figure 4: Corpus Scaling Curves (+ FULL after 100K)")
    parser.add_argument(
        "--base_dir", type=str,
        default="/leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results",
        help="Path to sigir_results directory",
    )
    parser.add_argument("--output", type=str, default="figure4_scaling.pdf")
    args = parser.parse_args()

    print(f"Base dir: {args.base_dir}")
    print("=" * 60)

    print("\nExtracting scaling data for E5 from phase3_scaling...")
    data = extract_scaling_data(args.base_dir)

    print(f"\nFound {len(data)} tasks:")
    for task in sorted(data.keys()):
        sizes = sorted([s for s in data[task].keys() if isinstance(s, int)])
        show_sizes = [("FULL" if s == FULL_X else str(s)) for s in sizes]
        print(f"  {task}: sizes = {show_sizes}")
        for s in sizes:
            entry = data[task][s]
            idx_t = entry.get("indexing_time", "?")
            lat = entry.get("query_latency_mean", "?")
            ndcg = entry.get("ndcg10", "?")
            size_name = "FULL" if s == FULL_X else str(s)

            if isinstance(idx_t, float):
                idx_t = f"{idx_t:.1f}s"
            if isinstance(lat, float):
                lat = f"{lat:.1f}ms"
            if isinstance(ndcg, float):
                ndcg = f"{ndcg:.4f}"

            print(f"    {size_name:>7}: idx={idx_t}, lat={lat}, nDCG@10={ndcg}")

    print(f"\n{'=' * 60}")
    plot_scaling(data, args.output)

    # Dump
    summary_path = args.output.replace(".pdf", "_data.json")
    dump = {}
    for task, sizes in data.items():
        dump[task] = {}
        for s, entry in sizes.items():
            key = "full" if s == FULL_X else str(s)
            dump[task][key] = {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in entry.items()}

    with open(summary_path, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[OK] Data: {summary_path}")

if __name__ == "__main__":
    main()
