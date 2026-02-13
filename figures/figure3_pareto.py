#!/usr/bin/env python3
"""
Figure 3: Pareto Frontier - Effectiveness vs Efficiency
Scatter plot: nDCG@10 (Y) vs QPS (X) per model from phase1_cache_build.
Pareto-optimal models connected with a frontier line.

This version keeps linear axes (NO log) and automatically "spreads" points
that overlap (deterministic beeswarm-like jitter) so they don't sit on each other.
Pareto is computed on TRUE values; only plotting coordinates are slightly adjusted.

Usage:
  python figure3_pareto.py --base_dir /leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results
  python figure3_pareto.py --output figure3_pareto.pdf --no_spread
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

VALID_MODELS = {
    "bge", "bm25", "contriever", "diver-retriever", "e5", "grit",
    "inst-l", "inst-xl", "nomic", "qwen", "qwen2", "rader",
    "reasonir", "sbert", "sf",
}

MODEL_DISPLAY = {
    "bge": "BGE", "bm25": "BM25", "contriever": "Contriever",
    "diver-retriever": "DIVER", "e5": "E5-Large", "grit": "GritLM",
    "inst-l": "Inst-L", "inst-xl": "Inst-XL", "nomic": "Nomic",
    "qwen": "Qwen", "qwen2": "Qwen2", "rader": "RADER",
    "reasonir": "ReasonIR", "sbert": "SBERT", "sf": "SFR",
}

# Category -> models (for color/marker grouping)
MODEL_CATEGORIES = {
    "Sparse":      ["bm25"],
    "Dense":       ["sbert", "contriever", "nomic", "bge", "e5"],
    "Instruct":    ["inst-l", "inst-xl", "sf"],
    "LLM-based":   ["grit", "qwen", "qwen2"],
    "Reasoning":   ["diver-retriever", "rader", "reasonir"],
}

CATEGORY_STYLE = {
    "Sparse":    {"color": "#888888", "marker": "X", "size": 90},
    "Dense":     {"color": "#1B6B93", "marker": "o", "size": 80},
    "Instruct":  {"color": "#E8963E", "marker": "s", "size": 80},
    "LLM-based": {"color": "#7B68EE", "marker": "^", "size": 90},
    "Reasoning": {"color": "#D64045", "marker": "D", "size": 85},
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

def extract_effectiveness_and_efficiency(base_dir):
    """
    Extract avg nDCG@10 and avg QPS per model from phase1_cache_build.
    Returns: {model: {'ndcg10': float, 'qps': float}}
    """
    phase_dir = os.path.join(base_dir, "phase1_cache_build")
    if not os.path.isdir(phase_dir):
        print(f"[ERROR] Missing: {phase_dir}")
        return {}

    model_ndcg = defaultdict(list)
    model_qps = defaultdict(list)

    for model in discover_subfolders(phase_dir):
        if model not in VALID_MODELS:
            continue
        model_path = os.path.join(phase_dir, model)

        for task in discover_subfolders(model_path):
            task_path = os.path.join(model_path, task)

            # nested subfolders
            for sub in discover_subfolders(task_path):
                sub_path = os.path.join(task_path, sub)

                rdata = load_json(os.path.join(sub_path, "results.json"))
                if rdata and "NDCG@10" in rdata:
                    model_ndcg[model].append(rdata["NDCG@10"])

                edata = load_json(os.path.join(sub_path, "efficiency.json"))
                if edata and "query_latency" in edata:
                    ql = edata["query_latency"]
                    if "total" in ql and "qps" in ql["total"]:
                        model_qps[model].append(ql["total"]["qps"])
                    elif "qps" in ql:
                        model_qps[model].append(ql["qps"])

            # also check directly in task folder
            rdata = load_json(os.path.join(task_path, "results.json"))
            if rdata and "NDCG@10" in rdata:
                model_ndcg[model].append(rdata["NDCG@10"])

            edata = load_json(os.path.join(task_path, "efficiency.json"))
            if edata and "query_latency" in edata:
                ql = edata["query_latency"]
                if "total" in ql and "qps" in ql["total"]:
                    model_qps[model].append(ql["total"]["qps"])
                elif "qps" in ql:
                    model_qps[model].append(ql["qps"])

    result = {}
    for model in model_ndcg:
        if model in model_qps and len(model_qps[model]) > 0:
            result[model] = {
                "ndcg10": float(np.mean(model_ndcg[model])),
                "qps": float(np.mean(model_qps[model])),
            }
    return result

def get_model_category(model):
    for cat, models in MODEL_CATEGORIES.items():
        if model in models:
            return cat
    return "Dense"

def compute_pareto_frontier(points):
    """
    points: list of (qps, ndcg10, model)
    Pareto-optimal = no other point has BOTH higher QPS AND higher nDCG@10.
    We want upper-right frontier.
    """
    sorted_pts = sorted(points, key=lambda p: p[0])  # QPS ascending
    pareto = []
    max_ndcg = -1.0
    for pt in reversed(sorted_pts):  # sweep from right to left
        if pt[1] >= max_ndcg:
            pareto.append(pt)
            max_ndcg = pt[1]
    pareto.sort(key=lambda p: p[0])
    return pareto

# ============================================================
# SPREAD / DE-OVERLAP (LINEAR AXIS)
# ============================================================

def spread_close_points(points, eps_x=2.5, eps_y=0.004, dx=3.5, dy=0.006):
    """
    Deterministic "beeswarm-ish" spreading in DATA coordinates.

    points: list of (qps, ndcg, model) in DATA coordinates.
    If a new point is too close to an already-placed point, move it around
    the original location in a small spiral until it doesn't overlap.

    eps_x/eps_y: overlap thresholds (data units)
    dx/dy: displacement scale (data units)
    Returns dict: model -> (qps_plot, ndcg_plot)
    """
    placed = []  # list of (x, y) already placed (plot coords)
    out = {}

    for qps, ndcg, model in points:
        x, y = qps, ndcg
        k = 0
        while any(abs(x - px) < eps_x and abs(y - py) < eps_y for px, py in placed):
            k += 1
            angle = k * (np.pi / 3.0)          # 60-degree steps
            radius = 1.0 + 0.25 * (k // 6)     # slowly expand
            x = qps + radius * dx * np.cos(angle)
            y = ndcg + radius * dy * np.sin(angle)

        placed.append((x, y))
        out[model] = (x, y)

    return out

# ============================================================
# PLOT
# ============================================================

def plot_pareto(data, output_path, spread=True, eps_x=2.5, eps_y=0.004, dx=3.5, dy=0.006):
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    # Build list of true points once
    all_points = []
    for cat, cat_models in MODEL_CATEGORIES.items():
        for model in cat_models:
            if model in data:
                all_points.append((data[model]["qps"], data[model]["ndcg10"], model))

    if len(all_points) == 0:
        print("[ERROR] No points to plot.")
        return

    # Plot coordinates (either same as true, or spread)
    if spread:
        disp = spread_close_points(all_points, eps_x=eps_x, eps_y=eps_y, dx=dx, dy=dy)
    else:
        disp = {m: (qps, ndcg) for (qps, ndcg, m) in all_points}

    # Scatter by category using plot coords
    for cat, cat_models in MODEL_CATEGORIES.items():
        style = CATEGORY_STYLE[cat]
        for model in cat_models:
            if model not in data:
                continue
            x, y = disp[model]
            ax.scatter(
                x, y,
                c=style["color"], marker=style["marker"],
                s=style["size"], edgecolors="white", linewidths=0.6,
                zorder=4
            )

    # Pareto frontier computed on TRUE points, drawn on plot coords
    pareto = compute_pareto_frontier(all_points)
    pareto_models = {m for _, _, m in pareto}

    if len(pareto) >= 2:
        pareto_xy = [disp[m] for _, _, m in pareto]
        ax.plot(
            [x for x, _ in pareto_xy],
            [y for _, y in pareto_xy],
            color="#333333", linewidth=1.3, linestyle="--",
            alpha=0.55, zorder=3
        )

    # Highlight Pareto points with ring (plot coords)
    for _, _, model in pareto:
        x, y = disp[model]
        cat = get_model_category(model)
        style = CATEGORY_STYLE[cat]
        ax.scatter(
            x, y,
            facecolors="none", edgecolors="#333333",
            s=style["size"] + 120, linewidths=1.3, zorder=5
        )

    # Manual label offsets (dx, dy in points) — keep your offsets
    LABEL_OFFSETS = {
        "diver-retriever": (-8, 10,  "center", "bottom"),
        "rader":           (12, -6,  "left",   "top"),
        "reasonir":        (-8, 8,   "right",  "bottom"),
        "qwen2":           (-10, 6, 'right', 'bottom'), #(8, 6,    "left",   "bottom"),
        "qwen":            (8, -3,   "left",   "top"),
        "inst-xl":         (14, -6,  'left',  'top'), #(8, 4,    "left",   "bottom"),
        "sf":              (-14, 10, 'right', 'bottom'), #(8, 5,    "left",   "bottom"),
        "e5":              (8, -6,   "left",   "top"),
        "sbert":           (0, 10,   "center", "bottom"),
        "bm25":            (-8, 6,   "right",  "bottom"),
        "nomic":           (0, 9,    "center", "bottom"),
        "inst-l":          (-8, -8,  "right",  "top"),
        "bge":             (0, -11,  "center", "top"),
        "contriever":      (0, 8,    "center", "bottom"),
        # "grit":          (8, 3,    "left",   "bottom"),
    }

    # Annotations (plot coords)
    for qps, ndcg, model in all_points:
        x, y = disp[model]
        display = MODEL_DISPLAY.get(model, model)
        cat = get_model_category(model)
        color = CATEGORY_STYLE[cat]["color"]
        fontweight = "bold" if model in pareto_models else "normal"
        fontsize = 10 if model in pareto_models else 9

        dxp, dyp, ha, va = LABEL_OFFSETS.get(model, (5, 5, "left", "bottom"))

        ax.annotate(
            display,
            xy=(x, y),
            xytext=(dxp, dyp),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            ha=ha, va=va,
            zorder=6,
            arrowprops=dict(
                arrowstyle="-",
                color="#AAAAAA",
                lw=0.6,
                shrinkA=0,
                shrinkB=3
            )
        )

    # Axis limits (based on plot coords so spread points don't clip)
    # xs = np.array([disp[m][0] for _, _, m in all_points], dtype=float)
    # ys = np.array([disp[m][1] for _, _, m in all_points], dtype=float)

    # xpad = (xs.max() - xs.min()) * 0.07 if xs.max() > xs.min() else 5.0
    # ypad = (ys.max() - ys.min()) * 0.12 if ys.max() > ys.min() else 0.01

    # ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    # ax.set_ylim(ys.min() - ypad, ys.max() + ypad)

    # # Axis labels
    # ax.set_xlabel("Queries Per Second (QPS) →", fontsize=12, fontweight="bold", color="#1a1a1a")
    # ax.set_ylabel("nDCG@10 →", fontsize=12, fontweight="bold", color="#1a1a1a")
    # ax.tick_params(axis="both", labelsize=10, colors="#333333")
    xs = np.array([disp[m][0] for _, _, m in all_points], dtype=float)
    ys = np.array([disp[m][1] for _, _, m in all_points], dtype=float)

    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()

    # MORE padding on the left side (this is what you want)
    xpad_left  = 30 #max(11.0, 0.22 * x_range)   # increase if still clipped
    xpad_right = max(10.0, 0.08 * x_range)

    ypad = max(0.01, 0.12 * y_range)

    ax.set_xlim(xs.min() - xpad_left, xs.max() + xpad_right)
    ax.set_ylim(ys.min() - ypad, ys.max() + ypad)

    # Keep xticks nice (start at 0 even if xlim is negative)
    step = 25
    xmax = xs.max() + xpad_right
    xticks = np.arange(0, (int(xmax // step) + 2) * step, step)
    ax.set_xticks(xticks)

    # Grid
    ax.grid(True, alpha=0.30, linewidth=0.6, color="#CCCCCC")
    ax.set_axisbelow(True)

    # "Ideal" arrow
    ax.annotate(
        "", xy=(0.95, 0.95), xytext=(0.82, 0.82),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2)
    )
    ax.text(
        0.97, 0.86, "Ideal", transform=ax.transAxes,
        fontsize=9, color="#999999", fontstyle="italic",
        ha="right", va="top"
    )

    # Legend
    import matplotlib.lines as mlines
    legend_handles = []
    for cat in MODEL_CATEGORIES:
        style = CATEGORY_STYLE[cat]
        handle = mlines.Line2D(
            [], [], color=style["color"],
            marker=style["marker"], linestyle="None",
            markersize=7.0, markeredgecolor="white",
            markeredgewidth=0.6, label=cat
        )
        legend_handles.append(handle)

    pareto_line = mlines.Line2D(
        [], [], color="#333333", linewidth=1.2,
        linestyle="--", alpha=0.55, label="Pareto"
    )
    legend_handles.append(pareto_line)

    legend = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        borderpad=0.5,
        columnspacing=0.9,
        handletextpad=0.4
    )
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_facecolor("white")

    # Spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#CCCCCC")
        ax.spines[spine].set_linewidth(0.7)

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
    parser = argparse.ArgumentParser(description="Figure 3: Pareto Frontier (linear axis + de-overlap spread)")
    parser.add_argument(
        "--base_dir", type=str,
        default="/leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results",
        help="Path to sigir_results directory",
    )
    parser.add_argument("--output", type=str, default="figure3_pareto.pdf")

    # Spread controls
    parser.add_argument("--no_spread", action="store_true", help="Disable de-overlap spreading")
    parser.add_argument("--eps_x", type=float, default=2.5, help="Overlap threshold in QPS (data units)")
    parser.add_argument("--eps_y", type=float, default=0.004, help="Overlap threshold in nDCG (data units)")
    parser.add_argument("--dx", type=float, default=3.5, help="Spread step in QPS (data units)")
    parser.add_argument("--dy", type=float, default=0.006, help="Spread step in nDCG (data units)")

    args = parser.parse_args()

    print(f"Base dir: {args.base_dir}")
    print("=" * 60)

    print("\nExtracting nDCG@10 + QPS from phase1_cache_build...")
    data = extract_effectiveness_and_efficiency(args.base_dir)

    print(f"\nFound {len(data)} models:")
    print(f"{'Model':<20} {'nDCG@10':>10} {'QPS':>10} {'Category':>12}")
    print("-" * 55)
    for model in sorted(data.keys(), key=lambda m: data[m]["ndcg10"], reverse=True):
        d = data[model]
        cat = get_model_category(model)
        print(f"  {MODEL_DISPLAY.get(model, model):<18} {d['ndcg10']:>10.4f} {d['qps']:>10.2f} {cat:>12}")

    # Pareto (true points)
    all_points = [(data[m]["qps"], data[m]["ndcg10"], m) for m in data]
    pareto = compute_pareto_frontier(all_points)
    print(f"\nPareto-optimal models ({len(pareto)}):")
    for qps, ndcg, model in pareto:
        print(f"  {MODEL_DISPLAY.get(model, model)}: nDCG@10={ndcg:.4f}, QPS={qps:.2f}")

    print(f"\n{'=' * 60}")
    plot_pareto(
        data,
        args.output,
        spread=(not args.no_spread),
        eps_x=args.eps_x,
        eps_y=args.eps_y,
        dx=args.dx,
        dy=args.dy,
    )

    # Dump summary json
    pareto_set = {p[2] for p in pareto}
    summary = {
        m: {
            "ndcg10": round(v["ndcg10"], 4),
            "qps": round(v["qps"], 2),
            "category": get_model_category(m),
            "pareto_optimal": m in pareto_set,
        }
        for m, v in data.items()
    }
    summary_path = args.output.replace(".pdf", "_data.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Data: {summary_path}")

if __name__ == "__main__":
    main()
