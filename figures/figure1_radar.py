"""
Figure 1: Data-Driven Evaluation Dimension Radar Chart
4 models x 4 dimensions (all with complete data)

Models: e5, bge, qwen2, reasonir
Dimensions:
  - Effectiveness  -> avg nDCG@10 from phase1_cache_build
  - Efficiency     -> avg QPS from phase1_cache_build
  - Robustness     -> avg nDCG@10 retention under perturbations from phase10_robustness
  - Calibration    -> avg AUROC@10 from phase17_confidence_auroc

Usage:
  python figure1_radar.py --base_dir /leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# Fixed model set
TARGET_MODELS = ['e5', 'bge', 'qwen2', 'reasonir']

# ============================================================
# DATA LOADING
# ============================================================

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def discover_subfolders(path):
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

# ============================================================
# PER-PHASE EXTRACTION
# ============================================================

def extract_effectiveness(base_dir):
    """Phase 1: avg nDCG@10 per model."""
    phase_dir = os.path.join(base_dir, 'phase1_cache_build')
    if not os.path.isdir(phase_dir):
        print(f"[WARN] Missing: {phase_dir}")
        return {}
    model_scores = defaultdict(list)
    for model in discover_subfolders(phase_dir):
        model_path = os.path.join(phase_dir, model)
        for task in discover_subfolders(model_path):
            task_path = os.path.join(model_path, task)
            for sub in discover_subfolders(task_path):
                rpath = os.path.join(task_path, sub, 'results.json')
                data = load_json(rpath)
                if data and 'NDCG@10' in data:
                    model_scores[model].append(data['NDCG@10'])
            rpath = os.path.join(task_path, 'results.json')
            data = load_json(rpath)
            if data and 'NDCG@10' in data:
                model_scores[model].append(data['NDCG@10'])
    return {m: np.mean(v) for m, v in model_scores.items() if v}


def extract_efficiency(base_dir):
    """Phase 1: avg QPS per model."""
    phase_dir = os.path.join(base_dir, 'phase1_cache_build')
    if not os.path.isdir(phase_dir):
        return {}
    model_qps = defaultdict(list)
    for model in discover_subfolders(phase_dir):
        model_path = os.path.join(phase_dir, model)
        for task in discover_subfolders(model_path):
            task_path = os.path.join(model_path, task)
            for sub in discover_subfolders(task_path):
                epath = os.path.join(task_path, sub, 'efficiency.json')
                data = load_json(epath)
                if data and 'query_latency' in data:
                    ql = data['query_latency']
                    if 'total' in ql and 'qps' in ql['total']:
                        model_qps[model].append(ql['total']['qps'])
                    elif 'qps' in ql:
                        model_qps[model].append(ql['qps'])
            epath = os.path.join(task_path, 'efficiency.json')
            data = load_json(epath)
            if data and 'query_latency' in data:
                ql = data['query_latency']
                if 'total' in ql and 'qps' in ql['total']:
                    model_qps[model].append(ql['total']['qps'])
                elif 'qps' in ql:
                    model_qps[model].append(ql['qps'])
    return {m: np.mean(v) for m, v in model_qps.items() if v}


def extract_robustness(base_dir):
    """Phase 10: avg perturbed nDCG@10 / original nDCG@10."""
    phase_dir = os.path.join(base_dir, 'phase10_robustness')
    if not os.path.isdir(phase_dir):
        print(f"[WARN] Missing: {phase_dir}")
        return {}
    orig = extract_effectiveness(base_dir)
    perturbation_types = ['adversarial', 'paraphrase', 'synonym', 'length']
    model_task_perturbed = defaultdict(lambda: defaultdict(list))

    for task in discover_subfolders(phase_dir):
        task_path = os.path.join(phase_dir, task)
        for run_folder in discover_subfolders(task_path):
            run_path = os.path.join(task_path, run_folder)
            data = None
            rpath = os.path.join(run_path, 'results.json')
            if os.path.exists(rpath):
                data = load_json(rpath)
            else:
                for sub in discover_subfolders(run_path):
                    rpath2 = os.path.join(run_path, sub, 'results.json')
                    data = load_json(rpath2)
                    if data:
                        break
            if data and 'NDCG@10' in data:
                for ptype in perturbation_types:
                    if f'_{ptype}' in run_folder:
                        model_name = run_folder.split(f'_{ptype}')[0]
                        model_task_perturbed[model_name][task].append(data['NDCG@10'])
                        break

    model_robustness = {}
    for model, tasks in model_task_perturbed.items():
        avg_perturbed = np.mean([s for scores in tasks.values() for s in scores])
        if model in orig and orig[model] > 0:
            model_robustness[model] = avg_perturbed / orig[model]
        else:
            model_robustness[model] = avg_perturbed
    return model_robustness


def extract_calibration(base_dir):
    """Phase 17: avg AUROC@10 per model."""
    phase_dir = os.path.join(base_dir, 'phase17_confidence_auroc')
    if not os.path.isdir(phase_dir):
        print(f"[WARN] Missing: {phase_dir}")
        return {}
    model_auroc = {}

    # Summary files first
    for fname in os.listdir(phase_dir):
        if fname.endswith('_auroc_summary.json'):
            model_name = fname.replace('_auroc_summary.json', '')
            data = load_json(os.path.join(phase_dir, fname))
            if data and 'mean_auroc_at_k' in data and '@10' in data['mean_auroc_at_k']:
                model_auroc[model_name] = data['mean_auroc_at_k']['@10']

    # Fallback: per-task
    per_task_scores = defaultdict(list)
    for task in discover_subfolders(phase_dir):
        task_path = os.path.join(phase_dir, task)
        for model in discover_subfolders(task_path):
            if model in model_auroc:
                continue
            apath = os.path.join(task_path, model, 'auroc_results.json')
            data = load_json(apath)
            if data and 'auroc_at_k' in data and '@10' in data['auroc_at_k']:
                per_task_scores[model].append(data['auroc_at_k']['@10'])
    for m, scores in per_task_scores.items():
        if m not in model_auroc:
            model_auroc[m] = np.mean(scores)
    return model_auroc


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_scores(scores_dict, higher_is_better=True):
    """Min-max normalize to [0.15, 1.0] so nothing collapses to center."""
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {k: 0.55 for k in scores_dict}
    floor = 0.15
    if higher_is_better:
        return {k: floor + (1.0 - floor) * (v - vmin) / (vmax - vmin) for k, v in scores_dict.items()}
    else:
        return {k: floor + (1.0 - floor) * (1.0 - (v - vmin) / (vmax - vmin)) for k, v in scores_dict.items()}


# ============================================================
# PLOTTING
# ============================================================

MODEL_DISPLAY = {
    'bge': 'BGE', 'e5': 'E5-Large', 'qwen2': 'Qwen2', 'reasonir': 'ReasonIR',
}

MODEL_COLORS = {
    'e5':       '#1B6B93',
    'bge':      '#D64045',
    'qwen2':    '#E8963E',
    'reasonir': '#5B8C5A',
}

MODEL_MARKERS = {
    'e5': 's', 'bge': 'o', 'qwen2': '^', 'reasonir': 'D',
}

MODEL_LINESTYLES = {
    'e5': '-', 'bge': '--', 'qwen2': '-.', 'reasonir': ':',
}


def plot_radar(model_data, output_path):
    categories = ['Effectiveness', 'Efficiency', 'Robustness', 'Calibration']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight='bold', color='#1a1a1a')

    # Radial ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=6, color='#AAAAAA')
    ax.set_ylim(0, 1.08)

    # Grid
    ax.yaxis.grid(True, color='#D8D8D8', linewidth=0.5, linestyle='-')
    ax.xaxis.grid(True, color='#D8D8D8', linewidth=0.5, linestyle='-')
    ax.spines['polar'].set_visible(False)

    # Plot each model
    for model in TARGET_MODELS:
        if model not in model_data:
            print(f"  [SKIP] {model} -- no data")
            continue
        dims = model_data[model]
        values = [dims.get(c, 0.0) for c in categories]
        values += values[:1]

        display = MODEL_DISPLAY.get(model, model)
        color = MODEL_COLORS.get(model, '#333')
        marker = MODEL_MARKERS.get(model, 'o')
        ls = MODEL_LINESTYLES.get(model, '-')

        ax.plot(angles, values, color=color, linewidth=1.8, linestyle=ls,
                marker=marker, markersize=6, markeredgecolor='white',
                markeredgewidth=0.5, label=display, zorder=3)
        ax.fill(angles, values, color=color, alpha=0.08, zorder=2)

    # Legend
    legend = ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.30),
        ncol=4,
        fontsize=7.5,
        frameon=True,
        fancybox=False,
        edgecolor='#CCCCCC',
        borderpad=0.6,
        columnspacing=0.8,
        handletextpad=0.3,
        handlelength=1.5,
    )
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    pdf_path = output_path if output_path.endswith('.pdf') else output_path + '.pdf'
    png_path = pdf_path.replace('.pdf', '.png')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Figure 1: Radar Chart')
    parser.add_argument('--base_dir', type=str,
                        default='/leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results',
                        help='Path to sigir_results directory')
    parser.add_argument('--output', type=str, default='figure1_radar_chart.pdf')
    args = parser.parse_args()

    print(f"Base dir: {args.base_dir}")
    print(f"Target models: {TARGET_MODELS}")
    print('=' * 60)

    # Extract
    print("\n[1/4] Effectiveness (phase1)...")
    effectiveness = extract_effectiveness(args.base_dir)
    print(f"  Found: {list(effectiveness.keys())}")

    print("\n[2/4] Efficiency (phase1)...")
    efficiency = extract_efficiency(args.base_dir)
    print(f"  Found: {list(efficiency.keys())}")

    print("\n[3/4] Robustness (phase10)...")
    robustness = extract_robustness(args.base_dir)
    print(f"  Found: {list(robustness.keys())}")

    print("\n[4/4] Calibration (phase17)...")
    calibration = extract_calibration(args.base_dir)
    print(f"  Found: {list(calibration.keys())}")

    # Filter to target models
    eff_filtered = {m: effectiveness[m] for m in TARGET_MODELS if m in effectiveness}
    qps_filtered = {m: efficiency[m] for m in TARGET_MODELS if m in efficiency}
    rob_filtered = {m: robustness[m] for m in TARGET_MODELS if m in robustness}
    cal_filtered = {m: calibration[m] for m in TARGET_MODELS if m in calibration}

    # Print raw
    print(f"\n{'=' * 60}")
    print("Raw scores (target models):")
    for m in TARGET_MODELS:
        print(f"  {m}:")
        if m in eff_filtered: print(f"    Effectiveness (nDCG@10): {eff_filtered[m]:.4f}")
        else: print(f"    Effectiveness: MISSING")
        if m in qps_filtered: print(f"    Efficiency (QPS):        {qps_filtered[m]:.2f}")
        else: print(f"    Efficiency: MISSING")
        if m in rob_filtered: print(f"    Robustness (retention):  {rob_filtered[m]:.4f}")
        else: print(f"    Robustness: MISSING")
        if m in cal_filtered: print(f"    Calibration (AUROC@10):  {cal_filtered[m]:.4f}")
        else: print(f"    Calibration: MISSING")

    # Normalize across target models only
    norm_eff = normalize_scores(eff_filtered, higher_is_better=True)
    norm_qps = normalize_scores(qps_filtered, higher_is_better=True)
    norm_rob = normalize_scores(rob_filtered, higher_is_better=True)
    norm_cal = normalize_scores(cal_filtered, higher_is_better=True)

    # Merge
    model_data = {}
    for model in TARGET_MODELS:
        dims = {}
        if model in norm_eff: dims['Effectiveness'] = norm_eff[model]
        if model in norm_qps: dims['Efficiency'] = norm_qps[model]
        if model in norm_rob: dims['Robustness'] = norm_rob[model]
        if model in norm_cal: dims['Calibration'] = norm_cal[model]
        model_data[model] = dims

    print(f"\nNormalized scores:")
    for m in TARGET_MODELS:
        dims = model_data.get(m, {})
        print(f"  {MODEL_DISPLAY.get(m, m)}: {', '.join(f'{k}={v:.3f}' for k, v in dims.items())}")

    # Plot
    print(f"\n{'=' * 60}")
    plot_radar(model_data, args.output)

    # Dump
    summary = {
        'raw': {
            'effectiveness': {k: round(v, 4) for k, v in eff_filtered.items()},
            'efficiency_qps': {k: round(v, 2) for k, v in qps_filtered.items()},
            'robustness': {k: round(v, 4) for k, v in rob_filtered.items()},
            'calibration': {k: round(v, 4) for k, v in cal_filtered.items()},
        },
        'normalized': {m: {k: round(v, 3) for k, v in d.items()} for m, d in model_data.items()}
    }
    summary_path = args.output.replace('.pdf', '_data.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Data: {summary_path}")


if __name__ == '__main__':
    main()