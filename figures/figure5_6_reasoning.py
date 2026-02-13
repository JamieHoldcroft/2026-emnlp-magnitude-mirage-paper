"""
Figure 5: Reasoning Overhead Trade-off (Grouped Bar Chart)
Figure 6: When is Reasoning Worth It? (Quadrant Chart)

Reads from:
  - phase1_cache_build  -> original nDCG@10 + QPS
  - phase6_reasoning    -> reasoning-enhanced nDCG@10 + QPS

Structure: phase6_reasoning/{task}/{model}_{reasoning_type}_reason/results.json + efficiency.json
Reasoning types: gpt4, llama3-70b, claude-3-opus, Gemini-1.0, grit

Usage:
  python figure5_6_reasoning.py --base_dir /leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results
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

VALID_MODELS = {
    'bge', 'bm25', 'contriever', 'diver-retriever', 'e5', 'grit',
    'inst-l', 'inst-xl', 'nomic', 'qwen', 'qwen2', 'rader',
    'reasonir', 'sbert', 'sf',
}

MODEL_DISPLAY = {
    'bge': 'BGE', 'bm25': 'BM25', 'contriever': 'Contriever',
    'diver-retriever': 'DIVER', 'e5': 'E5-Large', 'grit': 'GritLM',
    'inst-l': 'Inst-L', 'inst-xl': 'Inst-XL', 'nomic': 'Nomic',
    'qwen': 'Qwen', 'qwen2': 'Qwen2', 'rader': 'RADER',
    'reasonir': 'ReasonIR', 'sbert': 'SBERT', 'sf': 'SFR',
}

REASONING_TYPES = ['gpt4', 'llama3-70b', 'claude-3-opus', 'Gemini-1.0', 'grit']

REASONING_DISPLAY = {
    'gpt4': 'GPT-4',
    'llama3-70b': 'Llama-3-70B',
    'claude-3-opus': 'Claude-3',
    'Gemini-1.0': 'Gemini',
    'grit': 'GritLM',
}

REASONING_COLORS = {
    'gpt4': '#5B8C5A',
    'llama3-70b': '#7B68EE',
    'claude-3-opus': '#E8963E',
    'Gemini-1.0': '#1B6B93',
    'grit': '#FF6B9D',
}

# ============================================================
# DATA
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


def extract_original_scores(base_dir):
    """Phase 1: avg nDCG@10 and avg query latency per model."""
    phase_dir = os.path.join(base_dir, 'phase1_cache_build')
    if not os.path.isdir(phase_dir):
        return {}, {}

    model_ndcg = defaultdict(list)
    model_latency = defaultdict(list)

    for model in discover_subfolders(phase_dir):
        if model not in VALID_MODELS:
            continue
        model_path = os.path.join(phase_dir, model)
        for task in discover_subfolders(model_path):
            task_path = os.path.join(model_path, task)
            for sub in discover_subfolders(task_path):
                sub_path = os.path.join(task_path, sub)
                rdata = load_json(os.path.join(sub_path, 'results.json'))
                if rdata and 'NDCG@10' in rdata:
                    model_ndcg[model].append(rdata['NDCG@10'])
                edata = load_json(os.path.join(sub_path, 'efficiency.json'))
                if edata and 'query_latency' in edata:
                    ql = edata['query_latency']
                    if 'total' in ql and 'mean_ms' in ql['total']:
                        model_latency[model].append(ql['total']['mean_ms'])

    ndcg = {m: np.mean(v) for m, v in model_ndcg.items() if v}
    latency = {m: np.mean(v) for m, v in model_latency.items() if v}
    return ndcg, latency


def extract_reasoning_scores(base_dir):
    """
    Phase 6: nDCG@10 and latency per (model, reasoning_type), averaged across tasks.
    Structure: phase6_reasoning/{task}/{model}_{reasoning_type}_reason/
    Returns: {model: {reasoning_type: {'ndcg10': float, 'latency': float}}}
    """
    phase_dir = os.path.join(base_dir, 'phase6_reasoning')
    if not os.path.isdir(phase_dir):
        print(f"[ERROR] Missing: {phase_dir}")
        return {}

    # Collect: (model, reasoning_type) -> list of scores across tasks
    scores = defaultdict(lambda: defaultdict(lambda: {'ndcg': [], 'latency': []}))

    for task in discover_subfolders(phase_dir):
        task_path = os.path.join(phase_dir, task)
        for run_folder in discover_subfolders(task_path):
            if not run_folder.endswith('_reason'):
                continue

            # Parse: {model}_{reasoning_type}_reason
            name = run_folder[:-len('_reason')]  # remove _reason suffix

            # Find which reasoning type matches
            matched_rt = None
            matched_model = None
            for rt in REASONING_TYPES:
                suffix = f'_{rt}'
                if name.endswith(suffix):
                    candidate_model = name[:-len(suffix)]
                    if candidate_model in VALID_MODELS:
                        matched_rt = rt
                        matched_model = candidate_model
                        break

            if not matched_rt or not matched_model:
                continue

            run_path = os.path.join(task_path, run_folder)

            # Check for results directly or in subfolder
            rdata = load_json(os.path.join(run_path, 'results.json'))
            if not rdata:
                for sub in discover_subfolders(run_path):
                    rdata = load_json(os.path.join(run_path, sub, 'results.json'))
                    if rdata:
                        break

            edata = load_json(os.path.join(run_path, 'efficiency.json'))
            if not edata:
                for sub in discover_subfolders(run_path):
                    edata = load_json(os.path.join(run_path, sub, 'efficiency.json'))
                    if edata:
                        break

            if rdata and 'NDCG@10' in rdata:
                scores[matched_model][matched_rt]['ndcg'].append(rdata['NDCG@10'])

            if edata and 'query_latency' in edata:
                ql = edata['query_latency']
                if 'total' in ql and 'mean_ms' in ql['total']:
                    scores[matched_model][matched_rt]['latency'].append(ql['total']['mean_ms'])

    # Average across tasks
    result = {}
    for model, rts in scores.items():
        result[model] = {}
        for rt, vals in rts.items():
            entry = {}
            if vals['ndcg']:
                entry['ndcg10'] = np.mean(vals['ndcg'])
            if vals['latency']:
                entry['latency'] = np.mean(vals['latency'])
            if entry:
                result[model][rt] = entry

    return result


# ============================================================
# FIGURE 5: Grouped Bar Chart
# ============================================================

def plot_figure5(orig_ndcg, reasoning_data, output_path):
    """
    Grouped bar chart: original vs reasoning-enhanced nDCG@10 per model.
    One group per model, bars for original + each reasoning type.
    """
    # Select models that have reasoning data
    models = [m for m in sorted(reasoning_data.keys()) if m in orig_ndcg]

    # Sort by original nDCG@10
    models.sort(key=lambda m: orig_ndcg.get(m, 0))

    # Determine which reasoning types are present
    all_rts = set()
    for m in models:
        all_rts.update(reasoning_data[m].keys())
    rts = [rt for rt in REASONING_TYPES if rt in all_rts]

    print(f"  Models: {models}")
    print(f"  Reasoning types: {rts}")

    n_models = len(models)
    n_bars = 1 + len(rts)  # original + each reasoning type
    bar_width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x = np.arange(n_models)

    # Original bars
    orig_vals = [orig_ndcg.get(m, 0) for m in models]
    ax.bar(x - (n_bars - 1) * bar_width / 2, orig_vals, bar_width,
           color='#AAAAAA', edgecolor='white', linewidth=0.3,
           label='Original', zorder=3)

    # Reasoning bars
    for i, rt in enumerate(rts):
        vals = []
        for m in models:
            if rt in reasoning_data.get(m, {}):
                vals.append(reasoning_data[m][rt].get('ndcg10', 0))
            else:
                vals.append(0)

        offset = (i + 1 - (n_bars - 1) / 2) * bar_width
        color = REASONING_COLORS.get(rt, '#333')
        label = REASONING_DISPLAY.get(rt, rt)
        ax.bar(x + offset, vals, bar_width,
               color=color, edgecolor='white', linewidth=0.3,
               label=label, zorder=3)

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models],
                       fontsize=7.5, rotation=35, ha='right', fontweight='medium')
    ax.set_ylabel('nDCG@10', fontsize=10, fontweight='bold', color='#1a1a1a')
    ax.tick_params(axis='y', labelsize=7, colors='#333')

    # Grid
    ax.grid(True, axis='y', alpha=0.2, linewidth=0.4, color='#CCC')
    ax.set_axisbelow(True)

    # Spines
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#CCC')
        ax.spines[sp].set_linewidth(0.5)

    # Legend below
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                       ncol=3, fontsize=7, frameon=True, fancybox=False,
                       edgecolor='#CCC', borderpad=0.5, columnspacing=1.0,
                       handletextpad=0.4)
    legend.get_frame().set_linewidth(0.4)
    legend.get_frame().set_facecolor('white')

    plt.tight_layout()
    pdf_path = output_path if output_path.endswith('.pdf') else output_path + '.pdf'
    png_path = pdf_path.replace('.pdf', '.png')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Figure 5 saved: {pdf_path}")


# ============================================================
# FIGURE 6: Quadrant Chart
# ============================================================

def plot_figure6(orig_ndcg, orig_latency, reasoning_data, output_path):
    """
    Quadrant scatter: X = latency penalty (ms), Y = nDCG@10 gain.
    Only key models, labeled. Color = reasoning type.
    """
    KEY_MODELS = ['bge', 'e5', 'qwen2', 'reasonir', 'bm25']

    MODEL_SHORT = {
        'bge': 'BGE', 'e5': 'E5', 'qwen2': 'Qw2',
        'reasonir': 'RIR', 'bm25': 'BM25',
    }

    MODEL_MARKER = {
        'bge': 'o', 'e5': 's', 'qwen2': '^', 'reasonir': 'D', 'bm25': 'X',
    }

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    points = []  # (penalty, gain, model, rt) for labeling

    for model in KEY_MODELS:
        if model not in reasoning_data or model not in orig_ndcg:
            continue
        orig_n = orig_ndcg[model]
        orig_l = orig_latency.get(model, 0)
        marker = MODEL_MARKER.get(model, 'o')

        for rt, vals in reasoning_data[model].items():
            if 'ndcg10' not in vals:
                continue
            gain = vals['ndcg10'] - orig_n
            penalty = vals.get('latency', orig_l) - orig_l

            color = REASONING_COLORS.get(rt, '#333')

            ax.scatter(penalty, gain, c=color, marker=marker,
                       s=55, edgecolors='white', linewidths=0.4,
                       zorder=4, alpha=0.85)

            points.append((penalty, gain, model, rt))

    # Label each dot with short model name
    for penalty, gain, model, rt in points:
        short = MODEL_SHORT.get(model, model)
        color = REASONING_COLORS.get(rt, '#333')
        ax.annotate(short,
                    xy=(penalty, gain),
                    xytext=(4, 4),
                    textcoords='offset points',
                    fontsize=5, color=color,
                    fontweight='bold', alpha=0.8,
                    zorder=6)

    # Quadrant lines at 0
    ax.axhline(0, color='#999', linewidth=0.8, linestyle='-', zorder=2)
    ax.axvline(0, color='#999', linewidth=0.8, linestyle='-', zorder=2)

    # Quadrant labels
    ax.text(0.25, 0.88, 'High Gain\nLow Cost\n✓ BEST', transform=ax.transAxes,
            fontsize=6, fontstyle='italic', color='#2E7D32', fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='none', alpha=0.6))

    ax.text(0.78, 0.88, 'High Gain\nHigh Cost', transform=ax.transAxes,
            fontsize=6, fontstyle='italic', color='#E65100',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='none', alpha=0.6))

    ax.text(0.25, 0.12, 'Low Gain\nLow Cost', transform=ax.transAxes,
            fontsize=6, fontstyle='italic', color='#1565C0',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='none', alpha=0.6))

    ax.text(0.78, 0.12, 'Low Gain\nHigh Cost\n✗ WORST', transform=ax.transAxes,
            fontsize=6, fontstyle='italic', color='#C62828', fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='none', alpha=0.6))

    # Axis labels
    ax.set_xlabel('Latency Penalty (ms)', fontsize=9, fontweight='bold', color='#1a1a1a')
    ax.set_ylabel('nDCG@10 Gain', fontsize=9, fontweight='bold', color='#1a1a1a')
    ax.tick_params(axis='both', labelsize=7, colors='#333')

    ax.grid(True, alpha=0.15, linewidth=0.4, color='#CCC')
    ax.set_axisbelow(True)

    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#CCC')
        ax.spines[sp].set_linewidth(0.5)

    # Two-part legend: reasoning type (color) + model (marker shape)
    import matplotlib.lines as mlines
    handles_rt = []
    for rt in REASONING_TYPES:
        if rt in REASONING_COLORS:
            h = mlines.Line2D([], [], color=REASONING_COLORS[rt], marker='o',
                              linestyle='None', markersize=5,
                              markeredgecolor='white', markeredgewidth=0.3,
                              label=REASONING_DISPLAY.get(rt, rt))
            handles_rt.append(h)

    handles_model = []
    for m in KEY_MODELS:
        h = mlines.Line2D([], [], color='#666', marker=MODEL_MARKER.get(m, 'o'),
                          linestyle='None', markersize=5,
                          markeredgecolor='white', markeredgewidth=0.3,
                          label=MODEL_DISPLAY.get(m, m))
        handles_model.append(h)

    # Combine both legends
    all_handles = handles_rt + [mlines.Line2D([], [], linestyle='None')] + handles_model
    legend = ax.legend(handles=all_handles, loc='upper center',
                       bbox_to_anchor=(0.5, -0.16), ncol=3,
                       fontsize=5.5, frameon=True, fancybox=False,
                       edgecolor='#CCC', borderpad=0.4,
                       columnspacing=0.5, handletextpad=0.3)
    legend.get_frame().set_linewidth(0.4)
    legend.get_frame().set_facecolor('white')

    plt.tight_layout()
    pdf_path = output_path if output_path.endswith('.pdf') else output_path + '.pdf'
    png_path = pdf_path.replace('.pdf', '.png')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Figure 6 saved: {pdf_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Figure 5 & 6: Reasoning Overhead')
    parser.add_argument('--base_dir', type=str,
                        default='/leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results',
                        help='Path to sigir_results directory')
    parser.add_argument('--output5', type=str, default='figure5_reasoning_bars.pdf')
    parser.add_argument('--output6', type=str, default='figure6_reasoning_quadrant.pdf')
    args = parser.parse_args()

    print(f"Base dir: {args.base_dir}")
    print('=' * 60)

    # --- Original scores ---
    print("\n[1/2] Extracting original scores (phase1)...")
    orig_ndcg, orig_latency = extract_original_scores(args.base_dir)
    print(f"  {len(orig_ndcg)} models with nDCG@10")
    print(f"  {len(orig_latency)} models with latency")

    # --- Reasoning scores ---
    print("\n[2/2] Extracting reasoning scores (phase6)...")
    reasoning_data = extract_reasoning_scores(args.base_dir)
    print(f"  {len(reasoning_data)} models with reasoning data")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Reasoning overhead summary:")
    print(f"{'Model':<15} {'Original':>8} | ", end='')
    for rt in REASONING_TYPES:
        print(f" {REASONING_DISPLAY.get(rt, rt):>8}", end='')
    print()
    print('-' * 75)

    for model in sorted(reasoning_data.keys()):
        orig = orig_ndcg.get(model, 0)
        print(f"  {MODEL_DISPLAY.get(model, model):<13} {orig:>8.4f} | ", end='')
        for rt in REASONING_TYPES:
            if rt in reasoning_data[model]:
                val = reasoning_data[model][rt].get('ndcg10', 0)
                delta = val - orig
                sign = '+' if delta >= 0 else ''
                print(f" {val:>5.3f}({sign}{delta:.3f})", end='')
            else:
                print(f"     {'---':>8}", end='')
        print()

    # --- Plot ---
    print(f"\n{'=' * 60}")
    print("Generating Figure 5 (grouped bars)...")
    plot_figure5(orig_ndcg, reasoning_data, args.output5)

    print("\nGenerating Figure 6 (quadrant chart)...")
    plot_figure6(orig_ndcg, orig_latency, reasoning_data, args.output6)

    # Dump
    summary = {
        'original': {m: {'ndcg10': round(orig_ndcg.get(m, 0), 4),
                          'latency_ms': round(orig_latency.get(m, 0), 2)}
                     for m in orig_ndcg},
        'reasoning': {m: {rt: {k: round(v, 4) for k, v in vals.items()}
                          for rt, vals in rts.items()}
                      for m, rts in reasoning_data.items()}
    }
    with open('figure5_6_reasoning_data.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Data: figure5_6_reasoning_data.json")


if __name__ == '__main__':
    main()