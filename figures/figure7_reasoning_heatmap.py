"""
Figure 7: Reasoning Gain by Task Category
Heatmap: nDCG@10 gain (reasoning - original) across (task x reasoning_type).
Values averaged across all retrieval models. Tasks grouped by domain.

Usage:
  python figure7_reasoning_heatmap.py --base_dir /leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
from collections import defaultdict

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

VALID_MODELS = {
    'bge', 'bm25', 'contriever', 'diver-retriever', 'e5', 'grit',
    'inst-l', 'inst-xl', 'nomic', 'qwen', 'qwen2', 'rader',
    'reasonir', 'sbert', 'sf',
}

REASONING_TYPES = ['gpt4', 'llama3-70b', 'claude-3-opus', 'Gemini-1.0', 'grit']

REASONING_DISPLAY = {
    'gpt4': 'GPT',
    'llama3-70b': 'Llama',
    'claude-3-opus': 'Claude',
    'Gemini-1.0': 'Gemini',
    'grit': 'GritLM',
}

# Tasks grouped by domain
TASK_GROUPS = {
    'STEM': ['aops', 'leetcode', 'theoremqa_questions', 'theoremqa_theorems'],
    'Science': ['biology', 'earth_science'],
    'Social': ['economics', 'psychology'],
    'Technical': ['robotics', 'stackoverflow', 'pony', 'sustainable_living'],
}

TASK_DISPLAY = {
    'aops': 'AoPS', 'biology': 'Biology', 'earth_science': 'Earth Sci.',
    'economics': 'Economics', 'leetcode': 'LeetCode', 'pony': 'Pony',
    'psychology': 'Psychology', 'robotics': 'Robotics',
    'stackoverflow': 'StackOF', 'sustainable_living': 'Sust. Living',
    'theoremqa_questions': 'TheoremQA-Q', 'theoremqa_theorems': 'TheoremQA-T',
}

# Ordered task list (grouped)
TASK_ORDER = []
for grp in ['STEM', 'Science', 'Social', 'Technical']:
    TASK_ORDER.extend(TASK_GROUPS[grp])


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


def extract_original_per_task(base_dir):
    """
    Phase 1: nDCG@10 per (model, task).
    Returns: {model: {task: ndcg10}}
    """
    phase_dir = os.path.join(base_dir, 'phase1_cache_build')
    if not os.path.isdir(phase_dir):
        return {}

    data = defaultdict(dict)
    for model in discover_subfolders(phase_dir):
        if model not in VALID_MODELS:
            continue
        model_path = os.path.join(phase_dir, model)
        for task in discover_subfolders(model_path):
            task_path = os.path.join(model_path, task)
            for sub in discover_subfolders(task_path):
                rdata = load_json(os.path.join(task_path, sub, 'results.json'))
                if rdata and 'NDCG@10' in rdata:
                    data[model][task] = rdata['NDCG@10']
                    break
    return dict(data)


def extract_reasoning_per_task(base_dir):
    """
    Phase 6: nDCG@10 per (model, task, reasoning_type).
    Structure: phase6_reasoning/{task}/{model}_{reasoning_type}_reason/results.json
    Returns: {model: {task: {reasoning_type: ndcg10}}}
    """
    phase_dir = os.path.join(base_dir, 'phase6_reasoning')
    if not os.path.isdir(phase_dir):
        print(f"[ERROR] Missing: {phase_dir}")
        return {}

    data = defaultdict(lambda: defaultdict(dict))

    for task in discover_subfolders(phase_dir):
        task_path = os.path.join(phase_dir, task)
        for run_folder in discover_subfolders(task_path):
            if not run_folder.endswith('_reason'):
                continue
            name = run_folder[:-len('_reason')]

            matched_rt = None
            matched_model = None
            for rt in REASONING_TYPES:
                suffix = f'_{rt}'
                if name.endswith(suffix):
                    candidate = name[:-len(suffix)]
                    if candidate in VALID_MODELS:
                        matched_rt = rt
                        matched_model = candidate
                        break

            if not matched_rt:
                continue

            run_path = os.path.join(task_path, run_folder)
            rdata = load_json(os.path.join(run_path, 'results.json'))
            if not rdata:
                for sub in discover_subfolders(run_path):
                    rdata = load_json(os.path.join(run_path, sub, 'results.json'))
                    if rdata:
                        break

            if rdata and 'NDCG@10' in rdata:
                data[matched_model][task][matched_rt] = rdata['NDCG@10']

    return dict(data)


def compute_gain_matrix(orig, reasoning):
    """
    Compute avg nDCG@10 gain across models for each (task, reasoning_type).
    Returns: 2D array [n_tasks x n_reasoning_types]
    """
    tasks = TASK_ORDER
    rts = REASONING_TYPES

    matrix = np.full((len(tasks), len(rts)), np.nan)

    for i, task in enumerate(tasks):
        for j, rt in enumerate(rts):
            gains = []
            for model in reasoning:
                if (task in orig.get(model, {}) and
                    task in reasoning.get(model, {}) and
                    rt in reasoning[model].get(task, {})):
                    orig_val = orig[model][task]
                    reas_val = reasoning[model][task][rt]
                    gains.append(reas_val - orig_val)
            if gains:
                matrix[i, j] = np.mean(gains)

    return matrix, tasks, rts


# ============================================================
# PLOT
# ============================================================

def plot_heatmap(matrix, tasks, rts, output_path):
    n_tasks = len(tasks)
    n_rts = len(rts)
    fig_height = 0.3 * n_tasks + 0.9
    fig, ax = plt.subplots(figsize=(5.0, fig_height))
    fig.patch.set_facecolor('white')

    # Diverging colormap: red (loss) -> white (0) -> blue (gain)
    vmax = np.nanmax(np.abs(matrix))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    from matplotlib.colors import LinearSegmentedColormap
    colors_div = ['#C62828', '#EF9A9A', '#FFFFFF', '#90CAF9', '#1565C0']
    cmap = LinearSegmentedColormap.from_list('gain', colors_div, N=256)
    cmap.set_bad(color='#F0F0F0')

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    # Annotate cells
    for i in range(n_tasks):
        for j in range(n_rts):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            if abs(val) < 0.005:
                txt = '.00'
            else:
                txt = f'{val:+.2f}'
            intensity = abs(val) / vmax if vmax > 0 else 0
            color = 'white' if intensity > 0.55 else '#1a1a1a'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=7, color=color, fontweight='medium')

    # Axis labels
    task_labels = [TASK_DISPLAY.get(t, t) for t in tasks]
    rt_labels = [REASONING_DISPLAY.get(r, r) for r in rts]

    ax.set_xticks(range(n_rts))
    ax.set_xticklabels(rt_labels, fontsize=8, rotation=0, ha='center',
                       fontweight='medium', color='#1a1a1a')
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_labels, fontsize=7.5, fontweight='medium', color='#1a1a1a')

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Cell grid lines
    for i in range(n_tasks + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1.0)
    for j in range(n_rts + 1):
        ax.axvline(j - 0.5, color='white', linewidth=1.0)

    # Domain group separators + labels
    boundary = 0
    group_mids = {}
    for grp_name, grp_tasks in TASK_GROUPS.items():
        present = [t for t in grp_tasks if t in tasks]
        if present:
            start = boundary
            boundary += len(present)
            group_mids[grp_name] = (start + boundary - 1) / 2
            if boundary < n_tasks:
                ax.axhline(boundary - 0.5, color='#555', linewidth=1.5)

    # Group labels on right
    for grp_name, mid_y in group_mids.items():
        ax.text(n_rts + 0.1, mid_y, grp_name,
                ha='left', va='center', fontsize=7, fontweight='bold',
                color='#555', fontstyle='italic')

    ax.set_xlim(-0.5, n_rts + 0.8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.12, shrink=0.7)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label('Δ nDCG@10', fontsize=7.5, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_visible(False)

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
    parser = argparse.ArgumentParser(description='Figure 7: Reasoning Gain Heatmap')
    parser.add_argument('--base_dir', type=str,
                        default='/leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results',
                        help='Path to sigir_results directory')
    parser.add_argument('--output', type=str, default='figure7_reasoning_heatmap.pdf')
    args = parser.parse_args()

    print(f"Base dir: {args.base_dir}")
    print('=' * 60)

    print("\n[1/2] Extracting original per-task scores (phase1)...")
    orig = extract_original_per_task(args.base_dir)
    print(f"  {len(orig)} models")

    print("\n[2/2] Extracting reasoning per-task scores (phase6)...")
    reasoning = extract_reasoning_per_task(args.base_dir)
    print(f"  {len(reasoning)} models with reasoning data")

    # Compute gain matrix
    print(f"\n{'=' * 60}")
    print("Computing gain matrix (avg across models)...")
    matrix, tasks, rts = compute_gain_matrix(orig, reasoning)

    # Print summary
    print(f"\n{'Task':<18}", end='')
    for rt in rts:
        print(f" {REASONING_DISPLAY.get(rt, rt):>9}", end='')
    print()
    print('-' * (18 + 10 * len(rts)))
    for i, task in enumerate(tasks):
        print(f"  {TASK_DISPLAY.get(task, task):<16}", end='')
        for j in range(len(rts)):
            val = matrix[i, j]
            if np.isnan(val):
                print(f"     {'---':>6}", end='')
            else:
                sign = '+' if val >= 0 else ''
                print(f" {sign}{val:>8.4f}", end='')
        print()

    # Overall stats
    valid = matrix[~np.isnan(matrix)]
    print(f"\nOverall: mean gain = {np.mean(valid):+.4f}, "
          f"max = {np.max(valid):+.4f}, min = {np.min(valid):+.4f}")
    print(f"Positive gains: {np.sum(valid > 0)}/{len(valid)} "
          f"({100 * np.sum(valid > 0) / len(valid):.0f}%)")

    # Plot
    print(f"\n{'=' * 60}")
    plot_heatmap(matrix, tasks, rts, args.output)

    # Dump
    summary = {}
    for i, task in enumerate(tasks):
        summary[task] = {}
        for j, rt in enumerate(rts):
            val = matrix[i, j]
            summary[task][rt] = round(val, 5) if not np.isnan(val) else None
    with open(args.output.replace('.pdf', '_data.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Data saved")


if __name__ == '__main__':
    main()