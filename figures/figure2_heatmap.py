"""
Figure 2: Reproducibility Heatmap
nDCG@10 across all (model x task) combinations from phase1_cache_build.

Structure: phase1_cache_build/{model}/{task}/{task}_{model}_long_False/results.json

Usage:
  python figure2_heatmap.py --base_dir /leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results
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
# DATA EXTRACTION
# ============================================================

def extract_all_ndcg10(base_dir):
    """
    Extract nDCG@10 for every (model, task) pair from phase1_cache_build.
    Returns: dict of {model: {task: nDCG@10}}
    """
    phase_dir = os.path.join(base_dir, 'phase1_cache_build')
    if not os.path.isdir(phase_dir):
        print(f"[ERROR] Missing: {phase_dir}")
        return {}

    data = defaultdict(dict)

    for model in discover_subfolders(phase_dir):
        if model not in VALID_MODELS:
            continue  # skip backup, old, etc.
        model_path = os.path.join(phase_dir, model)
        for task in discover_subfolders(model_path):
            task_path = os.path.join(model_path, task)
            found = False
            # Check subfolders first (e.g., {task}_{model}_long_False/)
            for sub in discover_subfolders(task_path):
                rpath = os.path.join(task_path, sub, 'results.json')
                result = load_json(rpath)
                if result and 'NDCG@10' in result:
                    data[model][task] = result['NDCG@10']
                    found = True
                    break
            # Also check directly in task folder
            if not found:
                rpath = os.path.join(task_path, 'results.json')
                result = load_json(rpath)
                if result and 'NDCG@10' in result:
                    data[model][task] = result['NDCG@10']

    return dict(data)


# ============================================================
# DISPLAY NAMES
# ============================================================

# Only these model folder names are valid — skip backup, old, etc.
VALID_MODELS = {
    'bge', 'bm25', 'contriever', 'diver-retriever', 'e5', 'grit',
    'inst-l', 'inst-xl', 'nomic', 'qwen', 'qwen2', 'rader',
    'reasonir', 'sbert', 'sf',
}

MODEL_DISPLAY = {
    'bge': 'BGE',
    'bm25': 'BM25',
    'contriever': 'Contriever',
    'diver-retriever': 'DIVER',
    'e5': 'E5-Large',
    'grit': 'GritLM',
    'inst-l': 'Inst-L',
    'inst-xl': 'Inst-XL',
    'nomic': 'Nomic',
    'qwen': 'Qwen',
    'qwen2': 'Qwen2',
    'rader': 'RADER',
    'reasonir': 'ReasonIR',
    'sbert': 'SBERT',
    'sf': 'SFR',
}

TASK_DISPLAY = {
    'aops': 'AoPS',
    'biology': 'Biology',
    'earth_science': 'Earth Sci.',
    'economics': 'Economics',
    'leetcode': 'LeetCode',
    'pony': 'Pony',
    'psychology': 'Psychology',
    'robotics': 'Robotics',
    'stackoverflow': 'StackOF',
    'sustainable_living': 'Sust. Living',
    'theoremqa_questions': 'TheoremQA-Q',
    'theoremqa_theorems': 'TheoremQA-T',
}

# Preferred ordering: group by type
MODEL_ORDER = [
    'bm25',              # Sparse
    'sbert', 'contriever', 'nomic', 'bge', 'e5',  # Dense (smaller)
    'inst-l', 'inst-xl', 'sf',                      # Instruction-tuned
    'grit', 'qwen', 'qwen2',                        # LLM-based
    'diver-retriever', 'rader', 'reasonir',          # Reasoning-specialized
]

TASK_ORDER = [
    'aops', 'leetcode', 'theoremqa_questions', 'theoremqa_theorems',  # STEM
    'biology', 'earth_science', 'economics', 'psychology',             # Science/Social
    'robotics', 'stackoverflow', 'pony', 'sustainable_living',        # Technical/Other
]


# ============================================================
# PLOTTING
# ============================================================

def plot_heatmap(data, output_path):
    """
    Plot heatmap of nDCG@10 across (model x task).
    Sized for single-column SIGIR paper.
    """
    # Determine models and tasks present in data
    all_models = set(data.keys())
    all_tasks = set()
    for tasks in data.values():
        all_tasks.update(tasks.keys())

    # Use preferred order, only include those with data
    models = [m for m in MODEL_ORDER if m in all_models]
    # Add any models not in our order list
    models += sorted([m for m in all_models if m not in models])

    tasks = [t for t in TASK_ORDER if t in all_tasks]
    tasks += sorted([t for t in all_tasks if t not in tasks])

    print(f"  Models ({len(models)}): {models}")
    print(f"  Tasks ({len(tasks)}): {tasks}")

    # Build matrix (models = rows, tasks = columns)
    matrix = np.full((len(models), len(tasks)), np.nan)
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            if task in data.get(model, {}):
                matrix[i, j] = data[model][task]

    # Compute row averages (for annotation)
    row_avgs = np.nanmean(matrix, axis=1)

    # --- Figure ---
    # Full-width (two-column) figure for readability
    fig_width = 7.2
    fig_height = 0.32 * len(models) + 1.4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')

    # Custom colormap: white (low) -> blue (high)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#FFFFFF', '#DCEEFB', '#9EC5E8', '#4A90D9', '#1B4F8A', '#0A2647']
    cmap = LinearSegmentedColormap.from_list('custom_blue', colors_list, N=256)
    cmap.set_bad(color='#F0F0F0')  # gray for missing data

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # --- Annotate cells with values ---
    for i in range(len(models)):
        for j in range(len(tasks)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            # Dark text on light cells, white text on dark cells
            threshold = vmin + 0.6 * (vmax - vmin)
            color = 'white' if val > threshold else '#1a1a1a'
            fontweight = 'bold' if val > threshold else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight=fontweight)

    # --- Axis labels ---
    model_labels = [MODEL_DISPLAY.get(m, m) for m in models]
    task_labels = [TASK_DISPLAY.get(t, t) for t in tasks]

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(task_labels, fontsize=9, rotation=45, ha='right',
                       fontweight='medium', color='#1a1a1a')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels, fontsize=9, fontweight='medium', color='#1a1a1a')

    # Move x labels to top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # --- Add average column ---
    avg_x = len(tasks)  # position right after last task column
    for i, avg in enumerate(row_avgs):
        ax.text(avg_x + 0.3, i, f'{avg:.3f}', ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#1a1a1a',
                fontstyle='italic')
    ax.text(avg_x + 0.3, -0.8, 'Avg', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#1a1a1a', rotation=45)

    # Extend xlim to make room for avg column
    ax.set_xlim(-0.5, len(tasks) + 0.1)

    # --- Grid lines between cells ---
    for i in range(len(models) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.8)
    for j in range(len(tasks) + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.8)

    # --- Category separators (thicker lines) ---
    # Model category boundaries
    model_categories = {
        'Sparse': ['bm25'],
        'Dense': ['sbert', 'contriever', 'nomic', 'bge', 'e5'],
        'Instruct': ['inst-l', 'inst-xl', 'sf'],
        'LLM-based': ['grit', 'qwen', 'qwen2'],
        'Reasoning': ['diver-retriever', 'rader', 'reasonir'],
    }
    boundary_idx = 0
    for cat_name, cat_models in model_categories.items():
        present = [m for m in cat_models if m in models]
        if present:
            boundary_idx += len(present)
            if boundary_idx < len(models):
                ax.axhline(boundary_idx - 0.5, color='#666666', linewidth=1.2)

    # Task category boundaries (STEM | Science | Technical)
    task_cats = [
        ['aops', 'leetcode', 'theoremqa_questions', 'theoremqa_theorems'],
        ['biology', 'earth_science', 'economics', 'psychology'],
        ['robotics', 'stackoverflow', 'pony', 'sustainable_living'],
    ]
    t_boundary = 0
    for cat in task_cats:
        present = [t for t in cat if t in tasks]
        t_boundary += len(present)
        if t_boundary < len(tasks):
            ax.axvline(t_boundary - 0.5, color='#666666', linewidth=1.2)

    # --- Colorbar ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.08, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('nDCG@10', fontsize=9, fontweight='bold')

    # Remove outer spines
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
    parser = argparse.ArgumentParser(description='Figure 2: Reproducibility Heatmap')
    parser.add_argument('--base_dir', type=str,
                        default='/leonardo_scratch/fast/L-AUT_024/llm-retrieval/sigir_results',
                        help='Path to sigir_results directory')
    parser.add_argument('--output', type=str, default='figure2_heatmap.pdf')
    args = parser.parse_args()

    print(f"Base dir: {args.base_dir}")
    print('=' * 60)

    # Extract
    print("\nExtracting nDCG@10 from phase1_cache_build...")
    data = extract_all_ndcg10(args.base_dir)
    print(f"Found {len(data)} models")

    # Print summary
    for model in sorted(data.keys()):
        tasks = data[model]
        avg = np.mean(list(tasks.values()))
        print(f"  {model}: {len(tasks)} tasks, avg nDCG@10 = {avg:.4f}")

    # Plot
    print(f"\n{'=' * 60}")
    print("Generating heatmap...")
    plot_heatmap(data, args.output)

    # Dump data
    summary_path = args.output.replace('.pdf', '_data.json')
    dump = {}
    for m, tasks in data.items():
        dump[m] = {t: round(v, 5) for t, v in tasks.items()}
    with open(summary_path, 'w') as f:
        json.dump(dump, f, indent=2)
    print(f"[OK] Data: {summary_path}")


if __name__ == '__main__':
    main()