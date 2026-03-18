import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(base_dir, model, k):
    """Extracts AUROC scores for MaxScore and SMV across the three datasets."""
    datasets = ["BEIR", "TEMPO", "BRIGHT"] # Ordered by increasing cognitive complexity
    
    metrics = {"MaxScore": [], "SMV": []}
    
    for ds in datasets:
        file_path = Path(base_dir) / ds / f"{model}_qpp_results.json"
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                try:
                    max_val = data["metrics"]["MaxScore"][str(k)]["auroc"]
                    smv_val = data["metrics"]["SMV"][str(k)]["auroc"]
                    
                    # Handle cases where AUROC might be None
                    metrics["MaxScore"].append(max_val if max_val is not None else 0.5)
                    metrics["SMV"].append(smv_val if smv_val is not None else 0.5)
                except KeyError:
                    print(f"Warning: Missing metric data in {file_path}")
                    metrics["MaxScore"].append(0.5)
                    metrics["SMV"].append(0.5)
        else:
            print(f"Error: Could not find {file_path}")
            metrics["MaxScore"].append(0.5)
            metrics["SMV"].append(0.5)
            
    return datasets, metrics

def main():
    parser = argparse.ArgumentParser(description="Generate the 'Performance Cliff' Line Plot.")
    parser.add_argument("--model", type=str, required=True, help="Model internal name (e.g., 'bge', 'e5')")
    parser.add_argument("--display_name", type=str, default=None, help="Model name for the plot title (e.g., 'BGE-M3')")
    parser.add_argument("--k", type=int, default=25, help="Cutoff k to plot (default: 25)")
    parser.add_argument("--out", type=str, default="performance_cliff.png", help="Output filename (e.g., plot.png)")
    args = parser.parse_args()

    base_dir = "results/qpp_results"
    datasets, metrics = load_data(base_dir, args.model, args.k)
    
    display_name = args.display_name if args.display_name else args.model.upper()

    # --- Academic Plot Styling ---
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 14
    })

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # X-axis labels to display
    x_labels = ["Semantic\n(BEIR)", "Temporal\n(TEMPO)", "Logical\n(BRIGHT)"]
    x_positions = range(len(datasets))

    # Plot MaxScore (The Cliff) - Red, dashed, circle markers
    ax.plot(x_positions, metrics["MaxScore"], marker='o', markersize=9, 
            linestyle='--', linewidth=2.5, color='#D62728', 
            label='MaxScore ($s_1$)')

    # Plot SMV (The Stable Baseline) - Blue, solid, square markers
    ax.plot(x_positions, metrics["SMV"], marker='s', markersize=9, 
            linestyle='-', linewidth=2.5, color='#1F77B4', 
            label='L-SMV')

    # Add the "Random Guessing" baseline
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, zorder=0)
    ax.text(2.1, 0.51, 'Random Guessing (0.50)', color='gray', fontsize=11, ha='center')

    # Formatting axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(f"AUROC (Retrieval Success, NDCG@25 > 0)")
    ax.set_title(f"The Calibration Collapse of Magnitude-Based Confidence \n(ReasonIR Retriever)", pad=15, fontweight="bold")
    
    # Adjust Y-axis limits dynamically based on data, but ensure 0.45 is the minimum
    min_val = min(min(metrics["MaxScore"]), min(metrics["SMV"]))
    max_val = max(max(metrics["MaxScore"]), max(metrics["SMV"]))
    ax.set_ylim(max(0.40, min_val - 0.05), min(1.0, max_val + 0.05))

    # Legend
    ax.legend(loc="lower left", frameon=True, shadow=True, borderpad=1)

    # Clean up layout
    plt.tight_layout()

    # Save Plot
    plt.savefig(args.out, format="png", dpi=300, bbox_inches="tight")
    print(f"\nSuccess! Saved plot as {args.out}")

if __name__ == "__main__":
    main()