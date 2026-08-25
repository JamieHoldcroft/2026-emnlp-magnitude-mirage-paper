import sys
import json
import argparse
from pathlib import Path

def format_val(val):
    """Formats a float to .3f and removes the leading zero for academic tables (e.g., 0.694 -> .694)"""
    if val is None or val == "-":
        return ".000"
    
    formatted = f"{float(val):.3f}"
    if formatted.startswith("0."):
        return formatted[1:]
    elif formatted.startswith("-0."):
        return "-" + formatted[2:]
    return formatted

def main():
    parser = argparse.ArgumentParser(description="Format comprehensive QPP results into a massive LaTeX table.")
    parser.add_argument("--dataset", type=str, choices=["BEIR", "BRIGHT", "TEMPO"], required=True, 
                        help="The dataset family to generate the table for.")
    args = parser.parse_args()

    base_dir = Path("results/qpp_results")
    
    # The 4 RAG cutoffs
    cutoffs = ["5", "10", "25", "50"]
    
    # The exact metric names as they appear in your JSON
    qpp_metrics = [
        "MaxScore", "ScoreGap", "TopKStd", "NQC", 
        "MaxIterativeVariance", "SMV"
    ]
    
    # The evaluation measures
    eval_measures = ["auroc", "pearson", "spearman"]

    # Define the exact layout and mapping for the LaTeX table
    MODEL_CATEGORIES = {
        "Sparse Models": {
            "bm25": "BM25"
        },
        "Dense Models": {
            "bge": "BGE",
            "e5": "E5",
            "contriever": "Contriever",
            "sbert": "SBERT",
            "sf": "SFR",
            "qwen": "Qwen",
            "inst-l": "Instructor-Large"
        },
        "Reasoning Encoders": {
            "diver-retriever": "Diver-Retriever",
            "rader": "RaDeR",
            "reasonir": "ReasonIR"
        }
    }

    # 1. Gather Data based on the expected models
    results_data = {}
    for category, models in MODEL_CATEGORIES.items():
        for internal_name in models.keys():
            file_path = base_dir / args.dataset / f"{internal_name}_qpp_results.json"
            
            # Initialize empty nested dict for this model
            results_data[internal_name] = {k: {qm: {ev: "-" for ev in eval_measures} for qm in qpp_metrics} for k in cutoffs}
            
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                        for k in cutoffs:
                            for qm in qpp_metrics:
                                for ev in eval_measures:
                                    try:
                                        val = data["metrics"][qm][k][ev]
                                        if val is not None:
                                            results_data[internal_name][k][qm][ev] = val
                                    except KeyError:
                                        pass # Keep default "-"
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # 2. Generate LaTeX Table
    print("\n=== COPY THIS DIRECTLY INTO OVERLEAF ===\n")
    
    latex_code = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll ccc ccc ccc ccc ccc ccc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Model}} & \\multirow{2}{*}{\\textbf{$k$}} & \\multicolumn{3}{c}{\\textbf{MaxScore}} & \\multicolumn{3}{c}{\\textbf{ScoreGap}} & \\multicolumn{3}{c}{\\textbf{Top-$k$ Std}} & \\multicolumn{3}{c}{\\textbf{NQC}} & \\multicolumn{3}{c}{\\textbf{MaxIterVar}} & \\multicolumn{3}{c}{\\textbf{L-SMV}} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \\cmidrule(lr){15-17} \\cmidrule(lr){18-20}",
        "& & AUC & $r$ & $\\rho$ & AUC & $r$ & $\\rho$ & AUC & $r$ & $\\rho$ & AUC & $r$ & $\\rho$ & AUC & $r$ & $\\rho$ & AUC & $r$ & $\\rho$ \\\\",
        "\\midrule"
    ]
    
    for category, models in MODEL_CATEGORIES.items():
        # Optional: Add a category header row if you want them grouped visually
        latex_code.append(f"\\multicolumn{{20}}{{l}}{{\\textit{{{category}}}}} \\\\")
        latex_code.append("\\midrule")
        
        for internal_name, display_name in models.items():
            for i, k in enumerate(cutoffs):
                row_parts = []
                
                # First column: Model Name (only on the first row of the block)
                if i == 0:
                    row_parts.append(f"\\multirow{{{len(cutoffs)}}}{{*}}{{\\textbf{{{display_name}}}}}")
                else:
                    row_parts.append("")
                
                # Second column: k-cutoff
                row_parts.append(str(k))
                
                # Remaining columns: The metrics
                for qm in qpp_metrics:
                    auc = format_val(results_data[internal_name][k][qm]["auroc"])
                    r   = format_val(results_data[internal_name][k][qm]["pearson"])
                    rho = format_val(results_data[internal_name][k][qm]["spearman"])
                    row_parts.extend([auc, r, rho])
                
                # Join the row with '&' and end with '\\'
                latex_code.append(" & ".join(row_parts) + " \\\\")
            
            # Add a midrule after each model block to keep it readable
            latex_code.append("\\midrule")
            
    # Remove the very last midrule and close the table
    latex_code.pop() 
    
    latex_code.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}", # Closes resizebox
        f"\\caption{{Comprehensive zero-cost Query Performance Prediction evaluation across all target cutoffs ($k \\in \\{{5, 10, 25, 50\\}}$) for the \\textbf{{{args.dataset}}} benchmark. Predictive power is evaluated using AUROC (AUC), Pearson ($r$), and Spearman ($\\rho$).}}",
        f"\\label{{tab:massive_appendix_{args.dataset.lower()}}}",
        "\\end{table*}"
    ])
    
    print("\n".join(latex_code))
    print("\n========================================\n")

if __name__ == "__main__":
    main()