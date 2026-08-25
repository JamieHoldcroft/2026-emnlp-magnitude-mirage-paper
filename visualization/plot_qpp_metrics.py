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
    parser = argparse.ArgumentParser(description="Format QPP results into a nice console and LaTeX table.")
    parser.add_argument("--k", type=str, choices=["5", "10", "25", "50"], required=True, help="Cutoff k (e.g., 10)")
    parser.add_argument("--metric", type=str, choices=["pearson", "spearman", "auroc"], required=True, help="Evaluation metric")
    args = parser.parse_args()

    base_dir = Path("results/qpp_results")
    datasets = ["BEIR", "BRIGHT", "TEMPO"]
    qpp_metrics = ["MaxScore", "ScoreGap", "SMV"] 

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
            results_data[internal_name] = {}
            for ds in datasets:
                file_path = base_dir / ds / f"{internal_name}_qpp_results.json"
                vals = {"MaxScore": "-", "ScoreGap": "-", "SMV": "-"}
                
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            for qm in qpp_metrics:
                                try:
                                    val = data["metrics"][qm][args.k][args.metric]
                                    if val is not None:
                                        vals[qm] = val
                                except KeyError:
                                    pass
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                
                results_data[internal_name][ds] = vals

    # 2. Print Console Table
    print("\n" + "="*95)
    print(f" QPP RESULTS: {args.metric.upper()} @ {args.k}".center(95))
    print("="*95)
    
    header = f"| {'Model':<18} |"
    for ds in datasets:
        header += f" {ds.center(19)} |"
    print(header)
    
    sub_header = f"| {'':<18} |"
    for ds in datasets:
        sub_header += f" {'Max':<5} {'Gap':<5} {'SMV':<5}   |"
    print(sub_header)
    print("-" * 95)

    for category, models in MODEL_CATEGORIES.items():
        print(f"| {category:<91} |")
        for internal_name, display_name in models.items():
            row = f"|  {display_name:<17} |"
            for ds in datasets:
                max_s = format_val(results_data[internal_name][ds]['MaxScore'])
                gap = format_val(results_data[internal_name][ds]['ScoreGap'])
                smv = format_val(results_data[internal_name][ds]['SMV'])
                row += f" {max_s:<5} {gap:<5} {smv:<5}   |"
            print(row)
        print("-" * 95)
    print("\n")


    # 3. Print LaTeX Table
    print("=== COPY THIS DIRECTLY INTO OVERLEAF ===\n")
    
    latex_code = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt} % Adjust horizontal spacing",
        "\\begin{tabular}{l ccc ccc ccc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{3}{c}{\\textbf{BEIR (Semantic)}} & \\multicolumn{3}{c}{\\textbf{BRIGHT (Logical)}} & \\multicolumn{3}{c}{\\textbf{TEMPO (Temporal)}} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        "& Max Score & Score Gap & SMV & Max Score & Score Gap & SMV & Max Score & Score Gap & SMV \\\\",
        "\\midrule"
    ]
    
    for idx, (category, models) in enumerate(MODEL_CATEGORIES.items()):
        latex_code.append(f"\\textit{{{category}}} & & & & & & & & & \\\\")
        for internal_name, display_name in models.items():
            row_str = f"{display_name}"
            for ds in datasets:
                max_s = format_val(results_data[internal_name][ds]['MaxScore'])
                gap = format_val(results_data[internal_name][ds]['ScoreGap'])
                smv = format_val(results_data[internal_name][ds]['SMV'])
                row_str += f" & {max_s} & {gap} & {smv}"
            row_str += " \\\\"
            latex_code.append(row_str)
        
        # Add a midrule after sections, except for the last one
        if idx < len(MODEL_CATEGORIES) - 1:
            latex_code.append("\\midrule")
        
    latex_code.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Comparison of zero-cost QPP metrics across three cognitive tiers. We report {args.metric.upper()}@{args.k} scores for binary retrieval success ($NDCG@{args.k} > 0$). Max Score refers to ($s_1$), Score Gap refers to ($s_1 - s_{{{args.k}}}$), and SMV represents the Score Magnitude and Variance metric \\citep{{tao2014query}}.}}",
        "\\label{tab:main_results}",
        "\\end{table*}"
    ])
    
    print("\n".join(latex_code))
    print("\n========================================\n")

if __name__ == "__main__":
    main()