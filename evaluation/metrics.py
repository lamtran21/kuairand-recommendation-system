"""Model-agnostic ranking metrics.

I/O contract (minimal):
- Input to each metric:
  - y_true: dict[user_id -> set(video_id)] or equivalent structure
  - y_pred: dict[user_id -> list(video_id)] ranked list
  - k: int cutoff
- Output:
  - single float score per metric in [0, 1]

Expected metrics:
- recall_at_k
- ndcg_at_k
- hit_rate_at_k
- map_at_k

==================================================
USAGE GUIDE FOR OTHER TEAM MEMBERS (ALS, SVD, etc.)
==================================================
Please do not write your own evaluation loops in the
run_xxx.py scripts. Instead, import these functions!

Usage Example:
--------------
from evaluation.metrics import recall_at_k, ndcg_at_k, map_at_k

# 1. Load ground truth (y_true) from test_ground_truth.json
import json
with open("../data/processed/test_ground_truth.json", "r") as f:
    gt_raw = json.load(f)
y_true = {int(k): int(v) for k, v in gt_raw.items()}

# 2. Convert your prediction DataFrame into y_pred dictionary
# Example logic:
# y_pred = {}
# for row in rec_df.itertuples(index=False):
#     y_pred.setdefault(int(row.user_id), []).append(int(row.video_id))

# 3. Calculate metrics
recall_50 = recall_at_k(y_true, y_pred, k=50)
ndcg_50 = ndcg_at_k(y_true, y_pred, k=50)
map_50 = map_at_k(y_true, y_pred, k=50)

print(f"Recall@50: {recall_50:.4f}, NDCG@50: {ndcg_50:.4f}, MAP@50: {map_50:.4f}")
"""

import numpy as np


def recall_at_k(y_true: dict[int, int], y_pred: dict[int, list[int]], k: int = 10) -> float:
    """Return Recall@k."""
    vals = []
    for user, item in y_true.items():
        preds = y_pred.get(user, [])[:k]
        vals.append(1.0 if item in preds else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def ndcg_at_k(y_true: dict[int, int], y_pred: dict[int, list[int]], k: int = 10) -> float:
    """Return NDCG@k."""
    vals = []
    for user, item in y_true.items():
        preds = y_pred.get(user, [])[:k]
        if item in preds:
            rank = preds.index(item) + 1
            vals.append(1.0 / np.log2(rank + 1))
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def hit_rate_at_k(y_true: dict[int, int], y_pred: dict[int, list[int]], k: int = 10) -> float:
    """Return HitRate@k."""
    return recall_at_k(y_true, y_pred, k)


def map_at_k(y_true: dict[int, int], y_pred: dict[int, list[int]], k: int = 10) -> float:
    """Return MAP@k."""
    vals = []
    for user, item in y_true.items():
        preds = y_pred.get(user, [])[:k]
        if item in preds:
            rank = preds.index(item) + 1
            vals.append(1.0 / rank)
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def mrr_at_k(y_true: dict[int, int], y_pred: dict[int, list[int]], k: int = 10) -> float:
    """Return MRR@k (equivalent to MAP for leave-one-out but widely used terminology)."""
    return map_at_k(y_true, y_pred, k)


def item_coverage_at_k(y_pred: dict[int, list[int]], k: int = 10) -> int:
    """Return Absolute Item Coverage@k (number of unique items recommended)."""
    recommended_items = set()
    for preds in y_pred.values():
        recommended_items.update(preds[:k])
    return len(recommended_items)


# -------------------------------------------------------------
# AUTOMATED EVALUATION PIPELINE
# Run this script directly to recursively scan outputs/model_predictions
# and generate json metrics + comparison plot.
# -------------------------------------------------------------
import json
import argparse
import pandas as pd
from pathlib import Path
import sys

# Ensure current directory is in path so we can import evaluation.metrics
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Warning: matplotlib or seaborn not installed. Plotting will be skipped.")


def evaluate_model(pred_path: str, gt_path: str, output_path: str, k_list: list = [10, 20, 50, 100]):
    """
    Phase 1: Calculate metrics for a single model.
    Reads a single prediction file (CSV format), compares it with Ground Truth,
    calculates evaluation metrics, and saves as an independent JSON file.
    """
    pred_file = Path(pred_path)
    if not pred_file.exists():
        print(f"Prediction file not found: {pred_path}")
        return None
        
    print(f"Evaluating {pred_file.name}...")

    # Load Ground Truth
    with open(gt_path, "r") as f:
        gt_raw = json.load(f)
    y_true = {int(k): int(v) for k, v in gt_raw.items()}

    # Load Predictions
    if str(pred_file).endswith('.parquet'):
        rec_df = pd.read_parquet(pred_file, engine='fastparquet')
    else:
        rec_df = pd.read_csv(pred_file)
    y_pred = {}
    for row in rec_df.itertuples(index=False):
        u = int(row.user_id)
        v = int(row.video_id)
        if u in y_true:
            y_pred.setdefault(u, []).append(v)

    # Calculate Metrics
    # Use stem (filename without extension) and applying replacements to get a clean model name
    model_name = pred_file.name.replace("_recommendations.cf", "").replace("_recommendations.csv", "").replace("_recommendations.parquet", "").replace(".csv", "").replace(".parquet", "")
    metrics = {"model": model_name}
    
    for k in k_list:
        metrics[f"recall@{k}"] = recall_at_k(y_true, y_pred, k=k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(y_true, y_pred, k=k)
        metrics[f"mrr@{k}"] = mrr_at_k(y_true, y_pred, k=k)
        metrics[f"coverage@{k}"] = item_coverage_at_k(y_pred, k=k)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  -> Metrics saved to {Path(output_path).name}")
    return metrics


def generate_markdown_report(summary_df: pd.DataFrame, eval_dir: Path):
    """Generates an analytical Markdown report for the final paper."""
    report_path = eval_dir / "evaluation_report.md"
    
    # Sort models by NDCG@50
    sorted_df = summary_df.sort_values(by="ndcg@50", ascending=False)
    best_ndcg_model = sorted_df.iloc[0]["model"]
    best_ndcg_score = sorted_df.iloc[0]["ndcg@50"]
    
    sorted_df_recall = summary_df.sort_values(by="recall@50", ascending=False)
    best_recall_model = sorted_df_recall.iloc[0]["model"]
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Recommendation Models Evaluation Report\n\n")
        f.write("## 1. Overview\n")
        f.write("This report provides a comprehensive comparison of all recommendation models evaluated in the pipeline.\n\n")
        f.write("### Executive Summary\n")
        f.write(f"- **Best overall ranking model (NDCG@50):** `{best_ndcg_model}` with a score of {best_ndcg_score:.4f}.\n")
        f.write(f"- **Best recall model (Recall@50):** `{best_recall_model}` with a score of {sorted_df_recall.iloc[0]['recall@50']:.4f}.\n\n")
        
        f.write("## 2. Quantitative Comparison Table\n")
        f.write("The following table presents the full metrics across different cutoff points (K).\n\n")
        # Try to use pandas to_markdown which requires tabulate
        try:
            f.write(summary_df.to_markdown(index=False) + "\n\n")
        except ImportError:
            # Fallback if tabulate is not available
            f.write(summary_df.to_csv(index=False) + "\n\n")
        
        f.write("## 3. Visual Analysis\n")
        f.write("Please refer to the following generated charts in this directory:\n")
        f.write("- `model_comparison_bar_plot.png`: Direct comparison of top models at K=50.\n")
        f.write("- `model_comparison_radar.png`: **[NEW]** Multidimensional polygon comparison across Normalized Recall, NDCG, MRR, and Coverage.\n")
        f.write("- `accuracy_vs_diversity_bubble.png`: **[NEW]** Scatter plot demonstrating the trade-off between recommending popular relevant items and diverse niche items.\n")
        f.write("- `recall_trend_plot.png`: Line chart showing how Recall scales as K increases (10 to 100).\n")
        f.write("- `ndcg_trend_plot.png`: Line chart showing how ranking quality (NDCG) behaves across different K values.\n\n")
        
        f.write("## 4. Relative Improvement over Baseline\n")
        f.write("Comparing all models against the fundamental Popularity baseline (percentage improvement):\n\n")
        
        if "popularity" in summary_df["model"].values:
            pop_row = summary_df[summary_df["model"] == "popularity"].iloc[0]
            imp_df = summary_df.copy()
            for col in imp_df.columns:
                if col != "model":
                    imp_df[col] = ((imp_df[col] - pop_row[col]) / pop_row[col] * 100).map("{:+.2f}%".format)
            try:
                f.write(imp_df.to_markdown(index=False) + "\n\n")
            except ImportError:
                f.write(imp_df.to_csv(index=False) + "\n\n")
        else:
            f.write("*Popularity baseline not found for comparison.*\n\n")

        f.write("## 5. Conclusion & Insights\n")
        f.write("1. **Tree-based re-ranking** typically provides the highest performance improvements due to complex feature interactions.\n")
        f.write("2. **Matrix Factorization** (ALS/SVD) provides strong collaborative signals but differs widely in item coverage compared to ItemCF.\n")
        f.write("3. **Accuracy vs. Diversity Trade-off**: High scoring models might occasionally suffer from low coverage (popularity bias). Using the bubble chart helps determine the most balanced model.\n")

    print(f"Analytical markdown report generated at: {report_path.name}")


def compare_and_plot_models(eval_dir: Path, all_metrics: list):
    """
    Phase 2: Model comparison, visualization, and paper report generation.
    Generates summary CSV, multiple plots, and an analytical Markdown document.
    """
    if not all_metrics:
        print("No metrics available to compare.")
        return

    print("\n--- Summarizing Model Results ---")
    summary_df = pd.DataFrame(all_metrics)
    summary_path = eval_dir / "all_models_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV saved to: {summary_path.name}")
    print("\nOverview:")
    print(summary_df.to_string(index=False))

    # Generate Markdown Report inside evaluation subfolder
    generate_markdown_report(summary_df, eval_dir)

    # Draw comparison plots
    if PLOT_AVAILABLE:
        print("\n--- Generating Rich Visualizations for Paper ---")
        sns.set_theme(style="whitegrid", font_scale=1.1)
        
        # Plot 1: Bar Plot for K=50
        try:
            plot_df = summary_df[["model", "recall@50", "ndcg@50", "mrr@50"]].copy()
            plot_df = plot_df.melt(id_vars="model", var_name="Metric", value_name="Score")

            plt.figure(figsize=(10, 6))
            sns.barplot(data=plot_df, x="Metric", y="Score", hue="model", palette="Set2")
            plt.title("Model Comparison (Top-50 Recommendation Performance)", pad=15)
            plt.ylabel("Score")
            plt.xlabel("Evaluation Metric")
            plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(eval_dir / "model_comparison_bar_plot.png", dpi=300)
            plt.close()
            print("Generated: model_comparison_bar_plot.png")
        except KeyError as e:
            print(f"Skipping bar plot generation. {e}")

        # Plot 4: Diversity vs Accuracy Tradeoff scatter plot
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=summary_df, x="coverage@50", y="ndcg@50", hue="model", s=250, palette="Set2", style="model")
            
            # Add labels to points
            for i, row in summary_df.iterrows():
                plt.text(row["coverage@50"] * 1.01, row["ndcg@50"] * 1.0, row["model"], 
                         horizontalalignment='left', size='small', color='black', weight='semibold')

            plt.title("Accuracy vs. Diversity Trade-off (At K=50)", pad=15)
            plt.ylabel("Accuracy (NDCG@50) -> Higher is Better")
            plt.xlabel("Diversity (Unique Items Recommended) -> Higher is Better")
            plt.legend([],[], frameon=False)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(eval_dir / "accuracy_vs_diversity_bubble.png", dpi=300)
            plt.close()
            print("Generated: accuracy_vs_diversity_bubble.png")
        except KeyError as e:
            print(f"Skipping trade-off plot generation. {e}")

        # Plot 5: Radar Chart for Multi-dimensional comparison
        try:
            radar_cols = ["recall@50", "ndcg@50", "mrr@50", "coverage@50"]
            radar_df = summary_df[["model"] + radar_cols].copy()
            
            # Min-Max Scale for Radar bounds [0, 1]
            for col in radar_cols:
                col_min = radar_df[col].min()
                col_max = radar_df[col].max()
                if col_max - col_min > 0:
                    radar_df[col] = (radar_df[col] - col_min) / (col_max - col_min)
                else:
                    radar_df[col] = 1.0

            labels = np.array(['Recall', 'NDCG', 'MRR', 'Coverage (Diversity)'])
            num_vars = len(labels)
            
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            colors = sns.color_palette("Set2", len(radar_df))
            for i, row in radar_df.iterrows():
                values = row[radar_cols].values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=2, label=row["model"], color=colors[i])
                ax.fill(angles, values, alpha=0.15, color=colors[i])
                
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), labels)
            plt.title("Multidimensional Comparison (Min-Max Scaled at K=50)", pad=20, y=1.08)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            plt.savefig(eval_dir / "model_comparison_radar.png", dpi=300)
            plt.close()
            print("Generated: model_comparison_radar.png")
        except Exception as e:
            print(f"Skipping radar chart generation. {e}")

        # Helper function for Plot 2 & 3: Trend lines
        def plot_trend_line(metric_prefix: str, output_filename: str, title: str):
            columns = ["model"] + [f"{metric_prefix}@{k}" for k in [10, 20, 50, 100] if f"{metric_prefix}@{k}" in summary_df.columns]
            if len(columns) <= 1:
                return
            
            trend_df = summary_df[columns].copy()
            # Rename columns to just K values for x-axis scale representation
            rename_map = {f"{metric_prefix}@{k}": k for k in [10, 20, 50, 100] if f"{metric_prefix}@{k}" in summary_df.columns}
            trend_df.rename(columns=rename_map, inplace=True)
            trend_df = trend_df.melt(id_vars="model", var_name="K", value_name="Score")
            
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=trend_df, x="K", y="Score", hue="model", marker="o", linewidth=2.5, markersize=8, palette="Set2")
            plt.title(title, pad=15)
            plt.ylabel(f"{metric_prefix.upper()} Score")
            plt.xlabel("Cutoff (K)")
            plt.xticks([10, 20, 50, 100])
            plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(eval_dir / output_filename, dpi=300)
            plt.close()
            print(f"Generated: {output_filename}")

        plot_trend_line("recall", "recall_trend_plot.png", "Recall Trend (K=10 to 100)")
        plot_trend_line("ndcg", "ndcg_trend_plot.png", "NDCG Trend (K=10 to 100)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate & Compare Models")
    parser.add_argument("--gt", default=PROJECT_ROOT / "data/processed/test_ground_truth.json", help="Path to ground truth json")
    parser.add_argument("--preds_dir", default=PROJECT_ROOT / "outputs/model_predictions", help="Directory containing model predictions")
    parser.add_argument("--eval_dir", default=PROJECT_ROOT / "outputs/evaluation", help="Directory to save evaluation results")
    args = parser.parse_args()

    preds_dir = Path(args.preds_dir)
    eval_dir = Path(args.eval_dir)
    gt_path = Path(args.gt)

    if not preds_dir.exists():
        print(f"Predictions directory not found: {preds_dir}")
        return

    eval_dir.mkdir(parents=True, exist_ok=True)

    print("===== PHASE 1: EVALUATE INDIVIDUAL MODELS =====")
    # Find all prediction files (.cf, .csv, etc.)
    pred_files = list(preds_dir.glob("*_recommendations.*"))
    all_metrics = []
    if not pred_files:
        print(f"No prediction files found in {preds_dir}")
    else:
        for pred_file in pred_files:
            # Extract plain model name prefix
            model_name = pred_file.name.replace("_recommendations.cf", "").replace("_recommendations.csv", "").replace("_recommendations.parquet", "").replace(".csv", "").replace(".parquet", "")
            
            # Create a specific subdirectory for each model
            model_eval_dir = eval_dir / model_name
            model_eval_dir.mkdir(parents=True, exist_ok=True)
            output_json = model_eval_dir / f"{model_name}_metrics.json"
            
            # Compute metrics for individual model
            metrics = evaluate_model(pred_file, gt_path, output_json)
            if metrics:
                all_metrics.append(metrics)

    print("\n===== PHASE 2: COMPARE & PLOT =====")
    # Compare, summarize, and plot based on collected metrics
    compare_and_plot_models(eval_dir, all_metrics)


if __name__ == "__main__":
    main()
