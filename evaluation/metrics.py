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
    rec_df = pd.read_csv(pred_file)
    y_pred = {}
    for row in rec_df.itertuples(index=False):
        u = int(row.user_id)
        v = int(row.video_id)
        if u in y_true:
            y_pred.setdefault(u, []).append(v)

    # Calculate Metrics
    # Use stem (filename without extension) and applying replacements to get a clean model name
    model_name = pred_file.name.replace("_recommendations.cf", "").replace("_recommendations.csv", "").replace(".csv", "")
    metrics = {"model": model_name}
    
    for k in k_list:
        metrics[f"recall@{k}"] = recall_at_k(y_true, y_pred, k=k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(y_true, y_pred, k=k)
        metrics[f"map@{k}"] = map_at_k(y_true, y_pred, k=k)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  -> Metrics saved to {Path(output_path).name}")
    return metrics


def compare_and_plot_models(eval_dir: Path):
    """
    Phase 2: Model comparison and visualization.
    Reads all *_metrics.json results under eval_dir, generates a summary CSV,
    and plots performance comparison bar/line charts.
    """
    json_files = list(eval_dir.glob("*_metrics.json"))
    if not json_files:
        print("No metric JSON files found to compare.")
        return

    print("\n--- Summarizing Model Results ---")
    all_metrics = []
    for jf in json_files:
        with open(jf, "r") as f:
            all_metrics.append(json.load(f))

    # Generate summary DataFrame and save as CSV
    summary_df = pd.DataFrame(all_metrics)
    summary_path = eval_dir / "all_models_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV saved to: {summary_path.name}")
    print("\nOverview:")
    print(summary_df.to_string(index=False))

    # Draw comparison plots (if plotting libraries are available)
    if PLOT_AVAILABLE:
        print("\n--- Generating Model Comparison Plots ---")
        sns.set_theme(style="whitegrid")
        
        # Convert data to long format (melted dataframe) for seaborn plotting
        # Metrics to compare (e.g., Recall@50, NDCG@50)
        try:
            plot_df = summary_df[["model", "recall@50", "ndcg@50", "map@50"]].copy()
            plot_df = plot_df.melt(id_vars="model", var_name="Metric", value_name="Score")

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=plot_df, x="Metric", y="Score", hue="model")
            plt.title("Model Comparison (Top-50 Recommendation Performance)")
            plt.ylabel("Score")
            plt.xlabel("Evaluation Metric")
            plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            plot_path = eval_dir / "model_comparison_bar_plot.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"Bar plot saved to: {plot_path.name}")
        except KeyError as e:
            print(f"Skipping plot generation. Missing specific K metric columns: {e}")


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
    if not pred_files:
        print(f"No prediction files found in {preds_dir}")
    else:
        for pred_file in pred_files:
            # Extract plain model name prefix
            model_name = pred_file.name.replace("_recommendations.cf", "").replace("_recommendations.csv", "").replace(".csv", "")
            output_json = eval_dir / f"{model_name}_metrics.json"
            
            # Compute metrics for individual model
            evaluate_model(pred_file, gt_path, output_json)

    print("\n===== PHASE 2: COMPARE & PLOT =====")
    # Compare, summarize, and plot based on generated JSON files
    compare_and_plot_models(eval_dir)


if __name__ == "__main__":
    main()
