"""Two-stage recommendation with ALS + simple GBM re-ranking (no side features).

This script mirrors the original GBM setup that only uses:
    - ALS score and rank.
    - ItemCF score and rank (if available).
    - User interaction count.
    - Item popularity.

It is separated from ``run_two_stage_gbm.py`` so that the full
side-feature model and this simpler variant can be run and compared
independently without modifying each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gbm_reranker import GBMRerankerModel, score_candidates, train_reranker


EVAL_K = [10, 20, 50, 100, 200]


def recall_at_k(y_true: Dict[int, int], y_pred: Dict[int, List[int]], k: int) -> float:
    vals = []
    for user, item in y_true.items():
        preds = y_pred.get(user, [])[:k]
        vals.append(1.0 if item in preds else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def hit_rate_at_k(y_true: Dict[int, int], y_pred: Dict[int, List[int]], k: int) -> float:
    return recall_at_k(y_true, y_pred, k)


def ndcg_at_k(y_true: Dict[int, int], y_pred: Dict[int, List[int]], k: int) -> float:
    vals = []
    for user, item in y_true.items():
        preds = y_pred.get(user, [])[:k]
        if item in preds:
            rank = preds.index(item) + 1
            vals.append(1.0 / np.log2(rank + 1))
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def map_at_k(y_true: Dict[int, int], y_pred: Dict[int, List[int]], k: int) -> float:
    vals = []
    for user, item in y_true.items():
        preds = y_pred.get(user, [])[:k]
        if item in preds:
            rank = preds.index(item) + 1
            vals.append(1.0 / rank)
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def _make_pred_dict(rec_df: pd.DataFrame) -> Dict[int, List[int]]:
    y_pred: Dict[int, List[int]] = {}
    for row in rec_df.itertuples(index=False):
        y_pred.setdefault(int(row.user_id), []).append(int(row.video_id))
    return y_pred


def _evaluate_multi_k(
    y_true: Dict[int, int],
    y_pred: Dict[int, List[int]],
    eval_k_list: List[int],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in eval_k_list:
        out[f"recall@{k}"] = recall_at_k(y_true, y_pred, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(y_true, y_pred, k)
        out[f"ndcg@{k}"] = ndcg_at_k(y_true, y_pred, k)
        out[f"map@{k}"] = map_at_k(y_true, y_pred, k)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage recommendation: ALS candidates + simple GBM re-ranking.",
    )
    parser.add_argument(
        "--train-matrix",
        type=Path,
        default=PROJECT_ROOT / "data/processed/train_matrix.npz",
    )
    parser.add_argument(
        "--test-ground-truth",
        type=Path,
        default=PROJECT_ROOT / "data/processed/test_ground_truth.json",
    )
    parser.add_argument(
        "--als-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs/model_predictions/als_recommendations.csv",
    )
    parser.add_argument(
        "--itemcf-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs/model_predictions/itemcf_recommendations.cf",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/model_predictions",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/evaluation/two_stage_gbm",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models",
    )
    return parser.parse_args()


def _build_feature_table(
    train_matrix: sp.csr_matrix,
    y_true: Dict[int, int],
    als_df: pd.DataFrame,
    itemcf_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Construct feature table for GBM from ALS (and optional ItemCF) outputs.

    This variant only uses ALS/ItemCF scores + basic interaction stats.
    """
    als_df = als_df.copy()
    als_df.rename(columns={"score": "als_score", "rank": "als_rank"}, inplace=True)

    # Merge ItemCF scores if available.
    if itemcf_df is not None and not itemcf_df.empty:
        tmp = itemcf_df.rename(columns={"score": "itemcf_score", "rank": "itemcf_rank"})[
            ["user_id", "video_id", "itemcf_score", "itemcf_rank"]
        ]
        feat_df = als_df.merge(tmp, on=["user_id", "video_id"], how="left")
    else:
        feat_df = als_df.copy()
        feat_df["itemcf_score"] = np.nan
        feat_df["itemcf_rank"] = np.nan

    # User/item-level statistics from the interaction matrix.
    train_matrix = train_matrix.tocsr()
    user_interactions = np.diff(train_matrix.indptr).astype(np.int32)
    item_popularity = np.asarray(train_matrix.sum(axis=0)).ravel().astype(np.float32)

    feat_df["user_interactions"] = user_interactions[feat_df["user_id"].values]
    feat_df["item_popularity"] = item_popularity[feat_df["video_id"].values]

    # Fill any missing ItemCF values with neutral defaults.
    feat_df["itemcf_score"] = feat_df["itemcf_score"].fillna(0.0)
    feat_df["itemcf_rank"] = feat_df["itemcf_rank"].fillna(999.0)

    # Label: 1 if this (user, item) pair is the ground-truth next item.
    gt_arr = np.array([y_true.get(int(u), -1) for u in feat_df["user_id"].values])
    feat_df["label"] = (feat_df["video_id"].values == gt_arr).astype(np.int32)

    return feat_df


def main() -> None:
    args = parse_args()
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_matrix = sp.load_npz(args.train_matrix).tocsr()

    with open(args.test_ground_truth, "r") as f:
        gt_raw = json.load(f)
    y_true: Dict[int, int] = {int(k): int(v) for k, v in gt_raw.items()}

    als_df = pd.read_csv(args.als_predictions)

    itemcf_df: pd.DataFrame | None
    if args.itemcf_predictions.exists():
        itemcf_df = pd.read_csv(args.itemcf_predictions)
    else:
        itemcf_df = None

    feat_df = _build_feature_table(train_matrix, y_true, als_df, itemcf_df)

    feature_cols = [
        "als_score",
        "als_rank",
        "itemcf_score",
        "itemcf_rank",
        "user_interactions",
        "item_popularity",
    ]

    model: GBMRerankerModel = train_reranker(
        train_df=feat_df,
        feature_columns=feature_cols,
        label_column="label",
        random_state=42,
    )

    # Score all candidates and construct a re-ranked recommendation list.
    feat_df = feat_df.copy()
    feat_df["gbm_score"] = score_candidates(model, feat_df)

    rec_rows: list[dict[str, int | float]] = []
    max_k = max(EVAL_K)

    for user_id, group in feat_df.groupby("user_id"):
        group_sorted = group.sort_values("gbm_score", ascending=False)
        top = group_sorted.head(max_k)

        rank = 1
        for row in top.itertuples(index=False):
            rec_rows.append(
                {
                    "user_id": int(row.user_id),
                    "video_id": int(row.video_id),
                    "score": float(row.gbm_score),
                    "rank": rank,
                }
            )
            rank += 1

    rec_df = pd.DataFrame(rec_rows, columns=["user_id", "video_id", "score", "rank"])

    y_pred = _make_pred_dict(rec_df)
    metrics = _evaluate_multi_k(y_true, y_pred, EVAL_K)

    # Feature importances for interpretability.
    importances = dict(
        zip(model.feature_columns, model.gbm.feature_importances_.tolist()),
    )

    pred_path = args.pred_dir / "two_stage_gbm_simple_recommendations.csv"
    metrics_path = args.eval_dir / "two_stage_gbm_simple_metrics.json"
    model_path = args.model_dir / "two_stage_gbm_simple_model.pkl"
    fi_path = args.eval_dir / "two_stage_gbm_simple_feature_importances.json"

    rec_df.to_csv(pred_path, index=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(fi_path, "w") as f:
        json.dump(importances, f, indent=2)

    # Persist the model via pickle for potential reuse.
    import pickle

    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Two-stage simple GBM re-ranking complete.")
    print(f"Predictions: {pred_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Feature importances: {fi_path}")
    print(f"Model: {model_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

