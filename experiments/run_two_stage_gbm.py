"""Two-stage recommendation with ALS + gradient boosting re-ranking.

Stage 1 (already implemented elsewhere):
    - ALS generates top-K candidate items per user from the implicit
      interaction matrix (train_matrix.npz).

Stage 2 (this script):
    - Build a feature table for each (user, item) candidate using:
        * ALS score and rank.
        * ItemCF score and rank (if available).
        * User interaction count.
        * Item popularity.
    - Train a GradientBoostingClassifier to predict whether a candidate
      is the held-out ground-truth item (label = 1).
    - Re-rank ALS candidates by the GBM score.
    - Evaluate Recall@K / HitRate@K / NDCG@K / MAP@K and save outputs.

This does not modify the ALS implementation; it only consumes the
existing ALS and ItemCF outputs.
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


def _load_side_feature_tables() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load user and item side-feature tables if present.

    This function is intentionally conservative:
    - It only pulls in numeric / already-encoded columns that are
      broadly useful for ranking (no raw IDs or string ranges).
    - It returns None if a file is missing so the main pipeline
      can gracefully fall back to the existing feature set.
    """
    user_path = PROJECT_ROOT / "data/raw/user_features_pure.csv"
    video_basic_path = PROJECT_ROOT / "data/raw/video_features_basic_pure.csv"
    video_stat_path = PROJECT_ROOT / "data/raw/video_features_statistic_pure.csv"

    user_df: pd.DataFrame | None
    item_df: pd.DataFrame | None

    if user_path.exists():
        tmp = pd.read_csv(user_path)

        # Keep numeric / one-hot style columns only.
        base_user_numeric = [
            "is_lowactive_period",
            "is_live_streamer",
            "is_video_author",
            "follow_user_num",
            "fans_user_num",
            "friend_user_num",
            "register_days",
        ]
        onehot_cols = [c for c in tmp.columns if c.startswith("onehot_feat")]
        keep_cols = ["user_id"] + [c for c in base_user_numeric if c in tmp.columns] + onehot_cols
        user_df = tmp[keep_cols].copy()
    else:
        user_df = None

    if video_basic_path.exists() or video_stat_path.exists():
        parts: list[pd.DataFrame] = []
        if video_basic_path.exists():
            vb = pd.read_csv(video_basic_path)
            # Drop high-cardinality IDs and string columns; keep numeric basics.
            drop_cols = {"author_id", "video_type", "upload_dt", "upload_type", "music_id", "tag"}
            vb = vb[[c for c in vb.columns if c not in drop_cols]]
            parts.append(vb)
        if video_stat_path.exists():
            vs = pd.read_csv(video_stat_path)
            parts.append(vs)

        if parts:
            item_df = parts[0]
            for extra in parts[1:]:
                item_df = item_df.merge(extra, on="video_id", how="outer")
        else:
            item_df = None
    else:
        item_df = None

    return user_df, item_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage recommendation: ALS candidates + GBM re-ranking.",
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
    """Construct feature table for GBM from ALS, ItemCF, and side features."""
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

    # ------------------------------------------------------------------
    # Join user / item side features from KuaiRand tables (if available).
    # We map original IDs to matrix indices via user_map/item_map and
    # then merge by those indices so ALS, ItemCF, and GBM stay aligned.
    # ------------------------------------------------------------------
    user_side, item_side = _load_side_feature_tables()

    # Load mappings from original ids -> matrix indices.
    user_map_path = PROJECT_ROOT / "data/processed/user_map.json"
    item_map_path = PROJECT_ROOT / "data/processed/item_map.json"

    user_map: Dict[int, int] | None = None
    item_map: Dict[int, int] | None = None

    if user_side is not None and user_map_path.exists():
        with open(user_map_path, "r") as f:
            raw = json.load(f)
        user_map = {int(k): int(v) for k, v in raw.items()}

        user_side = user_side.copy()
        user_side["user_index"] = (
            user_side["user_id"].astype(int).map(user_map)  # type: ignore[arg-type]
        )
        user_side = user_side.dropna(subset=["user_index"])
        user_side["user_index"] = user_side["user_index"].astype(int)

        # Prefix feature columns for clarity.
        rename_cols = {
            c: f"user_{c}"
            for c in user_side.columns
            if c not in {"user_id", "user_index"}
        }
        user_side = user_side.rename(columns=rename_cols)

        feat_df = feat_df.merge(
            user_side.drop(columns=["user_id"]),
            left_on="user_id",
            right_on="user_index",
            how="left",
        )
        feat_df.drop(columns=["user_index"], inplace=True)

    if item_side is not None and item_map_path.exists():
        with open(item_map_path, "r") as f:
            raw = json.load(f)
        item_map = {int(k): int(v) for k, v in raw.items()}

        item_side = item_side.copy()
        item_side["item_index"] = (
            item_side["video_id"].astype(int).map(item_map)  # type: ignore[arg-type]
        )
        item_side = item_side.dropna(subset=["item_index"])
        item_side["item_index"] = item_side["item_index"].astype(int)

        rename_cols = {
            c: f"item_{c}"
            for c in item_side.columns
            if c not in {"video_id", "item_index"}
        }
        item_side = item_side.rename(columns=rename_cols)

        feat_df = feat_df.merge(
            item_side.drop(columns=["video_id"]),
            left_on="video_id",
            right_on="item_index",
            how="left",
        )
        feat_df.drop(columns=["item_index"], inplace=True)

    # Fill any remaining NaNs in side features with neutral defaults.
    side_feature_cols = [
        c for c in feat_df.columns if c.startswith("user_") or c.startswith("item_")
    ]
    if side_feature_cols:
        feat_df[side_feature_cols] = feat_df[side_feature_cols].fillna(0.0)

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

    base_feature_cols = [
        "als_score",
        "als_rank",
        "itemcf_score",
        "itemcf_rank",
        "user_interactions",
        "item_popularity",
    ]

    # Automatically pick numeric side-feature columns we added above.
    side_feature_cols = [
        c
        for c in feat_df.columns
        if (c.startswith("user_") or c.startswith("item_"))
        and np.issubdtype(feat_df[c].dtype, np.number)
    ]

    feature_cols = base_feature_cols + sorted(side_feature_cols)

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

    pred_path = args.pred_dir / "two_stage_gbm_recommendations.csv"
    metrics_path = args.eval_dir / "two_stage_gbm_metrics.json"
    model_path = args.model_dir / "two_stage_gbm_model.pkl"
    fi_path = args.eval_dir / "two_stage_gbm_feature_importances.json"

    rec_df.to_csv(pred_path, index=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(fi_path, "w") as f:
        json.dump(importances, f, indent=2)

    # Persist the model via pickle for potential reuse.
    import pickle

    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Two-stage GBM re-ranking complete.")
    print(f"Predictions: {pred_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Feature importances: {fi_path}")
    print(f"Model: {model_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
