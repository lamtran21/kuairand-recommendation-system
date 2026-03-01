"""Run script for ItemCF baseline on processed KuaiRand artifacts.

All hyper-parameters live at the top for easy tuning.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.itemcf import predict, train

# -----------------------------------------------------------------------------
# Tunable Hyper-Parameters
# -----------------------------------------------------------------------------
# Number of neighbors kept per item in the sparse similarity matrix.
GRID_NEIGHBOR_K = [100, 200, 400, 800]

# Shrinkage added in similarity denominator to reduce noisy high similarity on
# low-support item pairs.
GRID_SHRINK = [0.0, 10.0, 20.0, 50.0]

# Ranking cutoffs for evaluation. In leave-one-out with one positive per user,
# Recall@K equals HitRate@K, but we still report all metrics for completeness.
GRID_EVAL_K = [10, 20, 100, 200]

# Keep exported prediction file at top-100 even if evaluation uses larger K.
PREDICTION_TOP_K = 100

# Similarity optimization options:
# - cosine: baseline cosine on weighted implicit interactions.
# - adjusted_cosine: subtract each user's interaction mean to reduce user-scale bias.
# - jaccard: binary overlap/union to focus on co-consumption rather than intensity.
# - bm25_cosine: BM25-style reweighting before cosine to suppress popularity bias.
GRID_SIMILARITY = ["cosine", "jaccard", "bm25_cosine"]

# Kept fixed in this sweep to control runtime. You can still tune them later.
FIXED_SIGNIFICANCE_BETA = 0.0
FIXED_USE_IUF = False

# BM25 parameters (used only when similarity='bm25_cosine').
BM25_K1 = 1.2
BM25_B = 0.75

# Popularity penalty for post-scoring reweighting:
# final_score = cf_score / (log1p(item_pop + 1) ** gamma)
# gamma=0.0 disables this penalty.
POPULARITY_PENALTY_GAMMA = 0.5

# Time-decay is intentionally disabled here because current processed artifacts
# do not include per-interaction timestamps.
USE_TIME_DECAY = False
# Execution switches (set at top for quick control).
# If False and best metrics file exists, the script reuses best config and
# still refreshes metrics + predictions.
RUN_GRID_SEARCH = False

# Enable/disable evaluation metric computation and evaluation artifact writing.
RUN_EVALUATION = False


def recall_at_k(y_true: dict[int, int], y_pred: dict[int, list[int]], k: int) -> float:
    vals = []
    for user, item in y_true.items():
        preds = y_pred.get(user, [])[:k]
        vals.append(1.0 if item in preds else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def hit_rate_at_k(y_true: dict[int, int], y_pred: dict[int, list[int]], k: int) -> float:
    return recall_at_k(y_true, y_pred, k)


def ndcg_at_k(y_true: dict[int, int], y_pred: dict[int, list[int]], k: int) -> float:
    vals = []
    for user, item in y_true.items():
        preds = y_pred.get(user, [])[:k]
        if item in preds:
            rank = preds.index(item) + 1
            vals.append(1.0 / np.log2(rank + 1))
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def map_at_k(y_true: dict[int, int], y_pred: dict[int, list[int]], k: int) -> float:
    vals = []
    for user, item in y_true.items():
        preds = y_pred.get(user, [])[:k]
        if item in preds:
            rank = preds.index(item) + 1
            vals.append(1.0 / rank)
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ItemCF hyper-parameter search.")
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
        "--pred-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/model_predictions",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/evaluation/itemcf",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models",
    )
    return parser.parse_args()


def _make_pred_dict(rec_df: pd.DataFrame) -> dict[int, list[int]]:
    y_pred: dict[int, list[int]] = {}
    for row in rec_df.itertuples(index=False):
        y_pred.setdefault(int(row.user_id), []).append(int(row.video_id))
    return y_pred


def _evaluate_multi_k(
    y_true: dict[int, int],
    y_pred: dict[int, list[int]],
    eval_k_list: list[int],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in eval_k_list:
        out[f"recall@{k}"] = recall_at_k(y_true, y_pred, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(y_true, y_pred, k)
        out[f"ndcg@{k}"] = ndcg_at_k(y_true, y_pred, k)
        out[f"map@{k}"] = map_at_k(y_true, y_pred, k)
    return out


def _clip_topk_for_export(rec_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep only top-k rows per user for exported recommendation file."""
    if "rank" in rec_df.columns:
        out = rec_df[rec_df["rank"] <= k].copy()
    else:
        out = (
            rec_df.sort_values(["user_id", "score"], ascending=[True, False])
            .groupby("user_id", as_index=False)
            .head(k)
            .copy()
        )
    return out.sort_values(["user_id", "rank"], ascending=[True, True], kind="stable")


def main():
    """Load data, train ItemCF, optionally run grid/evaluation, and save outputs."""
    args = parse_args()
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_matrix = sp.load_npz(args.train_matrix).tocsr()

    grid_path = args.eval_dir / "itemcf_grid_results.csv"
    best_metrics_path = args.eval_dir / "itemcf_best_metrics.json"
    best_rec_path = args.pred_dir / "itemcf_recommendations.cf"
    best_model_path = args.model_dir / "itemcf_best_model.pkl"

    if (not RUN_GRID_SEARCH) and best_metrics_path.exists():
        with open(best_metrics_path, "r") as f:
            best_cfg = json.load(f)

        model = train(
            interactions=train_matrix,
            neighbor_k=int(best_cfg["neighbor_k"]),
            shrink=float(best_cfg["shrink"]),
            similarity=str(best_cfg["similarity"]),
            use_iuf=bool(best_cfg.get("use_iuf", FIXED_USE_IUF)),
            significance_beta=float(best_cfg.get("significance_beta", FIXED_SIGNIFICANCE_BETA)),
            bm25_k1=float(best_cfg.get("bm25_k1", BM25_K1)),
            bm25_b=float(best_cfg.get("bm25_b", BM25_B)),
        )

        rec_df = predict(
            model=model,
            interactions=train_matrix,
            user_ids=np.arange(train_matrix.shape[0], dtype=np.int64),
            k=max(PREDICTION_TOP_K, max(GRID_EVAL_K)),
            popularity_penalty_gamma=POPULARITY_PENALTY_GAMMA,
        )
        export_df = _clip_topk_for_export(rec_df, PREDICTION_TOP_K)
        export_df.to_csv(best_rec_path, index=False)
        with open(best_model_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.test_ground_truth, "r") as f:
            gt_raw = json.load(f)
        y_true = {int(k): int(v) for k, v in gt_raw.items()}
        eval_users = sorted(y_true.keys())
        eval_rec_df = predict(
            model=model,
            interactions=train_matrix,
            user_ids=eval_users,
            k=max(GRID_EVAL_K),
            popularity_penalty_gamma=POPULARITY_PENALTY_GAMMA,
        )
        y_pred = _make_pred_dict(eval_rec_df)
        metrics = _evaluate_multi_k(y_true, y_pred, GRID_EVAL_K)

        row = {
            "neighbor_k": int(best_cfg["neighbor_k"]),
            "shrink": float(best_cfg["shrink"]),
            "similarity": str(best_cfg["similarity"]),
            "significance_beta": float(best_cfg.get("significance_beta", FIXED_SIGNIFICANCE_BETA)),
            "use_iuf": bool(best_cfg.get("use_iuf", FIXED_USE_IUF)),
            "bm25_k1": float(best_cfg.get("bm25_k1", BM25_K1)),
            "bm25_b": float(best_cfg.get("bm25_b", BM25_B)),
            "popularity_penalty_gamma": POPULARITY_PENALTY_GAMMA,
            "num_eval_users": len(eval_users),
            **metrics,
        }
        with open(best_metrics_path, "w") as f:
            json.dump(row, f, indent=2)

        print("Best metrics found; skipped grid search and evaluation.")
        print(f"Predictions: {best_rec_path}")
        print(f"Metrics: {best_metrics_path}")
        print(f"Model: {best_model_path}")
        return

    with open(args.test_ground_truth, "r") as f:
        gt_raw = json.load(f)
    y_true = {int(k): int(v) for k, v in gt_raw.items()}
    eval_users = sorted(y_true.keys())
    max_eval_k = max(GRID_EVAL_K)

    if RUN_GRID_SEARCH:
        if not RUN_EVALUATION:
            raise ValueError("RUN_GRID_SEARCH=True requires RUN_EVALUATION=True.")

        grid_records = []
        best_by_recall100 = None
        best_rec_df = None
        best_model = None

        all_combos = list(
            itertools.product(
                GRID_NEIGHBOR_K,
                GRID_SHRINK,
                GRID_SIMILARITY,
            )
        )

        print(f"Grid size: {len(all_combos)} combinations")

        for idx, (neighbor_k, shrink, similarity) in enumerate(all_combos, start=1):
            t0 = time.time()

            model = train(
                interactions=train_matrix,
                neighbor_k=neighbor_k,
                shrink=shrink,
                similarity=similarity,
                use_iuf=FIXED_USE_IUF,
                significance_beta=FIXED_SIGNIFICANCE_BETA,
                bm25_k1=BM25_K1,
                bm25_b=BM25_B,
            )

            rec_df = predict(
                model=model,
                interactions=train_matrix,
                user_ids=eval_users,
                k=max_eval_k,
                popularity_penalty_gamma=POPULARITY_PENALTY_GAMMA,
            )

            y_pred = _make_pred_dict(rec_df)
            metrics = _evaluate_multi_k(y_true, y_pred, GRID_EVAL_K)

            row = {
                "neighbor_k": neighbor_k,
                "shrink": shrink,
                "similarity": similarity,
                "significance_beta": FIXED_SIGNIFICANCE_BETA,
                "use_iuf": FIXED_USE_IUF,
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
                "popularity_penalty_gamma": POPULARITY_PENALTY_GAMMA,
                "num_eval_users": len(eval_users),
                "elapsed_sec": time.time() - t0,
                **metrics,
            }
            grid_records.append(row)

            key = (row["recall@100"], row["ndcg@100"], row["map@100"])
            if best_by_recall100 is None or key > (
                best_by_recall100["recall@100"],
                best_by_recall100["ndcg@100"],
                best_by_recall100["map@100"],
            ):
                best_by_recall100 = row
                best_rec_df = rec_df.copy()
                best_model = model

            print(
                f"[{idx}/{len(all_combos)}] sim={similarity}, nk={neighbor_k}, "
                f"sh={shrink}, sig={FIXED_SIGNIFICANCE_BETA}, iuf={FIXED_USE_IUF} | "
                f"R@10={row['recall@10']:.4f}, R@20={row['recall@20']:.4f}, "
                f"R@100={row['recall@100']:.4f}, R@200={row['recall@200']:.4f}"
            )

        grid_df = pd.DataFrame(grid_records).sort_values(
            ["recall@100", "ndcg@100", "map@100"],
            ascending=False,
        )
        grid_df.to_csv(grid_path, index=False)
        with open(best_metrics_path, "w") as f:
            json.dump(best_by_recall100, f, indent=2)
        if best_rec_df is not None:
            _clip_topk_for_export(best_rec_df, PREDICTION_TOP_K).to_csv(best_rec_path, index=False)
        if best_model is not None:
            with open(best_model_path, "wb") as f:
                pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("ItemCF grid search complete.")
        print(f"Grid results: {grid_path}")
        print(f"Best metrics: {best_metrics_path}")
        print(f"Best recommendations: {best_rec_path}")
        print(f"Best model: {best_model_path}")
        print(json.dumps(best_by_recall100, indent=2))
        return

    model = train(
        interactions=train_matrix,
        neighbor_k=GRID_NEIGHBOR_K[0],
        shrink=GRID_SHRINK[0],
        similarity=GRID_SIMILARITY[0],
        use_iuf=FIXED_USE_IUF,
        significance_beta=FIXED_SIGNIFICANCE_BETA,
        bm25_k1=BM25_K1,
        bm25_b=BM25_B,
    )
    rec_df = predict(
        model=model,
        interactions=train_matrix,
        user_ids=np.arange(train_matrix.shape[0], dtype=np.int64),
        k=max(PREDICTION_TOP_K, max_eval_k),
        popularity_penalty_gamma=POPULARITY_PENALTY_GAMMA,
    )
    _clip_topk_for_export(rec_df, PREDICTION_TOP_K).to_csv(best_rec_path, index=False)
    with open(best_model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    if RUN_EVALUATION:
        eval_rec_df = predict(
            model=model,
            interactions=train_matrix,
            user_ids=eval_users,
            k=max_eval_k,
            popularity_penalty_gamma=POPULARITY_PENALTY_GAMMA,
        )
        y_pred = _make_pred_dict(eval_rec_df)
        metrics = _evaluate_multi_k(y_true, y_pred, GRID_EVAL_K)
        row = {
            "neighbor_k": GRID_NEIGHBOR_K[0],
            "shrink": GRID_SHRINK[0],
            "similarity": GRID_SIMILARITY[0],
            "significance_beta": FIXED_SIGNIFICANCE_BETA,
            "use_iuf": FIXED_USE_IUF,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "popularity_penalty_gamma": POPULARITY_PENALTY_GAMMA,
            "num_eval_users": len(eval_users),
            **metrics,
        }
        with open(best_metrics_path, "w") as f:
            json.dump(row, f, indent=2)
        print("Single-config run complete with evaluation.")
        print(f"Best metrics: {best_metrics_path}")
    else:
        print("Single-config run complete without evaluation.")

    print(f"Predictions: {best_rec_path}")
    print(f"Model: {best_model_path}")


if __name__ == "__main__":
    main()
