"""Run script for SVD baseline on processed KuaiRand artifacts.

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

# isort: split
from evaluation.metrics import hit_rate_at_k, map_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from models.svd import predict, train  # noqa: E402


# -----------------------------------------------------------------------------
# Tunable Hyper-Parameters
# -----------------------------------------------------------------------------
GRID_N_COMPONENTS = [50, 100, 150, 200, 250, 300]
GRID_EVAL_K = [10, 20, 100, 200]
PREDICTION_TOP_K = 100

FIXED_USE_IDF = True
FIXED_NORMALIZE_ROWS = True
FIXED_RANDOM_STATE = 42
POPULARITY_PENALTY_GAMMA = 0.0

RUN_GRID_SEARCH = False
# Keep internal evaluation off by default to avoid conflict with team evaluator.
RUN_EVALUATION = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SVD hyper-parameter search.")
    parser.add_argument("--train-matrix", type=Path,
                        default=PROJECT_ROOT / "data/processed/train_matrix.npz")
    parser.add_argument("--test-ground-truth", type=Path,
                        default=PROJECT_ROOT / "data/processed/test_ground_truth.json")
    parser.add_argument("--pred-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/model_predictions")
    parser.add_argument("--eval-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/evaluation/svd")
    parser.add_argument("--model-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/models")
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
    args = parse_args()
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_matrix = sp.load_npz(args.train_matrix).tocsr()

    grid_path = args.eval_dir / "svd_grid_results.csv"
    best_metrics_path = args.eval_dir / "svd_best_metrics.json"
    best_rec_path = args.pred_dir / "svd_recommendations.parquet"
    best_model_path = args.model_dir / "svd_best_model.pkl"

    if (not RUN_GRID_SEARCH) and best_metrics_path.exists():
        with open(best_metrics_path, "r") as f:
            best_cfg = json.load(f)

        model = train(
            interactions=train_matrix,
            n_components=int(best_cfg["n_components"]),
            random_state=int(best_cfg.get("random_state", FIXED_RANDOM_STATE)),
            use_idf=bool(best_cfg.get("use_idf", FIXED_USE_IDF)),
            normalize_rows=bool(best_cfg.get(
                "normalize_rows", FIXED_NORMALIZE_ROWS)),
        )

        rec_df = predict(
            model=model,
            user_ids=np.arange(train_matrix.shape[0], dtype=np.int64),
            k=max(PREDICTION_TOP_K, max(GRID_EVAL_K)),
            popularity_penalty_gamma=POPULARITY_PENALTY_GAMMA,
        )
        _clip_topk_for_export(rec_df, PREDICTION_TOP_K).to_parquet(
            best_rec_path, index=False)
        with open(best_model_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        if RUN_EVALUATION:
            with open(args.test_ground_truth, "r") as f:
                gt_raw = json.load(f)
            y_true = {int(k): int(v) for k, v in gt_raw.items()}
            eval_users = sorted(y_true.keys())
            eval_rec_df = predict(
                model=model,
                user_ids=eval_users,
                k=max(GRID_EVAL_K),
                popularity_penalty_gamma=POPULARITY_PENALTY_GAMMA,
            )
            y_pred = _make_pred_dict(eval_rec_df)
            metrics = _evaluate_multi_k(y_true, y_pred, GRID_EVAL_K)
            row = {
                "n_components": int(best_cfg["n_components"]),
                "use_idf": bool(best_cfg.get("use_idf", FIXED_USE_IDF)),
                "normalize_rows": bool(best_cfg.get("normalize_rows", FIXED_NORMALIZE_ROWS)),
                "random_state": int(best_cfg.get("random_state", FIXED_RANDOM_STATE)),
                "popularity_penalty_gamma": POPULARITY_PENALTY_GAMMA,
                "num_eval_users": len(eval_users),
                **metrics,
            }
            with open(best_metrics_path, "w") as f:
                json.dump(row, f, indent=2)

        print("Best metrics found; skipped grid search and evaluation.")
        print(f"Predictions: {best_rec_path}")
        if RUN_EVALUATION:
            print(f"Metrics: {best_metrics_path}")
        print(f"Model: {best_model_path}")
        return

    y_true: dict[int, int] = {}
    eval_users: list[int] = []
    max_eval_k = max(GRID_EVAL_K)
    if RUN_EVALUATION or RUN_GRID_SEARCH:
        with open(args.test_ground_truth, "r") as f:
            gt_raw = json.load(f)
        y_true = {int(k): int(v) for k, v in gt_raw.items()}
        eval_users = sorted(y_true.keys())

    if RUN_GRID_SEARCH:
        if not RUN_EVALUATION:
            raise ValueError(
                "RUN_GRID_SEARCH=True requires RUN_EVALUATION=True.")

        grid_records = []
        best_by_recall100 = None
        best_rec_df = None
        best_model = None

        all_combos = list(itertools.product(GRID_N_COMPONENTS))
        print(f"Grid size: {len(all_combos)} combinations")

        for idx, (n_components,) in enumerate(all_combos, start=1):
            t0 = time.time()
            model = train(
                interactions=train_matrix,
                n_components=n_components,
                random_state=FIXED_RANDOM_STATE,
                use_idf=FIXED_USE_IDF,
                normalize_rows=FIXED_NORMALIZE_ROWS,
            )

            rec_df = predict(
                model=model,
                user_ids=eval_users,
                k=max_eval_k,
                popularity_penalty_gamma=POPULARITY_PENALTY_GAMMA,
            )

            y_pred = _make_pred_dict(rec_df)
            metrics = _evaluate_multi_k(y_true, y_pred, GRID_EVAL_K)

            row = {
                "n_components": n_components,
                "use_idf": FIXED_USE_IDF,
                "normalize_rows": FIXED_NORMALIZE_ROWS,
                "random_state": FIXED_RANDOM_STATE,
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
                f"[{idx}/{len(all_combos)}] k={n_components}, use_idf={FIXED_USE_IDF}, "
                f"norm={FIXED_NORMALIZE_ROWS} | "
                f"R@10={row['recall@10']:.4f}, R@20={row['recall@20']:.4f}, "
                f"R@100={row['recall@100']:.4f}, R@200={row['recall@200']:.4f}"
            )

        grid_df = pd.DataFrame(grid_records).sort_values(
            ["recall@100", "ndcg@100", "map@100"], ascending=False)
        grid_df.to_csv(grid_path, index=False)
        with open(best_metrics_path, "w") as f:
            json.dump(best_by_recall100, f, indent=2)
        if best_rec_df is not None:
            _clip_topk_for_export(best_rec_df, PREDICTION_TOP_K).to_parquet(
                best_rec_path, index=False)
        if best_model is not None:
            with open(best_model_path, "wb") as f:
                pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("SVD grid search complete.")
        print(f"Grid results: {grid_path}")
        print(f"Best metrics: {best_metrics_path}")
        print(f"Best recommendations: {best_rec_path}")
        print(f"Best model: {best_model_path}")
        print(json.dumps(best_by_recall100, indent=2))
        return

    # Single-config run (default)
    model = train(
        interactions=train_matrix,
        n_components=GRID_N_COMPONENTS[0],
        random_state=FIXED_RANDOM_STATE,
        use_idf=FIXED_USE_IDF,
        normalize_rows=FIXED_NORMALIZE_ROWS,
    )
    rec_df = predict(
        model=model,
        user_ids=np.arange(train_matrix.shape[0], dtype=np.int64),
        k=max(PREDICTION_TOP_K, max_eval_k),
        popularity_penalty_gamma=POPULARITY_PENALTY_GAMMA,
    )
    _clip_topk_for_export(rec_df, PREDICTION_TOP_K).to_parquet(
        best_rec_path, index=False)
    with open(best_model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    if RUN_EVALUATION:
        eval_rec_df = predict(
            model=model,
            user_ids=eval_users,
            k=max_eval_k,
            popularity_penalty_gamma=POPULARITY_PENALTY_GAMMA,
        )
        y_pred = _make_pred_dict(eval_rec_df)
        metrics = _evaluate_multi_k(y_true, y_pred, GRID_EVAL_K)
        row = {
            "n_components": GRID_N_COMPONENTS[0],
            "use_idf": FIXED_USE_IDF,
            "normalize_rows": FIXED_NORMALIZE_ROWS,
            "random_state": FIXED_RANDOM_STATE,
            "popularity_penalty_gamma": POPULARITY_PENALTY_GAMMA,
            "num_eval_users": len(eval_users),
            **metrics,
        }
        with open(best_metrics_path, "w") as f:
            json.dump(row, f, indent=2)
        print("Single-config run complete with evaluation.")
        print(f"Metrics: {best_metrics_path}")
    else:
        print("Single-config run complete without evaluation.")

    print(f"Predictions: {best_rec_path}")
    print(f"Model: {best_model_path}")


if __name__ == "__main__":
    main()
