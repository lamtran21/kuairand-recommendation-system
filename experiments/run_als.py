"""Run script for ALS model on processed KuaiRand artifacts.

This script expects the preprocessing pipeline in ``experiments/preprocessing.py``
to have produced the following files under ``data/processed``:

- ``train_matrix.npz``: CSR user–item interaction matrix.
- ``test_ground_truth.json``: {user_index: item_index} mapping.
- ``user_map.json`` and ``item_map.json``: original id → matrix index.

The script:
- loads the training matrix and ground truth,
- trains an implicit ALS model,
- generates top‑K recommendations for all users,
- evaluates ranking metrics on the leave‑one‑out split,
- saves predictions, metrics, and the trained model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.als import ALSModel, predict, train


# ---------------------------------------------------------------------------
# Hyper-parameters / search space
# ---------------------------------------------------------------------------

# Single-config defaults (used when RUN_GRID_SEARCH = False)
LATENT_FACTORS = 64
REGULARIZATION = 0.1
ALPHA = 40.0
N_ITERS = 15

# Small grid for tuning. Keep size modest since ALS is relatively heavy.
GRID_FACTORS = [32, 64]
GRID_REG = [0.01, 0.1]
GRID_ALPHA = [20.0, 40.0]

# If True, sweep over GRID_* and write grid results.
RUN_GRID_SEARCH = True

# Ranking cutoffs to evaluate.
EVAL_K = [10, 20, 50, 100, 200]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run implicit ALS on processed KuaiRand data.")
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
        default=PROJECT_ROOT / "outputs/evaluation/als",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models",
    )
    return parser.parse_args()


def main() -> None:
    """Load data, train ALS model, predict, evaluate, and save outputs."""
    args = parse_args()
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_matrix = sp.load_npz(args.train_matrix).tocsr()

    with open(args.test_ground_truth, "r") as f:
        gt_raw = json.load(f)
    y_true: dict[int, int] = {int(k): int(v) for k, v in gt_raw.items()}

    max_k = max(EVAL_K)

    grid_path = args.eval_dir / "als_grid_results.csv"
    best_metrics_path = args.eval_dir / "als_best_metrics.json"
    best_pred_path = args.pred_dir / "als_recommendations.csv"
    best_model_path = args.model_dir / "als_model.pkl"

    if RUN_GRID_SEARCH:
        # Run a small grid search over ALS hyper-parameters.
        grid_records: list[dict[str, float | int]] = []
        best_row: dict[str, float | int] | None = None
        best_metrics: dict[str, float] | None = None
        best_rec_df: pd.DataFrame | None = None
        best_model: ALSModel | None = None

        combos = list(itertools.product(GRID_FACTORS, GRID_REG, GRID_ALPHA))
        print(f"ALS grid size: {len(combos)} combinations")

        eval_users = sorted(y_true.keys())

        for idx, (n_factors, reg, alpha) in enumerate(combos, start=1):
            print(
                f"[{idx}/{len(combos)}] factors={n_factors}, reg={reg}, alpha={alpha}"
            )

            model = train(
                interactions=train_matrix,
                n_factors=int(n_factors),
                regularization=float(reg),
                alpha=float(alpha),
                n_iters=N_ITERS,
                random_state=42,
            )

            rec_df = predict(
                model=model,
                interactions=train_matrix,
                user_ids=eval_users,
                k=max_k,
            )

            y_pred = _make_pred_dict(rec_df)
            metrics = _evaluate_multi_k(y_true, y_pred, EVAL_K)

            row: dict[str, float | int] = {
                "n_factors": int(n_factors),
                "regularization": float(reg),
                "alpha": float(alpha),
                **metrics,
            }
            grid_records.append(row)

            key = (
                metrics[f"recall@{max_k}"],
                metrics[f"ndcg@{max_k}"],
                metrics[f"map@{max_k}"],
            )
            if best_row is None or key > (
                best_row[f"recall@{max_k}"],
                best_row[f"ndcg@{max_k}"],
                best_row[f"map@{max_k}"],
            ):
                best_row = row
                best_metrics = metrics
                best_rec_df = rec_df.copy()
                best_model = model

        grid_df = pd.DataFrame(grid_records).sort_values(
            [f"recall@{max_k}", f"ndcg@{max_k}", f"map@{max_k}"],
            ascending=False,
        )
        grid_df.to_csv(grid_path, index=False)

        if best_rec_df is not None:
            best_rec_df.to_csv(best_pred_path, index=False)
        if best_model is not None:
            with open(best_model_path, "wb") as f:
                pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(best_metrics_path, "w") as f:
            json.dump(
                {
                    "best_config": {
                        "n_factors": int(best_row["n_factors"]),
                        "regularization": float(best_row["regularization"]),
                        "alpha": float(best_row["alpha"]),
                        "n_iters": N_ITERS,
                    },
                    "metrics": best_metrics,
                },
                f,
                indent=2,
            )

        print("ALS grid search complete.")
        print(f"Grid results: {grid_path}")
        print(f"Best metrics: {best_metrics_path}")
        print(f"Best predictions: {best_pred_path}")
        print(f"Best model: {best_model_path}")
        if best_metrics is not None:
            print(json.dumps(best_metrics, indent=2))
        return

    # Single-config training (no grid search).
    model: ALSModel = train(
        interactions=train_matrix,
        n_factors=LATENT_FACTORS,
        regularization=REGULARIZATION,
        alpha=ALPHA,
        n_iters=N_ITERS,
        random_state=42,
    )

    rec_df = predict(
        model=model,
        interactions=train_matrix,
        user_ids=np.arange(train_matrix.shape[0], dtype=np.int64),
        k=max_k,
    )

    y_pred = _make_pred_dict(rec_df)
    metrics = _evaluate_multi_k(y_true, y_pred, EVAL_K)

    rec_df.to_csv(best_pred_path, index=False)
    with open(best_metrics_path, "w") as f:
        json.dump(
            {
                "best_config": {
                    "n_factors": LATENT_FACTORS,
                    "regularization": REGULARIZATION,
                    "alpha": ALPHA,
                    "n_iters": N_ITERS,
                },
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    with open(best_model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("ALS single-config run complete.")
    print(f"Metrics: {best_metrics_path}")
    print(f"Predictions: {best_pred_path}")
    print(f"Model: {best_model_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
