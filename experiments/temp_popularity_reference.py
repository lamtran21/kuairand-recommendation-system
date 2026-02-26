"""Temporary popularity baseline evaluator (reference only).

This script intentionally does not use or modify models/popularity.py.
It reads processed artifacts and reports ranking metrics for a global
popularity recommender with per-user seen-item filtering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_MATRIX_PATH = PROJECT_ROOT / "data/processed/train_matrix.npz"
GT_PATH = PROJECT_ROOT / "data/processed/test_ground_truth.json"

EVAL_K_LIST = [10, 20, 50]


def recall_at_k(y_true: dict[int, int], y_pred: dict[int, list[int]], k: int) -> float:
    vals = []
    for user, item in y_true.items():
        vals.append(1.0 if item in y_pred.get(user, [])[:k] else 0.0)
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


def build_popularity_rankings(train_matrix: sp.csr_matrix) -> np.ndarray:
    # Popularity by interaction count (non-zero occurrences), not weight sum.
    pop = np.bincount(train_matrix.indices, minlength=train_matrix.shape[1]).astype(np.float32)
    return np.argsort(-pop)


def recommend_for_user(
    user_idx: int,
    train_matrix: sp.csr_matrix,
    global_ranked_items: np.ndarray,
    k: int,
) -> list[int]:
    seen = set(train_matrix[user_idx].indices.tolist())
    recs: list[int] = []
    for item in global_ranked_items:
        if item in seen:
            continue
        recs.append(int(item))
        if len(recs) >= k:
            break
    return recs


def main() -> None:
    train_matrix = sp.load_npz(TRAIN_MATRIX_PATH).tocsr()
    with open(GT_PATH, "r") as f:
        y_true_raw = json.load(f)
    y_true = {int(u): int(i) for u, i in y_true_raw.items()}

    eval_users = sorted(y_true.keys())
    max_k = max(EVAL_K_LIST)

    global_ranked = build_popularity_rankings(train_matrix)

    y_pred: dict[int, list[int]] = {}
    for u in eval_users:
        y_pred[u] = recommend_for_user(u, train_matrix, global_ranked, max_k)

    out = {
        "model": "popularity_count",
        "num_eval_users": len(eval_users),
        "train_shape": list(train_matrix.shape),
        "train_nnz": int(train_matrix.nnz),
    }

    for k in EVAL_K_LIST:
        out[f"recall@{k}"] = recall_at_k(y_true, y_pred, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(y_true, y_pred, k)
        out[f"ndcg@{k}"] = ndcg_at_k(y_true, y_pred, k)
        out[f"map@{k}"] = map_at_k(y_true, y_pred, k)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
