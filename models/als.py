"""ALS (implicit matrix factorization) model for implicit feedback.

This implementation follows the standard Hu, Koren & Volinsky style
alternating least squares (ALS) with confidence weights:

    c_ui = 1 + alpha * r_ui

where ``r_ui`` is a non‑negative implicit signal such as watch time.

I/O contract (minimal):
- Input to ``train``:
    - interactions: ``scipy.sparse.csr_matrix`` of shape [n_users, n_items]
      with non‑negative interaction strengths.
- Output of ``train``:
    - ALSModel with user and item latent factors.
- Input to ``predict``:
    - trained ALSModel
    - optional interactions matrix (to mask already‑seen items)
    - iterable of user ids and top‑k cutoff.
- Output of ``predict``:
    - pandas DataFrame with columns [user_id, video_id, score, rank].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class ALSModel:
    """Container for trained ALS artifacts."""

    user_factors: np.ndarray  # shape [n_users, n_factors]
    item_factors: np.ndarray  # shape [n_items, n_factors]
    alpha: float
    regularization: float


def _least_squares_step(
    interactions: sp.csr_matrix,
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    regularization: float,
) -> np.ndarray:
    """One ALS half‑step: solve for X given fixed Y.

    This uses the efficient formulation that reuses the dense YᵀY term and
    adds per‑row corrections based on the non‑zero confidence entries.
    """
    if not sp.isspmatrix_csr(interactions):
        interactions = interactions.tocsr()

    num_rows = interactions.shape[0]
    num_factors = Y.shape[1]

    Y = Y.astype(np.float32, copy=False)
    X_new = np.zeros((num_rows, num_factors), dtype=np.float32)

    YtY = Y.T @ Y
    I = np.eye(num_factors, dtype=np.float32)

    indptr = interactions.indptr
    indices = interactions.indices
    data = interactions.data.astype(np.float32, copy=False)

    for row in range(num_rows):
        start = indptr[row]
        end = indptr[row + 1]

        if start == end:
            # Row with no interactions keeps the default all‑zeros factors.
            continue

        item_indices = indices[start:end]
        r = data[start:end]

        # Confidence weights c_ui = 1 + alpha * r_ui
        c = 1.0 + alpha * r
        cu_minus_1 = c - 1.0

        # Latent factors of items this row interacted with
        Y_u = Y[item_indices]  # shape [nnz, num_factors]

        # A = YᵀY + Yᵀ (C_u - I) Y + λI
        A = YtY + (Y_u.T * cu_minus_1) @ Y_u + regularization * I
        # b = Yᵀ C_u p_u, with p_u = 1 for all non‑zero entries
        b = Y_u.T @ c

        X_new[row] = np.linalg.solve(A, b).astype(np.float32)

    return X_new


def train(
    interactions: sp.spmatrix,
    n_factors: int = 64,
    regularization: float = 0.1,
    alpha: float = 40.0,
    n_iters: int = 15,
    random_state: int | None = 42,
) -> ALSModel:
    """Train ALS-style factorization model on an implicit matrix.

    Parameters
    ----------
    interactions:
        User–item implicit feedback matrix (CSR preferred). Values must be
        non‑negative and encode interaction strength (e.g., log watch time).
    n_factors:
        Dimensionality of the latent space.
    regularization:
        L2 regularization strength λ applied to user and item factors.
    alpha:
        Confidence scaling parameter in c_ui = 1 + alpha * r_ui.
    n_iters:
        Number of alternating optimization iterations.
    random_state:
        Seed for the random number generator (for reproducibility).
    """
    if not sp.isspmatrix_csr(interactions):
        interactions = interactions.tocsr()
    interactions = interactions.astype(np.float32)

    num_users, num_items = interactions.shape

    if num_users == 0 or num_items == 0:
        raise ValueError("interactions matrix must be non‑empty.")

    rng = np.random.default_rng(random_state)
    user_factors = 0.01 * rng.standard_normal(
        size=(num_users, n_factors),
        dtype=np.float32,
    )
    item_factors = 0.01 * rng.standard_normal(
        size=(num_items, n_factors),
        dtype=np.float32,
    )

    for _ in range(n_iters):
        # Fix items, solve for users.
        user_factors = _least_squares_step(
            interactions=interactions,
            X=user_factors,
            Y=item_factors,
            alpha=alpha,
            regularization=regularization,
        )
        # Fix users, solve for items (transpose view).
        item_factors = _least_squares_step(
            interactions=interactions.T.tocsr(),
            X=item_factors,
            Y=user_factors,
            alpha=alpha,
            regularization=regularization,
        )

    return ALSModel(
        user_factors=user_factors,
        item_factors=item_factors,
        alpha=alpha,
        regularization=regularization,
    )


def predict(
    model: ALSModel,
    interactions: sp.csr_matrix | None = None,
    user_ids: Iterable[int] | None = None,
    k: int = 10,
) -> pd.DataFrame:
    """Return top-k recommendations from latent factors.

    Parameters
    ----------
    model:
        Trained ALSModel.
    interactions:
        Optional CSR matrix with the same shape used at train time. If
        provided, items already interacted with by a user are excluded
        from that user's recommendations.
    user_ids:
        Iterable of user indices (0‑based). If None, all users are scored.
    k:
        Number of items to recommend per user.
    """
    user_factors = model.user_factors
    item_factors = model.item_factors

    num_users, num_factors = user_factors.shape
    num_items = item_factors.shape[0]

    if interactions is not None:
        interactions = interactions.tocsr()
        if interactions.shape[0] != num_users or interactions.shape[1] != num_items:
            raise ValueError(
                "interactions shape does not match model factors: "
                f"{interactions.shape} vs ({num_users}, {num_items})",
            )

    if user_ids is None:
        user_ids_arr = np.arange(num_users, dtype=np.int64)
    else:
        user_ids_arr = np.asarray(list(user_ids), dtype=np.int64)

    rec_rows: list[dict[str, float | int]] = []

    for u in user_ids_arr:
        if u < 0 or u >= num_users:
            continue

        user_vec = user_factors[u]  # shape [num_factors]
        # Dense score vector over all items.
        scores = item_factors @ user_vec

        if interactions is not None:
            seen = interactions[u].indices
            if seen.size > 0:
                scores = scores.copy()
                scores[seen] = -np.inf

        if not np.any(np.isfinite(scores)):
            continue

        cand = min(k, num_items)
        top_idx = np.argpartition(scores, -cand)[-cand:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        rank = 1
        for item_idx in top_idx:
            score = float(scores[item_idx])
            if not np.isfinite(score):
                continue
            rec_rows.append(
                {
                    "user_id": int(u),
                    "video_id": int(item_idx),
                    "score": score,
                    "rank": rank,
                }
            )
            rank += 1
            if rank > k:
                break

    return pd.DataFrame(rec_rows, columns=["user_id", "video_id", "score", "rank"])
