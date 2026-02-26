"""Item-based collaborative filtering model for implicit feedback.

Tunable parameters are intentionally exposed so run scripts can perform
systematic hyper-parameter search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class ItemCFModel:
    """Container for trained ItemCF artifacts."""

    train_matrix: sp.csr_matrix
    similarity: sp.csr_matrix
    item_popularity: np.ndarray


def _apply_significance_weight(
    sim: sp.csr_matrix,
    co_counts: sp.csr_matrix,
    beta: float,
) -> sp.csr_matrix:
    """Apply significance weighting using aligned co-occurrence lookups."""
    if beta <= 0.0:
        return sim
    sim = sim.tocsr(copy=True)
    co_counts = co_counts.tocsr()
    for row in range(sim.shape[0]):
        start = sim.indptr[row]
        end = sim.indptr[row + 1]
        if start == end:
            continue
        cols = sim.indices[start:end]
        co = np.asarray(co_counts[row, cols].toarray()).ravel().astype(np.float32)
        sim.data[start:end] *= co / (co + beta)
    return sim


def _row_topk(sparse_mat: sp.csr_matrix, topk: int) -> sp.csr_matrix:
    """Keep only top-k values per row in a CSR matrix."""
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for row in range(sparse_mat.shape[0]):
        start = sparse_mat.indptr[row]
        end = sparse_mat.indptr[row + 1]
        row_cols = sparse_mat.indices[start:end]
        row_vals = sparse_mat.data[start:end]

        if row_vals.size == 0:
            continue

        if row_vals.size > topk:
            keep_idx = np.argpartition(row_vals, -topk)[-topk:]
            row_cols = row_cols[keep_idx]
            row_vals = row_vals[keep_idx]

        order = np.argsort(-row_vals)
        row_cols = row_cols[order]
        row_vals = row_vals[order]

        rows.extend([row] * row_vals.size)
        cols.extend(row_cols.tolist())
        vals.extend(row_vals.tolist())

    return sp.csr_matrix(
        (np.asarray(vals, dtype=np.float32), (rows, cols)),
        shape=sparse_mat.shape,
        dtype=np.float32,
    )


def _apply_iuf(x: sp.csr_matrix) -> sp.csr_matrix:
    """Apply inverse user frequency to down-weight globally popular items."""
    x = x.tocsr(copy=True)
    n_users = x.shape[0]
    item_df = np.bincount(x.indices, minlength=x.shape[1]).astype(np.float32)
    iuf = np.log1p(n_users / np.maximum(item_df, 1.0)).astype(np.float32)

    for u in range(x.shape[0]):
        start = x.indptr[u]
        end = x.indptr[u + 1]
        if start == end:
            continue
        cols = x.indices[start:end]
        x.data[start:end] *= iuf[cols]

    return x


def _bm25_weight(x: sp.csr_matrix, bm25_k1: float, bm25_b: float) -> sp.csr_matrix:
    """Apply BM25-like weighting before similarity computation."""
    x = x.tocsr(copy=True)
    n_users, n_items = x.shape

    item_df = np.bincount(x.indices, minlength=n_items).astype(np.float32)
    idf = np.log((n_users - item_df + 0.5) / (item_df + 0.5))
    idf = np.maximum(idf, 0.0).astype(np.float32)

    user_len = np.diff(x.indptr).astype(np.float32)
    avg_len = float(np.mean(user_len)) if n_users > 0 else 1.0
    avg_len = max(avg_len, 1e-8)

    for u in range(n_users):
        start = x.indptr[u]
        end = x.indptr[u + 1]
        if start == end:
            continue
        cols = x.indices[start:end]
        tf = x.data[start:end]
        norm = bm25_k1 * (1.0 - bm25_b + bm25_b * user_len[u] / avg_len)
        denom = tf + norm
        denom[denom == 0.0] = 1e-8
        x.data[start:end] = idf[cols] * (tf * (bm25_k1 + 1.0)) / denom

    return x


def _adjust_user_center(x: sp.csr_matrix) -> sp.csr_matrix:
    """Adjusted-cosine: subtract each user's non-zero mean from interacted items."""
    x = x.tocsr(copy=True)
    for u in range(x.shape[0]):
        start = x.indptr[u]
        end = x.indptr[u + 1]
        if start == end:
            continue
        row_vals = x.data[start:end]
        row_mean = row_vals.mean()
        x.data[start:end] = row_vals - row_mean
    x.eliminate_zeros()
    return x


def train(
    interactions: sp.csr_matrix,
    neighbor_k: int = 100,
    shrink: float = 10.0,
    similarity: str = "cosine",
    use_iuf: bool = False,
    significance_beta: float = 0.0,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
) -> ItemCFModel:
    """Build item-item similarity from a user-item matrix.

    Supported similarity modes:
    - cosine: standard cosine over weighted implicit matrix.
    - adjusted_cosine: remove user-level bias by centering each user row.
    - jaccard: binary overlap / union on interacted items.
    - bm25_cosine: BM25-style reweighting then cosine.

    Extra regularization options:
    - use_iuf: penalize very popular items via inverse user frequency.
    - significance_beta: significance weighting sim *= co_count/(co_count+beta).
    """
    if not sp.isspmatrix_csr(interactions):
        interactions = interactions.tocsr()

    x = interactions.astype(np.float32, copy=True)
    x.eliminate_zeros()

    sim_mode = similarity.strip().lower()
    if sim_mode not in {"cosine", "adjusted_cosine", "jaccard", "bm25_cosine"}:
        raise ValueError(f"Unsupported similarity mode: {similarity}")

    x_work = x
    if use_iuf:
        x_work = _apply_iuf(x_work)

    if sim_mode == "adjusted_cosine":
        x_work = _adjust_user_center(x_work)
    elif sim_mode == "bm25_cosine":
        x_work = _bm25_weight(x_work, bm25_k1=bm25_k1, bm25_b=bm25_b)

    if sim_mode == "jaccard":
        b = x.copy()
        b.data = np.ones_like(b.data, dtype=np.float32)
        co_counts = (b.T @ b).tocsr().astype(np.float32)
        co_counts = co_counts.tolil()
        co_counts.setdiag(0.0)
        co_counts = co_counts.tocsr()
        co_counts.eliminate_zeros()

        item_deg = np.asarray(b.sum(axis=0)).ravel().astype(np.float32)

        sim = co_counts.copy().astype(np.float32)
        for row in range(sim.shape[0]):
            start = sim.indptr[row]
            end = sim.indptr[row + 1]
            if start == end:
                continue
            cols = sim.indices[start:end]
            inter = sim.data[start:end]
            union = item_deg[row] + item_deg[cols] - inter
            union[union == 0.0] = 1e-8
            sim.data[start:end] = inter / (union + shrink)

        sim = _apply_significance_weight(sim, co_counts, significance_beta)

        sim = _row_topk(sim, topk=neighbor_k)
        sim.eliminate_zeros()
        item_popularity = np.bincount(x.indices, minlength=x.shape[1]).astype(np.float32)
        return ItemCFModel(train_matrix=x, similarity=sim, item_popularity=item_popularity)

    item_norms = np.sqrt(x_work.power(2).sum(axis=0)).A1.astype(np.float32)
    item_norms[item_norms == 0.0] = 1e-8

    sim = (x_work.T @ x_work).tocsr().astype(np.float32)
    sim = sim.tolil()
    sim.setdiag(0.0)
    sim = sim.tocsr()
    sim.eliminate_zeros()

    co_counts = None
    if significance_beta > 0.0:
        b = x.copy()
        b.data = np.ones_like(b.data, dtype=np.float32)
        co_counts = (b.T @ b).tocsr().astype(np.float32)
        co_counts = co_counts.tolil()
        co_counts.setdiag(0.0)
        co_counts = co_counts.tocsr()
        co_counts.eliminate_zeros()

    for row in range(sim.shape[0]):
        start = sim.indptr[row]
        end = sim.indptr[row + 1]
        if start == end:
            continue
        cols = sim.indices[start:end]
        denom = item_norms[row] * item_norms[cols] + shrink
        sim.data[start:end] /= denom
    if co_counts is not None:
        sim = _apply_significance_weight(sim, co_counts, significance_beta)

    sim = _row_topk(sim, topk=neighbor_k)
    sim.eliminate_zeros()

    item_popularity = np.bincount(x.indices, minlength=x.shape[1]).astype(np.float32)
    return ItemCFModel(train_matrix=x, similarity=sim, item_popularity=item_popularity)


def predict(
    model: ItemCFModel,
    interactions: sp.csr_matrix | None = None,
    user_ids=None,
    k: int = 10,
    popularity_penalty_gamma: float = 0.0,
):
    """Generate top-k item recommendations for each user."""
    x = model.train_matrix if interactions is None else interactions.tocsr()
    sim = model.similarity
    item_popularity = model.item_popularity

    if user_ids is None:
        user_ids = np.arange(x.shape[0], dtype=np.int64)
    else:
        user_ids = np.asarray(list(user_ids), dtype=np.int64)

    rec_rows = []
    n_items = x.shape[1]

    for u in user_ids:
        scores = (x[u] @ sim).toarray().ravel()
        if popularity_penalty_gamma > 0.0:
            pop_penalty = np.power(np.log1p(item_popularity + 1.0), popularity_penalty_gamma)
            pop_penalty[pop_penalty == 0.0] = 1.0
            scores = scores / pop_penalty
        seen = x[u].indices
        if seen.size > 0:
            scores[seen] = -np.inf

        valid_mask = np.isfinite(scores)
        if not np.any(valid_mask):
            continue

        cand = min(k, n_items)
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
