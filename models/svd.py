"""SVD-based collaborative filtering model for implicit feedback.

Tunable parameters are intentionally exposed so run scripts can perform
systematic hyper-parameter search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import diags
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


@dataclass
class SVDModel:
    """Container for trained SVD artifacts."""

    train_matrix: sp.csr_matrix
    # (n_users, k) – user factors scaled by singular values
    U: np.ndarray
    Vt: np.ndarray          # (k, n_items) – item factors
    item_popularity: np.ndarray


def _preprocess(
    x: sp.csr_matrix,
    use_idf: bool = True,
    normalize_rows: bool = True,
) -> sp.csr_matrix:
    """Clean and weight the user-item matrix before SVD decomposition.

    Steps (matching the notebook preprocessing):
    1. Remove explicit zeros left over from sparse storage.
    2. Binarize – implicit feedback signals engagement, not quantity.
    3. IDF column weighting – down-weight globally popular items so the
       model focuses on individual taste rather than popularity.
    4. L2 row normalization – equalize user activity levels so heavy
       listeners do not dominate the latent space.
    """
    x = x.copy().astype(np.float32)
    x.eliminate_zeros()

    # Binarize: implicit feedback only tells us *whether* a user engaged
    x.data = np.ones_like(x.data, dtype=np.float32)

    if use_idf:
        item_freq = np.array((x > 0).sum(axis=0)).flatten().astype(np.float32)
        idf = np.log1p(x.shape[0] / (item_freq + 1.0)).astype(np.float32)
        x = x.dot(diags(idf))

    if normalize_rows:
        x = normalize(x, norm="l2", axis=1)

    return x


def train(
    interactions: sp.csr_matrix,
    n_components: int = 100,
    random_state: int = 42,
    use_idf: bool = True,
    normalize_rows: bool = True,
) -> SVDModel:
    """Fit a Truncated SVD model on the user-item interaction matrix.

    Parameters
    ----------
    interactions:
        User-item CSR matrix (raw counts or binary).
    n_components:
        Number of latent factors (k). Larger k captures more variance but
        risks overfitting. Grid-search recommended range: 1–300.
    random_state:
        Seed for TruncatedSVD reproducibility.
    use_idf:
        Apply IDF column weighting to down-weight globally popular items.
    normalize_rows:
        L2-normalize user rows so heavy listeners do not dominate the
        latent space.
    """
    if not sp.isspmatrix_csr(interactions):
        interactions = interactions.tocsr()

    x = interactions.astype(np.float32, copy=True)
    x.eliminate_zeros()

    x_prep = _preprocess(x, use_idf=use_idf, normalize_rows=normalize_rows)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    # (n_users, k), already scaled by singular values
    U = svd.fit_transform(x_prep)
    Vt = svd.components_            # (k, n_items)

    item_popularity = np.bincount(
        x.indices, minlength=x.shape[1]).astype(np.float32)

    return SVDModel(train_matrix=x, U=U, Vt=Vt, item_popularity=item_popularity)


def predict(
    model: SVDModel,
    user_ids=None,
    k: int = 10,
    popularity_penalty_gamma: float = 0.0,
) -> pd.DataFrame:
    """Generate top-k item recommendations for each user.

    Parameters
    ----------
    model:
        Trained SVDModel artifact.
    user_ids:
        Iterable of integer user row indices to score.  Defaults to all
        users present in the training matrix.
    k:
        Number of recommendations per user.
    popularity_penalty_gamma:
        If > 0, penalize popular items by dividing scores by
        log(1 + pop)^gamma, mirroring the ItemCF popularity penalty.
    """
    x = model.train_matrix
    item_popularity = model.item_popularity

    if user_ids is None:
        user_ids = np.arange(x.shape[0], dtype=np.int64)
    else:
        user_ids = np.asarray(list(user_ids), dtype=np.int64)

    n_items = x.shape[1]
    cand = min(k, n_items)

    # Score all query users in one matrix multiply: (n_query, k) @ (k, n_items)
    U_sub = model.U[user_ids]
    scores = U_sub @ model.Vt       # (n_query, n_items)

    if popularity_penalty_gamma > 0.0:
        pop_penalty = np.power(
            np.log1p(item_popularity + 1.0), popularity_penalty_gamma
        ).astype(np.float32)
        pop_penalty[pop_penalty == 0.0] = 1.0
        scores = scores / pop_penalty[np.newaxis, :]

    # Mask items already seen in training
    seen_mask = x[user_ids].toarray().astype(bool)
    scores[seen_mask] = -np.inf

    rec_rows = []

    for i, u in enumerate(user_ids):
        row_scores = scores[i]

        valid_mask = np.isfinite(row_scores)
        if not np.any(valid_mask):
            continue

        top_idx = np.argpartition(row_scores, -cand)[-cand:]
        top_idx = top_idx[np.argsort(-row_scores[top_idx])]

        rank = 1
        for item_idx in top_idx:
            score = float(row_scores[item_idx])
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
