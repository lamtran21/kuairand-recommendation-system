from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class PopularityModel:
    train_matrix: sp.csr_matrix
    item_popularity: np.ndarray


def train(interactions: sp.csr_matrix) -> PopularityModel:
    if not sp.isspmatrix_csr(interactions):
        interactions = interactions.tocsr()

    interactions = interactions.astype(np.float32)
    interactions.eliminate_zeros()

    item_popularity = np.bincount(
        interactions.indices,
        minlength=interactions.shape[1]
    ).astype(np.float32)

    return PopularityModel(
        train_matrix=interactions,
        item_popularity=item_popularity
    )


def predict(
    model: PopularityModel,
    interactions: sp.csr_matrix | None = None,
    user_ids=None,
    k: int = 10,
):
    x = model.train_matrix if interactions is None else interactions.tocsr()
    item_pop = model.item_popularity

    if user_ids is None:
        user_ids = np.arange(x.shape[0])
    else:
        user_ids = np.asarray(list(user_ids))

    global_rank = np.argsort(-item_pop)

    rec_rows = []

    for u in user_ids:
        seen = set(x[u].indices)
        rank = 1

        for item in global_rank:
            if item in seen:
                continue

            rec_rows.append({
                "user_id": int(u),
                "video_id": int(item),
                "score": float(item_pop[item]),
                "rank": rank,
            })

            rank += 1
            if rank > k:
                break

    return pd.DataFrame(rec_rows)