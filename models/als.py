"""ALS (implicit matrix factorization) model.

I/O contract (minimal):
- Input to train: interactions DataFrame [user_id, video_id, weight].
- Output of train: model object + user/item id mappings.
- Input to predict: trained artifacts, iterable user_ids, int k.
- Output of predict: DataFrame [user_id, video_id, score, rank].
"""

from __future__ import annotations


def train(interactions):
    """Train ALS-style factorization model."""
    raise NotImplementedError


def predict(model, user_ids, k=10):
    """Return top-k recommendations from latent factors."""
    raise NotImplementedError
