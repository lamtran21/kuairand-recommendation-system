"""SVD-based collaborative filtering model.

I/O contract (minimal):
- Input to train: interactions DataFrame [user_id, video_id, rating_or_weight].
- Output of train: SVD model object + id mappings.
- Input to predict: trained model, iterable user_ids, int k.
- Output of predict: DataFrame [user_id, video_id, score, rank].
"""

from __future__ import annotations


def train(interactions):
    """Fit SVD model on user-item matrix."""
    raise NotImplementedError


def predict(model, user_ids, k=10):
    """Generate top-k recommendations."""
    raise NotImplementedError
