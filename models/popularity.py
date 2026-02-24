"""Popularity baseline model.

I/O contract (minimal):
- Input to train: interactions DataFrame with at least [user_id, video_id, label].
- Output of train: model dict with global item scores.
- Input to predict: model dict, iterable user_ids, int k.
- Output of predict: DataFrame [user_id, video_id, score, rank].
"""

from __future__ import annotations


def train(interactions):
    """Return popularity model from interaction logs."""
    raise NotImplementedError


def predict(model, user_ids, k=10):
    """Return top-k recommendations per user."""
    raise NotImplementedError
