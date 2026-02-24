"""Item-based collaborative filtering model.

I/O contract (minimal):
- Input to train: interactions DataFrame [user_id, video_id, label] (implicit feedback).
- Output of train: model object containing item-item similarity structure.
- Input to predict: trained model, user history DataFrame, iterable user_ids, int k.
- Output of predict: DataFrame [user_id, video_id, score, rank].
"""

from __future__ import annotations


def train(interactions):
    """Build item-item similarity from interactions."""
    raise NotImplementedError


def predict(model, interactions, user_ids, k=10):
    """Generate top-k item recommendations for each user."""
    raise NotImplementedError
