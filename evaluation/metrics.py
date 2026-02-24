"""Model-agnostic ranking metrics.

I/O contract (minimal):
- Input to each metric:
  - y_true: dict[user_id -> set(video_id)] or equivalent structure
  - y_pred: dict[user_id -> list(video_id)] ranked list
  - k: int cutoff
- Output:
  - single float score per metric in [0, 1]

Expected metrics:
- recall_at_k
- ndcg_at_k
- hit_rate_at_k
- map_at_k
"""

from __future__ import annotations


def recall_at_k(y_true, y_pred, k=10):
    """Return Recall@k."""
    raise NotImplementedError


def ndcg_at_k(y_true, y_pred, k=10):
    """Return NDCG@k."""
    raise NotImplementedError


def hit_rate_at_k(y_true, y_pred, k=10):
    """Return HitRate@k."""
    raise NotImplementedError


def map_at_k(y_true, y_pred, k=10):
    """Return MAP@k."""
    raise NotImplementedError
