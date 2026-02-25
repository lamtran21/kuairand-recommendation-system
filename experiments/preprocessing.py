"""
Build recommender system training artifacts using
leave-one-out temporal splitting.

Outputs:
    - Sparse training matrix (.npz)
    - User mapping (.json)
    - Item mapping (.json)
    - Test ground truth (.json)

Assumptions:
    - Implicit feedback setting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_PATH_1 = Path("../data/raw/log_standard_4_08_to_4_21_pure.csv")
RAW_PATH_2 = Path("../data/raw/log_standard_4_22_to_5_08_pure.csv")
OUTPUT_DIR = Path("../data/processed")

MIN_USER_INTERACTIONS = 4
MIN_ITEM_INTERACTIONS = 5
POS_THRESHOLD_MS = 3000


# ---------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------

def load_data(path1: Path, path2: Path) -> pd.DataFrame:
    """
    Load and concatenate raw interaction data.

    Parameters
    ----------
    path1 : Path
    path2 : Path

    Returns
    -------
    pd.DataFrame
    """
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df = pd.concat([df1, df2], ignore_index=True)

    required_cols = {"user_id", "video_id", "play_time_ms", "time_ms"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def filter_sparse_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove sparse users and items.

    Returns
    -------
    pd.DataFrame
    """
    df = df.groupby("video_id").filter(lambda x: len(x) >= MIN_ITEM_INTERACTIONS)
    df = df.groupby("user_id").filter(lambda x: len(x) >= MIN_USER_INTERACTIONS)

    return df


def compute_interaction_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw play time into an implicit feedback interaction score.

    This function converts `play_time_ms` into a weighted implicit signal
    suitable for collaborative filtering models.

    Transformation logic
    --------------------
    - If play_time_ms == 0:
        Interaction score = 0.0
        This represents explicit negative / skipped interaction.
        We do not apply log transformation to zero values to preserve
        true absence of engagement.

    - If play_time_ms > 0:
        Interaction score = log(1 + play_time_ms)

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe with an additional column:
        `interaction` (float32)
    """
    df = df.copy()

    play_time = df["play_time_ms"].astype(float)
    interaction = np.where(play_time == 0.0, 0.0, np.log1p(play_time))

    df["interaction"] = interaction.astype(np.float32)

    return df


def temporal_leave_one_out(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform leave-one-out split per user.

    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """
    df = df.sort_values("time_ms")

    train_parts = []
    test_parts = []

    for _, group in df.groupby("user_id"):
        group = group.sort_values("time_ms")

        train_parts.append(group.iloc[:-1])
        test_parts.append(group.iloc[-1:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df


def build_mappings(
    train_df: pd.DataFrame,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Create user and item index mappings from training data only.

    Returns
    -------
    user_map : Dict[int, int]
    item_map : Dict[int, int]
    """
    user_ids = train_df["user_id"].astype(int).unique()
    item_ids = train_df["video_id"].astype(int).unique()

    user_map = {int(u): int(i) for i, u in enumerate(user_ids)}
    item_map = {int(v): int(i) for i, v in enumerate(item_ids)}

    return user_map, item_map


def build_sparse_matrix(
    train_df: pd.DataFrame,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
) -> csr_matrix:
    """
    Construct CSR user-item interaction matrix.

    CSR is used because recommender datasets are extremely sparse
    (most users interact with only a small fraction of items).

    Storing data as a dense DataFrame or CSV would waste memory
    by storing mostly zeros. CSR instead stores only nonzero
    interactions and supports fast matrix factorization operations.

    Returns
    -------
    csr_matrix
    """
    rows = train_df["user_id"].map(user_map).values
    cols = train_df["video_id"].map(item_map).values
    data = train_df["interaction"].values

    n_users = len(user_map)
    n_items = len(item_map)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    return matrix


def build_test_ground_truth(
    test_df: pd.DataFrame,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
) -> Dict[int, int]:
    """
    Build ground truth dictionary for recommender evaluation.

    Ground truth represents the true future interaction that the model
    should rank highly during evaluation. We filter true interaction 
    as play_time_ms >= a threshold

    Since we use leave-one-out temporal splitting, each user has
    exactly one test interaction (the most recent one). Therefore,
    ground truth is stored as:

        { user_index : item_index }

    This structure is used to evaluate ranking metrics such as
    Recall@K and NDCG@K by checking whether the model can recover
    the held-out interaction.

    Returns
    -------
    Dict[user_index, item_index]
    """
    test_df = test_df.copy()

    test_df["relevant"] = (
        test_df["play_time_ms"] >= POS_THRESHOLD_MS
    ).astype(int)

    gt = {}

    for _, row in test_df.iterrows():
        if row["relevant"] == 1:
            user_id = int(row["user_id"])
            video_id = int(row["video_id"])

            if user_id in user_map and video_id in item_map:
                gt[user_map[user_id]] = item_map[video_id]

    return gt


def save_artifacts(
    train_matrix: csr_matrix,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
    test_ground_truth: Dict[int, int],
) -> None:
    """
    Persist artifacts to disk.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sp.save_npz(OUTPUT_DIR / "train_matrix.npz", train_matrix)

    with open(OUTPUT_DIR / "user_map.json", "w") as f:
        json.dump(user_map, f)

    with open(OUTPUT_DIR / "item_map.json", "w") as f:
        json.dump(item_map, f)

    with open(OUTPUT_DIR / "test_ground_truth.json", "w") as f:
        json.dump(test_ground_truth, f)


# ---------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------

def main() -> None:
    """
    Execute full dataset preparation pipeline.
    """
    df = load_data(RAW_PATH_1, RAW_PATH_2)
    df = filter_sparse_entities(df)
    df = compute_interaction_signal(df)

    train_df, test_df = temporal_leave_one_out(df)

    user_map, item_map = build_mappings(train_df)

    train_matrix = build_sparse_matrix(train_df, user_map, item_map)

    test_ground_truth = build_test_ground_truth(
        test_df,
        user_map,
        item_map,
    )

    save_artifacts(
        train_matrix,
        user_map,
        item_map,
        test_ground_truth,
    )

    print("Dataset successfully built.")
    print(f"Users: {len(user_map)}")
    print(f"Items: {len(item_map)}")
    print(f"Ground truth users: {len(test_ground_truth)}")


if __name__ == "__main__":
    main()