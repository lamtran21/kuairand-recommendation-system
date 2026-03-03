from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.popularity import train, predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-matrix",
        type=Path,
        default=PROJECT_ROOT / "data/train_matrix.npz",
    )
    parser.add_argument(
        "--test-ground-truth",
        type=Path,
        default=PROJECT_ROOT / "data/test_ground_truth.json",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/model_predictions",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_matrix = sp.load_npz(args.train_matrix).tocsr()

    model = train(train_matrix)

    rec_df = predict(
        model=model,
        interactions=train_matrix,
        user_ids=np.arange(train_matrix.shape[0]),
        k=200,
    )

    rec_path = args.pred_dir / "popularity_recommendations.cf"
    model_path = args.model_dir / "popularity_model.pkl"

    rec_df.to_csv(rec_path, index=False)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Popularity baseline complete.")
    print("Predictions saved to:", rec_path)
    print("Model saved to:", model_path)


if __name__ == "__main__":
    main()