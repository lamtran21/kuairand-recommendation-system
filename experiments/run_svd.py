"""Run script for SVD model.

I/O contract (minimal):
- Input files:
  - train interactions CSV path
  - test interactions CSV path
- Output files:
  - recommendations CSV to outputs/results/
  - metrics JSON/CSV to outputs/results/
  - optional model artifact to outputs/models/
"""

from __future__ import annotations


def main():
    """Load data, train SVD model, predict, evaluate, save outputs."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
