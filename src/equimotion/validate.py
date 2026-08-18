"""Validate the packaged train/test CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import validate_dataset

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate car-following trajectory datasets")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--test-csv", type=Path, default=ROOT / "data" / "test.csv")
    args = parser.parse_args()
    result = {
        "train": validate_dataset(args.train_csv, require_future_truth=True),
        "test": validate_dataset(args.test_csv, require_future_truth=False),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
