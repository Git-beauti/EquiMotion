"""Command-line evaluation for predictions with available future truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FUTURE_STEPS, HIST_STEPS, read_segments
from .metrics import make_score_args, score_segment, summarize

ROOT = Path(__file__).resolve().parents[2]


def load_predictions(path: Path) -> dict[int, np.ndarray]:
    frame = pd.read_csv(path, usecols=["Segment_ID", "Time_Index", "Pos_FAV"])
    if frame.duplicated(["Segment_ID", "Time_Index"]).any():
        raise ValueError("Prediction file contains duplicate keys")
    predictions = {}
    for segment_id, rows in frame.groupby("Segment_ID", sort=True):
        rows = rows.sort_values("Time_Index")
        if len(rows) != FUTURE_STEPS:
            raise ValueError(
                f"Segment {segment_id} has {len(rows)} predictions; expected {FUTURE_STEPS}"
            )
        values = rows["Pos_FAV"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Segment {segment_id} has non-finite predictions")
        predictions[int(segment_id)] = values
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a prediction CSV on segments whose future FAV truth is known"
    )
    parser.add_argument("--truth-csv", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "evaluation.json")
    parser.add_argument("--speed-threshold", type=float, default=5.0)
    parser.add_argument("--headway-threshold", type=float, default=25.0)
    parser.add_argument("--acc-threshold", type=float, default=5.0)
    parser.add_argument("--jerk-threshold", type=float, default=5.0)
    parser.add_argument("--ttc-threshold", type=float, default=3.0)
    args = parser.parse_args()

    segments = read_segments(args.truth_csv)
    predictions = load_predictions(args.predictions)
    missing = sorted(set(predictions) - set(segments))
    if missing:
        raise ValueError(f"Prediction Segment_IDs absent from truth data: {missing[:10]}")
    thresholds = make_score_args(strict=True)
    thresholds.speed_threshold = args.speed_threshold
    thresholds.headway_threshold = args.headway_threshold
    thresholds.acc_threshold = args.acc_threshold
    thresholds.jerk_threshold = args.jerk_threshold
    thresholds.ttc_threshold = args.ttc_threshold

    rows = []
    for segment_id, prediction in predictions.items():
        segment = segments[segment_id]
        truth = segment["Pos_FAV"].iloc[HIST_STEPS:].to_numpy(dtype=float)
        if len(truth) != FUTURE_STEPS or not np.isfinite(truth).all():
            raise ValueError(f"Segment {segment_id} does not expose complete future truth")
        row = score_segment(segment, prediction, thresholds)
        row["segment_id"] = segment_id
        rows.append(row)

    report = {
        "score_type": "local_estimate",
        "formula_source": "https://ieee-et3-challenge.com/evaluation/",
        "threshold_disclosure": (
            "The public page names but does not numerically publish its normalization "
            "thresholds; command-line values are local estimates."
        ),
        "thresholds": vars(thresholds),
        "segments": len(rows),
        "summary": summarize([{k: v for k, v in row.items() if k != "segment_id"} for row in rows]),
        "per_segment": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
