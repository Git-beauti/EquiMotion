"""Loading and integrity checks for the car-following trajectory dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

DT = 0.1
HIST_STEPS = 100
FUTURE_STEPS = 200
TOTAL_STEPS = HIST_STEPS + FUTURE_STEPS
REQUIRED_COLUMNS = (
    "Segment_ID",
    "Time_Index",
    "ID_LV",
    "Type_LV",
    "Pos_LV",
    "Speed_LV",
    "Acc_LV",
    "ID_FAV",
    "Pos_FAV",
    "Speed_FAV",
    "Acc_FAV",
    "Spatial_Gap",
    "Spatial_Headway",
    "Speed_Diff",
)


def read_segments(csv_path: Path) -> Dict[int, pd.DataFrame]:
    frame = pd.read_csv(csv_path, usecols=list(REQUIRED_COLUMNS))
    frame = frame.sort_values(["Segment_ID", "Time_Index"], kind="mergesort")
    return {
        int(segment_id): segment.reset_index(drop=True)
        for segment_id, segment in frame.groupby("Segment_ID", sort=False)
    }


def vehicle_length(segment: pd.DataFrame) -> float:
    """Estimate the combined half-length term exposed by headway minus gap."""
    difference = (segment["Spatial_Headway"] - segment["Spatial_Gap"]).iloc[:HIST_STEPS]
    value = float(np.nanmedian(difference.to_numpy(dtype=float)))
    return value if np.isfinite(value) and value > 0.0 else 4.5


def validate_dataset(csv_path: Path, *, require_future_truth: bool) -> dict[str, object]:
    frame = pd.read_csv(csv_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame.sort_values(["Segment_ID", "Time_Index"], kind="mergesort")
    sizes = frame.groupby("Segment_ID", sort=True).size()
    if sizes.empty or not np.all(sizes.to_numpy() == TOTAL_STEPS):
        raise ValueError("Every Segment_ID must contain exactly 300 rows")
    if frame.duplicated(["Segment_ID", "Time_Index"]).any():
        raise ValueError("Duplicate (Segment_ID, Time_Index) keys detected")

    observed = frame.groupby("Segment_ID", sort=True).head(HIST_STEPS)
    visible_columns = [
        "Pos_LV",
        "Speed_LV",
        "Acc_LV",
        "Pos_FAV",
        "Speed_FAV",
        "Acc_FAV",
        "Spatial_Gap",
        "Speed_Diff",
    ]
    if not np.isfinite(observed[visible_columns].to_numpy(dtype=float)).all():
        raise ValueError("Observed trajectory values contain NaN or infinity")
    if require_future_truth:
        future = frame.groupby("Segment_ID", sort=True).tail(FUTURE_STEPS)
        truth_columns = ["Pos_FAV", "Speed_FAV", "Acc_FAV"]
        if not np.isfinite(future[truth_columns].to_numpy(dtype=float)).all():
            raise ValueError("Training future FAV truth contains NaN or infinity")
    return {
        "path": str(csv_path.resolve()),
        "rows": int(len(frame)),
        "segments": int(len(sizes)),
        "steps_per_segment": TOTAL_STEPS,
        "future_truth_required": require_future_truth,
    }
