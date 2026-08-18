"""Run the strict train-only EquiMotion inference and ComfortGuard pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor, cho_solve

from . import training as train

ROOT = Path(__file__).resolve().parents[2]

DT = 0.1
HISTORY = 100
FUTURE = 200
TOTAL = 300
FIXED_SMOOTH_LAMBDA = 24723409.822928518
COMFORT_TARGET = 0.99005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_dataset(path: Path, name: str) -> Path:
    resolved = path.resolve()
    expected = (ROOT / "data" / name).resolve()
    if resolved != expected:
        raise ValueError(f"Expected {expected}, got {resolved}")
    return resolved


def build_test_tensors(test_csv: Path, checkpoint: dict):
    columns = [
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
        "Speed_Diff",
    ]
    frame = pd.read_csv(test_csv, usecols=columns).sort_values(["Segment_ID", "Time_Index"])
    sizes = frame.groupby("Segment_ID", sort=True).size().to_numpy()
    if not np.all(sizes == TOTAL):
        raise ValueError("Every test.csv segment must contain 300 rows")
    ids = np.sort(frame["Segment_ID"].unique().astype(np.int64))
    n = len(ids)
    arrays = {
        name: frame[name].to_numpy(dtype=np.float32).reshape(n, TOTAL) for name in columns[2:]
    }
    if not np.isfinite(arrays["Pos_FAV"][:, :HISTORY]).all():
        raise ValueError("Observed test FAV positions contain non-finite values")
    for name in ("Pos_FAV", "Speed_FAV", "Acc_FAV", "Spatial_Gap", "Speed_Diff"):
        if not np.isfinite(arrays[name][:, :HISTORY]).all():
            raise ValueError(f"Observed test column {name} contains non-finite values")

    features = np.zeros((n, TOTAL, 36), dtype=np.float32)
    features[:, :, 0] = arrays["Speed_LV"]
    features[:, :, 1] = arrays["Acc_LV"]
    features[:, :, 2] = arrays["Pos_LV"] - arrays["Pos_LV"][:, 99:100]
    features[:, :, 3] = arrays["Type_LV"]
    features[:, :, 4] = np.linspace(-1.0, 1.0, TOTAL, dtype=np.float32)
    features[:, :HISTORY, 5] = arrays["Speed_FAV"][:, :HISTORY]
    features[:, :HISTORY, 6] = arrays["Acc_FAV"][:, :HISTORY]
    features[:, :HISTORY, 7] = arrays["Spatial_Gap"][:, :HISTORY]
    features[:, :HISTORY, 8] = arrays["Speed_Diff"][:, :HISTORY]
    features[:, :HISTORY, 9] = arrays["Pos_FAV"][:, :HISTORY] - arrays["Pos_FAV"][:, 99:100]
    for channel, name in (
        (5, "Speed_FAV"),
        (6, "Acc_FAV"),
        (7, "Spatial_Gap"),
        (8, "Speed_Diff"),
    ):
        features[:, HISTORY:, channel] = arrays[name][:, 99:100]

    identity = np.eye(13, dtype=np.float32)
    lv_index = np.clip(np.rint(arrays["ID_LV"]).astype(np.int64) + 1, 0, 12)
    fav_visible = arrays["ID_FAV"].copy()
    fav_visible[:, HISTORY:] = fav_visible[:, 99:100]
    fav_index = np.clip(np.rint(fav_visible).astype(np.int64) + 1, 0, 12)
    features[:, :, 10:23] = identity[lv_index]
    features[:, :, 23:36] = identity[fav_index]

    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    normalized = ((features - feature_mean) / feature_std).astype(np.float32)
    x0 = arrays["Pos_FAV"][:, 99].astype(np.float64)
    v0 = arrays["Speed_FAV"][:, 99].astype(np.float32)
    gap0 = arrays["Spatial_Gap"][:, 99].astype(np.float32)
    tensors = (
        torch.from_numpy(normalized),
        torch.from_numpy(arrays["Speed_LV"][:, HISTORY:].astype(np.float32)),
        torch.from_numpy(x0),
        torch.from_numpy(v0),
        torch.from_numpy(gap0),
    )
    lv_future_position = arrays["Pos_LV"][:, HISTORY:].astype(np.float64)
    vehicle_length = (
        arrays["Pos_LV"][:, 99] - arrays["Pos_FAV"][:, 99] - arrays["Spatial_Gap"][:, 99]
    ).astype(np.float64)
    future_time = frame["Time_Index"].to_numpy(dtype=np.float64).reshape(n, TOTAL)[:, HISTORY:]
    return ids, tensors, lv_future_position, vehicle_length, future_time


def load_model(checkpoint: dict, device: torch.device):
    config = train.StateSpaceConfig(**checkpoint["model_config"])
    model = train.FutureConditionedStateSpace(
        config, train.make_speed_smoother(config.speed_smooth_lambda)
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def smooth_positions(values: np.ndarray, strength: float) -> np.ndarray:
    difference = np.diff(np.eye(FUTURE), n=3, axis=0)
    factor = cho_factor(
        np.eye(FUTURE) + strength * (difference.T @ difference),
        check_finite=False,
    )
    return cho_solve(factor, values.T, check_finite=False).T


def comfort_rows(values: np.ndarray) -> np.ndarray:
    speed = np.diff(values, axis=1) / DT
    acceleration = np.diff(speed, axis=1) / DT
    jerk = np.diff(acceleration, axis=1) / DT
    return np.maximum(0.0, 1.0 - np.sqrt(np.mean(jerk * jerk, axis=1)) / 5.0)


def safety_rows(
    values: np.ndarray, lv_position: np.ndarray, vehicle_length: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gap = lv_position - values - vehicle_length[:, None]
    pred_speed = np.diff(values, axis=1) / DT
    lv_speed = np.diff(lv_position, axis=1) / DT
    closing = pred_speed - lv_speed
    ttc = np.full_like(closing, np.inf)
    mask = closing > 1.0e-9
    ttc[mask] = gap[:, :-1][mask] / closing[mask]
    violation = np.mean(ttc < 3.0, axis=1)
    collision = np.any(gap < 0.0, axis=1)
    safety = np.where(collision, 0.0, 1.0 - violation)
    return safety, gap, closing


def safety_guard(
    candidate: np.ndarray,
    baseline: np.ndarray,
    lv_position: np.ndarray,
    vehicle_length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_safety, _, _ = safety_rows(baseline, lv_position, vehicle_length)
    candidate_safety, gap, closing = safety_rows(candidate, lv_position, vehicle_length)
    shifts = np.zeros(len(candidate), dtype=np.float64)
    for row in np.flatnonzero(candidate_safety + 1.0e-12 < baseline_safety):
        allowed = int(round((1.0 - baseline_safety[row]) * (FUTURE - 1)))
        requirements = 3.0 * closing[row] - gap[row, :-1]
        requirements = requirements[closing[row] > 1.0e-9]
        ttc_shift = 0.0
        if len(requirements) > allowed:
            ordered = np.sort(requirements)[::-1]
            ttc_shift = max(0.0, float(ordered[allowed]) + 1.0e-6)
        collision_shift = max(0.0, float(0.05 - np.min(gap[row])))
        shifts[row] = max(ttc_shift, collision_shift)
    guarded = candidate - shifts[:, None]
    guarded_safety, _, _ = safety_rows(guarded, lv_position, vehicle_length)
    if np.any(guarded_safety + 1.0e-12 < baseline_safety):
        raise RuntimeError("Prediction-only Safety guard failed to preserve baseline Safety")
    return guarded, shifts


def diagnostics(
    values: np.ndarray, lv_position: np.ndarray, vehicle_length: np.ndarray
) -> dict[str, float]:
    safety, gap, _ = safety_rows(values, lv_position, vehicle_length)
    comfort = comfort_rows(values)
    return {
        "segments": int(len(values)),
        "predicted_safety_mean": float(np.mean(safety)),
        "predicted_comfort_mean": float(np.mean(comfort)),
        "predicted_comfort_min": float(np.min(comfort)),
        "min_gap_min": float(np.min(gap)),
        "min_gap_p05": float(np.quantile(np.min(gap, axis=1), 0.05)),
        "jerk_rms_mean": float(np.mean((1.0 - comfort) * 5.0)),
    }


def write_predictions(
    ids: np.ndarray,
    time: np.ndarray,
    values: np.ndarray,
    path: Path,
) -> None:
    output = pd.DataFrame(
        {
            "Segment_ID": np.repeat(ids, FUTURE),
            "Time_Index": time.reshape(-1),
            "Pos_FAV": values.reshape(-1),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EquiMotion trajectory predictions")
    parser.add_argument("--test-csv", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint produced by equimotion-train",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "predictions.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts" / "inference_report.json",
    )
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        default=ROOT / "artifacts" / "test_predictions.npz",
    )
    args = parser.parse_args()

    test_csv = require_dataset(args.test_csv, "test.csv")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not checkpoint.get("full_train", False):
        raise ValueError("Release checkpoint is not marked as full-train")
    train_ids = np.asarray(checkpoint["train_ids"])
    if len(np.unique(train_ids)) != 4517:
        raise ValueError("Release checkpoint does not contain all 4517 train Segment_IDs")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ids, tensors, lv_position, vehicle_length, future_time = build_test_tensors(
        test_csv, checkpoint
    )
    model = load_model(checkpoint, device)
    index = np.arange(len(ids), dtype=np.int64)
    core_predictions = train.predict(model, tensors, index, args.batch_size)

    smooth_lambda = FIXED_SMOOTH_LAMBDA
    candidate = smooth_positions(core_predictions, smooth_lambda)
    while float(np.mean(comfort_rows(candidate))) < COMFORT_TARGET:
        smooth_lambda *= 1.2
        candidate = smooth_positions(core_predictions, smooth_lambda)
    predictions, shifts = safety_guard(candidate, core_predictions, lv_position, vehicle_length)
    if float(np.mean(comfort_rows(predictions))) <= 0.990:
        raise RuntimeError("EquiMotion failed the Comfort > 0.990 release gate")

    write_predictions(ids, future_time, predictions, args.output)
    args.prediction_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.prediction_cache,
        ids=ids.astype(np.int32),
        pred=predictions,
        smooth_lambda=smooth_lambda,
        safety_shifts=shifts,
    )
    report = {
        "model_name": "EquiMotion",
        "data_boundary": {
            "training_and_targets": "data/train.csv only",
            "full_train_segments": 4517,
            "inference": "data/test.csv visible features only",
            "test_future_fav_labels_used": False,
        },
        "test_prediction_diagnostics": {
            "before_comfort_guard": diagnostics(core_predictions, lv_position, vehicle_length),
            "equimotion": diagnostics(predictions, lv_position, vehicle_length),
            "smooth_lambda": float(smooth_lambda),
            "safety_shifted_segments": int(np.sum(shifts > 0.0)),
            "safety_shift_max_m": float(np.max(shifts)),
        },
        "artifacts": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint.resolve()),
            "test_csv_sha256": sha256(test_csv),
            "predictions": str(args.output.resolve()),
            "predictions_sha256": sha256(args.output.resolve()),
        },
        "future_truth_disclosure": (
            "data/test.csv has no future FAV truth, so reference-based metrics "
            "cannot be measured locally."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
