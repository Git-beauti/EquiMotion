"""Local trajectory-accuracy, safety, and comfort metrics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from .data import DT, FUTURE_STEPS, HIST_STEPS, vehicle_length


def make_score_args(strict: bool = True):
    """Return configurable local thresholds for the public score equations.

    The organizer publishes the equations but not the numerical normalization
    thresholds. These values are therefore estimates, not hidden-server values.
    """
    scale = 1.0 if strict else 2.0
    return SimpleNamespace(
        speed_threshold=5.0 * scale,
        headway_threshold=25.0 * scale,
        acc_threshold=5.0 * scale,
        jerk_threshold=5.0 * scale,
        ttc_threshold=3.0,
    )


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def score_segment(segment, pred_pos: np.ndarray, args=None) -> Dict[str, float]:
    if args is None:
        args = make_score_args(strict=True)
    pred_pos = np.asarray(pred_pos, dtype=float)
    if pred_pos.shape != (FUTURE_STEPS,) or not np.isfinite(pred_pos).all():
        raise ValueError(f"Expected {FUTURE_STEPS} finite predicted positions")

    future = slice(HIST_STEPS, HIST_STEPS + FUTURE_STEPS)
    true_pos = segment["Pos_FAV"].iloc[future].to_numpy(dtype=float)
    lv_pos = segment["Pos_LV"].iloc[future].to_numpy(dtype=float)
    if not np.isfinite(true_pos).all():
        raise ValueError("Reference FAV future is unavailable")

    pred_speed = np.diff(pred_pos) / DT
    true_speed = np.diff(true_pos) / DT
    lv_speed = np.diff(lv_pos) / DT
    pred_acc = np.diff(pred_speed) / DT
    true_acc = np.diff(true_speed) / DT
    pred_jerk = np.diff(pred_acc) / DT

    speed_rmse = rmse(pred_speed - true_speed)
    headway_rmse = rmse((lv_pos - pred_pos) - (lv_pos - true_pos))
    acc_rmse = rmse(pred_acc - true_acc)
    jerk_rms = rmse(pred_jerk)
    accuracy = (
        0.4 * max(0.0, 1.0 - speed_rmse / args.speed_threshold)
        + 0.4 * max(0.0, 1.0 - headway_rmse / args.headway_threshold)
        + 0.2 * max(0.0, 1.0 - acc_rmse / args.acc_threshold)
    )

    gap = lv_pos - pred_pos - vehicle_length(segment)
    closing_speed = pred_speed - lv_speed
    ttc = np.full_like(closing_speed, np.inf)
    closing = closing_speed > 1.0e-9
    ttc[closing] = gap[:-1][closing] / closing_speed[closing]
    violation_ratio = float(np.mean(ttc < args.ttc_threshold))
    safety = 0.0 if np.any(gap < 0.0) else 1.0 - violation_ratio
    comfort = max(0.0, 1.0 - jerk_rms / args.jerk_threshold)
    final = 0.5 * accuracy + 0.3 * safety + 0.2 * comfort
    return {
        "accuracy": float(accuracy),
        "safety": float(safety),
        "comfort": float(comfort),
        "final": float(final),
        "speed_rmse": speed_rmse,
        "headway_rmse": headway_rmse,
        "acc_rmse": acc_rmse,
        "jerk_rms": jerk_rms,
        "min_gap": float(np.min(gap)),
        "ttc_violation_ratio": violation_ratio,
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("At least one scored segment is required")
    output: Dict[str, float] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        output[f"{key}_mean"] = float(np.mean(values))
        if key in {"final", "accuracy", "safety", "comfort", "min_gap"}:
            for name, quantile in (("p05", 0.05), ("p50", 0.50), ("p95", 0.95)):
                output[f"{key}_{name}"] = float(np.quantile(values, quantile))
    return output
