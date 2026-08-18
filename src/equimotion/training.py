"""Training pipeline and neural architecture for EquiMotion."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor, cho_solve
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data import read_segments
from .metrics import make_score_args, score_segment, summarize

ROOT = Path(__file__).resolve().parents[2]

HIST_STEPS = 100
FUTURE_STEPS = 200
TOTAL_STEPS = 300
DT = 0.1
TARGET_SMOOTH_LAMBDA = 1000.0
SEED = 20260714
DEFAULT_SPLIT_SEED = 20260615


@dataclass
class ModelConfig:
    input_channels: int = 36
    hidden_channels: int = 40
    blocks: int = 6
    kernel_size: int = 3
    dropout: float = 0.05
    speed_smooth_lambda: float = 300.0
    max_speed_residual: float = 15.0
    max_position_coefficient: float = 12.0
    position_residual_head: bool = False
    position_smooth_lambda: float = 10000.0
    max_position_residual: float = 10.0
    identity_heads: bool = False
    global_position_head: bool = False
    global_pool_bins: int = 20
    global_decoder_width: int = 256
    delay_mixture: bool = False
    max_delay_steps: int = 30
    attention_layers: int = 0
    attention_heads: int = 8


@dataclass
class GRUConfig:
    input_channels: int = 36
    history_hidden: int = 64
    future_hidden: int = 64
    layers: int = 2
    dropout: float = 0.08
    speed_smooth_lambda: float = 300.0
    max_speed_residual: float = 15.0
    max_position_coefficient: float = 12.0


@dataclass
class StateSpaceConfig:
    input_channels: int = 36
    hidden_channels: int = 128
    layers: int = 2
    dropout: float = 0.08
    speed_smooth_lambda: float = 300.0
    position_smooth_lambda: float = 10000.0
    max_acceleration: float = 4.0
    max_position_residual: float = 10.0
    max_position_coefficient: float = 12.0
    hybrid_direct: bool = False
    max_speed_residual: float = 15.0
    bidirectional_history: bool = False
    gap_position_head: bool = False
    max_gap_residual: float = 12.0
    final_position_smooth_lambda: float = 0.0
    joint_future_layers: int = 0
    joint_future_heads: int = 8
    identity_film: bool = False
    identity_gate_bias: bool = False
    identity_trajectory_bias: bool = False


@dataclass
class GapMixerConfig:
    input_channels: int = 36
    hidden_channels: int = 160
    layers: int = 2
    blocks: int = 6
    dropout: float = 0.06
    speed_smooth_lambda: float = 300.0
    position_smooth_lambda: float = 10000.0
    max_speed_residual: float = 15.0
    max_gap_residual: float = 12.0
    max_position_residual: float = 12.0
    max_position_coefficient: float = 12.0


def require_train_csv(path: Path) -> Path:
    resolved = path.resolve()
    expected = (ROOT / "data" / "train.csv").resolve()
    if resolved != expected:
        raise ValueError(
            f"Training input must be the repository train.csv: {expected}; got {resolved}"
        )
    return resolved


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def smooth_positions(values: np.ndarray, smooth_lambda: float) -> np.ndarray:
    difference = np.diff(np.eye(FUTURE_STEPS), n=3, axis=0)
    factor = cho_factor(
        np.eye(FUTURE_STEPS) + smooth_lambda * (difference.T @ difference), check_finite=False
    )
    return cho_solve(factor, values.T, check_finite=False).T


def build_cache(train_csv: Path, cache_path: Path, target_smooth_lambda: float) -> None:
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
    frame = pd.read_csv(train_csv, usecols=columns).sort_values(["Segment_ID", "Time_Index"])
    sizes = frame.groupby("Segment_ID", sort=True).size().to_numpy()
    if not np.all(sizes == TOTAL_STEPS):
        raise ValueError("Every train.csv segment must contain exactly 300 rows")

    ids = np.sort(frame["Segment_ID"].unique().astype(np.int64))
    n = len(ids)
    arrays = {
        name: frame[name].to_numpy(dtype=np.float32).reshape(n, TOTAL_STEPS) for name in columns[2:]
    }
    # Keep target positions in float64 until after converting them to relative
    # coordinates. Absolute roadway coordinates lose meaningful derivatives if
    # they are quantized to float32 before differencing.
    pos_fav_exact = frame["Pos_FAV"].to_numpy(dtype=np.float64).reshape(n, TOTAL_STEPS)
    speed_fav_exact = frame["Speed_FAV"].to_numpy(dtype=np.float64).reshape(n, TOTAL_STEPS)

    pos_lv = arrays["Pos_LV"]
    speed_lv = arrays["Speed_LV"]
    acc_lv = arrays["Acc_LV"]
    pos_fav = arrays["Pos_FAV"]
    speed_fav = arrays["Speed_FAV"]
    acc_fav = arrays["Acc_FAV"]
    gap = arrays["Spatial_Gap"]
    speed_diff = arrays["Speed_Diff"]
    type_lv = arrays["Type_LV"]

    features = np.zeros((n, TOTAL_STEPS, 36), dtype=np.float32)
    features[:, :, 0] = speed_lv
    features[:, :, 1] = acc_lv
    features[:, :, 2] = pos_lv - pos_lv[:, 99:100]
    features[:, :, 3] = type_lv
    features[:, :, 4] = np.linspace(-1.0, 1.0, TOTAL_STEPS, dtype=np.float32)
    features[:, :HIST_STEPS, 5] = speed_fav[:, :HIST_STEPS]
    features[:, :HIST_STEPS, 6] = acc_fav[:, :HIST_STEPS]
    features[:, :HIST_STEPS, 7] = gap[:, :HIST_STEPS]
    features[:, :HIST_STEPS, 8] = speed_diff[:, :HIST_STEPS]
    features[:, :HIST_STEPS, 9] = pos_fav[:, :HIST_STEPS] - pos_fav[:, 99:100]
    features[:, HIST_STEPS:, 5] = speed_fav[:, 99:100]
    features[:, HIST_STEPS:, 6] = acc_fav[:, 99:100]
    features[:, HIST_STEPS:, 7] = gap[:, 99:100]
    features[:, HIST_STEPS:, 8] = speed_diff[:, 99:100]
    features[:, HIST_STEPS:, 9] = 0.0

    lv_id_index = np.clip(np.rint(arrays["ID_LV"]).astype(np.int64) + 1, 0, 12)
    fav_id_visible = arrays["ID_FAV"].copy()
    fav_id_visible[:, HIST_STEPS:] = fav_id_visible[:, 99:100]
    fav_id_index = np.clip(np.rint(fav_id_visible).astype(np.int64) + 1, 0, 12)
    identity = np.eye(13, dtype=np.float32)
    features[:, :, 10:23] = identity[lv_id_index]
    features[:, :, 23:36] = identity[fav_id_index]

    raw_future = pos_fav_exact[:, HIST_STEPS:]
    target_pos = smooth_positions(raw_future, target_smooth_lambda)
    x0 = pos_fav_exact[:, HIST_STEPS - 1]
    v0 = speed_fav_exact[:, HIST_STEPS - 1]
    target_pos_rel = target_pos - x0[:, None]
    target_speed = np.empty_like(target_pos_rel)
    target_speed[:, 0] = target_pos_rel[:, 0] / DT
    target_speed[:, 1:] = np.diff(target_pos_rel, axis=1) / DT
    target_acc = np.empty_like(target_speed)
    target_acc[:, 0] = (target_speed[:, 0] - v0) / DT
    target_acc[:, 1:] = np.diff(target_speed, axis=1) / DT

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        ids=ids,
        features=features,
        target_pos=target_pos.astype(np.float64),
        target_pos_rel=target_pos_rel.astype(np.float32),
        target_speed=target_speed.astype(np.float32),
        target_acc=target_acc.astype(np.float32),
        raw_future=raw_future.astype(np.float64),
        x0=x0,
        v0=v0.astype(np.float32),
        lv_future_speed=speed_lv[:, HIST_STEPS:],
        target_smooth_lambda=np.asarray([target_smooth_lambda], dtype=np.float64),
    )


def build_augmented_cache(
    train_csv: Path,
    cache_path: Path,
    target_smooth_lambda: float,
    cuts: Tuple[int, ...],
) -> None:
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
    frame = pd.read_csv(train_csv, usecols=columns).sort_values(["Segment_ID", "Time_Index"])
    sizes = frame.groupby("Segment_ID", sort=True).size().to_numpy()
    if not np.all(sizes == TOTAL_STEPS):
        raise ValueError("Every train.csv segment must contain exactly 300 rows")

    segment_ids = np.sort(frame["Segment_ID"].unique().astype(np.int64))
    n = len(segment_ids)
    arrays = {
        name: frame[name].to_numpy(dtype=np.float32).reshape(n, TOTAL_STEPS) for name in columns[2:]
    }
    pos_fav_exact = frame["Pos_FAV"].to_numpy(dtype=np.float64).reshape(n, TOTAL_STEPS)
    speed_fav_exact = frame["Speed_FAV"].to_numpy(dtype=np.float64).reshape(n, TOTAL_STEPS)
    identity = np.eye(13, dtype=np.float32)
    feature_rows = []
    raw_rows = []
    x0_rows = []
    v0_rows = []
    lv_speed_rows = []

    for cut in cuts:
        history_index = np.clip(np.arange(cut - HIST_STEPS, cut), 0, TOTAL_STEPS - 1)
        future_index = np.arange(cut, cut + FUTURE_STEPS)
        sequence_index = np.concatenate([history_index, future_index])
        features = np.zeros((n, TOTAL_STEPS, 36), dtype=np.float32)
        features[:, :, 0] = arrays["Speed_LV"][:, sequence_index]
        features[:, :, 1] = arrays["Acc_LV"][:, sequence_index]
        features[:, :, 2] = arrays["Pos_LV"][:, sequence_index] - arrays["Pos_LV"][:, cut - 1 : cut]
        features[:, :, 3] = arrays["Type_LV"][:, sequence_index]
        features[:, :, 4] = np.linspace(-1.0, 1.0, TOTAL_STEPS, dtype=np.float32)
        features[:, :HIST_STEPS, 5] = arrays["Speed_FAV"][:, history_index]
        features[:, :HIST_STEPS, 6] = arrays["Acc_FAV"][:, history_index]
        features[:, :HIST_STEPS, 7] = arrays["Spatial_Gap"][:, history_index]
        features[:, :HIST_STEPS, 8] = arrays["Speed_Diff"][:, history_index]
        features[:, :HIST_STEPS, 9] = (
            arrays["Pos_FAV"][:, history_index] - arrays["Pos_FAV"][:, cut - 1 : cut]
        )
        for channel, name in (
            (5, "Speed_FAV"),
            (6, "Acc_FAV"),
            (7, "Spatial_Gap"),
            (8, "Speed_Diff"),
        ):
            features[:, HIST_STEPS:, channel] = arrays[name][:, cut - 1 : cut]

        lv_id = np.clip(
            np.rint(arrays["ID_LV"][:, sequence_index]).astype(np.int64) + 1,
            0,
            12,
        )
        fav_id = np.clip(
            np.rint(arrays["ID_FAV"][:, history_index]).astype(np.int64) + 1,
            0,
            12,
        )
        features[:, :, 10:23] = identity[lv_id]
        features[:, :HIST_STEPS, 23:36] = identity[fav_id]
        final_fav_id = fav_id[:, -1:]
        features[:, HIST_STEPS:, 23:36] = identity[np.broadcast_to(final_fav_id, (n, FUTURE_STEPS))]

        feature_rows.append(features)
        raw_rows.append(pos_fav_exact[:, future_index])
        x0_rows.append(pos_fav_exact[:, cut - 1])
        v0_rows.append(speed_fav_exact[:, cut - 1])
        lv_speed_rows.append(arrays["Speed_LV"][:, future_index])

    features = np.concatenate(feature_rows, axis=0)
    raw_future = np.concatenate(raw_rows, axis=0)
    x0 = np.concatenate(x0_rows, axis=0)
    v0 = np.concatenate(v0_rows, axis=0)
    lv_future_speed = np.concatenate(lv_speed_rows, axis=0)
    target_pos = smooth_positions(raw_future, target_smooth_lambda)
    target_pos_rel = target_pos - x0[:, None]
    target_speed = np.empty_like(target_pos_rel)
    target_speed[:, 0] = target_pos_rel[:, 0] / DT
    target_speed[:, 1:] = np.diff(target_pos_rel, axis=1) / DT
    target_acc = np.empty_like(target_speed)
    target_acc[:, 0] = (target_speed[:, 0] - v0) / DT
    target_acc[:, 1:] = np.diff(target_speed, axis=1) / DT

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        ids=np.tile(segment_ids, len(cuts)),
        cut_points=np.repeat(np.asarray(cuts, dtype=np.int16), n),
        features=features,
        target_pos=target_pos.astype(np.float64),
        target_pos_rel=target_pos_rel.astype(np.float32),
        target_speed=target_speed.astype(np.float32),
        target_acc=target_acc.astype(np.float32),
        raw_future=raw_future.astype(np.float64),
        x0=x0,
        v0=v0.astype(np.float32),
        lv_future_speed=lv_future_speed,
        target_smooth_lambda=np.asarray([target_smooth_lambda], dtype=np.float64),
        augmentation_cuts=np.asarray(cuts, dtype=np.int16),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        groups = 5 if channels % 5 == 0 else math.gcd(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(F.gelu(self.norm1(self.conv1(values))))
        hidden = self.dropout(F.gelu(self.norm2(self.conv2(hidden))))
        return values + hidden


class FutureConditionedGapMixer(nn.Module):
    FUTURE_CHANNELS = [0, 1, 2, 3, 4, *range(10, 36)]

    def __init__(self, config: GapMixerConfig, speed_smoother: np.ndarray):
        super().__init__()
        self.config = config
        recurrent_dropout = config.dropout if config.layers > 1 else 0.0
        self.history_gru = nn.GRU(
            input_size=config.input_channels,
            hidden_size=config.hidden_channels,
            num_layers=config.layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.future_gru = nn.GRU(
            input_size=len(self.FUTURE_CHANNELS),
            hidden_size=config.hidden_channels // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.input_projection = nn.Sequential(
            nn.Linear(config.hidden_channels * 2, config.hidden_channels),
            nn.LayerNorm(config.hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    config.hidden_channels,
                    kernel_size=3,
                    dilation=2**index,
                    dropout=config.dropout,
                )
                for index in range(config.blocks)
            ]
        )
        self.speed_head = nn.Sequential(
            nn.Conv1d(config.hidden_channels, config.hidden_channels, 1),
            nn.GELU(),
            nn.Conv1d(config.hidden_channels, 1, 1),
        )
        self.gap_head = nn.Sequential(
            nn.Conv1d(config.hidden_channels, config.hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.hidden_channels, 1, 3, padding=1),
        )
        self.position_gate = nn.Sequential(
            nn.Conv1d(config.hidden_channels, config.hidden_channels // 2, 1),
            nn.GELU(),
            nn.Conv1d(config.hidden_channels // 2, 1, 1),
        )
        nn.init.constant_(self.position_gate[-1].bias, 2.0)
        self.position_head = nn.Sequential(
            nn.Conv1d(config.hidden_channels, config.hidden_channels, 1),
            nn.GELU(),
            nn.Conv1d(config.hidden_channels, 1, 1),
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(config.hidden_channels * 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 4),
        )
        self.register_buffer("speed_smoother", torch.from_numpy(speed_smoother), persistent=True)
        self.register_buffer(
            "position_smoother",
            torch.from_numpy(make_position_smoother(config.position_smooth_lambda)),
            persistent=True,
        )
        time = torch.linspace(1.0 / FUTURE_STEPS, 1.0, FUTURE_STEPS)
        basis = torch.stack(
            [
                time,
                time.square(),
                torch.sin(torch.pi * time),
                4.0 * time * (1.0 - time),
            ],
            dim=1,
        )
        self.register_buffer("trajectory_basis", basis, persistent=True)

    def forward(
        self,
        features: torch.Tensor,
        lv_future_speed: torch.Tensor,
        v0: torch.Tensor,
        gap0: torch.Tensor | None = None,
    ):
        if gap0 is None:
            raise ValueError("Gap mixer requires the observed final-history gap")
        _, history_state = self.history_gru(features[:, :HIST_STEPS])
        history_context = history_state[-1]
        future_input = features[:, HIST_STEPS:, self.FUTURE_CHANNELS]
        future_encoded, _ = self.future_gru(future_input)
        history_rows = history_context[:, None, :].expand(-1, FUTURE_STEPS, -1)
        hidden = self.input_projection(torch.cat([history_rows, future_encoded], dim=2)).transpose(
            1, 2
        )
        for block in self.blocks:
            hidden = block(hidden)

        speed_residual = self.config.max_speed_residual * torch.tanh(
            self.speed_head(hidden)[:, 0] / 4.0
        )
        raw_speed = torch.clamp_min(lv_future_speed + speed_residual, 0.0)
        speed = F.softplus(raw_speed @ self.speed_smoother.T, beta=4.0)
        speed_position = DT * torch.cumsum(speed, dim=1)

        gap_rollout = gap0[:, None] + DT * torch.cumsum(lv_future_speed - speed, dim=1)
        gap_correction = self.config.max_gap_residual * torch.tanh(
            self.gap_head(hidden)[:, 0] / 5.0
        )
        predicted_gap = gap_rollout + gap_correction @ self.position_smoother.T
        lv_position_rel = DT * torch.cumsum(lv_future_speed, dim=1)
        gap_position_rel = lv_position_rel + gap0[:, None] - predicted_gap
        speed_position_weight = torch.sigmoid(self.position_gate(hidden)[:, 0])
        position_rel = (
            speed_position_weight * speed_position
            + (1.0 - speed_position_weight) * gap_position_rel
        )

        local_position = self.config.max_position_residual * torch.tanh(
            self.position_head(hidden)[:, 0] / 5.0
        )
        position_rel = position_rel + local_position @ self.position_smoother.T
        pooled = torch.cat([history_context, hidden.mean(dim=2), hidden.amax(dim=2)], dim=1)
        coefficients = self.config.max_position_coefficient * torch.tanh(
            self.trajectory_head(pooled) / 4.0
        )
        position_rel = position_rel + coefficients @ self.trajectory_basis.T
        scored_speed = torch.cat(
            [position_rel[:, :1] / DT, torch.diff(position_rel, dim=1) / DT], dim=1
        )
        acceleration = torch.cat(
            [
                (scored_speed[:, :1] - v0[:, None]) / DT,
                torch.diff(scored_speed, dim=1) / DT,
            ],
            dim=1,
        )
        return position_rel, scored_speed, acceleration, gap_position_rel


def make_speed_smoother(smooth_lambda: float) -> np.ndarray:
    difference = np.diff(np.eye(FUTURE_STEPS), n=2, axis=0)
    factor = cho_factor(
        np.eye(FUTURE_STEPS) + smooth_lambda * (difference.T @ difference), check_finite=False
    )
    return cho_solve(factor, np.eye(FUTURE_STEPS), check_finite=False).astype(np.float32)


def make_position_smoother(smooth_lambda: float) -> np.ndarray:
    difference = np.diff(np.eye(FUTURE_STEPS), n=3, axis=0)
    factor = cho_factor(
        np.eye(FUTURE_STEPS) + smooth_lambda * (difference.T @ difference), check_finite=False
    )
    return cho_solve(factor, np.eye(FUTURE_STEPS), check_finite=False).astype(np.float32)


class FutureConditionedTCN(nn.Module):
    def __init__(self, config: ModelConfig, speed_smoother: np.ndarray):
        super().__init__()
        self.config = config
        self.input = nn.Conv1d(config.input_channels, config.hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(config.hidden_channels, config.kernel_size, 2**index, config.dropout)
                for index in range(config.blocks)
            ]
        )
        if config.attention_layers > 0:
            attention_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_channels,
                nhead=config.attention_heads,
                dim_feedforward=config.hidden_channels * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.attention = nn.TransformerEncoder(
                attention_layer,
                num_layers=config.attention_layers,
                norm=nn.LayerNorm(config.hidden_channels),
                enable_nested_tensor=False,
            )
        self.context = nn.Sequential(
            nn.Linear(config.hidden_channels * 2, config.hidden_channels),
            nn.GELU(),
            nn.Linear(config.hidden_channels, config.hidden_channels),
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(HIST_STEPS * 7, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(160, config.hidden_channels),
        )
        if config.delay_mixture:
            self.delay_head = nn.Sequential(
                nn.Linear(config.hidden_channels, 64),
                nn.GELU(),
                nn.Linear(64, config.max_delay_steps + 1),
            )
        self.head = nn.Sequential(
            nn.Conv1d(config.hidden_channels, config.hidden_channels, 1),
            nn.GELU(),
            nn.Conv1d(config.hidden_channels, 13 if config.identity_heads else 1, 1),
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(config.hidden_channels * 3, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(96, 52 if config.identity_heads else 4),
        )
        if config.position_residual_head:
            self.position_residual = nn.Sequential(
                nn.Conv1d(config.hidden_channels, config.hidden_channels, 1),
                nn.GELU(),
                nn.Conv1d(config.hidden_channels, 13 if config.identity_heads else 1, 1),
            )
        if config.global_position_head:
            global_input_width = config.hidden_channels * (config.global_pool_bins + 1)
            self.global_position_decoder = nn.Sequential(
                nn.Linear(global_input_width, config.global_decoder_width),
                nn.LayerNorm(config.global_decoder_width),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.global_decoder_width, FUTURE_STEPS),
            )
        if config.position_residual_head or config.global_position_head:
            self.register_buffer(
                "position_smoother",
                torch.from_numpy(make_position_smoother(config.position_smooth_lambda)),
                persistent=True,
            )
        t = torch.linspace(1.0 / FUTURE_STEPS, 1.0, FUTURE_STEPS)
        trajectory_basis = torch.stack(
            [t, t.square(), torch.sin(torch.pi * t), 4.0 * t * (1.0 - t)],
            dim=1,
        )
        self.register_buffer("trajectory_basis", trajectory_basis, persistent=True)
        self.register_buffer("speed_smoother", torch.from_numpy(speed_smoother), persistent=True)

    def forward(
        self,
        features: torch.Tensor,
        lv_future_speed: torch.Tensor,
        v0: torch.Tensor,
        gap0: torch.Tensor | None = None,
    ):
        hidden = self.input(features.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        if self.config.attention_layers > 0:
            hidden = self.attention(hidden.transpose(1, 2)).transpose(1, 2)
        history = hidden[:, :, :HIST_STEPS]
        pooled = torch.cat([history.mean(dim=2), history.amax(dim=2)], dim=1)
        history_raw = features[:, :HIST_STEPS, [0, 1, 5, 6, 7, 8, 9]].reshape(features.shape[0], -1)
        driver_context = self.context(pooled) + self.history_encoder(history_raw)
        hidden = hidden + driver_context.unsqueeze(2)
        fav_index = features[:, HIST_STEPS - 1, 23:36].argmax(dim=1)
        residual_channels = self.head(hidden)[:, :, HIST_STEPS:]
        if self.config.identity_heads:
            residual_logits = residual_channels.gather(
                1, fav_index[:, None, None].expand(-1, 1, FUTURE_STEPS)
            )[:, 0]
        else:
            residual_logits = residual_channels[:, 0]
        residual = self.config.max_speed_residual * torch.tanh(residual_logits / 5.0)
        if self.config.delay_mixture:
            delay_weights = F.softmax(self.delay_head(driver_context), dim=1)
            delayed_speeds = []
            for delay in range(self.config.max_delay_steps + 1):
                if delay == 0:
                    delayed = lv_future_speed
                else:
                    delayed = torch.cat(
                        [lv_future_speed[:, :1].expand(-1, delay), lv_future_speed[:, :-delay]],
                        dim=1,
                    )
                delayed_speeds.append(delayed)
            delayed_bank = torch.stack(delayed_speeds, dim=1)
            lv_baseline = torch.sum(delay_weights[:, :, None] * delayed_bank, dim=1)
        else:
            lv_baseline = lv_future_speed
        raw_speed = lv_baseline + residual
        speed = raw_speed @ self.speed_smoother.T
        speed = F.softplus(speed, beta=4.0)
        position_rel = DT * torch.cumsum(speed, dim=1)
        future_hidden = hidden[:, :, HIST_STEPS:]
        full_context = torch.cat(
            [driver_context, future_hidden.mean(dim=2), future_hidden.amax(dim=2)],
            dim=1,
        )
        coefficient_logits = self.trajectory_head(full_context)
        if self.config.identity_heads:
            coefficient_logits = coefficient_logits.reshape(-1, 13, 4).gather(
                1, fav_index[:, None, None].expand(-1, 1, 4)
            )[:, 0]
        coefficients = self.config.max_position_coefficient * torch.tanh(coefficient_logits / 4.0)
        position_rel = position_rel + coefficients @ self.trajectory_basis.T
        if self.config.position_residual_head:
            position_channels = self.position_residual(future_hidden)
            if self.config.identity_heads:
                position_logits = position_channels.gather(
                    1, fav_index[:, None, None].expand(-1, 1, FUTURE_STEPS)
                )[:, 0]
            else:
                position_logits = position_channels[:, 0]
            raw_position_residual = self.config.max_position_residual * torch.tanh(
                position_logits / 5.0
            )
            position_rel = position_rel + raw_position_residual @ self.position_smoother.T
        if self.config.global_position_head:
            global_input = torch.cat(
                [
                    driver_context,
                    F.adaptive_avg_pool1d(future_hidden, self.config.global_pool_bins).flatten(1),
                ],
                dim=1,
            )
            global_logits = self.global_position_decoder(global_input)
            global_residual = self.config.max_position_residual * torch.tanh(global_logits / 5.0)
            position_rel = position_rel + global_residual @ self.position_smoother.T
        scored_speed = torch.cat(
            [position_rel[:, :1] / DT, torch.diff(position_rel, dim=1) / DT], dim=1
        )
        acceleration = torch.cat(
            [(scored_speed[:, :1] - v0[:, None]) / DT, torch.diff(scored_speed, dim=1) / DT], dim=1
        )
        return position_rel, scored_speed, acceleration


class FutureConditionedGRU(nn.Module):
    FUTURE_CHANNELS = [0, 1, 2, 3, 4, *range(10, 36)]

    def __init__(self, config: GRUConfig, speed_smoother: np.ndarray):
        super().__init__()
        self.config = config
        recurrent_dropout = config.dropout if config.layers > 1 else 0.0
        self.history_gru = nn.GRU(
            input_size=config.input_channels,
            hidden_size=config.history_hidden,
            num_layers=config.layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.future_gru = nn.GRU(
            input_size=len(self.FUTURE_CHANNELS),
            hidden_size=config.future_hidden,
            num_layers=config.layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=True,
        )
        decoder_width = config.history_hidden + 2 * config.future_hidden
        self.decoder = nn.Sequential(
            nn.Linear(decoder_width, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(decoder_width, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(96, 4),
        )
        self.register_buffer("speed_smoother", torch.from_numpy(speed_smoother), persistent=True)
        t = torch.linspace(1.0 / FUTURE_STEPS, 1.0, FUTURE_STEPS)
        trajectory_basis = torch.stack(
            [t, t.square(), torch.sin(torch.pi * t), 4.0 * t * (1.0 - t)],
            dim=1,
        )
        self.register_buffer("trajectory_basis", trajectory_basis, persistent=True)

    def forward(
        self,
        features: torch.Tensor,
        lv_future_speed: torch.Tensor,
        v0: torch.Tensor,
        gap0: torch.Tensor | None = None,
    ):
        _, history_state = self.history_gru(features[:, :HIST_STEPS])
        history_context = history_state[-1]
        future_input = features[:, HIST_STEPS:, self.FUTURE_CHANNELS]
        future_encoded, _ = self.future_gru(future_input)
        history_repeated = history_context[:, None, :].expand(-1, FUTURE_STEPS, -1)
        decoded = torch.cat([future_encoded, history_repeated], dim=2)
        residual = self.config.max_speed_residual * torch.tanh(
            self.decoder(decoded).squeeze(2) / 5.0
        )
        speed = (lv_future_speed + residual) @ self.speed_smoother.T
        speed = F.softplus(speed, beta=4.0)
        position_rel = DT * torch.cumsum(speed, dim=1)

        future_summary = torch.cat([future_encoded.mean(dim=1), history_context], dim=1)
        coefficients = self.config.max_position_coefficient * torch.tanh(
            self.trajectory_head(future_summary) / 4.0
        )
        position_rel = position_rel + coefficients @ self.trajectory_basis.T
        scored_speed = torch.cat(
            [position_rel[:, :1] / DT, torch.diff(position_rel, dim=1) / DT], dim=1
        )
        acceleration = torch.cat(
            [(scored_speed[:, :1] - v0[:, None]) / DT, torch.diff(scored_speed, dim=1) / DT], dim=1
        )
        return position_rel, scored_speed, acceleration


class FutureConditionedStateSpace(nn.Module):
    FUTURE_CHANNELS = [0, 1, 2, 3, 4, *range(10, 36)]

    def __init__(self, config: StateSpaceConfig, speed_smoother: np.ndarray):
        super().__init__()
        self.config = config
        recurrent_dropout = config.dropout if config.layers > 1 else 0.0
        history_hidden = (
            config.hidden_channels // 2 if config.bidirectional_history else config.hidden_channels
        )
        self.history_gru = nn.GRU(
            input_size=config.input_channels,
            hidden_size=history_hidden,
            num_layers=config.layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=config.bidirectional_history,
        )
        self.future_gru = nn.GRU(
            input_size=len(self.FUTURE_CHANNELS),
            hidden_size=config.hidden_channels // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        if config.identity_film:
            self.fav_film = nn.Embedding(13, config.hidden_channels * 2)
            self.lv_film = nn.Embedding(13, config.hidden_channels * 2)
            nn.init.zeros_(self.fav_film.weight)
            nn.init.zeros_(self.lv_film.weight)
        else:
            self.fav_film = None
            self.lv_film = None
        if config.identity_gate_bias:
            self.fav_gate_bias = nn.Embedding(13, 2)
            self.lv_gate_bias = nn.Embedding(13, 2)
            nn.init.zeros_(self.fav_gate_bias.weight)
            nn.init.zeros_(self.lv_gate_bias.weight)
        else:
            self.fav_gate_bias = None
            self.lv_gate_bias = None
        if config.identity_trajectory_bias:
            self.fav_trajectory_bias = nn.Embedding(13, 4)
            self.lv_trajectory_bias = nn.Embedding(13, 4)
            nn.init.zeros_(self.fav_trajectory_bias.weight)
            nn.init.zeros_(self.lv_trajectory_bias.weight)
        else:
            self.fav_trajectory_bias = None
            self.lv_trajectory_bias = None
        state_width = config.hidden_channels + 5
        self.state_cell = nn.GRUCell(state_width, config.hidden_channels)
        self.acceleration_head = nn.Sequential(
            nn.Linear(config.hidden_channels, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.position_head = nn.Sequential(
            nn.Linear(config.hidden_channels, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        if config.hybrid_direct:
            hybrid_width = config.hidden_channels * 3
            self.direct_speed_head = nn.Sequential(
                nn.Linear(hybrid_width, config.hidden_channels),
                nn.LayerNorm(config.hidden_channels),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_channels, 1),
            )
            self.speed_gate_head = nn.Sequential(
                nn.Linear(hybrid_width, config.hidden_channels // 2),
                nn.GELU(),
                nn.Linear(config.hidden_channels // 2, 1),
            )
            nn.init.constant_(self.speed_gate_head[-1].bias, 2.0)
        else:
            self.direct_speed_head = None
            self.speed_gate_head = None
        if config.gap_position_head:
            hybrid_width = config.hidden_channels * 3
            self.gap_correction_head = nn.Sequential(
                nn.Conv1d(hybrid_width, config.hidden_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(config.hidden_channels, 1, kernel_size=3, padding=1),
            )
            self.gap_position_gate = nn.Sequential(
                nn.Linear(hybrid_width, config.hidden_channels // 2),
                nn.GELU(),
                nn.Linear(config.hidden_channels // 2, 1),
            )
            nn.init.constant_(self.gap_position_gate[-1].bias, 2.0)
        else:
            self.gap_correction_head = None
            self.gap_position_gate = None
        if config.joint_future_layers > 0:
            joint_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_channels,
                nhead=config.joint_future_heads,
                dim_feedforward=config.hidden_channels * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.joint_future_input = nn.Linear(hybrid_width, config.hidden_channels)
            self.joint_future_position = nn.Parameter(
                torch.zeros(1, FUTURE_STEPS, config.hidden_channels)
            )
            self.joint_future_decoder = nn.TransformerEncoder(
                joint_layer,
                num_layers=config.joint_future_layers,
                norm=nn.LayerNorm(config.hidden_channels),
                enable_nested_tensor=False,
            )
            self.joint_future_output = nn.Sequential(
                nn.LayerNorm(config.hidden_channels),
                nn.Linear(config.hidden_channels, hybrid_width),
            )
            self.joint_position_head = nn.Sequential(
                nn.LayerNorm(config.hidden_channels),
                nn.Linear(config.hidden_channels, 1),
            )
            nn.init.normal_(self.joint_future_position, std=0.01)
            nn.init.zeros_(self.joint_future_output[-1].weight)
            nn.init.zeros_(self.joint_future_output[-1].bias)
            nn.init.zeros_(self.joint_position_head[-1].weight)
            nn.init.zeros_(self.joint_position_head[-1].bias)
        else:
            self.joint_future_input = None
            self.joint_future_position = None
            self.joint_future_decoder = None
            self.joint_future_output = None
            self.joint_position_head = None
        self.trajectory_head = nn.Sequential(
            nn.Linear(config.hidden_channels * 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 4),
        )
        self.register_buffer("speed_smoother", torch.from_numpy(speed_smoother), persistent=True)
        self.register_buffer(
            "position_smoother",
            torch.from_numpy(make_position_smoother(config.position_smooth_lambda)),
            persistent=True,
        )
        if config.final_position_smooth_lambda > 0.0:
            self.register_buffer(
                "final_position_smoother",
                torch.from_numpy(make_position_smoother(config.final_position_smooth_lambda)),
                persistent=True,
            )
        else:
            self.final_position_smoother = None
        t = torch.linspace(1.0 / FUTURE_STEPS, 1.0, FUTURE_STEPS)
        basis = torch.stack([t, t.square(), torch.sin(torch.pi * t), 4.0 * t * (1.0 - t)], dim=1)
        self.register_buffer("trajectory_basis", basis, persistent=True)

    def forward(
        self,
        features: torch.Tensor,
        lv_future_speed: torch.Tensor,
        v0: torch.Tensor,
        gap0: torch.Tensor | None = None,
    ):
        if gap0 is None:
            raise ValueError("State-space model requires the observed final-history gap")
        _, history_state = self.history_gru(features[:, :HIST_STEPS])
        if self.config.bidirectional_history:
            batch_size = features.shape[0]
            history_state = history_state.reshape(
                self.config.layers, 2, batch_size, self.config.hidden_channels // 2
            )
            history_context = torch.cat([history_state[-1, 0], history_state[-1, 1]], dim=1)
        else:
            history_context = history_state[-1]
        future_input = features[:, HIST_STEPS:, self.FUTURE_CHANNELS]
        future_encoded, _ = self.future_gru(future_input)
        fav_index = features[:, HIST_STEPS - 1, 23:36].argmax(dim=1)
        lv_index = features[:, HIST_STEPS:, 10:23].argmax(dim=2)
        if self.fav_film is not None and self.lv_film is not None:
            fav_gamma, fav_beta = self.fav_film(fav_index).chunk(2, dim=1)
            history_context = history_context * (1.0 + 0.1 * torch.tanh(fav_gamma)) + 0.1 * fav_beta
            lv_gamma, lv_beta = self.lv_film(lv_index).chunk(2, dim=2)
            future_encoded = future_encoded * (1.0 + 0.1 * torch.tanh(lv_gamma)) + 0.1 * lv_beta

        lv_acceleration = torch.cat(
            [torch.zeros_like(lv_future_speed[:, :1]), torch.diff(lv_future_speed, dim=1) / DT],
            dim=1,
        )
        hidden = history_context
        speed_state = v0
        gap_state = gap0
        hidden_rows = []
        raw_speed_rows = []
        gap_rows = []
        for step in range(FUTURE_STEPS):
            lv_speed_step = lv_future_speed[:, step]
            state_input = torch.cat(
                [
                    future_encoded[:, step],
                    lv_speed_step[:, None],
                    lv_acceleration[:, step : step + 1],
                    speed_state[:, None],
                    gap_state[:, None],
                    (lv_speed_step - speed_state)[:, None],
                ],
                dim=1,
            )
            hidden = self.state_cell(state_input, hidden)
            acceleration = self.config.max_acceleration * torch.tanh(
                self.acceleration_head(hidden)[:, 0] / 2.0
            )
            speed_state = torch.clamp_min(speed_state + DT * acceleration, 0.0)
            gap_state = gap_state + DT * (lv_speed_step - speed_state)
            hidden_rows.append(hidden)
            raw_speed_rows.append(speed_state)
            gap_rows.append(gap_state)

        decoded = torch.stack(hidden_rows, dim=1)
        raw_speed = torch.stack(raw_speed_rows, dim=1)
        joint_hidden = None
        if self.direct_speed_head is not None and self.speed_gate_head is not None:
            history_rows = history_context[:, None, :].expand(-1, FUTURE_STEPS, -1)
            hybrid_features = torch.cat([history_rows, future_encoded, decoded], dim=2)
            if self.joint_future_decoder is not None:
                joint_hidden = self.joint_future_decoder(
                    self.joint_future_input(hybrid_features) + self.joint_future_position
                )
                hybrid_features = hybrid_features + self.joint_future_output(joint_hidden)
            direct_residual = self.config.max_speed_residual * torch.tanh(
                self.direct_speed_head(hybrid_features)[:, :, 0] / 4.0
            )
            direct_speed = torch.clamp_min(lv_future_speed + direct_residual, 0.0)
            speed_gate_logits = self.speed_gate_head(hybrid_features)[:, :, 0]
            if self.fav_gate_bias is not None and self.lv_gate_bias is not None:
                identity_gate_bias = self.fav_gate_bias(fav_index)[:, None, :] + self.lv_gate_bias(
                    lv_index
                )
                speed_gate_logits = speed_gate_logits + identity_gate_bias[:, :, 0]
            else:
                identity_gate_bias = None
            state_weight = torch.sigmoid(speed_gate_logits)
            raw_speed = state_weight * raw_speed + (1.0 - state_weight) * direct_speed
        speed = F.softplus(raw_speed @ self.speed_smoother.T, beta=4.0)
        position_rel = DT * torch.cumsum(speed, dim=1)
        gap_position_rel = None
        if self.gap_correction_head is not None and self.gap_position_gate is not None:
            gap_rollout = torch.stack(gap_rows, dim=1)
            gap_correction = self.config.max_gap_residual * torch.tanh(
                self.gap_correction_head(hybrid_features.transpose(1, 2))[:, 0] / 5.0
            )
            predicted_gap = gap_rollout + gap_correction @ self.position_smoother.T
            lv_position_rel = DT * torch.cumsum(lv_future_speed, dim=1)
            gap_position_rel = lv_position_rel + gap0[:, None] - predicted_gap
            gap_gate_logits = self.gap_position_gate(hybrid_features)[:, :, 0]
            if identity_gate_bias is not None:
                gap_gate_logits = gap_gate_logits + identity_gate_bias[:, :, 1]
            speed_position_weight = torch.sigmoid(gap_gate_logits)
            position_rel = (
                speed_position_weight * position_rel
                + (1.0 - speed_position_weight) * gap_position_rel
            )

        local_position = self.config.max_position_residual * torch.tanh(
            self.position_head(decoded)[:, :, 0] / 5.0
        )
        if joint_hidden is not None:
            joint_position = (
                0.5
                * self.config.max_position_residual
                * torch.tanh(self.joint_position_head(joint_hidden)[:, :, 0] / 5.0)
            )
            local_position = local_position + joint_position
        position_rel = position_rel + local_position @ self.position_smoother.T
        summary = torch.cat(
            [history_context, future_encoded.mean(dim=1), decoded.mean(dim=1)],
            dim=1,
        )
        coefficients = self.config.max_position_coefficient * torch.tanh(
            self.trajectory_head(summary) / 4.0
        )
        if self.fav_trajectory_bias is not None and self.lv_trajectory_bias is not None:
            coefficients = coefficients + self.fav_trajectory_bias(fav_index)
            coefficients = coefficients + self.lv_trajectory_bias(lv_index).mean(dim=1)
        position_rel = position_rel + coefficients @ self.trajectory_basis.T
        if self.final_position_smoother is not None:
            position_rel = position_rel @ self.final_position_smoother.T
        scored_speed = torch.cat(
            [position_rel[:, :1] / DT, torch.diff(position_rel, dim=1) / DT], dim=1
        )
        acceleration = torch.cat(
            [(scored_speed[:, :1] - v0[:, None]) / DT, torch.diff(scored_speed, dim=1) / DT],
            dim=1,
        )
        if gap_position_rel is not None:
            return position_rel, scored_speed, acceleration, gap_position_rel
        return position_rel, scored_speed, acceleration


def normalize_features(
    features: np.ndarray, train_index: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[train_index].mean(axis=(0, 1), keepdims=True)
    std = features[train_index].std(axis=(0, 1), keepdims=True)
    std[std < 1e-5] = 1.0
    normalized = (features - mean) / std
    return (
        normalized.astype(np.float32),
        mean.reshape(-1).astype(np.float32),
        std.reshape(-1).astype(np.float32),
    )


def loss_function(
    pred,
    target,
    mode: str = "legacy",
    jerk_weight: float = 0.002,
    position_multiplier: float = 1.0,
    gap_aux_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_pos, pred_speed, pred_acc = pred[:3]
    target_pos, target_speed, target_acc = target
    gap_aux = None
    if len(pred) > 3 and gap_aux_weight > 0.0:
        gap_aux = torch.sqrt(torch.mean((pred[3] - target_pos).square(), dim=1) + 1e-8).mean()
    if mode == "metric":
        # Match the public Accuracy formula: each trajectory contributes its
        # normalized RMSE, with finite differences taken inside the future
        # horizon exactly as in the scorer.
        scored_pred_speed = torch.diff(pred_pos, dim=1) / DT
        scored_target_speed = torch.diff(target_pos, dim=1) / DT
        scored_pred_acc = torch.diff(scored_pred_speed, dim=1) / DT
        scored_target_acc = torch.diff(scored_target_speed, dim=1) / DT
        position = torch.sqrt(torch.mean((pred_pos - target_pos).square(), dim=1) + 1e-8).mean()
        speed = torch.sqrt(
            torch.mean((scored_pred_speed - scored_target_speed).square(), dim=1) + 1e-8
        ).mean()
        acceleration = torch.sqrt(
            torch.mean((scored_pred_acc - scored_target_acc).square(), dim=1) + 1e-8
        ).mean()
        jerk = torch.diff(scored_pred_acc, dim=1) / DT
        jerk_loss = torch.mean(jerk.square())
        total = (
            position_multiplier * 0.4 * position / 25.0
            + 0.4 * speed / 5.0
            + 0.2 * acceleration / 5.0
            + jerk_weight * jerk_loss
        )
        if gap_aux is not None:
            total = total + gap_aux_weight * gap_aux / 25.0
        return total, {
            "loss": float(total.detach()),
            "position": float(position.detach()),
            "speed": float(speed.detach()),
            "acceleration": float(acceleration.detach()),
            "jerk": float(torch.sqrt(jerk_loss.detach())),
            "gap_aux": 0.0 if gap_aux is None else float(gap_aux.detach()),
        }
    pos_loss = F.smooth_l1_loss((pred_pos - target_pos) / 5.0, torch.zeros_like(pred_pos), beta=1.0)
    speed_loss = F.smooth_l1_loss(pred_speed - target_speed, torch.zeros_like(pred_speed), beta=0.5)
    acc_loss = F.smooth_l1_loss(pred_acc - target_acc, torch.zeros_like(pred_acc), beta=0.25)
    jerk = torch.diff(pred_acc, dim=1) / DT
    jerk_loss = torch.mean(jerk.square())
    total = pos_loss + 0.70 * speed_loss + 0.08 * acc_loss + jerk_weight * jerk_loss
    if gap_aux is not None:
        total = total + gap_aux_weight * gap_aux / 25.0
    return total, {
        "loss": float(total.detach()),
        "position": float(pos_loss.detach()),
        "speed": float(speed_loss.detach()),
        "acceleration": float(acc_loss.detach()),
        "jerk": float(torch.sqrt(jerk_loss.detach())),
        "gap_aux": 0.0 if gap_aux is None else float(gap_aux.detach()),
    }


@torch.no_grad()
def predict(
    model: nn.Module, tensors: Tuple[torch.Tensor, ...], index: np.ndarray, batch_size: int
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    dataset = TensorDataset(*(tensor[index] for tensor in tensors))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []
    for batch in loader:
        features, lv_speed, x0, v0 = batch[:4]
        gap0 = batch[4] if len(batch) > 4 else None
        features = features.to(device, non_blocking=True)
        lv_speed = lv_speed.to(device, non_blocking=True)
        v0_device = v0.to(device, non_blocking=True)
        gap0_device = None if gap0 is None else gap0.to(device, non_blocking=True)
        output = model(features, lv_speed, v0_device, gap0_device)
        position_rel = output[0]
        position = (
            position_rel.cpu().numpy().astype(np.float64)
            + x0.cpu().numpy().astype(np.float64)[:, None]
        )
        rows.append(position)
    return np.vstack(rows)


def evaluate_positions(segments, ids: np.ndarray, positions: np.ndarray) -> Dict[str, float]:
    args = make_score_args(strict=True)
    rows = []
    for sid, prediction in zip(ids, positions):
        row = score_segment(segments[int(sid)], prediction, args)
        row["segment_id"] = int(sid)
        rows.append(row)
    return summarize(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EquiMotion using data/train.csv only")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts" / "train_tensors.npz")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "artifacts" / "equimotion_trained.pt"
    )
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "training_report.json")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--val-size", type=int, default=678)
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Train on every data/train.csv segment for a fixed number of epochs",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--architecture",
        choices=(
            "tcn",
            "gru",
            "state",
            "hybrid",
            "hybrid-comfort",
            "hybrid-bi",
            "hybrid-gap",
            "hybrid-gap-comfort",
            "hybrid-gap-joint",
            "hybrid-gap-film",
            "hybrid-gap-idbias",
            "hybrid-gap-idtraj",
            "mixer-gap",
        ),
        default="hybrid-gap-idtraj",
    )
    parser.add_argument("--target-smooth-lambda", type=float, default=TARGET_SMOOTH_LAMBDA)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--reuse-init-normalization", action="store_true")
    parser.add_argument(
        "--freeze-backbone-for-film",
        action="store_true",
        help="Train only the FAV/LV FiLM embeddings",
    )
    parser.add_argument(
        "--trainable-prefixes",
        type=str,
        default="",
        help="Comma-separated parameter prefixes to train while freezing all other parameters",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--speed-smooth-lambda", type=float, default=300.0)
    parser.add_argument("--model-seed", type=int, default=SEED)
    parser.add_argument("--loss-mode", choices=("legacy", "metric"), default="metric")
    parser.add_argument("--jerk-loss-weight", type=float, default=0.001)
    parser.add_argument("--min-comfort", type=float, default=0.0)
    parser.add_argument("--gap-aux-weight", type=float, default=0.2)
    parser.add_argument("--position-loss-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--fav-id", type=int, help="Train and validate a specialist for one visible FAV identity"
    )
    parser.add_argument("--position-residual-head", action="store_true")
    parser.add_argument("--position-smooth-lambda", type=float, default=10000.0)
    parser.add_argument("--final-position-smooth-lambda", type=float, default=0.0)
    parser.add_argument("--max-position-residual", type=float, default=12.0)
    parser.add_argument("--upweight-fav-ids", type=str, default="")
    parser.add_argument("--upweight-factor", type=float, default=1.0)
    parser.add_argument("--identity-heads", action="store_true")
    parser.add_argument("--global-position-head", action="store_true")
    parser.add_argument("--global-pool-bins", type=int, default=20)
    parser.add_argument("--global-decoder-width", type=int, default=256)
    parser.add_argument("--delay-mixture", action="store_true")
    parser.add_argument("--max-delay-steps", type=int, default=30)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--normalization-all-train", action="store_true")
    parser.add_argument("--attention-layers", type=int, default=0)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--joint-future-layers", type=int, default=2)
    parser.add_argument("--joint-future-heads", type=int, default=8)
    parser.add_argument(
        "--augmentation-cuts",
        type=str,
        default="",
        help="Comma-separated prediction cut points; grouped by Segment_ID",
    )
    parser.add_argument(
        "--augmented-cut-weight",
        type=float,
        default=1.0,
        help="Sampling weight for non-100 augmented cuts",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    if args.min_learning_rate < 0.0:
        raise ValueError("--min-learning-rate must be non-negative")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("--min-learning-rate cannot exceed --learning-rate")

    train_csv = require_train_csv(args.train_csv)
    set_seed(args.model_seed)
    torch.set_num_threads(max(1, min(12, torch.get_num_threads())))
    augmentation_cuts = tuple(
        int(value) for value in args.augmentation_cuts.split(",") if value.strip()
    )
    if augmentation_cuts:
        if 100 not in augmentation_cuts:
            raise ValueError("--augmentation-cuts must include the official cut point 100")
        if any(cut < 1 or cut > 100 for cut in augmentation_cuts):
            raise ValueError("Augmentation cut points must be between 1 and 100")
        augmentation_cuts = tuple(sorted(set(augmentation_cuts)))
    if args.rebuild_cache or not args.cache.exists():
        print("building strict train-only tensor cache", flush=True)
        if augmentation_cuts:
            build_augmented_cache(
                train_csv,
                args.cache,
                args.target_smooth_lambda,
                augmentation_cuts,
            )
        else:
            build_cache(train_csv, args.cache, args.target_smooth_lambda)
    if args.prepare_only:
        return

    cache = np.load(args.cache)
    cached_target_lambda = (
        float(cache["target_smooth_lambda"][0])
        if "target_smooth_lambda" in cache.files
        else TARGET_SMOOTH_LAMBDA
    )
    if not np.isclose(cached_target_lambda, args.target_smooth_lambda):
        raise ValueError(
            f"Cache target lambda {cached_target_lambda:g} != requested "
            f"{args.target_smooth_lambda:g}; use --rebuild-cache"
        )
    cached_cuts = (
        tuple(int(value) for value in cache["augmentation_cuts"])
        if "augmentation_cuts" in cache.files
        else ()
    )
    if cached_cuts != augmentation_cuts:
        raise ValueError(
            f"Cache augmentation cuts {cached_cuts} != requested {augmentation_cuts}; "
            "use --rebuild-cache"
        )
    ids = cache["ids"].astype(np.int64)
    rng = np.random.default_rng(args.split_seed)
    unique_ids = np.unique(ids)
    permutation = rng.permutation(len(unique_ids))
    validation_ids = unique_ids[permutation[: args.val_size]]
    if args.full_train:
        validation_ids = np.asarray([], dtype=unique_ids.dtype)
        val_index = np.asarray([], dtype=np.int64)
        train_index = np.arange(len(ids), dtype=np.int64)
    elif "cut_points" in cache.files:
        cut_points = cache["cut_points"].astype(np.int64)
        validation_mask = np.isin(ids, validation_ids)
        val_index = np.flatnonzero(validation_mask & (cut_points == 100))
        train_index = np.flatnonzero(~validation_mask)
    else:
        val_index = np.sort(permutation[: args.val_size])
        train_index = np.sort(permutation[args.val_size :])
    all_train_index = train_index.copy()
    fav_identity = np.argmax(cache["features"][:, HIST_STEPS - 1, 23:36], axis=1) - 1
    if args.fav_id is not None:
        train_index = train_index[fav_identity[train_index] == args.fav_id]
        val_index = val_index[fav_identity[val_index] == args.fav_id]
        if len(train_index) == 0 or (not args.full_train and len(val_index) == 0):
            raise ValueError(f"No train/validation samples found for FAV identity {args.fav_id}")
        print(
            f"specialist FAV={args.fav_id} train_segments={len(train_index)} "
            f"validation_segments={len(val_index)}",
            flush=True,
        )
    normalization_index = all_train_index if args.normalization_all_train else train_index
    initial_for_normalization = None
    if args.reuse_init_normalization:
        if args.init_checkpoint is None:
            raise ValueError("--reuse-init-normalization requires --init-checkpoint")
        initial_for_normalization = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        feature_mean = np.asarray(initial_for_normalization["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(initial_for_normalization["feature_std"], dtype=np.float32)
        features = ((cache["features"].astype(np.float32) - feature_mean) / feature_std).astype(
            np.float32
        )
    else:
        features, feature_mean, feature_std = normalize_features(
            cache["features"].astype(np.float32), normalization_index
        )

    tensors = (
        torch.from_numpy(features),
        torch.from_numpy(cache["lv_future_speed"].astype(np.float32)),
        torch.from_numpy(cache["x0"].astype(np.float64)),
        torch.from_numpy(cache["v0"].astype(np.float32)),
        torch.from_numpy(cache["features"][:, HIST_STEPS - 1, 7].astype(np.float32)),
    )
    targets = (
        torch.from_numpy(cache["target_pos_rel"].astype(np.float32)),
        torch.from_numpy(cache["target_speed"].astype(np.float32)),
        torch.from_numpy(cache["target_acc"].astype(np.float32)),
    )
    train_dataset = TensorDataset(*(tensor[train_index] for tensor in (*tensors, *targets)))
    generator = torch.Generator().manual_seed(args.model_seed)
    upweight_ids = [int(value) for value in args.upweight_fav_ids.split(",") if value.strip()]
    sample_weights = np.ones(len(train_index), dtype=np.float64)
    weighted_sampling = False
    sampler_size = len(train_index)
    if "cut_points" in cache.files and not np.isclose(args.augmented_cut_weight, 1.0):
        if args.augmented_cut_weight <= 0.0:
            raise ValueError("--augmented-cut-weight must be positive")
        augmented_mask = cache["cut_points"][train_index] != 100
        sample_weights[augmented_mask] = args.augmented_cut_weight
        sampler_size = max(1, int(round(float(sample_weights.sum()))))
        weighted_sampling = True
        print(
            f"augmented cut weight={args.augmented_cut_weight:g} samples_per_epoch={sampler_size}",
            flush=True,
        )
    if upweight_ids and args.upweight_factor > 1.0:
        sample_weights[np.isin(fav_identity[train_index], upweight_ids)] *= args.upweight_factor
        weighted_sampling = True
    if weighted_sampling:
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights),
            num_samples=sampler_size,
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        if upweight_ids and args.upweight_factor > 1.0:
            print(
                f"upweighted FAV identities={upweight_ids} factor={args.upweight_factor:g}",
                flush=True,
            )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
        )

    if args.architecture == "gru":
        config = GRUConfig(dropout=args.dropout, speed_smooth_lambda=args.speed_smooth_lambda)
        model = FutureConditionedGRU(config, make_speed_smoother(config.speed_smooth_lambda))
    elif args.architecture == "mixer-gap":
        config = GapMixerConfig(
            hidden_channels=args.hidden_channels,
            blocks=args.blocks,
            dropout=args.dropout,
            speed_smooth_lambda=args.speed_smooth_lambda,
            position_smooth_lambda=args.position_smooth_lambda,
            max_position_residual=args.max_position_residual,
        )
        model = FutureConditionedGapMixer(config, make_speed_smoother(config.speed_smooth_lambda))
    elif args.architecture in {
        "state",
        "hybrid",
        "hybrid-comfort",
        "hybrid-bi",
        "hybrid-gap",
        "hybrid-gap-comfort",
        "hybrid-gap-joint",
        "hybrid-gap-film",
        "hybrid-gap-idbias",
        "hybrid-gap-idtraj",
    }:
        config = StateSpaceConfig(
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            speed_smooth_lambda=args.speed_smooth_lambda,
            position_smooth_lambda=args.position_smooth_lambda,
            max_position_residual=args.max_position_residual,
            hybrid_direct=args.architecture
            in {
                "hybrid",
                "hybrid-comfort",
                "hybrid-bi",
                "hybrid-gap",
                "hybrid-gap-comfort",
                "hybrid-gap-joint",
                "hybrid-gap-film",
                "hybrid-gap-idbias",
                "hybrid-gap-idtraj",
            },
            bidirectional_history=args.architecture == "hybrid-bi",
            gap_position_head=args.architecture
            in {
                "hybrid-gap",
                "hybrid-gap-comfort",
                "hybrid-gap-joint",
                "hybrid-gap-film",
                "hybrid-gap-idbias",
                "hybrid-gap-idtraj",
            },
            identity_film=args.architecture == "hybrid-gap-film",
            identity_gate_bias=args.architecture == "hybrid-gap-idbias",
            identity_trajectory_bias=args.architecture == "hybrid-gap-idtraj",
            final_position_smooth_lambda=args.final_position_smooth_lambda,
            joint_future_layers=args.joint_future_layers
            if args.architecture == "hybrid-gap-joint"
            else 0,
            joint_future_heads=args.joint_future_heads,
        )
        model = FutureConditionedStateSpace(config, make_speed_smoother(config.speed_smooth_lambda))
    else:
        config = ModelConfig(
            hidden_channels=args.hidden_channels,
            blocks=args.blocks,
            dropout=args.dropout,
            speed_smooth_lambda=args.speed_smooth_lambda,
            position_residual_head=args.position_residual_head,
            position_smooth_lambda=args.position_smooth_lambda,
            max_position_residual=args.max_position_residual,
            identity_heads=args.identity_heads,
            global_position_head=args.global_position_head,
            global_pool_bins=args.global_pool_bins,
            global_decoder_width=args.global_decoder_width,
            delay_mixture=args.delay_mixture,
            max_delay_steps=args.max_delay_steps,
            attention_layers=args.attention_layers,
            attention_heads=args.attention_heads,
        )
        model = FutureConditionedTCN(config, make_speed_smoother(config.speed_smooth_lambda))
    if args.init_checkpoint is not None:
        initial = (
            initial_for_normalization
            if initial_for_normalization is not None
            else torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        )
        initial_architecture = initial.get("architecture", args.architecture)
        state_to_hybrid = initial_architecture == "state" and args.architecture == "hybrid"
        hybrid_to_gap = initial_architecture == "hybrid" and args.architecture == "hybrid-gap"
        hybrid_to_comfort = (
            initial_architecture == "hybrid" and args.architecture == "hybrid-comfort"
        )
        gap_to_comfort = (
            initial_architecture == "hybrid-gap" and args.architecture == "hybrid-gap-comfort"
        )
        gap_to_joint = (
            initial_architecture == "hybrid-gap" and args.architecture == "hybrid-gap-joint"
        )
        gap_to_film = (
            initial_architecture == "hybrid-gap" and args.architecture == "hybrid-gap-film"
        )
        gap_to_idbias = (
            initial_architecture == "hybrid-gap" and args.architecture == "hybrid-gap-idbias"
        )
        gap_to_idtraj = (
            initial_architecture == "hybrid-gap" and args.architecture == "hybrid-gap-idtraj"
        )
        transfer_initialization = (
            state_to_hybrid
            or hybrid_to_gap
            or hybrid_to_comfort
            or gap_to_comfort
            or gap_to_joint
            or gap_to_film
            or gap_to_idbias
            or gap_to_idtraj
        )
        if initial_architecture != args.architecture and not transfer_initialization:
            raise ValueError("Initial checkpoint architecture does not match --architecture")
        incompatible = model.load_state_dict(
            initial["state_dict"], strict=not transfer_initialization
        )
        if transfer_initialization:
            unexpected = list(incompatible.unexpected_keys)
            missing = list(incompatible.missing_keys)
            if state_to_hybrid:
                allowed_prefixes = ("direct_speed_head.", "speed_gate_head.")
            elif hybrid_to_gap:
                allowed_prefixes = ("gap_correction_head.", "gap_position_gate.")
            elif gap_to_joint:
                allowed_prefixes = (
                    "joint_future_input.",
                    "joint_future_position",
                    "joint_future_decoder.",
                    "joint_future_output.",
                    "joint_position_head.",
                )
            elif gap_to_film:
                allowed_prefixes = ("fav_film.", "lv_film.")
            elif gap_to_idbias:
                allowed_prefixes = ("fav_gate_bias.", "lv_gate_bias.")
            elif gap_to_idtraj:
                allowed_prefixes = (
                    "fav_trajectory_bias.",
                    "lv_trajectory_bias.",
                )
            else:
                allowed_prefixes = ("final_position_smoother",)
            if unexpected or any(not key.startswith(allowed_prefixes) for key in missing):
                raise ValueError(
                    "State-to-hybrid initialization mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        print(f"initialized model from {args.init_checkpoint}", flush=True)
    trainable_prefixes = tuple(
        prefix.strip() for prefix in args.trainable_prefixes.split(",") if prefix.strip()
    )
    if args.freeze_backbone_for_film and trainable_prefixes:
        raise ValueError(
            "--freeze-backbone-for-film and --trainable-prefixes are mutually exclusive"
        )
    if args.freeze_backbone_for_film:
        if not isinstance(config, StateSpaceConfig) or not config.identity_film:
            raise ValueError("--freeze-backbone-for-film requires an identity-FiLM architecture")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("fav_film.") or name.startswith("lv_film."))
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print(
            f"frozen backbone; trainable FiLM parameters={trainable_parameters}",
            flush=True,
        )
    elif trainable_prefixes:
        matched_prefixes = set()
        for name, parameter in model.named_parameters():
            matches = [prefix for prefix in trainable_prefixes if name.startswith(prefix)]
            parameter.requires_grad_(bool(matches))
            matched_prefixes.update(matches)
        missing_prefixes = sorted(set(trainable_prefixes) - matched_prefixes)
        if missing_prefixes:
            raise ValueError(f"No model parameters matched trainable prefixes: {missing_prefixes}")
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if trainable_parameters == 0:
            raise ValueError("--trainable-prefixes selected zero parameters")
        print(
            "partially frozen model; "
            f"trainable parameters={trainable_parameters} prefixes={trainable_prefixes}",
            flush=True,
        )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment")
    model.to(device)
    ema_model = None
    if args.ema_decay > 0.0:
        if not 0.0 < args.ema_decay < 1.0:
            raise ValueError("--ema-decay must be between 0 and 1")
        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
    print(
        f"training device={device} model_parameters={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_learning_rate,
    )
    score_segments = read_segments(train_csv)

    best = {"accuracy_mean": -math.inf}
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums: Dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            (
                features_b,
                lv_speed_b,
                x0_b,
                v0_b,
                gap0_b,
                target_pos_b,
                target_speed_b,
                target_acc_b,
            ) = batch
            features_b = features_b.to(device, non_blocking=True)
            lv_speed_b = lv_speed_b.to(device, non_blocking=True)
            v0_b = v0_b.to(device, non_blocking=True)
            gap0_b = gap0_b.to(device, non_blocking=True)
            target_pos_b = target_pos_b.to(device, non_blocking=True)
            target_speed_b = target_speed_b.to(device, non_blocking=True)
            target_acc_b = target_acc_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(features_b, lv_speed_b, v0_b, gap0_b)
            loss, parts = loss_function(
                output,
                (target_pos_b, target_speed_b, target_acc_b),
                mode=args.loss_mode,
                jerk_weight=args.jerk_loss_weight,
                position_multiplier=args.position_loss_multiplier,
                gap_aux_weight=args.gap_aux_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            if ema_model is not None:
                with torch.no_grad():
                    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                        ema_parameter.mul_(args.ema_decay).add_(
                            parameter, alpha=1.0 - args.ema_decay
                        )
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
            batches += 1
        scheduler.step()

        if args.full_train:
            selected_model = ema_model if ema_model is not None else model
            selected_name = "ema" if ema_model is not None else "raw"
            epoch_row = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": {key: value / batches for key, value in sums.items()},
            }
            history.append(epoch_row)
            best = {
                "epoch": epoch,
                "source": selected_name,
                "training_loss": float(epoch_row["train"]["loss"]),
            }
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": selected_model.state_dict(),
                    "architecture": args.architecture,
                    "model_config": asdict(config),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "train_ids": ids[train_index],
                    "val_ids": ids[val_index],
                    "best": best,
                    "data_boundary": "data/train.csv only",
                    "target_smooth_lambda": args.target_smooth_lambda,
                    "full_train": True,
                },
                args.checkpoint,
            )
            print(
                f"epoch={epoch:02d} loss={epoch_row['train']['loss']:.5f} "
                f"full_train_segments={len(np.unique(ids[train_index]))}",
                flush=True,
            )
            continue

        val_positions = predict(model, tensors, val_index, args.batch_size)
        metrics = evaluate_positions(score_segments, ids[val_index], val_positions)
        candidates = [("raw", model, metrics)]
        ema_metrics = None
        if ema_model is not None:
            ema_positions = predict(ema_model, tensors, val_index, args.batch_size)
            ema_metrics = evaluate_positions(score_segments, ids[val_index], ema_positions)
            candidates.append(("ema", ema_model, ema_metrics))
        eligible = [item for item in candidates if item[2]["comfort_mean"] >= args.min_comfort]
        selected = max(eligible, key=lambda item: item[2]["accuracy_mean"]) if eligible else None
        epoch_row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": {key: value / batches for key, value in sums.items()},
            "validation": {
                key: float(value) for key, value in metrics.items() if key.endswith("_mean")
            },
        }
        if ema_metrics is not None:
            epoch_row["ema_validation"] = {
                key: float(value) for key, value in ema_metrics.items() if key.endswith("_mean")
            }
        history.append(epoch_row)
        ema_text = "" if ema_metrics is None else f" ema_acc={ema_metrics['accuracy_mean']:.6f}"
        print(
            f"epoch={epoch:02d} loss={epoch_row['train']['loss']:.5f} "
            f"acc={metrics['accuracy_mean']:.6f} safety={metrics['safety_mean']:.6f} "
            f"comfort={metrics['comfort_mean']:.6f} final={metrics['final_mean']:.6f}{ema_text}",
            flush=True,
        )
        if selected is not None and selected[2]["accuracy_mean"] > best["accuracy_mean"]:
            selected_name, selected_model, selected_metrics = selected
            best = {
                key: float(value)
                for key, value in selected_metrics.items()
                if key.endswith("_mean")
            }
            best["epoch"] = epoch
            best["source"] = selected_name
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": selected_model.state_dict(),
                    "architecture": args.architecture,
                    "model_config": asdict(config),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "train_ids": ids[train_index],
                    "val_ids": ids[val_index],
                    "best": best,
                    "data_boundary": "data/train.csv only",
                    "target_smooth_lambda": args.target_smooth_lambda,
                },
                args.checkpoint,
            )

    if not args.full_train and best["accuracy_mean"] == -math.inf:
        raise RuntimeError(f"No validation checkpoint reached --min-comfort {args.min_comfort:.6f}")

    report = {
        "run_name": (
            f"equimotion_{args.architecture}_{'full_train' if args.full_train else 'validation'}"
        ),
        "data_boundary": {"training_and_targets": str(train_csv), "test_labels_used": False},
        "model_config": asdict(config),
        "training": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "split": {
            "seed": args.split_seed,
            "train_segments": int(len(np.unique(ids[train_index]))),
            "train_samples": int(len(train_index)),
            "validation_segments": int(len(np.unique(ids[val_index]))),
            "validation_samples": int(len(val_index)),
        },
        "best": best,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps({"best": best, "elapsed_seconds": report["elapsed_seconds"]}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
