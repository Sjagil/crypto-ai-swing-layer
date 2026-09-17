"""
Production/research-grade causal TCN + GRU challenger for
Sjagil/crypto-ai-swing-layer, using Sjagil/crypto as the market-data runtime.

Drop this file at:
    src/crypto_ai_swing/models/tcn_gru.py

Repository assumptions
----------------------
Main repo:
    https://github.com/Sjagil/crypto-ai-swing-layer

Complementary runtime/data repo:
    https://github.com/Sjagil/crypto

This module deliberately does NOT pretend a sequence model is a normal
row-wise sklearn estimator. The existing AgentDataset contains multiple markets
interleaved by feature_time, so a correct temporal model must preserve market
boundaries and chronology.

Main guarantees
---------------
1. Strictly causal convolutions.
2. Unidirectional GRU.
3. Sequences are built independently per market.
4. No sequence can cross from BTC into ETH/etc.
5. Scaler/imputer are fitted on training data only.
6. External validation/test predictions may use ONLY earlier context.
7. Early stopping uses an inner chronological, purged tail of TRAIN only.
8. No random train/test split.
9. Probability calibration is fitted only on a dedicated calibration segment.
10. Threshold selection happens after calibration on a separate selection segment.
11. Test remains untouched until the final evaluation.
12. Artifact is CPU-portable and joblib-safe.
13. Apple Silicon MPS, CUDA, and CPU are supported.
14. First lookback-1 rows per market are not fake-padded for training.
15. The module can fetch OHLCV through the repo's existing CryptoLibraryBridge.

The TCN extracts local and multi-scale temporal motifs. The GRU receives the
causal TCN representation and models longer state dependence.

This module is intended as a SHADOW/CHALLENGER model until it has earned
promotion through the repository's existing validation and governance gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import copy
import json
import math
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from crypto_ai_swing.agents.calibration import (
    ProbabilityCalibrator,
    fit_probability_calibrator,
    purged_calibration_selection_split,
)
from crypto_ai_swing.agents.dataset import (
    AgentDataset,
    DEFAULT_FEATURES,
    build_agent_dataset,
    purged_chronological_split,
)
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.data.features import build_features


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TCNGRUConfig:
    # Sequence geometry.
    lookback: int = 64
    tcn_channels: tuple[int, ...] = (64, 64, 96)
    kernel_size: int = 3
    gru_hidden: int = 96
    gru_layers: int = 1

    # Regularization.
    dropout: float = 0.15
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    class_balance: bool = True

    # Optimizer/training.
    learning_rate: float = 3e-4
    batch_size: int = 256
    epochs: int = 60
    patience: int = 8
    min_delta: float = 1e-4
    grad_clip: float = 1.0

    # Inner TRAIN-only validation.
    inner_validation_fraction: float = 0.15
    inner_purge_bars: int = 8

    # Probability/threshold selection.
    calibration_method: str = "sigmoid"
    threshold_min: float = 0.50
    threshold_max: float = 0.75
    threshold_step: float = 0.025

    # Economic selection constraints.
    minimum_selected: int = 40
    minimum_markets: int = 3
    minimum_positive_market_fraction: float = 0.50

    # Reproducibility/runtime.
    seed: int = 17
    device: str = "auto"
    num_workers: int = 0
    deterministic: bool = True

    def validate(self) -> None:
        if self.lookback < 8:
            raise ValueError("lookback must be >= 8")
        if not self.tcn_channels or any(int(v) <= 0 for v in self.tcn_channels):
            raise ValueError("tcn_channels must contain positive integers")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")
        if self.gru_hidden <= 0 or self.gru_layers <= 0:
            raise ValueError("GRU dimensions must be positive")
        if not 0.0 <= self.dropout < 0.8:
            raise ValueError("dropout must be in [0, 0.8)")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if not 0.05 <= self.inner_validation_fraction <= 0.35:
            raise ValueError("inner_validation_fraction must be in [0.05, 0.35]")
        if self.threshold_step <= 0:
            raise ValueError("threshold_step must be positive")
        if self.threshold_min >= self.threshold_max:
            raise ValueError("threshold_min must be < threshold_max")


@dataclass(frozen=True)
class TrainingHistoryRow:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_brier: float
    validation_auc: float | None


@dataclass(frozen=True)
class SequencePrediction:
    probability: pd.Series
    valid_rows: int
    total_rows: int


@dataclass(frozen=True)
class TCNGRUTrainingResult:
    status: str
    artifact_path: Path
    manifest_path: Path
    metrics: dict[str, Any]
    dataset_id: str
    row_count: int
    markets: tuple[str, ...]
    timeframe: str
    horizon_bars: int


# ---------------------------------------------------------------------------
# Reproducibility/device
# ---------------------------------------------------------------------------


def _set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _resolve_device(requested: str) -> torch.device:
    value = str(requested).strip().lower()
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """Conv1d with left-only receptive field."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=self.left_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.left_padding > 0:
            y = y[..., :-self.left_padding]
        return y


class TemporalResidualBlock(nn.Module):
    """
    Causal residual TCN block.

    LayerNorm is applied over channels at each time point instead of BatchNorm,
    avoiding running batch statistics across changing market regimes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation,
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    @staticmethod
    def _layer_norm_time(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        # [B,C,T] -> [B,T,C] -> LN(C) -> [B,C,T]
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        y = self.conv1(x)
        y = self._layer_norm_time(y, self.norm1)
        y = self.activation(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self._layer_norm_time(y, self.norm2)
        y = self.activation(y)
        y = self.dropout(y)

        return self.activation(y + residual)


class TCNGRUNetwork(nn.Module):
    """
    Causal TCN -> unidirectional GRU -> binary logit.

    The TCN learns local/multi-scale filters. Dilations grow as 1,2,4,...
    The GRU consumes the filtered representation and maintains a dynamic
    hidden state across the lookback sequence.
    """

    def __init__(
        self,
        input_dim: int,
        config: TCNGRUConfig,
    ) -> None:
        super().__init__()
        config.validate()

        blocks: list[nn.Module] = []
        in_channels = int(input_dim)

        for level, out_channels in enumerate(config.tcn_channels):
            blocks.append(
                TemporalResidualBlock(
                    in_channels,
                    int(out_channels),
                    kernel_size=config.kernel_size,
                    dilation=2**level,
                    dropout=config.dropout,
                )
            )
            in_channels = int(out_channels)

        self.tcn = nn.Sequential(*blocks)

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=(
                config.dropout
                if config.gru_layers > 1
                else 0.0
            ),
        )

        hidden2 = max(16, config.gru_hidden // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(config.gru_hidden),
            nn.Dropout(config.dropout),
            nn.Linear(config.gru_hidden, hidden2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,F]
        z = x.transpose(1, 2)  # [B,F,T]
        z = self.tcn(z)
        z = z.transpose(1, 2)  # [B,T,C]
        z, _ = self.gru(z)
        return self.head(z[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# Preprocessing + safe sequence construction
# ---------------------------------------------------------------------------


class TrainOnlyStandardizer:
    """
    Median imputation + standardization fitted ONLY from training observations.
    """

    def __init__(self) -> None:
        self.median_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "TrainOnlyStandardizer":
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or not len(x):
            raise ValueError("standardizer requires a non-empty 2D matrix")

        median = np.nanmedian(x, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)

        clean = x.copy()
        bad = ~np.isfinite(clean)
        if bad.any():
            rows, cols = np.where(bad)
            clean[rows, cols] = median[cols]

        mean = clean.mean(axis=0)
        scale = clean.std(axis=0, ddof=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)

        self.median_ = median.astype(np.float64)
        self.mean_ = mean.astype(np.float64)
        self.scale_ = scale.astype(np.float64)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("standardizer is not fitted")

        x = np.asarray(values, dtype=np.float64).copy()
        bad = ~np.isfinite(x)
        if bad.any():
            rows, cols = np.where(bad)
            x[rows, cols] = self.median_[cols]

        result = (x - self.mean_) / self.scale_
        return result.astype(np.float32)

    def state(self) -> dict[str, list[float]]:
        if self.median_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("standardizer is not fitted")
        return {
            "median": self.median_.tolist(),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "TrainOnlyStandardizer":
        obj = cls()
        obj.median_ = np.asarray(state["median"], dtype=np.float64)
        obj.mean_ = np.asarray(state["mean"], dtype=np.float64)
        obj.scale_ = np.asarray(state["scale"], dtype=np.float64)
        return obj


def _feature_times(frame: pd.DataFrame) -> pd.Series:
    if "feature_time" in frame.columns:
        return pd.to_datetime(frame["feature_time"], utc=True, errors="raise")
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(
            pd.to_datetime(frame.index, utc=True),
            index=frame.index,
        )
    raise ValueError("frame needs feature_time or a DatetimeIndex")


def _validate_agent_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    require_target: bool,
    target_column: str = "target_alpha",
) -> None:
    required = {"market", "feature_time", *feature_columns}
    if require_target:
        required.add(target_column)

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if frame.empty:
        raise ValueError("frame is empty")

    times = _feature_times(frame)
    if times.isna().any():
        raise ValueError("feature_time contains invalid values")


@dataclass(frozen=True)
class SequenceBatch:
    x: np.ndarray
    row_positions: np.ndarray
    markets: np.ndarray
    times_ns: np.ndarray

    @property
    def rows(self) -> int:
        return int(len(self.row_positions))


def _sequence_batch_from_combined_frame(
    frame: pd.DataFrame,
    scaled_values: np.ndarray,
    *,
    lookback: int,
    target_position_mask: np.ndarray | None = None,
) -> SequenceBatch:
    """
    Build complete lookback windows independently per market.

    target_position_mask:
        Optional boolean mask over rows of ``frame``. Context rows may be present
        in ``frame`` to provide history, while only rows with mask=True are
        returned as predictions/training observations.
    """
    if len(frame) != len(scaled_values):
        raise ValueError("frame and scaled_values are not aligned")

    markets = frame["market"].astype(str).str.upper().to_numpy(dtype=object)
    times = pd.to_datetime(
        frame["feature_time"],
        utc=True,
        errors="raise",
    ).astype("int64").to_numpy()

    if target_position_mask is None:
        target_position_mask = np.ones(len(frame), dtype=bool)
    else:
        target_position_mask = np.asarray(target_position_mask, dtype=bool)
        if len(target_position_mask) != len(frame):
            raise ValueError("target_position_mask length mismatch")

    x_rows: list[np.ndarray] = []
    positions: list[int] = []
    out_markets: list[str] = []
    out_times: list[int] = []

    # Stable market-specific ordering. Nothing crosses a market boundary.
    for market in pd.unique(markets):
        market_positions = np.flatnonzero(markets == market)
        order = np.argsort(times[market_positions], kind="stable")
        ordered_positions = market_positions[order]

        if len(ordered_positions) < int(lookback):
            continue

        market_values = scaled_values[ordered_positions]

        for local_end in range(int(lookback) - 1, len(ordered_positions)):
            global_position = int(ordered_positions[local_end])
            if not target_position_mask[global_position]:
                continue

            local_start = local_end - int(lookback) + 1
            window = market_values[local_start : local_end + 1]

            # defensive integrity check
            if window.shape[0] != int(lookback):
                raise RuntimeError("sequence length invariant violated")

            x_rows.append(window.astype(np.float32, copy=False))
            positions.append(global_position)
            out_markets.append(str(market))
            out_times.append(int(times[global_position]))

    if not x_rows:
        feature_count = int(scaled_values.shape[1])
        return SequenceBatch(
            x=np.empty((0, int(lookback), feature_count), dtype=np.float32),
            row_positions=np.empty(0, dtype=np.int64),
            markets=np.empty(0, dtype=object),
            times_ns=np.empty(0, dtype=np.int64),
        )

    return SequenceBatch(
        x=np.stack(x_rows).astype(np.float32, copy=False),
        row_positions=np.asarray(positions, dtype=np.int64),
        markets=np.asarray(out_markets, dtype=object),
        times_ns=np.asarray(out_times, dtype=np.int64),
    )


def _concat_context_target(
    context: pd.DataFrame | None,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Concatenate earlier context with target while marking rows that belong to target.

    Duplicate (market, feature_time) observations are resolved in favor of target.
    """
    if context is None or context.empty:
        combined = target.copy()
        mask = np.ones(len(combined), dtype=bool)
        return combined, mask

    c = context.copy()
    t = target.copy()
    c["__is_target__"] = False
    t["__is_target__"] = True

    combined = pd.concat([c, t], axis=0, ignore_index=False)
    combined["__market_norm__"] = combined["market"].astype(str).str.upper()
    combined["__time_norm__"] = pd.to_datetime(
        combined["feature_time"],
        utc=True,
        errors="raise",
    )

    # Sort context first and target second for duplicate keys, then keep target.
    combined = combined.sort_values(
        ["__time_norm__", "__market_norm__", "__is_target__"],
        kind="stable",
    )
    combined = combined.drop_duplicates(
        subset=["__market_norm__", "__time_norm__"],
        keep="last",
    )
    combined = combined.sort_values(
        ["__time_norm__", "__market_norm__"],
        kind="stable",
    )

    mask = combined["__is_target__"].to_numpy(dtype=bool)
    combined = combined.drop(
        columns=["__is_target__", "__market_norm__", "__time_norm__"]
    )
    return combined, mask


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class TCNGRUModel:
    """
    Sequence-native wrapper, not a fake row-wise estimator.

    fit_frame() operates on the repository's AgentDataset frame.
    predict_proba_frame() can receive earlier context so sequence history is
    continuous across train/validation/test boundaries.
    """

    requires_sequence_history = True
    model_family = "tcn_gru"

    def __init__(
        self,
        feature_columns: Sequence[str] = DEFAULT_FEATURES,
        *,
        config: TCNGRUConfig | None = None,
    ) -> None:
        self.feature_columns = tuple(str(x) for x in feature_columns)
        self.config = config or TCNGRUConfig()
        self.config.validate()

        self.standardizer = TrainOnlyStandardizer()
        self.network: TCNGRUNetwork | None = None
        self.history: list[TrainingHistoryRow] = []
        self.best_epoch: int | None = None
        self.best_inner_validation_loss: float | None = None
        self.fitted_at: str | None = None
        self.training_device: str | None = None

    # -------------------------
    # Inner TRAIN-only split
    # -------------------------

    def _inner_train_validation_masks(
        self,
        frame: pd.DataFrame,
        *,
        horizon_bars: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        times = pd.DatetimeIndex(
            pd.to_datetime(frame["feature_time"], utc=True)
            .drop_duplicates()
            .sort_values()
        )

        if len(times) < max(40, self.config.lookback * 2):
            raise ValueError(
                "insufficient unique training timestamps for TCN+GRU inner split"
            )

        cut = int(len(times) * (1.0 - self.config.inner_validation_fraction))
        cut = max(self.config.lookback, min(len(times) - 1, cut))

        purge = max(
            1,
            int(horizon_bars),
            int(self.config.inner_purge_bars),
        )

        train_end = max(1, cut - purge)
        val_start = min(len(times), cut + purge)

        train_times = set(times[:train_end])
        val_times = set(times[val_start:])

        row_times = pd.to_datetime(frame["feature_time"], utc=True)
        train_mask = row_times.isin(train_times).to_numpy(dtype=bool)
        val_mask = row_times.isin(val_times).to_numpy(dtype=bool)

        if train_mask.sum() < 100 or val_mask.sum() < 50:
            raise ValueError(
                "inner chronological split produced too few observations"
            )

        return train_mask, val_mask

    # -------------------------
    # Fitting
    # -------------------------

    def fit_frame(
        self,
        train_frame: pd.DataFrame,
        *,
        target_column: str = "target_alpha",
        horizon_bars: int = 4,
    ) -> "TCNGRUModel":
        _validate_agent_frame(
            train_frame,
            self.feature_columns,
            require_target=True,
            target_column=target_column,
        )

        y_all = train_frame[target_column].astype(int).to_numpy()
        if set(np.unique(y_all)).difference({0, 1}):
            raise ValueError("target must be binary 0/1")
        if len(np.unique(y_all)) < 2:
            raise ValueError("target has only one class")

        _set_seed(self.config.seed, self.config.deterministic)

        inner_train_mask, inner_val_mask = self._inner_train_validation_masks(
            train_frame,
            horizon_bars=horizon_bars,
        )

        raw_features = train_frame.loc[
            :, self.feature_columns
        ].to_numpy(dtype=np.float64, copy=True)

        # preprocessing fit ONLY on inner training observations
        self.standardizer.fit(raw_features[inner_train_mask])
        scaled = self.standardizer.transform(raw_features)

        train_sequences = _sequence_batch_from_combined_frame(
            train_frame,
            scaled,
            lookback=self.config.lookback,
            target_position_mask=inner_train_mask,
        )

        # Inner validation can use all earlier TRAIN observations as context.
        val_sequences = _sequence_batch_from_combined_frame(
            train_frame,
            scaled,
            lookback=self.config.lookback,
            target_position_mask=inner_val_mask,
        )

        if train_sequences.rows < 100:
            raise ValueError(
                f"too few complete TCN+GRU train sequences: "
                f"{train_sequences.rows}"
            )
        if val_sequences.rows < 30:
            raise ValueError(
                f"too few complete TCN+GRU validation sequences: "
                f"{val_sequences.rows}"
            )

        y_train = y_all[train_sequences.row_positions].astype(np.float32)
        y_val = y_all[val_sequences.row_positions].astype(np.float32)

        device = _resolve_device(self.config.device)
        self.training_device = str(device)

        network = TCNGRUNetwork(
            input_dim=len(self.feature_columns),
            config=self.config,
        ).to(device)

        positive = float(y_train.sum())
        negative = float(len(y_train) - positive)

        if self.config.class_balance and positive > 0.0:
            pos_weight = max(1e-4, negative / positive)
        else:
            pos_weight = 1.0

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=device)
        )

        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        generator = torch.Generator()
        generator.manual_seed(self.config.seed)

        train_dataset = TensorDataset(
            torch.from_numpy(train_sequences.x),
            torch.from_numpy(y_train),
        )

        loader = DataLoader(
            train_dataset,
            batch_size=min(self.config.batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=(device.type == "cuda"),
            generator=generator,
        )

        val_x = torch.from_numpy(val_sequences.x)
        val_y = torch.from_numpy(y_val)

        best_state: dict[str, torch.Tensor] | None = None
        best_loss = math.inf
        best_epoch = 0
        stale = 0
        history: list[TrainingHistoryRow] = []

        for epoch in range(1, self.config.epochs + 1):
            network.train()

            total_loss = 0.0
            seen = 0

            for xb, yb in loader:
                xb = xb.to(
                    device,
                    non_blocking=(device.type == "cuda"),
                )
                yb = yb.to(
                    device,
                    non_blocking=(device.type == "cuda"),
                )

                optimizer.zero_grad(set_to_none=True)
                logits = network(xb)

                if self.config.label_smoothing > 0.0:
                    s = float(self.config.label_smoothing)
                    smooth_y = yb * (1.0 - s) + 0.5 * s
                    loss = criterion(logits, smooth_y)
                else:
                    loss = criterion(logits, yb)

                loss.backward()

                if self.config.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(
                        network.parameters(),
                        self.config.grad_clip,
                    )

                optimizer.step()

                n = len(xb)
                total_loss += float(loss.detach().cpu()) * n
                seen += n

            train_loss = total_loss / max(1, seen)

            network.eval()
            probabilities: list[np.ndarray] = []
            val_losses: list[float] = []
            val_counts: list[int] = []

            with torch.inference_mode():
                for start in range(
                    0,
                    len(val_x),
                    self.config.batch_size,
                ):
                    xb = val_x[
                        start : start + self.config.batch_size
                    ].to(device)
                    yb = val_y[
                        start : start + self.config.batch_size
                    ].to(device)

                    logits = network(xb)
                    loss = criterion(logits, yb)

                    p = torch.sigmoid(logits).detach().cpu().numpy()
                    probabilities.append(p)

                    val_losses.append(float(loss.detach().cpu()))
                    val_counts.append(len(xb))

            val_probability = np.concatenate(probabilities)
            val_loss = float(
                np.average(val_losses, weights=val_counts)
            )
            val_brier = float(
                brier_score_loss(y_val.astype(int), val_probability)
            )
            val_auc = (
                float(roc_auc_score(y_val, val_probability))
                if len(np.unique(y_val)) >= 2
                else None
            )

            history.append(
                TrainingHistoryRow(
                    epoch=epoch,
                    train_loss=float(train_loss),
                    validation_loss=val_loss,
                    validation_brier=val_brier,
                    validation_auc=val_auc,
                )
            )

            # Early stopping on loss, not on an economic threshold,
            # to keep the TRAIN-internal stopping rule separate from
            # later validation threshold selection.
            if val_loss < best_loss - self.config.min_delta:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in network.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= self.config.patience:
                    break

        if best_state is None:
            raise RuntimeError("TCN+GRU training produced no checkpoint")

        network.load_state_dict(best_state)
        network.to(torch.device("cpu"))
        network.eval()

        self.network = network
        self.history = history
        self.best_epoch = best_epoch
        self.best_inner_validation_loss = float(best_loss)
        self.fitted_at = datetime.now(UTC).isoformat()

        return self

    # -------------------------
    # Prediction
    # -------------------------

    def _check_fitted(self) -> None:
        if self.network is None:
            raise RuntimeError("TCNGRUModel is not fitted")

    def _predict_sequences(
        self,
        sequences: np.ndarray,
        *,
        device: str = "cpu",
    ) -> np.ndarray:
        self._check_fitted()
        assert self.network is not None

        if not len(sequences):
            return np.empty(0, dtype=np.float64)

        run_device = _resolve_device(device)
        model = self.network.to(run_device)
        model.eval()

        values: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(
                0,
                len(sequences),
                self.config.batch_size,
            ):
                xb = torch.from_numpy(
                    sequences[start : start + self.config.batch_size]
                ).to(run_device)

                logits = model(xb)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                values.append(p.astype(np.float64, copy=False))

        # Always leave artifact/model CPU-portable.
        model.to(torch.device("cpu"))
        return np.concatenate(values)

    def predict_proba_frame(
        self,
        target_frame: pd.DataFrame,
        *,
        context_frame: pd.DataFrame | None = None,
        device: str = "cpu",
        probability_name: str = "tcn_gru_probability_raw",
    ) -> SequencePrediction:
        """
        Predict target rows while optionally prepending strictly earlier context.

        Context is essential around validation/test boundaries. It prevents the
        model from losing the previous lookback bars while still ensuring the
        model cannot see anything after each prediction timestamp.
        """
        self._check_fitted()

        _validate_agent_frame(
            target_frame,
            self.feature_columns,
            require_target=False,
        )

        if context_frame is not None and not context_frame.empty:
            _validate_agent_frame(
                context_frame,
                self.feature_columns,
                require_target=False,
            )

            # Hard causality check by market:
            # context may overlap old timestamps but must never contain timestamps
            # after the maximum target timestamp for that same market.
            for market in sorted(
                target_frame["market"].astype(str).str.upper().unique()
            ):
                target_market = target_frame[
                    target_frame["market"].astype(str).str.upper() == market
                ]
                context_market = context_frame[
                    context_frame["market"].astype(str).str.upper() == market
                ]
                if target_market.empty or context_market.empty:
                    continue

                target_min = pd.to_datetime(
                    target_market["feature_time"],
                    utc=True,
                ).min()
                # overlapping context is tolerated and deduped in favor of target,
                # but future context is forbidden
                if pd.to_datetime(
                    context_market["feature_time"],
                    utc=True,
                ).max() > pd.to_datetime(
                    target_frame["feature_time"],
                    utc=True,
                ).max():
                    raise ValueError(
                        f"context contains future rows for market {market}"
                    )

        combined, target_mask = _concat_context_target(
            context_frame,
            target_frame,
        )

        raw = combined.loc[
            :, self.feature_columns
        ].to_numpy(dtype=np.float64, copy=True)
        scaled = self.standardizer.transform(raw)

        sequences = _sequence_batch_from_combined_frame(
            combined,
            scaled,
            lookback=self.config.lookback,
            target_position_mask=target_mask,
        )

        probability_values = self._predict_sequences(
            sequences.x,
            device=device,
        )

        # Use a stable MultiIndex key to map combined positions back to target rows.
        combined_market = combined["market"].astype(str).str.upper()
        combined_time = pd.to_datetime(
            combined["feature_time"],
            utc=True,
        )

        prediction_map: dict[tuple[str, int], float] = {}
        for pos, probability in zip(
            sequences.row_positions,
            probability_values,
        ):
            key = (
                str(combined_market.iloc[int(pos)]),
                int(combined_time.iloc[int(pos)].value),
            )
            prediction_map[key] = float(probability)

        target_probability = []
        for market, ts in zip(
            target_frame["market"].astype(str).str.upper(),
            pd.to_datetime(target_frame["feature_time"], utc=True),
        ):
            target_probability.append(
                prediction_map.get(
                    (str(market), int(ts.value)),
                    np.nan,
                )
            )

        series = pd.Series(
            target_probability,
            index=target_frame.index,
            dtype=float,
            name=probability_name,
        )

        return SequencePrediction(
            probability=series,
            valid_rows=int(series.notna().sum()),
            total_rows=int(len(series)),
        )

    def predict_latest_features(
        self,
        market: str,
        feature_frame: pd.DataFrame,
        *,
        device: str = "cpu",
    ) -> float | None:
        """
        Runtime helper for AgentRuntime integration.

        ``feature_frame`` should contain a chronological feature history for ONE
        market, not only the final row. Returns None until lookback rows exist.
        """
        self._check_fitted()

        if feature_frame.empty:
            return None

        work = feature_frame.copy()

        if "feature_time" not in work.columns:
            if not isinstance(work.index, pd.DatetimeIndex):
                raise ValueError(
                    "feature_frame needs feature_time or DatetimeIndex"
                )
            work["feature_time"] = pd.to_datetime(
                work.index,
                utc=True,
            )

        work["market"] = str(market).upper()

        missing = [
            c for c in self.feature_columns
            if c not in work.columns
        ]
        if missing:
            raise ValueError(
                f"runtime feature frame missing columns: {missing}"
            )

        # Only recent history is necessary.
        work = work.sort_values("feature_time").tail(
            max(self.config.lookback, self.config.lookback + 8)
        )

        prediction = self.predict_proba_frame(
            work,
            context_frame=None,
            device=device,
        ).probability.dropna()

        if prediction.empty:
            return None
        return float(prediction.iloc[-1])

    def predict_latest_ohlcv(
        self,
        market: str,
        ohlcv_frame: pd.DataFrame,
        *,
        device: str = "cpu",
    ) -> float | None:
        """
        Convenience method: OHLCV -> existing repo feature builder -> probability.
        """
        features = (
            build_features(ohlcv_frame)
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=list(self.feature_columns))
        )
        return self.predict_latest_features(
            market,
            features,
            device=device,
        )

    # -------------------------
    # Portable state
    # -------------------------

    def state_dict(self) -> dict[str, Any]:
        self._check_fitted()
        assert self.network is not None

        return {
            "schema_version": "tcn_gru_model_v2",
            "feature_columns": list(self.feature_columns),
            "config": asdict(self.config),
            "standardizer": self.standardizer.state(),
            "network_state": {
                key: value.detach().cpu()
                for key, value in self.network.state_dict().items()
            },
            "history": [asdict(row) for row in self.history],
            "best_epoch": self.best_epoch,
            "best_inner_validation_loss": (
                self.best_inner_validation_loss
            ),
            "fitted_at": self.fitted_at,
            "training_device": self.training_device,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> "TCNGRUModel":
        if state.get("schema_version") != "tcn_gru_model_v2":
            raise ValueError(
                f"unsupported model schema: {state.get('schema_version')}"
            )

        config_raw = dict(state["config"])
        config_raw["tcn_channels"] = tuple(
            config_raw["tcn_channels"]
        )
        config = TCNGRUConfig(**config_raw)

        obj = cls(
            feature_columns=tuple(state["feature_columns"]),
            config=config,
        )
        obj.standardizer = TrainOnlyStandardizer.from_state(
            state["standardizer"]
        )

        network = TCNGRUNetwork(
            input_dim=len(obj.feature_columns),
            config=config,
        )
        network.load_state_dict(state["network_state"])
        network.to(torch.device("cpu"))
        network.eval()

        obj.network = network
        obj.history = [
            TrainingHistoryRow(**dict(row))
            for row in state.get("history", [])
        ]
        obj.best_epoch = state.get("best_epoch")
        obj.best_inner_validation_loss = state.get(
            "best_inner_validation_loss"
        )
        obj.fitted_at = state.get("fitted_at")
        obj.training_device = state.get("training_device")
        return obj


# ---------------------------------------------------------------------------
# Calibration + economic threshold selection
# ---------------------------------------------------------------------------


@dataclass
class CalibratedTCNGRU:
    """
    Joblib-safe calibrated wrapper.

    Unlike the repo's generic CalibratedClassifier, this wrapper remains
    sequence-aware and accepts context frames.
    """

    model: TCNGRUModel
    calibrator: ProbabilityCalibrator

    requires_sequence_history: bool = field(default=True, init=False)
    model_family: str = field(default="tcn_gru", init=False)

    def predict_proba_frame(
        self,
        target_frame: pd.DataFrame,
        *,
        context_frame: pd.DataFrame | None = None,
        device: str = "cpu",
        probability_name: str = "tcn_gru_probability",
    ) -> SequencePrediction:
        raw = self.model.predict_proba_frame(
            target_frame,
            context_frame=context_frame,
            device=device,
        )

        out = raw.probability.copy()
        mask = out.notna()

        if mask.any():
            out.loc[mask] = self.calibrator.transform(
                out.loc[mask].to_numpy(float)
            )

        out.name = probability_name
        return SequencePrediction(
            probability=out,
            valid_rows=int(mask.sum()),
            total_rows=len(out),
        )

    def predict_latest_ohlcv(
        self,
        market: str,
        ohlcv_frame: pd.DataFrame,
        *,
        device: str = "cpu",
    ) -> float | None:
        features = (
            build_features(ohlcv_frame)
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=list(self.model.feature_columns))
        )
        if features.empty:
            return None

        features = features.copy()
        features["market"] = str(market).upper()
        features["feature_time"] = pd.to_datetime(
            features.index,
            utc=True,
        )

        pred = self.predict_proba_frame(
            features,
            device=device,
        ).probability.dropna()

        return None if pred.empty else float(pred.iloc[-1])


def _finite_probability_rows(
    prediction: SequencePrediction,
    frame: pd.DataFrame,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    mask = prediction.probability.notna().to_numpy(dtype=bool)
    positions = np.flatnonzero(mask)
    selected_frame = frame.iloc[positions].copy()

    y = selected_frame[target_column].astype(int).to_numpy()
    p = prediction.probability.iloc[positions].to_numpy(float)

    if not len(y):
        raise ValueError("no complete sequences available")

    return y, p, selected_frame


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return (
        float(roc_auc_score(y, p))
        if len(np.unique(y)) >= 2
        else None
    )


def _probability_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    pred = p >= threshold

    return {
        "rows": int(len(y)),
        "positive_fraction": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "auc": _auc(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(
            log_loss(
                y,
                np.column_stack([1.0 - p, p]),
                labels=[0, 1],
            )
        ),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
    }


def _selected_economics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    *,
    cost_floor: float,
) -> dict[str, Any]:
    selected = probability >= float(threshold)

    if not selected.any():
        return {
            "threshold": float(threshold),
            "count": 0,
            "market_count": 0,
            "mean_return": None,
            "mean_net": None,
            "conservative_mean_net": None,
            "market_balanced_mean_net": None,
            "positive_market_fraction": None,
        }

    realized = frame["target_forward_return"].to_numpy(float)[
        selected
    ]
    markets = frame["market"].astype(str).to_numpy()[selected]
    net = realized - float(cost_floor)

    by_market: list[float] = []
    for market in sorted(set(markets)):
        values = net[markets == market]
        if len(values):
            by_market.append(float(np.mean(values)))

    mean_net = float(np.mean(net))

    if len(net) >= 2:
        se = float(np.std(net, ddof=1) / np.sqrt(len(net)))
        conservative = mean_net - se
    else:
        conservative = mean_net

    return {
        "threshold": float(threshold),
        "count": int(selected.sum()),
        "market_count": len(by_market),
        "mean_return": float(np.mean(realized)),
        "mean_net": mean_net,
        "conservative_mean_net": float(conservative),
        "market_balanced_mean_net": (
            float(np.mean(by_market))
            if by_market
            else None
        ),
        "positive_market_fraction": (
            float(np.mean(np.asarray(by_market) > 0.0))
            if by_market
            else None
        ),
    }


def _threshold_grid(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    config: TCNGRUConfig,
    cost_floor: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    values = np.arange(
        config.threshold_min,
        config.threshold_max + config.threshold_step / 2.0,
        config.threshold_step,
    )

    for threshold in values:
        economics = _selected_economics(
            frame,
            probability,
            float(threshold),
            cost_floor=cost_floor,
        )

        eligible = bool(
            economics["count"] >= config.minimum_selected
            and economics["market_count"] >= config.minimum_markets
            and economics["mean_net"] is not None
            and economics["mean_net"] > 0.0
            and economics["conservative_mean_net"] is not None
            and economics["conservative_mean_net"] > 0.0
            and economics["market_balanced_mean_net"] is not None
            and economics["market_balanced_mean_net"] > 0.0
            and economics["positive_market_fraction"] is not None
            and economics["positive_market_fraction"]
            >= config.minimum_positive_market_fraction
        )

        rows.append(
            {
                **economics,
                "economically_eligible": eligible,
            }
        )

    eligible_rows = [
        row for row in rows
        if row["economically_eligible"]
    ]

    pool = eligible_rows or rows

    chosen = max(
        pool,
        key=lambda row: (
            1 if row["economically_eligible"] else 0,
            float(
                row["conservative_mean_net"]
                if row["conservative_mean_net"] is not None
                else -999.0
            ),
            float(
                row["market_balanced_mean_net"]
                if row["market_balanced_mean_net"] is not None
                else -999.0
            ),
            int(row["count"]),
        ),
    )

    return {
        "chosen": chosen,
        "grid": rows,
    }


# ---------------------------------------------------------------------------
# High-level repo-specific challenger trainer
# ---------------------------------------------------------------------------


class TCNGRUChallengerTrainer:
    """
    End-to-end trainer using the repository's current data flow.

    It writes its OWN challenger artifact under:
        output/crypto_ai_swing/agents/tcn_gru/

    It intentionally does not overwrite the main AgentTrainer bundle or grant
    live authority.
    """

    def __init__(
        self,
        settings,
        *,
        config: TCNGRUConfig | None = None,
    ) -> None:
        self.settings = settings
        self.config = config or TCNGRUConfig()
        self.config.validate()
        self.crypto = CryptoLibraryBridge(
            settings.crypto_repo_root
        )

    def _fetch_dataset(
        self,
        markets: Sequence[str],
        *,
        timeframe: str,
        horizon_bars: int,
        minimum_net_move_bps: float,
    ) -> AgentDataset:
        selected = [
            str(market).upper()
            for market in markets
            if str(market).strip()
        ]
        if not selected:
            raise ValueError("at least one market is required")

        # Training must use the canonical persisted historical store, not the
        # short live/runtime lookback. This keeps temporal training reproducible
        # and aligned with the MTF historical contract.
        frames = self.crypto.historical_ohlcv_many(
            selected,
            timeframe,
            provider="bitvavo",
        )

        frames = {
            market: frame
            for market, frame in frames.items()
            if frame is not None and not frame.empty
        }

        if not frames:
            raise ValueError("no OHLCV frames returned by crypto repo")

        return build_agent_dataset(
            frames,
            horizon_bars=horizon_bars,
            minimum_net_move_bps=minimum_net_move_bps,
        )

    def train(
        self,
        *,
        markets: Sequence[str],
        timeframe: str = "1h",
        horizon_bars: int = 4,
        minimum_rows: int = 2500,
        minimum_net_move_bps: float = 65.0,
        expires_days: int = 30,
    ) -> TCNGRUTrainingResult:
        dataset = self._fetch_dataset(
            markets,
            timeframe=timeframe,
            horizon_bars=horizon_bars,
            minimum_net_move_bps=minimum_net_move_bps,
        )

        if len(dataset.frame) < int(minimum_rows):
            raise ValueError(
                f"insufficient causal rows: "
                f"{len(dataset.frame)} < {minimum_rows}"
            )

        train, validation, test = purged_chronological_split(
            dataset
        )

        calibration_frame, selection_frame = (
            purged_calibration_selection_split(
                validation,
                horizon_bars=horizon_bars,
                calibration_fraction=0.50,
            )
        )

        model = TCNGRUModel(
            feature_columns=dataset.feature_columns,
            config=self.config,
        )
        model.fit_frame(
            train,
            horizon_bars=horizon_bars,
        )

        # CALIBRATION:
        # use TRAIN as historical context; only calibration rows contribute
        # to the calibration fit.
        calibration_prediction = model.predict_proba_frame(
            calibration_frame,
            context_frame=train,
            device="cpu",
        )
        y_cal, p_cal_raw, cal_aligned = _finite_probability_rows(
            calibration_prediction,
            calibration_frame,
            "target_alpha",
        )

        calibrator = fit_probability_calibrator(
            p_cal_raw,
            y_cal,
            method=self.config.calibration_method,
        )

        calibrated = CalibratedTCNGRU(
            model=model,
            calibrator=calibrator,
        )

        # SELECTION:
        # context may include train + earlier calibration only.
        selection_context = pd.concat(
            [train, calibration_frame],
            axis=0,
        )
        selection_prediction = calibrated.predict_proba_frame(
            selection_frame,
            context_frame=selection_context,
            device="cpu",
        )
        y_sel, p_sel, selection_aligned = _finite_probability_rows(
            selection_prediction,
            selection_frame,
            "target_alpha",
        )

        cost_floor = float(minimum_net_move_bps) / 10_000.0

        threshold_plan = _threshold_grid(
            selection_aligned,
            p_sel,
            config=self.config,
            cost_floor=cost_floor,
        )
        threshold = float(
            threshold_plan["chosen"]["threshold"]
        )

        # FINAL TEST:
        # test can consume every observation that happened before test.
        # No future test row is used to form an earlier test sequence.
        test_context = pd.concat(
            [train, validation],
            axis=0,
        )
        test_prediction = calibrated.predict_proba_frame(
            test,
            context_frame=test_context,
            device="cpu",
        )
        y_test, p_test, test_aligned = _finite_probability_rows(
            test_prediction,
            test,
            "target_alpha",
        )

        test_economics = _selected_economics(
            test_aligned,
            p_test,
            threshold,
            cost_floor=cost_floor,
        )

        metrics = {
            "model_family": "tcn_gru",
            "status": "SHADOW_CHALLENGER",
            "architecture": {
                "input_features": len(dataset.feature_columns),
                "feature_columns": list(dataset.feature_columns),
                "lookback": self.config.lookback,
                "tcn_channels": list(
                    self.config.tcn_channels
                ),
                "tcn_dilations": [
                    2**i
                    for i in range(
                        len(self.config.tcn_channels)
                    )
                ],
                "kernel_size": self.config.kernel_size,
                "gru_hidden": self.config.gru_hidden,
                "gru_layers": self.config.gru_layers,
                "bidirectional": False,
                "causal_convolution": True,
            },
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "rows": len(dataset.frame),
                "markets": list(dataset.markets),
                "time_start": dataset.time_start,
                "time_end": dataset.time_end,
                "timeframe": timeframe,
                "horizon_bars": horizon_bars,
                "minimum_net_move_bps": minimum_net_move_bps,
            },
            "split_rows": {
                "train": len(train),
                "validation": len(validation),
                "calibration": len(calibration_frame),
                "selection": len(selection_frame),
                "test": len(test),
            },
            "sequence_rows": {
                "calibration": int(
                    calibration_prediction.valid_rows
                ),
                "selection": int(
                    selection_prediction.valid_rows
                ),
                "test": int(test_prediction.valid_rows),
            },
            "inner_training": {
                "best_epoch": model.best_epoch,
                "best_validation_loss": (
                    model.best_inner_validation_loss
                ),
                "epochs_run": len(model.history),
                "history": [
                    asdict(row)
                    for row in model.history
                ],
            },
            "calibration": {
                "method": self.config.calibration_method,
                "raw": _probability_metrics(
                    y_cal,
                    p_cal_raw,
                ),
                "calibrated": _probability_metrics(
                    y_cal,
                    calibrator.transform(p_cal_raw),
                ),
            },
            "selection": {
                "probability": _probability_metrics(
                    y_sel,
                    p_sel,
                ),
                "threshold_plan": threshold_plan,
            },
            "test": {
                "probability": _probability_metrics(
                    y_test,
                    p_test,
                    threshold=threshold,
                ),
                "threshold": threshold,
                "economics": test_economics,
            },
            "integrity": {
                "closed_candles_source": True,
                "market_isolated_sequences": True,
                "causal_convolutions": True,
                "unidirectional_gru": True,
                "train_only_preprocessing_fit": True,
                "chronological_split": True,
                "purged_outer_split": True,
                "purged_inner_early_stopping_split": True,
                "dedicated_probability_calibration": True,
                "dedicated_threshold_selection": True,
                "test_used_once_after_selection": True,
                "automatic_live_promotion": False,
                "live_decision_influence": False,
            },
        }

        trained_at = datetime.now(UTC)

        artifact_payload = {
            "schema_version": "tcn_gru_challenger_bundle_v2",
            "status": "SHADOW_CHALLENGER",
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "trained_at": trained_at.isoformat(),
            "expires_at": (
                trained_at + timedelta(days=int(expires_days))
            ).isoformat(),
            "dataset_id": dataset.dataset_id,
            "feature_columns": list(dataset.feature_columns),
            "markets": list(dataset.markets),
            "timeframe": timeframe,
            "horizon_bars": int(horizon_bars),
            "minimum_net_move_bps": float(
                minimum_net_move_bps
            ),
            "probability_threshold": threshold,
            "calibration_method": (
                self.config.calibration_method
            ),
            "model_state": model.state_dict(),
            "calibrator": calibrator,
            "metrics": metrics,
        }

        identity_payload = {
            key: value
            for key, value in artifact_payload.items()
            if key not in {"model_state", "calibrator"}
        }

        identity = sha256(
            json.dumps(
                identity_payload,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

        root = (
            Path(self.settings.project_root)
            / "output/crypto_ai_swing/agents/tcn_gru"
        )
        artdir = root / "artifacts" / identity
        artdir.mkdir(parents=True, exist_ok=True)

        artifact_path = artdir / "bundle.joblib"
        joblib.dump(artifact_payload, artifact_path)

        artifact_hash = sha256(
            artifact_path.read_bytes()
        ).hexdigest()

        manifest = {
            **identity_payload,
            "artifact_hash": artifact_hash,
            "artifact_path": str(
                artifact_path.resolve()
            ),
        }

        manifest_path = artdir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )

        pointer = root / "latest.pointer.json"
        pointer.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        pointer.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "tcn_gru_challenger_pointer_v2"
                    ),
                    "status": "SHADOW_CHALLENGER",
                    "artifact_path": str(
                        artifact_path.resolve()
                    ),
                    "manifest_path": str(
                        manifest_path.resolve()
                    ),
                    "artifact_hash": artifact_hash,
                    "live_decision_influence": False,
                    "automatic_live_promotion": False,
                    "updated_at": trained_at.isoformat(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return TCNGRUTrainingResult(
            status="SHADOW_CHALLENGER",
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            metrics=metrics,
            dataset_id=dataset.dataset_id,
            row_count=len(dataset.frame),
            markets=dataset.markets,
            timeframe=timeframe,
            horizon_bars=horizon_bars,
        )


# ---------------------------------------------------------------------------
# Artifact loading/runtime
# ---------------------------------------------------------------------------


@dataclass
class LoadedTCNGRUChallenger:
    model: CalibratedTCNGRU
    threshold: float
    metadata: dict[str, Any]

    def predict_latest_ohlcv(
        self,
        market: str,
        ohlcv_frame: pd.DataFrame,
        *,
        device: str = "cpu",
    ) -> dict[str, Any]:
        probability = self.model.predict_latest_ohlcv(
            market,
            ohlcv_frame,
            device=device,
        )

        return {
            "market": str(market).upper(),
            "probability": probability,
            "threshold": self.threshold,
            "signal": (
                None
                if probability is None
                else bool(probability >= self.threshold)
            ),
            "status": self.metadata.get(
                "status",
                "SHADOW_CHALLENGER",
            ),
            "live_decision_influence": False,
        }


def load_tcn_gru_challenger(
    path: str | Path,
) -> LoadedTCNGRUChallenger:
    payload = joblib.load(Path(path))

    if not isinstance(payload, dict):
        raise ValueError("TCN+GRU artifact is not a dict")
    if payload.get("schema_version") != (
        "tcn_gru_challenger_bundle_v2"
    ):
        raise ValueError(
            f"unsupported TCN+GRU artifact schema: "
            f"{payload.get('schema_version')}"
        )

    model = TCNGRUModel.from_state_dict(
        payload["model_state"]
    )

    calibrator = payload["calibrator"]
    if not isinstance(
        calibrator,
        ProbabilityCalibrator,
    ):
        raise TypeError(
            "artifact calibrator has unexpected type"
        )

    wrapped = CalibratedTCNGRU(
        model=model,
        calibrator=calibrator,
    )

    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state", "calibrator"}
    }

    return LoadedTCNGRUChallenger(
        model=wrapped,
        threshold=float(
            payload["probability_threshold"]
        ),
        metadata=metadata,
    )


def load_latest_tcn_gru_challenger(
    settings,
) -> LoadedTCNGRUChallenger | None:
    pointer = (
        Path(settings.project_root)
        / "output/crypto_ai_swing/agents/tcn_gru"
        / "latest.pointer.json"
    )

    if not pointer.is_file():
        return None

    raw = json.loads(
        pointer.read_text(encoding="utf-8")
    )
    artifact = Path(str(raw["artifact_path"]))

    if not artifact.is_file():
        raise FileNotFoundError(
            f"TCN+GRU artifact missing: {artifact}"
        )

    expected_hash = raw.get("artifact_hash")
    if expected_hash:
        actual_hash = sha256(
            artifact.read_bytes()
        ).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                "TCN+GRU artifact hash mismatch"
            )

    return load_tcn_gru_challenger(artifact)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def default_tcn_gru_config(
    *,
    horizon_bars: int = 8,
) -> TCNGRUConfig:
    """
    Default geared toward the repo's 1h swing-agent workflow.

    lookback=64 means roughly 64 model bars of temporal context. This is kept
    intentionally smaller than the separate Kronos lookback because TCN+GRU is
    a learned discriminative classifier, not a pretrained forecaster.
    """
    return TCNGRUConfig(
        lookback=64,
        tcn_channels=(64, 64, 96),
        kernel_size=3,
        gru_hidden=96,
        gru_layers=1,
        dropout=0.15,
        learning_rate=3e-4,
        weight_decay=1e-4,
        batch_size=256,
        epochs=60,
        patience=8,
        inner_validation_fraction=0.15,
        inner_purge_bars=max(1, int(horizon_bars)),
        calibration_method="sigmoid",
        threshold_min=0.50,
        threshold_max=0.75,
        threshold_step=0.025,
        minimum_selected=40,
        minimum_markets=3,
        minimum_positive_market_fraction=0.50,
        seed=17,
        device="auto",
    )


def load_live_tcn_gru(
    settings,
) -> LoadedTCNGRUChallenger | None:
    """Load only an evidence-qualified live TCN+GRU artifact.

    Candidate artifacts never receive live influence just because they exist.
    The governor writes ``live.pointer.json`` only after qualification.
    """
    pointer = (
        Path(settings.project_root)
        / "output/crypto_ai_swing/agents/tcn_gru"
        / "live.pointer.json"
    )
    if not pointer.is_file():
        return None
    raw = json.loads(pointer.read_text(encoding="utf-8"))
    if raw.get("qualified") is not True:
        return None
    artifact = Path(str(raw.get("artifact_path") or ""))
    if not artifact.is_file():
        raise FileNotFoundError(f"qualified TCN+GRU artifact missing: {artifact}")
    expected_hash = raw.get("artifact_hash")
    if expected_hash:
        actual_hash = sha256(artifact.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("TCN+GRU live artifact hash mismatch")
    loaded = load_tcn_gru_challenger(artifact)
    loaded.metadata["status"] = "LIVE_QUALIFIED"
    loaded.metadata["live_decision_influence"] = True
    return loaded
