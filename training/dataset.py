"""
LOB Dataset — Reads from TimescaleDB lob_5s continuous aggregate,
applies transformations, and produces sliding-window training samples.

Supports both 5-level (legacy) and 40-level LOB data.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import asyncpg
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from training.features import engineer_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "lob_user"
    db_password: str = "lob_password"
    db_name: str = "lob_data"

    # LOB structure
    lob_levels: int = 40

    # Sequence params (from data-requirements.md)
    context_length: int = 120    # 10 minutes at 5s intervals
    prediction_length: int = 24  # 2 minutes
    stride: int = 60             # 50% overlap

    # Split ratios (chronological)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    # test_ratio = 1 - train - val = 0.2

    # Exchanges and pairs
    exchanges: list = None
    pairs: list = None

    # Feature pipeline
    feature_version: str = "v2"     # "v1" -> engineer_features; "v2" -> engineer_features_v2
    savgol_window: int = 11         # V2 default (was hardcoded 21 in build_dataloaders)
    # Data source
    source: str = "db"              # "db" -> lob_5s; "parquet" -> parquet_dir
    parquet_dir: str = "lob_data"

    def __post_init__(self):
        if self.exchanges is None:
            self.exchanges = [
                "binance_spot", "binance_perp", "bybit_spot", "kucoin_spot"
            ]
        if self.pairs is None:
            self.pairs = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "WLD-USDT"]

    @property
    def window_size(self) -> int:
        return self.context_length + self.prediction_length

    @property
    def n_base_features(self) -> int:
        """Number of base LOB features per timestep (price + volume for each level, both sides)."""
        return self.lob_levels * 4  # bid_price, bid_vol, ask_price, ask_vol per level

    @property
    def n_features(self) -> int:
        """Total base features from DB per timestep."""
        return self.n_base_features + 2  # + mid_price, spread

    @property
    def n_enriched_features(self) -> int:
        """base (4N+2) + derived. V1 derived=N+9 (=5N+11). V2 adds 8 momentum (=5N+19)."""
        return self.lob_levels * 5 + (19 if self.feature_version == "v2" else 11)

    @property
    def warmup_trim(self) -> int:
        """Rows to drop at the head of each split (largest feature lookback)."""
        return max(self.savgol_window, 60)  # 60 = longest momentum horizon (log_return_60)


# ---------------------------------------------------------------------------
# Exchange / symbol encoding
# ---------------------------------------------------------------------------

EXCHANGE_MAP = {
    "binance_spot": 0,
    "binance_perp": 1,
    "bybit_spot": 2,
    "kucoin_spot": 3,
}

SYMBOL_MAP = {
    "BTC-USDT": 0,
    "ETH-USDT": 1,
    "SOL-USDT": 2,
    "WLD-USDT": 3,
}

MARKET_TYPE_MAP = {
    "binance_spot": 0,   # spot
    "binance_perp": 1,   # perpetual
    "bybit_spot": 0,     # spot
    "kucoin_spot": 0,     # spot
}


# ---------------------------------------------------------------------------
# Data fetching from TimescaleDB
# ---------------------------------------------------------------------------

def _build_feature_columns(n_levels: int) -> list[str]:
    """Build the list of feature column names for N LOB levels."""
    cols = []
    for i in range(1, n_levels + 1):
        cols.append(f"bid_price_{i}")
        cols.append(f"bid_volume_{i}")
    for i in range(1, n_levels + 1):
        cols.append(f"ask_price_{i}")
        cols.append(f"ask_volume_{i}")
    cols.extend(["mid_price", "spread"])
    return cols


async def _fetch_stream_data(
    config: DataConfig,
    exchange: str,
    symbol: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch 5s-resampled data for one exchange/symbol stream from TimescaleDB.

    Returns:
        features: np.ndarray of shape (T, n_features) — float64
        timestamps: np.ndarray of shape (T,) — float64 unix timestamps
    """
    feature_cols = _build_feature_columns(config.lob_levels)
    col_str = ", ".join(feature_cols)

    where_clauses = ["exchange = $1", "symbol = $2"]
    params = [exchange, symbol]
    idx = 3

    # Filter out rows without full depth when training at >5 levels
    if config.lob_levels > 5:
        where_clauses.append(f"bid_price_{config.lob_levels} > 0")

    if start_time:
        where_clauses.append(f"bucket >= ${idx}")
        params.append(start_time)
        idx += 1
    if end_time:
        where_clauses.append(f"bucket < ${idx}")
        params.append(end_time)
        idx += 1

    where_str = " AND ".join(where_clauses)

    query = f"""
        SELECT bucket, {col_str}
        FROM lob_5s
        WHERE {where_str}
        ORDER BY bucket ASC
    """

    conn = await asyncpg.connect(
        host=config.db_host, port=config.db_port,
        user=config.db_user, password=config.db_password,
        database=config.db_name,
    )
    try:
        rows = await conn.fetch(query, *params)
    finally:
        await conn.close()

    if not rows:
        return np.empty((0, len(feature_cols))), np.empty((0,))

    n_cols = len(feature_cols)
    timestamps = np.array([r["bucket"].timestamp() for r in rows], dtype=np.float64)
    features = np.zeros((len(rows), n_cols), dtype=np.float64)

    for i, row in enumerate(rows):
        for j, col in enumerate(feature_cols):
            val = row[col]
            features[i, j] = val if val is not None else 0.0

    return features, timestamps


def fetch_stream_data(
    config: DataConfig,
    exchange: str,
    symbol: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synchronous wrapper around _fetch_stream_data."""
    return asyncio.get_event_loop().run_until_complete(
        _fetch_stream_data(config, exchange, symbol, start_time, end_time)
    )


# ---------------------------------------------------------------------------
# Scaling & Transformations
# ---------------------------------------------------------------------------

class LOBScaler:
    """Z-score normalization (per-column mean/std).

    Fits on training data only; transforms train/val/test consistently.
    Preserves cross-column relationships needed for structure loss
    via inverse_transform capability.
    """

    def __init__(self, n_levels: int = 40):
        self.n_levels = n_levels
        self.fitted = False
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit on training data and return z-score transformed copy."""
        self._means = data.mean(axis=0)
        self._stds = data.std(axis=0)
        self._stds = np.where(self._stds == 0, 1.0, self._stds)
        self.fitted = True
        return (data - self._means) / self._stds

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform using previously fitted parameters."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")
        return (data - self._means) / self._stds

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse z-score transform back to original scale."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted.")
        return data * self._stds + self._means

    def save(self, path: str):
        """Save scaler parameters."""
        np.savez(path,
                 means=self._means,
                 stds=self._stds,
                 n_levels=np.array([self.n_levels]))

    @classmethod
    def load(cls, path: str) -> "LOBScaler":
        """Load scaler from saved parameters."""
        data = np.load(path)
        scaler = cls(n_levels=int(data["n_levels"][0]))
        scaler._means = data["means"]
        scaler._stds = data["stds"]
        scaler.fitted = True
        return scaler


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class LOBDataset(Dataset):
    """Sliding-window dataset for LOB time series.

    Each sample contains:
        - context: (context_length, n_features) — input sequence
        - target: (prediction_length, n_features) — prediction target
        - metadata: dict with exchange_id, symbol_id, market_type_id, timestamp
    """

    def __init__(
        self,
        features: np.ndarray,
        timestamps: np.ndarray,
        exchange: str,
        symbol: str,
        config: DataConfig,
    ):
        """
        Args:
            features: (T, n_features) scaled feature array
            timestamps: (T,) unix timestamps
            exchange: exchange name string
            symbol: symbol name string
            config: DataConfig with window parameters
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.timestamps = timestamps
        self.exchange = exchange
        self.symbol = symbol
        self.config = config

        self.exchange_id = EXCHANGE_MAP.get(exchange, 0)
        self.symbol_id = SYMBOL_MAP.get(symbol, 0)
        self.market_type_id = MARKET_TYPE_MAP.get(exchange, 0)

        # Pre-compute valid window start indices, skipping windows that span gaps
        total_len = len(features)
        window = config.window_size
        stride = config.stride
        expected_interval = 5.0  # 5-second buckets
        max_allowed_gap = expected_interval * 2.5  # 12.5s — tolerates one missed tick

        # Find gap positions: indices where timestamp jump exceeds threshold
        ts_diffs = np.diff(timestamps)
        gap_mask = ts_diffs > max_allowed_gap  # True at positions with gaps

        self.indices = []
        skipped = 0
        for start_idx in range(0, total_len - window + 1, stride):
            # Check if any gap falls within this window (start_idx to start_idx+window-1)
            window_gaps = gap_mask[start_idx:start_idx + window - 1]
            if window_gaps.any():
                skipped += 1
            else:
                self.indices.append(start_idx)

        if skipped > 0:
            logger.info(
                f"  {exchange}/{symbol}: skipped {skipped} windows spanning gaps "
                f"({len(self.indices)} valid)"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        start = self.indices[idx]
        ctx_end = start + self.config.context_length
        tgt_end = ctx_end + self.config.prediction_length

        context = self.features[start:ctx_end]      # (120, F)
        target = self.features[ctx_end:tgt_end]      # (24, F)
        ts = self.timestamps[start]

        return {
            "context": context,
            "target": target,
            "exchange_id": self.exchange_id,
            "symbol_id": self.symbol_id,
            "market_type_id": self.market_type_id,
            "timestamp": ts,
        }


# ---------------------------------------------------------------------------
# Multi-stream dataset (combines all exchange/symbol pairs)
# ---------------------------------------------------------------------------

class MultiStreamLOBDataset(Dataset):
    """Combines LOBDatasets from multiple exchange/symbol streams."""

    def __init__(self, datasets: list[LOBDataset]):
        self.datasets = datasets
        # Build cumulative index mapping
        self._cum_lengths = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self._cum_lengths.append(total)
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict:
        # Binary search for which sub-dataset
        for i, cum_len in enumerate(self._cum_lengths):
            if idx < cum_len:
                offset = self._cum_lengths[i - 1] if i > 0 else 0
                return self.datasets[i][idx - offset]
        raise IndexError(f"Index {idx} out of range")


# ---------------------------------------------------------------------------
# Pipeline: fetch, split, scale, create DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: DataConfig,
    batch_size: int = 64,
    num_workers: int = 0,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Full pipeline: fetch data, split, scale, return train/val/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, metadata
        metadata contains scalers, stream info, and split boundaries
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    train_datasets = []
    val_datasets = []
    test_datasets = []
    scalers = {}
    stream_info = {}

    for exchange in config.exchanges:
        for symbol in config.pairs:
            key = f"{exchange}_{symbol}"
            logger.info(f"Fetching data for {key}...")

            features, timestamps = loop.run_until_complete(
                _fetch_stream_data(config, exchange, symbol, start_time, end_time)
            )

            if len(features) < config.window_size:
                logger.warning(
                    f"  {key}: only {len(features)} samples, need {config.window_size}. Skipping."
                )
                continue

            # Feature engineering (OFI, volume, price features + SG smoothing)
            features, derived_names = engineer_features(
                features, n_levels=config.lob_levels,
                apply_smoothing=True, savgol_window=21, savgol_poly=3,
            )
            logger.info(f"  {key}: enriched {features.shape[1]} features ({len(derived_names)} derived)")

            # Chronological split
            n = len(features)
            train_end = int(n * config.train_ratio)
            val_end = int(n * (config.train_ratio + config.val_ratio))

            train_feat = features[:train_end]
            val_feat = features[train_end:val_end]
            test_feat = features[val_end:]

            train_ts = timestamps[:train_end]
            val_ts = timestamps[train_end:val_end]
            test_ts = timestamps[val_end:]

            # Fit scaler on training data only
            scaler = LOBScaler(n_levels=config.lob_levels)
            train_scaled = scaler.fit_transform(train_feat)
            val_scaled = scaler.transform(val_feat)
            test_scaled = scaler.transform(test_feat)

            scalers[key] = scaler

            # Create datasets
            if len(train_scaled) >= config.window_size:
                train_datasets.append(
                    LOBDataset(train_scaled, train_ts, exchange, symbol, config)
                )
            if len(val_scaled) >= config.window_size:
                val_datasets.append(
                    LOBDataset(val_scaled, val_ts, exchange, symbol, config)
                )
            if len(test_scaled) >= config.window_size:
                test_datasets.append(
                    LOBDataset(test_scaled, test_ts, exchange, symbol, config)
                )

            stream_info[key] = {
                "total_samples": n,
                "train_samples": len(train_feat),
                "val_samples": len(val_feat),
                "test_samples": len(test_feat),
                "train_windows": len(train_datasets[-1]) if train_datasets and train_datasets[-1].exchange == exchange and train_datasets[-1].symbol == symbol else 0,
            }

            logger.info(
                f"  {key}: {n} total, train={len(train_feat)}, "
                f"val={len(val_feat)}, test={len(test_feat)}"
            )

    loop.close()

    if not train_datasets:
        raise ValueError("No streams had enough data for even one training window")

    train_loader = DataLoader(
        MultiStreamLOBDataset(train_datasets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        MultiStreamLOBDataset(val_datasets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    ) if val_datasets else None

    test_loader = DataLoader(
        MultiStreamLOBDataset(test_datasets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    ) if test_datasets else None

    metadata = {
        "scalers": scalers,
        "stream_info": stream_info,
        "config": config,
        "n_train_windows": sum(len(ds) for ds in train_datasets),
        "n_val_windows": sum(len(ds) for ds in val_datasets),
        "n_test_windows": sum(len(ds) for ds in test_datasets),
    }

    return train_loader, val_loader, test_loader, metadata
