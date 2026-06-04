"""Parquet-backed data source (Colab cannot reach the local TimescaleDB)."""
from pathlib import Path
import numpy as np
import pandas as pd

from training.dataset import _build_feature_columns


def fetch_parquet_stream(parquet_dir, exchange, symbol, n_levels):
    """Return (features (T, 4N+2) float64, timestamps (T,) float64) from an exported parquet."""
    path = Path(parquet_dir) / f"{exchange}_{symbol}.parquet"
    df = pd.read_parquet(path)
    feature_cols = _build_feature_columns(n_levels)
    feats = df[feature_cols].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(df["bucket"]).astype("int64").to_numpy() / 1e9  # unix seconds
    return feats, ts
