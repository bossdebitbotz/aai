"""Parquet-backed data source (Colab cannot reach the local TimescaleDB)."""
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

from training.dataset import _build_feature_columns


def fetch_parquet_stream(parquet_dir, exchange, symbol, n_levels,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None):
    """Return (features (T, 4N+2) float64, timestamps (T,) float64) from an exported parquet.

    Optional [start_time, end_time) bounds filter on the `bucket` column (half-open,
    matching the DB path's `bucket >= start AND bucket < end`)."""
    path = Path(parquet_dir) / f"{exchange}_{symbol}.parquet"
    df = pd.read_parquet(path)
    buckets = pd.to_datetime(df["bucket"], utc=True)
    if start_time is not None:
        df = df[buckets >= pd.Timestamp(start_time)]
        buckets = buckets[buckets >= pd.Timestamp(start_time)]
    if end_time is not None:
        mask = buckets < pd.Timestamp(end_time)
        df = df[mask]
    feature_cols = _build_feature_columns(n_levels)
    feats = df[feature_cols].to_numpy(dtype=np.float64)
    # unix seconds, resolution-robust: pandas datetime64 is ns-backed (astype int64 / 1e6 gave
    # MILLISECONDS -> 1000x too large -> gap-detection rejected every window). Divide by a
    # Timedelta so it's correct regardless of ns/us backend.
    ts = ((pd.to_datetime(df["bucket"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
          / pd.Timedelta(seconds=1)).to_numpy()
    finite_rows = np.isfinite(feats).all(axis=1)
    feats = feats[finite_rows]
    ts = ts[finite_rows]
    return feats, ts
