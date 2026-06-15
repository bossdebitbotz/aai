# training/test_walkforward.py
import sys, datetime as dt
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
import pandas as pd
from training.data_source import fetch_parquet_stream


def test_parquet_time_bounds(tmp_path):
    # tiny synthetic parquet with the columns _build_feature_columns(1) expects
    from training.dataset import _build_feature_columns
    cols = _build_feature_columns(1)              # 1 level -> bid/ask price+vol + mid + spread
    n = 100
    base = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    df = pd.DataFrame({c: np.arange(n, dtype=float) + 1.0 for c in cols})
    df["bucket"] = [base + dt.timedelta(seconds=5 * i) for i in range(n)]
    p = tmp_path / "binance_perp_BTC-USDT.parquet"
    df.to_parquet(p)

    # unbounded -> all rows
    feats, ts = fetch_parquet_stream(str(tmp_path), "binance_perp", "BTC-USDT", 1)
    assert len(feats) == n
    # bounded [t10, t40) -> rows 10..39
    start = base + dt.timedelta(seconds=5 * 10)
    end = base + dt.timedelta(seconds=5 * 40)
    feats_b, ts_b = fetch_parquet_stream(str(tmp_path), "binance_perp", "BTC-USDT", 1,
                                         start_time=start, end_time=end)
    assert len(feats_b) == 30
    assert ts_b[0] == start.timestamp()
    assert ts_b[-1] == (end - dt.timedelta(seconds=5)).timestamp()
