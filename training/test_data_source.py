# training/test_data_source.py
import sys, logging, numpy as np, pandas as pd
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.data_source import fetch_parquet_stream

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_fetch_parquet_stream(tmp_path):
    n_levels = 40
    cols = ["bucket"]
    for i in range(1, n_levels + 1):
        cols += [f"bid_price_{i}", f"bid_volume_{i}", f"ask_price_{i}", f"ask_volume_{i}"]
    cols += ["mid_price", "spread"]
    df = pd.DataFrame(np.random.rand(50, len(cols)) + 1.0, columns=cols)
    df["bucket"] = pd.date_range("2026-03-12", periods=50, freq="5s")
    f = tmp_path / "binance_perp_BTC-USDT.parquet"
    df.to_parquet(f, index=False)
    feats, ts = fetch_parquet_stream(str(tmp_path), "binance_perp", "BTC-USDT", n_levels)
    assert feats.shape == (50, n_levels * 4 + 2)
    assert ts.shape == (50,)
    logger.info("PASS: test_fetch_parquet_stream")


def test_fetch_parquet_drops_nonfinite_rows(tmp_path):
    n_levels = 40
    cols = ["bucket"]
    for i in range(1, n_levels + 1):
        cols += [f"bid_price_{i}", f"bid_volume_{i}", f"ask_price_{i}", f"ask_volume_{i}"]
    cols += ["mid_price", "spread"]
    df = pd.DataFrame(np.random.rand(50, len(cols)) + 1.0, columns=cols)
    df["bucket"] = pd.date_range("2026-03-12", periods=50, freq="5s")
    # Corrupt two feature rows: one NaN, one inf.
    df.loc[7, "bid_price_3"] = np.nan
    df.loc[20, "ask_volume_10"] = np.inf
    f = tmp_path / "binance_perp_BTC-USDT.parquet"
    df.to_parquet(f, index=False)
    feats, ts = fetch_parquet_stream(str(tmp_path), "binance_perp", "BTC-USDT", n_levels)
    assert feats.shape == (48, n_levels * 4 + 2)
    assert ts.shape == (48,)
    assert np.isfinite(feats).all()
    logger.info("PASS: test_fetch_parquet_drops_nonfinite_rows")
