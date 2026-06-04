# training/test_integration_v2.py
import sys, logging, tempfile, numpy as np, pandas as pd, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.dataset import DataConfig, build_dataloaders

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def _write_stream(d, exchange, symbol, n_levels=5, T=600):
    cols = ["bucket"]
    for i in range(1, n_levels + 1):
        cols += [f"bid_price_{i}", f"bid_volume_{i}", f"ask_price_{i}", f"ask_volume_{i}"]
    cols += ["mid_price", "spread"]
    rng = np.random.default_rng(0)
    arr = np.cumsum(rng.normal(0, 0.01, size=(T, len(cols) - 1)), axis=0) + 100.0
    df = pd.DataFrame(arr, columns=cols[1:])
    df.insert(0, "bucket", pd.date_range("2026-03-12", periods=T, freq="5s"))
    df.to_parquet(f"{d}/{exchange}_{symbol}.parquet", index=False)

def test_parquet_pipeline_builds_v2_loaders():
    with tempfile.TemporaryDirectory() as d:
        _write_stream(d, "binance_perp", "BTC-USDT", n_levels=5, T=600)
        cfg = DataConfig(lob_levels=5, feature_version="v2", savgol_window=11,
                         source="parquet", parquet_dir=d,
                         exchanges=["binance_perp"], pairs=["BTC-USDT"],
                         context_length=20, prediction_length=24, stride=10)
        train, val, test, meta = build_dataloaders(cfg, batch_size=8)
        batch = next(iter(train))
        assert batch["context"].shape[-1] == 5 * 5 + 19  # 44 features at 5 levels
        assert meta["n_train_windows"] > 0
        logger.info("PASS: test_parquet_pipeline_builds_v2_loaders")
