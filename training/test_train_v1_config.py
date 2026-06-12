# training/test_train_v1_config.py
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import build_data_config


def test_build_data_config_parquet_and_savgol():
    """Colab baseline run: parquet source + V2-matched savgol window, still V1 features."""
    args = Namespace(levels=40, exchanges=None, pairs=None,
                     source="parquet", parquet_dir="lob_data", savgol_window=11)
    cfg = build_data_config(args)
    assert cfg.feature_version == "v1"
    assert cfg.lob_levels == 40
    assert cfg.source == "parquet"
    assert cfg.parquet_dir == "lob_data"
    assert cfg.savgol_window == 11


def test_build_data_config_defaults_preserve_legacy_v1():
    """Default invocation must keep historic V1 behavior: db source, savgol 21."""
    args = Namespace(levels=40, exchanges="binance_spot", pairs="BTC-USDT",
                     source="db", parquet_dir="lob_data", savgol_window=21)
    cfg = build_data_config(args)
    assert cfg.feature_version == "v1"
    assert cfg.source == "db"
    assert cfg.savgol_window == 21
    assert cfg.exchanges == ["binance_spot"]
    assert cfg.pairs == ["BTC-USDT"]
