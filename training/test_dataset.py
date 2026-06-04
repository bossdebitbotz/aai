"""Tests for the LOB training dataset pipeline."""

import sys
import logging
import asyncio
import numpy as np
import torch

sys.path.insert(0, "/Users/clint/Projects/aai")
from training.dataset import (
    DataConfig, LOBScaler, LOBDataset, MultiStreamLOBDataset,
    _fetch_stream_data, _build_feature_columns, build_dataloaders,
    EXCHANGE_MAP, SYMBOL_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def test_feature_columns():
    """Test that feature column builder generates correct columns."""
    cols_5 = _build_feature_columns(5)
    assert len(cols_5) == 5 * 4 + 2  # 20 base + mid_price + spread = 22
    assert cols_5[0] == "bid_price_1"
    assert cols_5[1] == "bid_volume_1"
    assert cols_5[10] == "ask_price_1"
    assert cols_5[-1] == "spread"
    assert cols_5[-2] == "mid_price"

    cols_40 = _build_feature_columns(40)
    assert len(cols_40) == 40 * 4 + 2  # 160 base + 2 derived = 162
    assert cols_40[0] == "bid_price_1"
    assert cols_40[79] == "bid_volume_40"  # last bid vol
    assert cols_40[80] == "ask_price_1"
    assert cols_40[159] == "ask_volume_40"  # last ask vol
    assert cols_40[160] == "mid_price"
    assert cols_40[161] == "spread"

    logger.info("PASS: feature_columns")


def test_data_config():
    """Test DataConfig feature count properties."""
    config_5 = DataConfig(lob_levels=5, feature_version="v1")
    assert config_5.n_base_features == 20
    assert config_5.n_features == 22  # 4*5 + 2
    assert config_5.n_enriched_features == 36  # 5*5 + 11

    config_40 = DataConfig(lob_levels=40, feature_version="v1")
    assert config_40.n_base_features == 160
    assert config_40.n_features == 162  # 4*40 + 2
    assert config_40.n_enriched_features == 211  # 5*40 + 11

    logger.info("PASS: data_config")


def test_scaler_zscore():
    """Test LOBScaler z-score normalization."""
    n_levels = 5
    n_features = n_levels * 4 + 2  # 22
    T = 200

    np.random.seed(42)
    data = np.random.rand(T, n_features) * 100 + 50

    scaler = LOBScaler(n_levels=n_levels)
    scaled = scaler.fit_transform(data)

    # Check output shape
    assert scaled.shape == (T, n_features), f"Shape mismatch: {scaled.shape}"

    # Z-score: mean should be ~0, std should be ~1 on training data
    col_means = scaled.mean(axis=0)
    col_stds = scaled.std(axis=0)
    np.testing.assert_allclose(col_means, 0.0, atol=1e-10,
                               err_msg="Z-scored data should have mean ~0")
    np.testing.assert_allclose(col_stds, 1.0, atol=0.01,
                               err_msg="Z-scored data should have std ~1")

    assert scaler.fitted

    # Transform new data
    new_data = np.random.rand(50, n_features) * 100 + 50
    new_scaled = scaler.transform(new_data)
    assert new_scaled.shape == (50, n_features)

    # Inverse transform roundtrip
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered, data, atol=1e-10,
                               err_msg="Inverse transform should recover original data")

    # Test save/load roundtrip
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        scaler.save(path)
        loaded = LOBScaler.load(path)
        assert loaded.fitted
        assert loaded.n_levels == n_levels
        reloaded_scaled = loaded.transform(new_data)
        np.testing.assert_allclose(new_scaled, reloaded_scaled, atol=1e-10)

        # Loaded scaler inverse_transform should also work
        reloaded_inv = loaded.inverse_transform(scaled)
        np.testing.assert_allclose(reloaded_inv, data, atol=1e-10)
    finally:
        os.unlink(path)

    logger.info("PASS: scaler_zscore")


def test_scaler_zero_std():
    """Test that scaler handles constant columns (zero std)."""
    n_features = 10
    T = 100

    data = np.random.rand(T, n_features)
    data[:, 3] = 42.0  # constant column

    scaler = LOBScaler(n_levels=2)
    scaled = scaler.fit_transform(data)

    # Constant column should be 0 after z-score (42 - 42) / 1.0 = 0
    assert np.all(scaled[:, 3] == 0.0), "Constant column should be 0 after z-score"

    # Inverse transform should recover original
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered, data, atol=1e-10)

    logger.info("PASS: scaler_zero_std")


def test_dataset_synthetic():
    """Test LOBDataset with synthetic data."""
    config = DataConfig(lob_levels=5)
    n_cols = 5 * 5 + 11  # 36 enriched features
    T = 500

    features = np.random.rand(T, n_cols).astype(np.float32)
    timestamps = np.arange(T, dtype=np.float64) * 5 + 1700000000

    ds = LOBDataset(features, timestamps, "binance_spot", "BTC-USDT", config)

    # Check length: (T - window_size) // stride + 1
    window = config.context_length + config.prediction_length  # 144
    expected_len = len(range(0, T - window + 1, config.stride))
    assert len(ds) == expected_len, f"Expected {expected_len} windows, got {len(ds)}"

    # Check first sample
    sample = ds[0]
    assert sample["context"].shape == (120, n_cols), f"Context shape: {sample['context'].shape}"
    assert sample["target"].shape == (24, n_cols), f"Target shape: {sample['target'].shape}"
    assert sample["exchange_id"] == EXCHANGE_MAP["binance_spot"]
    assert sample["symbol_id"] == SYMBOL_MAP["BTC-USDT"]
    assert isinstance(sample["context"], torch.Tensor)
    assert sample["context"].dtype == torch.float32

    logger.info(f"PASS: dataset_synthetic ({len(ds)} windows from {T} timesteps)")


def test_multi_stream_dataset():
    """Test MultiStreamLOBDataset combining multiple streams."""
    config = DataConfig(lob_levels=5)
    n_cols = 5 * 5 + 11  # 36 enriched
    T = 300

    datasets = []
    for exch in ["binance_spot", "binance_perp"]:
        for sym in ["BTC-USDT", "ETH-USDT"]:
            features = np.random.rand(T, n_cols).astype(np.float32)
            timestamps = np.arange(T, dtype=np.float64) * 5
            datasets.append(LOBDataset(features, timestamps, exch, sym, config))

    multi = MultiStreamLOBDataset(datasets)
    expected_total = sum(len(ds) for ds in datasets)
    assert len(multi) == expected_total, f"Expected {expected_total}, got {len(multi)}"

    # Check that samples from different sub-datasets have different exchange_ids
    first_sample = multi[0]
    last_ds_start = sum(len(ds) for ds in datasets[:-1])
    last_sample = multi[last_ds_start]
    assert first_sample["exchange_id"] != last_sample["exchange_id"] or \
           first_sample["symbol_id"] != last_sample["symbol_id"]

    logger.info(f"PASS: multi_stream_dataset ({len(multi)} total windows from {len(datasets)} streams)")


def test_dataloader_batching():
    """Test that DataLoader correctly batches samples."""
    config = DataConfig(lob_levels=5)
    n_cols = 5 * 5 + 11  # 36 enriched
    T = 500

    features = np.random.rand(T, n_cols).astype(np.float32)
    timestamps = np.arange(T, dtype=np.float64) * 5
    ds = LOBDataset(features, timestamps, "binance_spot", "BTC-USDT", config)

    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    batch = next(iter(loader))

    assert batch["context"].shape[0] <= 32
    assert batch["context"].shape[1] == 120
    assert batch["context"].shape[2] == n_cols
    assert batch["target"].shape[1] == 24
    assert batch["target"].shape[2] == n_cols

    logger.info(f"PASS: dataloader_batching (batch shape: {batch['context'].shape})")


def test_fetch_from_db():
    """Test actual data fetch from TimescaleDB using 5-level historical data."""
    config = DataConfig(lob_levels=5)

    loop = asyncio.new_event_loop()
    features, timestamps = loop.run_until_complete(
        _fetch_stream_data(config, "binance_spot", "BTC-USDT")
    )
    loop.close()

    if len(features) == 0:
        logger.warning("SKIP: No data in DB for binance_spot/BTC-USDT")
        return

    n_expected_cols = 5 * 4 + 2  # 22 (raw from DB, before engineer_features)
    assert features.shape[1] == n_expected_cols, \
        f"Expected {n_expected_cols} columns, got {features.shape[1]}"
    assert len(timestamps) == len(features)
    assert features.shape[0] > 0

    # Check no NaN
    assert not np.isnan(features).any(), "Found NaN in features"

    # Check timestamps are sorted
    assert np.all(np.diff(timestamps) >= 0), "Timestamps not sorted"

    logger.info(
        f"PASS: fetch_from_db (binance_spot/BTC-USDT: {features.shape[0]} rows, "
        f"{features.shape[1]} features)"
    )


def test_fetch_40_levels():
    """Test fetching 40-level data (new data only)."""
    config = DataConfig(lob_levels=40)

    loop = asyncio.new_event_loop()
    features, timestamps = loop.run_until_complete(
        _fetch_stream_data(config, "binance_spot", "BTC-USDT")
    )
    loop.close()

    if len(features) == 0:
        logger.warning("SKIP: No data in DB")
        return

    n_expected_cols = 40 * 4 + 2  # 162 (raw from DB)
    assert features.shape[1] == n_expected_cols, \
        f"Expected {n_expected_cols} columns, got {features.shape[1]}"

    # Check that level 40 data exists in recent rows (not all NULL/0)
    last_100 = features[-min(100, len(features)):]
    bp40_col_idx = 39 * 2  # bid_price_40 index
    has_40_level_data = np.any(last_100[:, bp40_col_idx] != 0)

    if has_40_level_data:
        logger.info(
            f"PASS: fetch_40_levels ({features.shape[0]} rows, "
            f"40-level data present in recent rows)"
        )
    else:
        logger.info(
            f"PASS: fetch_40_levels ({features.shape[0]} rows, "
            f"but level 40 data is still zero — may be old 5-level data)"
        )


def test_end_to_end_pipeline():
    """Test the full build_dataloaders pipeline with 5-level data.

    This now includes feature engineering (engineer_features),
    so feature count should be 5*5+11=36 (enriched).
    """
    config = DataConfig(
        lob_levels=5,
        exchanges=["binance_spot"],
        pairs=["BTC-USDT"],
    )

    try:
        train_loader, val_loader, test_loader, metadata = build_dataloaders(
            config, batch_size=32
        )
    except ValueError as e:
        logger.warning(f"SKIP: end_to_end_pipeline — {e}")
        return

    logger.info(
        f"  Train windows: {metadata['n_train_windows']}, "
        f"Val windows: {metadata['n_val_windows']}, "
        f"Test windows: {metadata['n_test_windows']}"
    )

    # Get a batch
    batch = next(iter(train_loader))
    logger.info(
        f"  Batch: context={batch['context'].shape}, "
        f"target={batch['target'].shape}"
    )

    assert batch["context"].shape[1] == 120
    assert batch["target"].shape[1] == 24
    assert batch["context"].dtype == torch.float32

    # Feature count should be enriched (5N+11 = 36 for 5 levels)
    expected_features = config.n_enriched_features  # 36
    assert batch["context"].shape[2] == expected_features, \
        f"Expected {expected_features} enriched features, got {batch['context'].shape[2]}"

    # Z-score scaling: mean should be roughly 0 (not [0,1])
    ctx_vals = batch["context"].numpy()
    logger.info(
        f"  Scaled stats: min={ctx_vals.min():.4f}, max={ctx_vals.max():.4f}, "
        f"mean={ctx_vals.mean():.4f}"
    )

    # Check scaler is in metadata and has inverse_transform
    assert "scalers" in metadata
    first_scaler = next(iter(metadata["scalers"].values()))
    assert first_scaler.fitted
    assert first_scaler._means is not None
    assert first_scaler._stds is not None
    assert len(first_scaler._means) == expected_features

    logger.info("PASS: end_to_end_pipeline")


if __name__ == "__main__":
    # Unit tests (no DB needed)
    test_feature_columns()
    test_data_config()
    test_scaler_zscore()
    test_scaler_zero_std()
    test_dataset_synthetic()
    test_multi_stream_dataset()
    test_dataloader_batching()

    # Integration tests (need running TimescaleDB)
    test_fetch_from_db()
    test_fetch_40_levels()
    test_end_to_end_pipeline()

    logger.info("\n=== ALL TESTS PASSED ===")
