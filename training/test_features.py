"""Tests for the feature engineering pipeline."""

import sys
import logging
import numpy as np

sys.path.insert(0, "/Users/clint/Projects/aai")
from training.features import (
    get_column_indices, apply_savgol_smoothing, compute_ofi,
    compute_aggregate_ofi, compute_volume_features, compute_price_features,
    compute_cross_exchange_features, engineer_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _make_synthetic_lob(n_levels: int, T: int, seed: int = 42) -> np.ndarray:
    """Create realistic synthetic LOB data."""
    np.random.seed(seed)
    n_cols = n_levels * 4 + 2  # bid_p, bid_v, ask_p, ask_v per level + mid + spread

    data = np.zeros((T, n_cols), dtype=np.float64)
    idx = get_column_indices(n_levels)

    # Generate realistic prices
    mid = 70000.0 + np.cumsum(np.random.randn(T) * 5)  # BTC-like

    for k in range(n_levels):
        data[:, idx["bid_prices"][k]] = mid - (k + 1) * 0.5  # descending bids
        data[:, idx["ask_prices"][k]] = mid + (k + 1) * 0.5  # ascending asks
        data[:, idx["bid_volumes"][k]] = np.abs(np.random.randn(T) * 10 + 5)
        data[:, idx["ask_volumes"][k]] = np.abs(np.random.randn(T) * 10 + 5)

    data[:, idx["mid_price"]] = mid
    data[:, idx["spread"]] = data[:, idx["ask_prices"][0]] - data[:, idx["bid_prices"][0]]

    return data


def test_column_indices():
    """Test column index mapping."""
    idx = get_column_indices(5)
    assert len(idx["bid_prices"]) == 5
    assert len(idx["ask_prices"]) == 5
    assert idx["bid_prices"][0] == 0      # bid_price_1
    assert idx["bid_volumes"][0] == 1     # bid_volume_1
    assert idx["ask_prices"][0] == 10     # ask_price_1 (after 5*2=10 bid cols)
    assert idx["mid_price"] == 20         # after 5*4=20 bid+ask cols
    assert idx["spread"] == 21

    idx40 = get_column_indices(40)
    assert len(idx40["bid_prices"]) == 40
    assert idx40["ask_prices"][0] == 80   # after 40*2=80 bid cols
    assert idx40["mid_price"] == 160
    assert idx40["spread"] == 161

    logger.info("PASS: column_indices")


def test_savgol_smoothing():
    """Test Savitzky-Golay smoothing."""
    data = _make_synthetic_lob(5, 200)
    idx = get_column_indices(5)

    smoothed = apply_savgol_smoothing(data, n_levels=5, window_length=21, polyorder=3)

    # Shape preserved
    assert smoothed.shape == data.shape

    # Price columns should be smoothed (different from original)
    bp1_orig = data[:, idx["bid_prices"][0]]
    bp1_smooth = smoothed[:, idx["bid_prices"][0]]
    assert not np.allclose(bp1_orig, bp1_smooth), "Smoothing had no effect"

    # Volume columns should be unchanged
    bv1_orig = data[:, idx["bid_volumes"][0]]
    bv1_smooth = smoothed[:, idx["bid_volumes"][0]]
    np.testing.assert_array_equal(bv1_orig, bv1_smooth)

    # Smoothed should have less variance
    assert bp1_smooth.std() <= bp1_orig.std() * 1.01  # allow tiny float tolerance

    # Short data should return copy without error
    short = _make_synthetic_lob(5, 10)
    result = apply_savgol_smoothing(short, 5, window_length=21)
    assert result.shape == short.shape
    np.testing.assert_array_equal(result, short)

    logger.info("PASS: savgol_smoothing")


def test_ofi():
    """Test Order Flow Imbalance computation."""
    data = _make_synthetic_lob(5, 100)
    ofi = compute_ofi(data, n_levels=5)

    assert ofi.shape == (100, 5), f"OFI shape: {ofi.shape}"

    # First row should be zero (no previous state)
    assert ofi[0, :].sum() == 0.0

    # OFI should be finite
    assert np.all(np.isfinite(ofi))

    # Aggregate
    agg = compute_aggregate_ofi(ofi)
    assert agg.shape == (100,)
    np.testing.assert_allclose(agg, ofi.sum(axis=1))

    logger.info("PASS: ofi (shape: {})".format(ofi.shape))


def test_ofi_40_levels():
    """Test OFI with 40 levels."""
    data = _make_synthetic_lob(40, 100)
    ofi = compute_ofi(data, n_levels=40)
    assert ofi.shape == (100, 40)
    assert np.all(np.isfinite(ofi))

    logger.info("PASS: ofi_40_levels")


def test_volume_features():
    """Test volume feature computation."""
    data = _make_synthetic_lob(5, 100)
    vol = compute_volume_features(data, n_levels=5)

    assert "cumulative_bid_volume" in vol
    assert "cumulative_ask_volume" in vol
    assert "volume_ratio" in vol
    assert "volume_imbalance_total" in vol

    for name, arr in vol.items():
        assert arr.shape == (100,), f"{name} shape: {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{name} has non-finite values"

    # Volume ratio should be positive
    assert np.all(vol["volume_ratio"] > 0)

    # Imbalance should be in [-1, 1]
    assert np.all(vol["volume_imbalance_total"] >= -1.0)
    assert np.all(vol["volume_imbalance_total"] <= 1.0)

    logger.info("PASS: volume_features")


def test_price_features():
    """Test price feature computation."""
    data = _make_synthetic_lob(5, 100)
    pf = compute_price_features(data, n_levels=5)

    assert "price_imbalance" in pf
    assert "spread_ratio" in pf
    assert "depth_bid" in pf
    assert "depth_ask" in pf

    for name, arr in pf.items():
        assert arr.shape == (100,), f"{name} shape: {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{name} has non-finite values"

    # Price imbalance should be positive (ask > bid)
    assert np.all(pf["price_imbalance"] > 0)

    # Spread ratio should be positive
    assert np.all(pf["spread_ratio"] > 0)

    # Depth should be positive (our synthetic data has proper ordering)
    assert np.all(pf["depth_bid"] > 0)
    assert np.all(pf["depth_ask"] > 0)

    logger.info("PASS: price_features")


def test_cross_exchange_features():
    """Test cross-exchange feature computation."""
    n_levels = 5
    T = 100

    # Create aligned data for multiple streams
    stream_data = {
        "binance_spot_BTC-USDT": _make_synthetic_lob(n_levels, T, seed=1),
        "binance_perp_BTC-USDT": _make_synthetic_lob(n_levels, T, seed=2),
        "bybit_spot_BTC-USDT": _make_synthetic_lob(n_levels, T, seed=3),
    }

    cross = compute_cross_exchange_features(stream_data, n_levels, "binance_spot")

    assert len(cross) > 0, "No cross-exchange features computed"

    for name, arr in cross.items():
        assert arr.shape == (T,), f"{name} shape: {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{name} has non-finite values"

    # Check that price diffs exist for binance_spot vs others
    diff_keys = [k for k in cross if "mid_price_diff_" in k and "bps" not in k]
    assert len(diff_keys) >= 2, f"Expected >=2 price diff features, got {len(diff_keys)}"

    logger.info(f"PASS: cross_exchange_features ({len(cross)} features)")


def test_engineer_features_5_levels():
    """Test full pipeline with 5 levels."""
    data = _make_synthetic_lob(5, 200)
    enriched, names = engineer_features(data, n_levels=5)

    expected_base = 5 * 4 + 2  # 22
    expected_derived = 5 + 1 + 4 + 4  # 5 OFI levels + 1 aggregate + 4 volume + 4 price = 14
    expected_total = expected_base + expected_derived

    assert enriched.shape == (200, expected_total), \
        f"Expected shape (200, {expected_total}), got {enriched.shape}"
    assert len(names) == expected_derived, \
        f"Expected {expected_derived} derived names, got {len(names)}"

    # Check names
    assert "ofi_level_1" in names
    assert "ofi_aggregate" in names
    assert "cumulative_bid_volume" in names
    assert "price_imbalance" in names
    assert "depth_bid" in names

    # Check all finite
    assert np.all(np.isfinite(enriched)), "Enriched features contain non-finite values"

    logger.info(f"PASS: engineer_features_5_levels ({enriched.shape[1]} total features)")


def test_engineer_features_40_levels():
    """Test full pipeline with 40 levels."""
    data = _make_synthetic_lob(40, 200)
    enriched, names = engineer_features(data, n_levels=40)

    expected_base = 40 * 4 + 2  # 162
    expected_derived = 40 + 1 + 4 + 4  # 40 OFI + 1 aggregate + 4 volume + 4 price = 49
    expected_total = expected_base + expected_derived

    assert enriched.shape == (200, expected_total), \
        f"Expected shape (200, {expected_total}), got {enriched.shape}"
    assert len(names) == expected_derived

    assert np.all(np.isfinite(enriched))

    logger.info(f"PASS: engineer_features_40_levels ({enriched.shape[1]} total features)")


def test_engineer_features_no_smoothing():
    """Test pipeline without smoothing."""
    data = _make_synthetic_lob(5, 200)
    enriched, names = engineer_features(data, n_levels=5, apply_smoothing=False)

    # Base columns should be unchanged
    np.testing.assert_array_equal(enriched[:, :22], data)

    logger.info("PASS: engineer_features_no_smoothing")


def test_pipeline_with_real_data():
    """Test pipeline with actual DB data if available."""
    import asyncio
    from training.dataset import _fetch_stream_data, DataConfig

    config = DataConfig(lob_levels=5)
    loop = asyncio.new_event_loop()
    features, timestamps = loop.run_until_complete(
        _fetch_stream_data(config, "binance_spot", "BTC-USDT")
    )
    loop.close()

    if len(features) < 30:
        logger.warning("SKIP: pipeline_with_real_data — not enough DB data")
        return

    enriched, names = engineer_features(features, n_levels=5)

    assert enriched.shape[0] == features.shape[0]
    assert enriched.shape[1] > features.shape[1]
    assert np.all(np.isfinite(enriched)), "Real data enriched features have non-finite values"

    logger.info(
        f"PASS: pipeline_with_real_data "
        f"({features.shape[0]} rows, {features.shape[1]} -> {enriched.shape[1]} features)"
    )


if __name__ == "__main__":
    test_column_indices()
    test_savgol_smoothing()
    test_ofi()
    test_ofi_40_levels()
    test_volume_features()
    test_price_features()
    test_cross_exchange_features()
    test_engineer_features_5_levels()
    test_engineer_features_40_levels()
    test_engineer_features_no_smoothing()
    test_pipeline_with_real_data()

    logger.info("\n=== ALL FEATURE TESTS PASSED ===")
