"""Tests for the V2 feature engineering pipeline (momentum features)."""

import sys
import logging
import numpy as np

sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.features import get_column_indices
from training.features_v2 import compute_momentum_features, engineer_features_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

N_LEVELS = 40
T = 200


def _make_fake_40level(T: int = 200, seed: int = 42) -> np.ndarray:
    """Create synthetic 40-level LOB data with shape (T, 162)."""
    np.random.seed(seed)
    idx = get_column_indices(N_LEVELS)
    data = np.zeros((T, N_LEVELS * 4 + 2), dtype=np.float64)

    mid = 70000.0 + np.cumsum(np.random.randn(T) * 5)

    for k in range(N_LEVELS):
        data[:, idx["bid_prices"][k]] = mid - (k + 1) * 0.5
        data[:, idx["ask_prices"][k]] = mid + (k + 1) * 0.5
        data[:, idx["bid_volumes"][k]] = np.abs(np.random.randn(T) * 10 + 5)
        data[:, idx["ask_volumes"][k]] = np.abs(np.random.randn(T) * 10 + 5)

    data[:, idx["mid_price"]] = mid
    data[:, idx["spread"]] = data[:, idx["ask_prices"][0]] - data[:, idx["bid_prices"][0]]
    return data


def test_momentum_features_shape():
    """compute_momentum_features returns 8 features, each (T,), no NaN."""
    data = _make_fake_40level(T)
    mom = compute_momentum_features(data, N_LEVELS)

    assert len(mom) == 8, f"Expected 8 momentum features, got {len(mom)}"
    for name, arr in mom.items():
        assert arr.shape == (T,), f"{name} shape {arr.shape} != ({T},)"
        assert not np.any(np.isnan(arr)), f"{name} contains NaN"

    logger.info("PASS: test_momentum_features_shape")


def test_momentum_features_names():
    """All 8 expected momentum feature names are present."""
    data = _make_fake_40level(T)
    mom = compute_momentum_features(data, N_LEVELS)

    expected = {
        "log_return_1", "log_return_6", "log_return_12", "log_return_60",
        "ofi_roc_6", "price_velocity_6", "price_acceleration_6",
        "realized_vol_12",
    }
    assert set(mom.keys()) == expected, f"Keys mismatch: {set(mom.keys())} vs {expected}"

    logger.info("PASS: test_momentum_features_names")


def test_engineer_features_v2_output_shape():
    """engineer_features_v2 produces (T, 219) features with no NaN."""
    data = _make_fake_40level(T)
    enriched, names = engineer_features_v2(data, N_LEVELS)

    assert enriched.shape == (T, 219), f"Shape {enriched.shape} != (200, 219)"
    assert not np.any(np.isnan(enriched)), "Enriched features contain NaN"

    logger.info(f"PASS: test_engineer_features_v2_output_shape ({enriched.shape})")


def test_savgol_window_reduced():
    """Default savgol_window for V2 is 11, not 21."""
    import inspect
    sig = inspect.signature(engineer_features_v2)
    default = sig.parameters["savgol_window"].default
    assert default == 11, f"savgol_window default is {default}, expected 11"

    logger.info("PASS: test_savgol_window_reduced")


if __name__ == "__main__":
    test_momentum_features_shape()
    test_momentum_features_names()
    test_engineer_features_v2_output_shape()
    test_savgol_window_reduced()

    logger.info("\n=== ALL V2 FEATURE TESTS PASSED ===")
