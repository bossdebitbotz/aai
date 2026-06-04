"""
V2 Feature Engineering Pipeline for LOB data.

Extends the original features.py with momentum/micro-structure features:
- Log returns at multiple horizons
- OFI rate of change
- Price velocity & acceleration
- Realized volatility

Imports and reuses all functions from training.features.
"""

import numpy as np
from training.features import (
    get_column_indices,
    apply_savgol_smoothing,
    compute_ofi,
    compute_aggregate_ofi,
    compute_volume_features,
    compute_price_features,
    engineer_features,
)


# ---------------------------------------------------------------------------
# Momentum / Micro-structure Features
# ---------------------------------------------------------------------------

def compute_momentum_features(
    features: np.ndarray,
    n_levels: int,
) -> dict[str, np.ndarray]:
    """Compute momentum and micro-structure features from LOB data.

    Returns dict with 8 features, each of shape (T,):
        - log_return_1:  1-step log return of mid-price
        - log_return_6:  6-step log return
        - log_return_12: 12-step log return
        - log_return_60: 60-step log return
        - ofi_roc_6:     rate of change of aggregate OFI over 6 steps
        - price_velocity_6:    6-step price velocity (mid-price diff / 6)
        - price_acceleration_6: second derivative (velocity diff)
        - realized_vol_12:     rolling 12-step std of 1-step log returns
    """
    idx = get_column_indices(n_levels)
    mid = features[:, idx["mid_price"]]
    T = len(mid)

    result: dict[str, np.ndarray] = {}

    # --- Log returns at multiple horizons ---
    safe_mid = np.where(mid > 0, mid, 1.0)
    for horizon in (1, 6, 12, 60):
        lr = np.zeros(T, dtype=np.float64)
        if T > horizon:
            lr[horizon:] = np.log(safe_mid[horizon:] / safe_mid[:-horizon])
        result[f"log_return_{horizon}"] = lr

    # --- OFI rate of change ---
    ofi_levels = compute_ofi(features, n_levels)
    agg_ofi = compute_aggregate_ofi(ofi_levels)
    ofi_roc = np.zeros(T, dtype=np.float64)
    step = 6
    if T > step:
        ofi_roc[step:] = agg_ofi[step:] - agg_ofi[:-step]
    result["ofi_roc_6"] = ofi_roc

    # --- Price velocity (6-step) ---
    velocity = np.zeros(T, dtype=np.float64)
    if T > step:
        velocity[step:] = (mid[step:] - mid[:-step]) / step
    result["price_velocity_6"] = velocity

    # --- Price acceleration (derivative of velocity) ---
    acceleration = np.zeros(T, dtype=np.float64)
    acceleration[1:] = np.diff(velocity)
    result["price_acceleration_6"] = acceleration

    # --- Realized volatility (rolling 12-step std of 1-step returns) ---
    log_ret_1 = result["log_return_1"]
    window = 12
    realized_vol = np.zeros(T, dtype=np.float64)
    if T >= window:
        # Use cumsum trick for efficient rolling std
        cumsum = np.cumsum(log_ret_1)
        cumsum2 = np.cumsum(log_ret_1 ** 2)
        for t in range(window, T):
            s = cumsum[t] - cumsum[t - window]
            s2 = cumsum2[t] - cumsum2[t - window]
            var = s2 / window - (s / window) ** 2
            realized_vol[t] = np.sqrt(max(var, 0.0))
    result["realized_vol_12"] = realized_vol

    return result


# ---------------------------------------------------------------------------
# V2 Full Feature Engineering Pipeline
# ---------------------------------------------------------------------------

def engineer_features_v2(
    features: np.ndarray,
    n_levels: int,
    apply_smoothing: bool = True,
    savgol_window: int = 11,
    savgol_poly: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Apply V2 feature engineering pipeline to a single stream.

    Same as original engineer_features but:
    - savgol_window defaults to 11 (reduced from 21)
    - Appends 8 momentum features after the original derived features
    - Total for 40 levels: 162 base + 49 original derived + 8 momentum = 219

    Args:
        features: (T, F_base) raw feature array from DB
        n_levels: number of LOB levels
        apply_smoothing: whether to apply Savitzky-Golay smoothing
        savgol_window: SG window length (default 11, reduced from 21 in V1)
        savgol_poly: SG polynomial order

    Returns:
        enriched: (T, F_base + F_derived + F_momentum) enriched features
        derived_names: list of names for all derived columns (original + momentum)
    """
    # Delegate to original pipeline with the V2 savgol_window default
    enriched, derived_names = engineer_features(
        features, n_levels,
        apply_smoothing=apply_smoothing,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
    )

    T = len(features)
    if T == 0:
        return enriched, derived_names

    # Compute momentum features on the (possibly smoothed) enriched data
    # Use the base columns from enriched (first n_levels*4+2 columns)
    base_cols = n_levels * 4 + 2
    base_features = enriched[:, :base_cols]
    momentum = compute_momentum_features(base_features, n_levels)

    # Append momentum features
    momentum_arrays = []
    momentum_names = []
    for name, arr in momentum.items():
        momentum_arrays.append(arr)
        momentum_names.append(name)

    if momentum_arrays:
        momentum_block = np.column_stack(momentum_arrays)
        enriched = np.hstack([enriched, momentum_block])
        derived_names = derived_names + momentum_names

    return enriched, derived_names
