"""
Feature Engineering Pipeline for LOB data.

Implements:
1. Savitzky-Golay smoothing for price columns
2. Multi-level Order Flow Imbalance (OFI)
3. Volume imbalance, cumulative volumes
4. Cross-exchange features (price differential, spread ratio)
5. Price imbalance, spread ratio
"""

import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Feature column index helpers
# ---------------------------------------------------------------------------

def get_column_indices(n_levels: int) -> dict:
    """Return named index ranges for the LOB feature array.

    Feature layout from dataset.py:
        bid_price_1, bid_volume_1, bid_price_2, bid_volume_2, ..., (n_levels pairs)
        ask_price_1, ask_volume_1, ask_price_2, ask_volume_2, ..., (n_levels pairs)
        mid_price, spread
    """
    bid_prices = [i * 2 for i in range(n_levels)]
    bid_volumes = [i * 2 + 1 for i in range(n_levels)]

    offset = n_levels * 2
    ask_prices = [offset + i * 2 for i in range(n_levels)]
    ask_volumes = [offset + i * 2 + 1 for i in range(n_levels)]

    base = n_levels * 4
    mid_price_idx = base
    spread_idx = base + 1

    return {
        "bid_prices": bid_prices,
        "bid_volumes": bid_volumes,
        "ask_prices": ask_prices,
        "ask_volumes": ask_volumes,
        "mid_price": mid_price_idx,
        "spread": spread_idx,
    }


# ---------------------------------------------------------------------------
# 1. Savitzky-Golay Smoothing
# ---------------------------------------------------------------------------

def apply_savgol_smoothing(
    features: np.ndarray,
    n_levels: int,
    window_length: int = 21,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to price columns.

    Research shows this is the single highest-impact preprocessing step
    for LOB prediction (arXiv:2506.05764).

    Args:
        features: (T, F) raw feature array
        n_levels: number of LOB levels
        window_length: SG window (must be odd). 21 = ~105s at 5s intervals
        polyorder: polynomial order for SG filter

    Returns:
        smoothed: (T, F) with price columns smoothed
    """
    if len(features) < window_length:
        return features.copy()

    idx = get_column_indices(n_levels)
    smoothed = features.copy()

    # Smooth all price columns
    price_cols = idx["bid_prices"] + idx["ask_prices"] + [idx["mid_price"], idx["spread"]]
    for col in price_cols:
        smoothed[:, col] = savgol_filter(features[:, col], window_length, polyorder)

    return smoothed


# ---------------------------------------------------------------------------
# 2. Order Flow Imbalance (OFI)
# ---------------------------------------------------------------------------

def compute_ofi(features: np.ndarray, n_levels: int) -> np.ndarray:
    """Compute multi-level Order Flow Imbalance.

    OFI measures the net buying/selling pressure from LOB changes.
    For each level k:
        OFI_k(t) = Δbid_volume_k(t) * I(bid_price_k(t) >= bid_price_k(t-1))
                 - Δask_volume_k(t) * I(ask_price_k(t) <= ask_price_k(t-1))

    Where Δ denotes change from previous timestep and I is indicator function.

    Research: Multi-level OFI reduces RMSE by 68-74% (Kolm & Turiel, 2023).

    Args:
        features: (T, F) raw feature array
        n_levels: number of LOB levels

    Returns:
        ofi: (T, n_levels) OFI per level. First row is zero.
    """
    idx = get_column_indices(n_levels)
    T = len(features)
    ofi = np.zeros((T, n_levels), dtype=np.float64)

    for k in range(n_levels):
        bp = features[:, idx["bid_prices"][k]]
        bv = features[:, idx["bid_volumes"][k]]
        ap = features[:, idx["ask_prices"][k]]
        av = features[:, idx["ask_volumes"][k]]

        # Changes
        d_bv = np.diff(bv, prepend=bv[0])
        d_av = np.diff(av, prepend=av[0])
        d_bp = np.diff(bp, prepend=bp[0])
        d_ap = np.diff(ap, prepend=ap[0])

        # Bid side: volume increase when price held or improved
        bid_ofi = np.where(d_bp >= 0, d_bv, -bv)
        # Ask side: volume decrease when price held or improved (lower)
        ask_ofi = np.where(d_ap <= 0, d_av, -av)

        ofi[:, k] = bid_ofi - ask_ofi

    return ofi


def compute_aggregate_ofi(ofi_levels: np.ndarray) -> np.ndarray:
    """Aggregate multi-level OFI into a single signal.

    Args:
        ofi_levels: (T, n_levels) per-level OFI

    Returns:
        agg_ofi: (T,) aggregated OFI (sum across levels)
    """
    return ofi_levels.sum(axis=1)


# ---------------------------------------------------------------------------
# 3. Volume Features
# ---------------------------------------------------------------------------

def compute_volume_features(features: np.ndarray, n_levels: int) -> dict[str, np.ndarray]:
    """Compute volume-based derived features.

    Returns:
        dict with keys:
            - cumulative_bid_volume: (T,) sum of all bid volumes
            - cumulative_ask_volume: (T,) sum of all ask volumes
            - volume_ratio: (T,) bid/ask volume ratio
            - volume_imbalance_total: (T,) (bid - ask) / (bid + ask)
    """
    idx = get_column_indices(n_levels)

    cum_bid = sum(features[:, i] for i in idx["bid_volumes"])
    cum_ask = sum(features[:, i] for i in idx["ask_volumes"])

    denom = cum_bid + cum_ask
    safe_denom = np.where(denom > 0, denom, 1.0)

    return {
        "cumulative_bid_volume": cum_bid,
        "cumulative_ask_volume": cum_ask,
        "volume_ratio": np.where(cum_ask > 0, cum_bid / cum_ask, 1.0),
        "volume_imbalance_total": (cum_bid - cum_ask) / safe_denom,
    }


# ---------------------------------------------------------------------------
# 4. Price Features
# ---------------------------------------------------------------------------

def compute_price_features(features: np.ndarray, n_levels: int) -> dict[str, np.ndarray]:
    """Compute price-based derived features.

    Returns:
        dict with keys:
            - price_imbalance: (T,) (ask1 - bid1) / (ask1 + bid1)
            - spread_ratio: (T,) spread / mid_price
            - depth_bid: (T,) bid_price_1 - bid_price_N
            - depth_ask: (T,) ask_price_N - ask_price_1
    """
    idx = get_column_indices(n_levels)

    bp1 = features[:, idx["bid_prices"][0]]
    ap1 = features[:, idx["ask_prices"][0]]
    mid = features[:, idx["mid_price"]]
    spread = features[:, idx["spread"]]

    denom = ap1 + bp1
    safe_denom = np.where(denom > 0, denom, 1.0)
    safe_mid = np.where(mid > 0, mid, 1.0)

    result = {
        "price_imbalance": (ap1 - bp1) / safe_denom,
        "spread_ratio": spread / safe_mid,
    }

    # Depth spread (price range across all levels)
    if n_levels > 1:
        bp_last = features[:, idx["bid_prices"][-1]]
        ap_last = features[:, idx["ask_prices"][-1]]
        result["depth_bid"] = bp1 - bp_last
        result["depth_ask"] = ap_last - ap1

    return result


# ---------------------------------------------------------------------------
# 5. Cross-Exchange Features
# ---------------------------------------------------------------------------

def compute_cross_exchange_features(
    stream_data: dict[str, np.ndarray],
    n_levels: int,
    reference_exchange: str = "binance_spot",
) -> dict[str, np.ndarray]:
    """Compute cross-exchange features.

    Args:
        stream_data: dict mapping "exchange_symbol" -> (T, F) aligned feature arrays
                     All arrays must have the same length T (time-aligned).
        n_levels: number of LOB levels
        reference_exchange: base exchange for differentials

    Returns:
        dict with cross-exchange feature arrays
    """
    idx = get_column_indices(n_levels)
    result = {}

    # Find reference streams
    ref_streams = {k: v for k, v in stream_data.items() if k.startswith(reference_exchange)}

    for ref_key, ref_data in ref_streams.items():
        ref_symbol = ref_key.split("_", 2)[-1] if "_" in ref_key else ref_key
        ref_mid = ref_data[:, idx["mid_price"]]
        ref_spread = ref_data[:, idx["spread"]]

        for other_key, other_data in stream_data.items():
            if other_key == ref_key:
                continue
            other_symbol = other_key.split("_", 2)[-1] if "_" in other_key else other_key
            if other_symbol != ref_symbol:
                continue

            other_mid = other_data[:, idx["mid_price"]]
            other_spread = other_data[:, idx["spread"]]

            safe_ref_mid = np.where(ref_mid > 0, ref_mid, 1.0)

            pair_name = f"{ref_key}_vs_{other_key}"
            result[f"mid_price_diff_{pair_name}"] = ref_mid - other_mid
            result[f"mid_price_diff_bps_{pair_name}"] = \
                (ref_mid - other_mid) / safe_ref_mid * 10000
            result[f"spread_ratio_{pair_name}"] = np.where(
                other_spread > 0, ref_spread / other_spread, 1.0
            )

    return result


# ---------------------------------------------------------------------------
# Full Feature Engineering Pipeline
# ---------------------------------------------------------------------------

def engineer_features(
    features: np.ndarray,
    n_levels: int,
    apply_smoothing: bool = True,
    savgol_window: int = 21,
    savgol_poly: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Apply full feature engineering pipeline to a single stream.

    Takes raw (T, F_base) features and returns (T, F_base + F_derived) features.

    Args:
        features: (T, F_base) raw feature array from DB
        n_levels: number of LOB levels
        apply_smoothing: whether to apply Savitzky-Golay smoothing
        savgol_window: SG window length
        savgol_poly: SG polynomial order

    Returns:
        enriched: (T, F_base + F_derived) enriched features
        derived_names: list of names for the derived columns
    """
    T = len(features)
    if T == 0:
        return features, []

    # Step 1: Savitzky-Golay smoothing on prices
    if apply_smoothing and T >= savgol_window:
        features = apply_savgol_smoothing(features, n_levels, savgol_window, savgol_poly)

    derived_arrays = []
    derived_names = []

    # Step 2: Multi-level OFI
    ofi = compute_ofi(features, n_levels)
    for k in range(n_levels):
        derived_arrays.append(ofi[:, k])
        derived_names.append(f"ofi_level_{k+1}")

    # Aggregate OFI
    agg_ofi = compute_aggregate_ofi(ofi)
    derived_arrays.append(agg_ofi)
    derived_names.append("ofi_aggregate")

    # Step 3: Volume features
    vol_feats = compute_volume_features(features, n_levels)
    for name, arr in vol_feats.items():
        derived_arrays.append(arr)
        derived_names.append(name)

    # Step 4: Price features
    price_feats = compute_price_features(features, n_levels)
    for name, arr in price_feats.items():
        derived_arrays.append(arr)
        derived_names.append(name)

    # Stack derived features
    derived = np.column_stack(derived_arrays) if derived_arrays else np.empty((T, 0))

    enriched = np.hstack([features, derived])
    return enriched, derived_names
