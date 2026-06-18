#!/usr/bin/env python3
"""Feature-parity diagnostic for the paper-trading harness.

The model was trained on features smoothed with a CENTERED Savitzky-Golay filter
(scipy default, window=11 -> uses +/-5 future buckets). A live harness can't see
the future. This script proves that computing features on a trailing buffer and
reading the bucket LAG steps back from "now" reproduces the training features
bit-for-bit (because the centered SG window for that bucket is then fully
available). It also shows how badly a no-lag (LAG=0) computation diverges.

Run: .venv/bin/python executor/paper/feature_parity_check.py
"""
import sys
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
import pandas as pd
from training.features_v2 import engineer_features_v2

N_LEVELS = 40
SAVGOL = 11
BUF = 240          # trailing buffer length (buckets) the live harness keeps
MID_IDX = 160

def raw_from_parquet(path):
    df = pd.read_parquet(path)
    return df.drop(columns=["bucket"]).to_numpy(np.float64)  # (T, 162): LOB(160)+mid+spread

def live_decision_features(raw, n, lag):
    """What the harness computes at real-time index n, deciding for bucket n-lag."""
    buf = raw[n - BUF: n + 1]                                  # trailing buffer ending at now=n
    feats, _ = engineer_features_v2(buf, N_LEVELS, apply_smoothing=True, savgol_window=SAVGOL)
    return feats[BUF - lag]                                    # row for decision bucket (n-lag)

def main():
    path = "/Volumes/Docker-SSD/projects/aaiwdbback/aai/lob_data/binance_perp_BTC-USDT.parquet"
    raw = raw_from_parquet(path)
    # drop non-finite rows (same data-quality filter as training)
    raw = raw[np.isfinite(raw).all(axis=1)]
    print(f"stream rows: {len(raw)}")

    # ground truth: training-style features over the FULL stream (centered SG)
    train_feats, names = engineer_features_v2(raw, N_LEVELS, apply_smoothing=True, savgol_window=SAVGOL)
    F = train_feats.shape[1]
    print(f"features: {F}  (expect 219)")

    # sample interior decision points (need BUF history + room)
    rng = np.arange(BUF + 10, len(raw) - 10, max(1, (len(raw) - BUF - 20) // 200))[:200]

    for lag in (0, 6):
        diffs = []
        for n in rng:
            live = live_decision_features(raw, n, lag)
            truth = train_feats[n - lag]
            # relative diff on a robust scale (per-feature) -> use abs diff, report max & where
            diffs.append(np.abs(live - truth))
        D = np.vstack(diffs)                                   # (n_points, F)
        max_abs = D.max()
        worst_feat = int(D.max(axis=0).argmax())
        # focus on smoothed mid + momentum block (last 8 cols) which are the risk
        mid_max = D[:, MID_IDX].max()
        mom_max = D[:, -8:].max()
        print(f"\nLAG={lag} ({lag*5}s behind real-time): "
              f"max|Δ| any feature = {max_abs:.3e} (feat #{worst_feat}={names[worst_feat-(F-len(names))] if worst_feat>=F-len(names) else 'base/raw'}) | "
              f"mid Δ = {mid_max:.3e} | momentum-block Δ = {mom_max:.3e}")
    print("\nInterpretation: LAG=6 should be ~numerical-zero (bit-exact parity); "
          "LAG=0 should diverge on the smoothed/momentum features (the savgol edge effect).")

if __name__ == "__main__":
    main()
