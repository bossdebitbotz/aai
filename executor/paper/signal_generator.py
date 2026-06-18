#!/usr/bin/env python3
"""Live signal generator: trailing raw buffer -> features (30s lag for parity) ->
saved-scaler scale -> 120-window -> model -> 3 head signs -> FROZEN gate stack.

Stateless and buffer-based so it's identical for live and replay/parity. The
decision bucket is DECISION_LAG buckets back from the end of the buffer ("now"),
which makes the centered Savitzky-Golay window for that bucket fully available ->
features bit-exact to training (proven in feature_parity_check.py).
"""
from __future__ import annotations
import sys, json, pickle, functools
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np, torch
from training.features_v2 import engineer_features_v2
from training.model_v2 import CompoundAttentionModelV2
from executor.paper import strategy as S
from executor.paper import db_source as DBS

RUN = "/Volumes/Docker-SSD/projects/aaiwdbback/aai/experiments/v2_balanced"
N_LEVELS, MID, SPR, CTX, STRIDE = 40, 160, 161, 120, 23
LAG = S.DECISION_LAG                      # 6 buckets (~30s)
HEAD_HORIZON = (0, 1, 2)                  # 30s / 1m / 2m heads
# history needed before the decision bucket: max(context+momentum-warmup, ER lookback)
HIST_NEEDED = max(CTX + 60, (S.ER_KER + 1) * STRIDE) + 5
MIN_BUFFER = HIST_NEEDED + LAG + 1
EX_ID = {"binance_spot": 0, "binance_perp": 1, "bybit_spot": 2, "kucoin_spot": 3}
SY_ID = {"BTC-USDT": 0, "ETH-USDT": 1, "SOL-USDT": 2, "WLD-USDT": 3}


@functools.lru_cache(maxsize=1)
def _load():
    cfg = json.load(open(f"{RUN}/config.json"))
    ck = torch.load(f"{RUN}/checkpoints/best.pt", map_location="cpu", weights_only=False)
    pl = ck.get("prediction_length", cfg["prediction_length"])
    m = CompoundAttentionModelV2(n_levels=40, n_features=cfg["n_features"], context_length=cfg["context_length"],
        prediction_length=pl, d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"], dropout=cfg.get("dropout", 0.1))
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    scalers = pickle.load(open(f"{RUN}/scalers.pkl", "rb"))
    return m, scalers


MAX_BUCKET_GAP_S = 12.5          # matches training (5s * 2.5): a context with a bigger jump is rejected
ER_SPAN_TOL = 1.5                # decision-mid lookback span must be <= 1.5x expected (catches restart gap)


def _gap_ok(buckets, didx) -> bool:
    """True iff the model context is gap-free (<=12.5s jumps) AND the ER lookback span
    is contiguous (not spanning a large outage / the restart gap)."""
    if buckets is None:
        return True
    # 1) model context (120 buckets ending at didx): no jump > MAX_BUCKET_GAP_S (training rule)
    ctx_b = buckets[didx - CTX + 1: didx + 1]
    for a, b in zip(ctx_b, ctx_b[1:]):
        if (b - a).total_seconds() > MAX_BUCKET_GAP_S:
            return False
    # 2) ER lookback: oldest sampled decision-mid must be within ~expected span (no big gap)
    lo = didx - S.ER_KER * STRIDE
    expected_s = S.ER_KER * STRIDE * 5.0
    if (buckets[didx] - buckets[lo]).total_seconds() > expected_s * ER_SPAN_TOL:
        return False
    return True


def generate_from_buffer(raw_buffer: np.ndarray, exchange: str, symbol: str,
                         decision_offset: int = LAG, buckets=None) -> dict:
    """Compute the FROZEN strategy's desired signal for the decision bucket
    `decision_offset` rows back from the end of `raw_buffer` (chronological, (T,162)).
    If `buckets` (timestamps) is given, enforces the gap guard (training-matched).
    Returns dict with signal + all intermediates (for parity/debugging).
    """
    model, scalers = _load()
    sc = scalers[f"{exchange}_{symbol}"]
    T = len(raw_buffer)
    didx = T - 1 - decision_offset                      # decision bucket index in buffer
    if didx - (CTX - 1) < 60 or didx - S.ER_KER * STRIDE < 0:
        return {"signal": 0, "reason": "insufficient_history", "didx": didx, "T": T}
    if not _gap_ok(buckets, didx):
        return {"signal": 0, "reason": "warming_up_or_gap", "didx": didx, "T": T}

    feats, _ = engineer_features_v2(raw_buffer, N_LEVELS, apply_smoothing=True, savgol_window=11)
    scaled = (feats - sc._means) / sc._stds             # exact z-score (winsorize skipped; rare-extreme only)
    ctx = scaled[didx - CTX + 1: didx + 1]              # 120 rows ending AT the decision bucket
    assert ctx.shape == (CTX, feats.shape[1]), ctx.shape

    with torch.no_grad():
        out = model(torch.from_numpy(ctx[None]).float(),
                    torch.tensor([EX_ID[exchange]]), torch.tensor([SY_ID[symbol]]))
    dl = (out[1] if isinstance(out, (tuple, list)) else out).reshape(3, 3).numpy()
    s = [S.head_sign(dl[h]) for h in HEAD_HORIZON]      # [s0, s1, s2]

    scaled_mid_ctx = ctx[:, MID]
    # decision-mid series: raw mid sampled every STRIDE back from the decision bucket
    dec_mids = raw_buffer[didx::-STRIDE, MID][: S.ER_KER + 1][::-1]   # chronological
    sig = S.decide_signal(s[0], s[1], s[2], scaled_mid_ctx, dec_mids)
    return {
        "signal": int(sig), "s0": s[0], "s1": s[1], "s2": s[2],
        "vol": float(S.context_vol(scaled_mid_ctx)),
        "trend_bp": float(S.trend_return_bp(dec_mids)),
        "er": float(S.efficiency_ratio(dec_mids)),
        "mid": float(raw_buffer[didx, MID]), "spread": float(abs(raw_buffer[didx, SPR])),
        "didx": didx, "T": T,
        # gate inputs, returned so an exit overlay (risk/sizing) reuses the SAME arrays
        # the gate used (SSOT — no recompute). Base path ignores these.
        "scaled_mid_ctx": scaled_mid_ctx, "dec_mids": dec_mids,
    }


def generate_live(exchange: str, symbol: str, as_of=None) -> dict:
    """Fetch the trailing buffer from the live DB and generate the current signal."""
    buckets, raw = DBS.fetch_buffer(exchange, symbol, n=MIN_BUFFER + 200, end_time=as_of)
    if len(raw) < MIN_BUFFER:
        return {"signal": 0, "reason": f"buffer_too_small({len(raw)})"}
    return generate_from_buffer(raw, exchange, symbol, decision_offset=LAG, buckets=buckets)


if __name__ == "__main__":
    print("MIN_BUFFER:", MIN_BUFFER)
    r = generate_live("binance_perp", "BTC-USDT")   # as_of=None -> latest available (DB frozen at 2026-06-10)
    print(r)
