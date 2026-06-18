#!/usr/bin/env python3
"""End-to-end signal pipeline: raw LOB -> features -> scale -> model -> gates -> inventory.

This is the local validation of the full inference chain (parity-checked features +
saved-scaler scaling + model + the FROZEN strategy brain). For LIVE use, the same
chain runs on a trailing DB buffer with DECISION_LAG (proven bit-exact in
feature_parity_check.py). Here we replay it over a historical parquet stream to
confirm the chain runs and produces coherent gated signals + PnL on this machine.

NOTE: training winsorized features to train 0.1/99.9 pct bounds before scaling;
those bounds are not persisted, so we skip winsorize (affects only rare extremes;
the z-score scaling via saved means/stds is exact). Flagged for later refinement.

Run: .venv/bin/python executor/paper/signal_pipeline.py [stream] [limit]
"""
import sys, json, pickle, time
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np, pandas as pd, torch
from training.features_v2 import engineer_features_v2
from training.model_v2 import CompoundAttentionModelV2
from executor.paper import strategy as S

RUN = "/Volumes/Docker-SSD/projects/aaiwdbback/aai/experiments/v2_balanced"
N_LEVELS, MID, SPR, CTX, STRIDE, WARMUP = 40, 160, 161, 120, 23, 60
FEE_BP = 3.0


def load_model():
    cfg = json.load(open(f"{RUN}/config.json"))
    ck = torch.load(f"{RUN}/checkpoints/best.pt", map_location="cpu", weights_only=False)
    pl = ck.get("prediction_length", cfg["prediction_length"])
    m = CompoundAttentionModelV2(n_levels=40, n_features=cfg["n_features"], context_length=cfg["context_length"],
        prediction_length=pl, d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"], dropout=cfg.get("dropout", 0.1))
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


def run(stream="binance_perp_BTC-USDT", limit=15000, rawtail=150000):
    sc = pickle.load(open(f"{RUN}/scalers.pkl", "rb"))[stream]
    raw = pd.read_parquet(f"/Volumes/Docker-SSD/projects/aaiwdbback/aai/lob_data/{stream}.parquet").drop(columns=["bucket"]).to_numpy(np.float64)
    raw = raw[np.isfinite(raw).all(axis=1)]
    if rawtail and len(raw) > rawtail:
        raw = raw[-rawtail:]                                       # recent slice -> fast feature build (live uses small buffer anyway)
    feats, _ = engineer_features_v2(raw, N_LEVELS, apply_smoothing=True, savgol_window=11)
    scaled = (feats - sc._means) / sc._stds                       # exact z-score (skip winsorize)
    T = len(scaled)
    # decision indices: every STRIDE, after context+warmup; take the most recent `limit`
    didx = np.arange(CTX + WARMUP, T, STRIDE)
    if limit: didx = didx[-limit:]
    print(f"{stream}: {T} rows -> {len(didx)} decisions (stride {STRIDE})")

    model = load_model()
    # batched CPU inference of the direction head
    ex_id = {"binance_spot":0,"binance_perp":1,"bybit_spot":2,"kucoin_spot":3}["_".join(stream.split("_")[:2])]
    sy_id = {"BTC-USDT":0,"ETH-USDT":1,"SOL-USDT":2,"WLD-USDT":3}[stream.split("_")[-1] if "USDT" in stream.split("_")[-1] else "-".join(stream.split("_")[-2:])]
    signs = np.zeros((len(didx), 3), int)
    t0 = time.time()
    B = 256
    with torch.no_grad():
        for b in range(0, len(didx), B):
            idx = didx[b:b+B]
            ctx = np.stack([scaled[t-CTX:t] for t in idx])
            out = model(torch.from_numpy(ctx).float(),
                        torch.full((len(idx),), ex_id, dtype=torch.long),
                        torch.full((len(idx),), sy_id, dtype=torch.long))
            dl = (out[1] if isinstance(out,(tuple,list)) else out).reshape(-1,3,3).numpy()
            pred = dl.argmax(2)
            signs[b:b+len(idx)] = np.where(pred==2,1,np.where(pred==0,-1,0))
    print(f"inference {len(didx)} windows in {time.time()-t0:.0f}s")

    # run the FROZEN strategy over the decision series
    dec_mid = raw[didx, MID]; dec_spr = np.abs(raw[didx, SPR])
    inv = S.Inventory()
    n_gate_pass = 0
    for i, t in enumerate(didx):
        s0,s1,s2 = signs[i]
        scaled_mid_ctx = scaled[t-CTX:t, MID]
        mids_hist = dec_mid[:i+1]                                  # trailing decision-mids
        sig = S.decide_signal(s0,s1,s2, scaled_mid_ctx, mids_hist)
        gated = sig != 0
        if gated: n_gate_pass += 1
        inv.on_decision(sig if gated else 0, gated, dec_mid[i], dec_spr[i], FEE_BP)
    # diagnostics
    agree = np.array([S.base_signal(*signs[i]) for i in range(len(didx))])
    print(f"head-2 sign dist: down={np.mean(signs[:,2]==-1):.2f} flat={np.mean(signs[:,2]==0):.2f} up={np.mean(signs[:,2]==1):.2f}")
    print(f"3-horizon agreement rate: {np.mean(agree!=0):.3f} | gated-trade decisions: {n_gate_pass} ({n_gate_pass/len(didx)*100:.1f}%)")
    print(f"INVENTORY: net={inv.net_pnl_bp:+.0f} bp-units (marked {inv.marked_pnl_bp:+.0f}, cost {inv.realized_cost_bp:.0f}) | final pos {inv.position:+.0f}")


if __name__ == "__main__":
    stream = sys.argv[1] if len(sys.argv) > 1 else "binance_perp_BTC-USDT"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    run(stream, limit)
