#!/usr/bin/env python3
"""Nested, model-retrained, rolling-window walk-forward harness for the
inventory + trailing-stop strategy.

Heavy steps (retrain, signal precompute) run on GPU/Colab; the pure pieces
(fold ranges, strategy sim, inner sweep, aggregation) are TDD'd locally and
operate on cached per-decision signals.

See docs/superpowers/specs/2026-06-15-trailing-stop-walkforward-design.md.
"""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train_start: dt.datetime
    tune_start: dt.datetime
    train_end: dt.datetime      # == test_start (rolling boundary)
    test_start: dt.datetime
    test_end: dt.datetime

    @property
    def tune_end(self) -> dt.datetime:
        return self.train_end


def make_folds(data_start: dt.datetime, data_end: dt.datetime,
               train_days: int = 45, test_days: int = 8, tune_days: int = 7) -> list[Fold]:
    """Rolling fixed-width folds: train[t..t+train] (last tune_days held out for
    inner param selection) then test[t+train .. t+train+test]. Step = test_days.
    Returns only fully-contained folds (no overlap of test with train; never
    runs past data_end)."""
    folds: list[Fold] = []
    train_w = dt.timedelta(days=train_days)
    test_w = dt.timedelta(days=test_days)
    tune_w = dt.timedelta(days=tune_days)
    t = data_start
    while t + train_w + test_w <= data_end:
        train_start = t
        train_end = t + train_w
        f = Fold(train_start=train_start, tune_start=train_end - tune_w,
                 train_end=train_end, test_start=train_end, test_end=train_end + test_w)
        # leakage guards
        assert f.train_start < f.tune_start < f.train_end, "tune slice must sit inside train"
        assert f.test_start >= f.train_end, "test must not overlap train"
        folds.append(f)
        t = t + test_w
    return folds


import numpy as np
import sys
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from executor.paper.strategy import Inventory
from executor.paper import risk as R

FEE_BP = 3.0


def _ctx_for_vol(v: float) -> np.ndarray:
    """A context whose context_vol (std of first-diffs) == v.
    diffs = [-v, +v] -> std = v ; values = [0, -v, 0]."""
    return np.array([0.0, -v, 0.0])


def simulate(signs: np.ndarray, mids: np.ndarray, spreads: np.ndarray,
             vol: np.ndarray, params: dict, fee_bp: float = FEE_BP):
    """Run the inventory + trailing-stop strategy over one slice of cached
    per-decision signals. `vol` is the precomputed context_vol per decision
    (scaled-mid first-diff std). Returns (net_bp_series (N,), n_trades).

    The vol gate needs a scaled-mid CONTEXT, but we only cached the scalar
    context_vol per decision; `_ctx_for_vol` rebuilds a 3-point context whose
    context_vol equals vol[i] exactly."""
    inv = Inventory()
    stop = R.TrailingStop(k=params["k"], L=params["L"], cooldown_n=params["cooldown_n"])
    N = len(mids)
    net = np.zeros(N, dtype=np.float64)
    n_trades = 0
    for i in range(N):
        ctx = _ctx_for_vol(float(vol[i]))
        prev_pos = inv.position
        out = R.step_with_stop(int(signs[i, 0]), int(signs[i, 1]), int(signs[i, 2]),
                               ctx, mids[: i + 1], float(mids[i]), float(spreads[i]),
                               fee_bp, inv, stop)
        if inv.position != prev_pos:
            n_trades += 1
        net[i] = out["net_bp"]
    return net, n_trades


def metrics(net_series: np.ndarray, n_trades: int) -> dict:
    """Per-decision PnL increments -> Sharpe + max drawdown (bp)."""
    net_series = np.asarray(net_series, dtype=np.float64)
    rets = np.diff(net_series, prepend=0.0)
    sharpe = float(rets.mean() / rets.std()) if rets.std() > 1e-12 else 0.0
    running_max = np.maximum.accumulate(net_series) if len(net_series) else np.array([0.0])
    max_dd = float((running_max - net_series).max()) if len(net_series) else 0.0
    return {"net_bp": float(net_series[-1]) if len(net_series) else 0.0,
            "sharpe": sharpe, "max_dd_bp": max_dd,
            "n_trades": int(n_trades), "n_decisions": int(len(net_series))}


import itertools


def param_grid(ks=(1.0, 1.5, 2.0, 2.5, 3.0), Ls=(12, 24), cooldowns=(0, 2, 5)) -> list[dict]:
    return [dict(k=k, L=L, cooldown_n=c) for k, L, c in itertools.product(ks, Ls, cooldowns)]


def sweep(signs, mids, spreads, vol, grid: list[dict], fee_bp: float = FEE_BP):
    """Evaluate every param set on the (tune) slice. Returns (best_params, table).
    Objective: net_bp primary, sharpe tie-break (favours flat/robust regions)."""
    table = []
    for p in grid:
        net, n_trades = simulate(signs, mids, spreads, vol, p, fee_bp)
        m = metrics(net, n_trades)
        table.append({**p, **m})
    best_row = max(table, key=lambda r: (r["net_bp"], r["sharpe"]))
    best = {"k": best_row["k"], "L": best_row["L"], "cooldown_n": best_row["cooldown_n"]}
    return best, table


def save_signals(path: str, signs, mids, spreads, vol, ts):
    np.savez(path, signs=signs, mids=mids, spreads=spreads, vol=vol, ts=ts)


def load_signals(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in ("signs", "mids", "spreads", "vol", "ts")}


def aggregate(fold_results: list[dict]) -> dict:
    """Stitch per-fold OOS net series into one continuous curve + per-fold table."""
    per_fold = []
    stitched = []
    offset = 0.0
    n_profitable = 0
    for fr in fold_results:
        ns = np.asarray(fr["net_series"], dtype=np.float64)
        fold_net = float(ns[-1]) if len(ns) else 0.0
        if fold_net > 0:
            n_profitable += 1
        m = metrics(ns, fr["n_trades"])
        per_fold.append({"fold": fr["fold"], "best": fr["best"], **m})
        stitched.extend((ns + offset).tolist())
        offset += fold_net
    stitched = np.asarray(stitched, dtype=np.float64)
    overall = metrics(stitched, sum(fr["n_trades"] for fr in fold_results))
    overall["n_folds"] = len(fold_results)
    overall["n_profitable_folds"] = n_profitable
    return {"overall": overall, "per_fold": per_fold}


import pickle
from pathlib import Path
import torch


def train_fold(fold: "Fold", out_dir: str, parquet_dir: str = "lob_data",
               levels: int = 40, epochs: int = 50, batch_size: int = 16,
               accum_steps: int = 8, lr: float = 1e-3,
               d_model: int = 66, n_heads: int = 3, n_layers: int = 3,
               patience: int = 10, seed: int = 0) -> dict:
    """Retrain the V2 model on fold.train_start..fold.train_end (parquet, bounded).
    Saves best.pt + scalers.pkl into out_dir. Returns {best_val_dir_acc, run_dir}.

    Reuses train_v2's per-epoch helpers so training logic stays single-sourced.
    """
    import datetime as dt
    from training.dataset import DataConfig, build_dataloaders
    from training.model_v2 import CompoundAttentionModelV2, LOBLossV2
    from training.model import WarmupDecayScheduler
    from training.train_v2 import run_one_epoch_v2, evaluate_v2_loader, get_device

    torch.manual_seed(seed)
    device = get_device()
    n_features = levels * 5 + 19
    d_ff = d_model * 4
    out = Path(out_dir); (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(lob_levels=levels, feature_version="v2", savgol_window=11,
                     source="parquet", parquet_dir=parquet_dir,
                     train_ratio=0.85, val_ratio=0.15)   # within-window train/val (no internal test)
    # Bound to the fold's TRAIN window only.
    train_loader, val_loader, _, meta = build_dataloaders(
        cfg, batch_size=batch_size, start_time=fold.train_start, end_time=fold.train_end)

    model = CompoundAttentionModelV2(
        n_levels=levels, n_features=n_features, context_length=cfg.context_length,
        prediction_length=cfg.prediction_length, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, dropout=0.1).to(device)
    first_scaler = next(iter(meta["scalers"].values()))
    loss_fn = LOBLossV2(n_levels=levels, use_feature_weights=True, mid_price_idx=levels * 4,
                        scaler_means=first_scaler._means, scaler_stds=first_scaler._stds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = WarmupDecayScheduler(optimizer, warmup_steps=1000, decay_factor=0.8, decay_every=5000)

    with open(out / "scalers.pkl", "wb") as f:
        pickle.dump(meta["scalers"], f)

    best_val = -1.0
    no_improve = 0
    for epoch in range(1, epochs + 1):
        run_one_epoch_v2(model, loss_fn, optimizer, scheduler, train_loader, device, accum_steps)
        va = evaluate_v2_loader(model, loss_fn, val_loader, device)
        score = va["dir_acc_mean"]
        is_best = score > best_val
        if is_best:
            best_val = score
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "n_features": n_features,
                        "context_length": cfg.context_length, "prediction_length": cfg.prediction_length,
                        "direction_horizons": list(loss_fn.direction_horizons),
                        "mid_price_idx": loss_fn.mid_price_idx, "feature_version": "v2",
                        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff},
                       out / "checkpoints" / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    # config.json for downstream loaders (mirrors experiments/<run>/config.json shape)
    (out / "config.json").write_text(__import__("json").dumps(
        {"n_features": n_features, "context_length": cfg.context_length,
         "prediction_length": cfg.prediction_length, "d_model": d_model, "n_heads": n_heads,
         "n_layers": n_layers, "d_ff": d_ff, "dropout": 0.1}))
    return {"best_val_dir_acc": best_val, "run_dir": str(out)}


import json

N_LEVELS_DEFAULT, MID_IDX, SPR_IDX, CTX, STRIDE, WARMUP = 40, 160, 161, 120, 23, 60


def precompute_signals(run_dir: str, stream: str, start, end, parquet_dir: str = "lob_data",
                       levels: int = N_LEVELS_DEFAULT, batch: int = 256) -> dict:
    """Run the fold model over [start, end) for one stream and return per-decision
    arrays {signs (M,3), mids (M,), spreads (M,), vol (M,), ts (M,)}.

    Features use centered savgol (training-faithful; the live harness reproduces
    this causally via DECISION_LAG). Winsorize is skipped (z-score exact), matching
    the live path — we backtest what we trade.
    """
    import torch
    from training.features_v2 import engineer_features_v2
    from training.model_v2 import CompoundAttentionModelV2
    from training.data_source import fetch_parquet_stream
    from training.dataset import EXCHANGE_MAP, SYMBOL_MAP
    from executor.paper import strategy as S

    cfg = json.load(open(f"{run_dir}/config.json"))
    ck = torch.load(f"{run_dir}/checkpoints/best.pt", map_location="cpu", weights_only=False)
    pl = ck.get("prediction_length", cfg["prediction_length"])
    model = CompoundAttentionModelV2(n_levels=levels, n_features=cfg["n_features"],
        context_length=cfg["context_length"], prediction_length=pl, d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], d_ff=cfg["d_ff"], dropout=cfg.get("dropout", 0.1))
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    import pickle
    sc = pickle.load(open(f"{run_dir}/scalers.pkl", "rb"))[stream]
    # stream = "<exchange>_<symbol>", e.g. "binance_perp_BTC-USDT" -> exch="binance_perp", sym="BTC-USDT"
    parts = stream.split("_")
    exch, sym = "_".join(parts[:2]), parts[-1]
    ex_id, sy_id = EXCHANGE_MAP[exch], SYMBOL_MAP[sym]
    raw, _ = fetch_parquet_stream(parquet_dir, exch, sym, levels, start_time=start, end_time=end)
    feats, _ = engineer_features_v2(raw, levels, apply_smoothing=True, savgol_window=11)
    scaled = (feats - sc._means) / sc._stds
    T = len(scaled)
    didx = np.arange(CTX + WARMUP, T, STRIDE)

    signs = np.zeros((len(didx), 3), np.int8)
    with torch.no_grad():
        for b in range(0, len(didx), batch):
            idx = didx[b:b + batch]
            ctx = np.stack([scaled[t - CTX:t] for t in idx])
            out = model(torch.from_numpy(ctx).float(),
                        torch.full((len(idx),), ex_id, dtype=torch.long),
                        torch.full((len(idx),), sy_id, dtype=torch.long))
            dl = (out[1] if isinstance(out, (tuple, list)) else out).reshape(-1, 3, 3).numpy()
            pred = dl.argmax(2)
            signs[b:b + len(idx)] = np.where(pred == 2, 1, np.where(pred == 0, -1, 0))
    mids = raw[didx, MID_IDX].astype(np.float64)
    spreads = np.abs(raw[didx, SPR_IDX]).astype(np.float64)
    vol = np.array([S.context_vol(scaled[t - CTX:t, MID_IDX]) for t in didx], dtype=np.float64)
    ts = didx.astype(np.float64)
    return dict(signs=signs, mids=mids, spreads=spreads, vol=vol, ts=ts)
