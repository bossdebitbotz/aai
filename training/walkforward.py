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
