#!/usr/bin/env python3
"""FROZEN paper-trade strategy logic (gate stack + inventory).

Pure functions of model head-signs + the trailing decision-mid series — no model
or DB dependency, so it is unit-testable in isolation and is the single shared
'brain' for both backtest and the live harness.

All constants are FROZEN to the spec (training data/FROZEN_strategy_spec_2026-06-14.md).
Do NOT tune these on paper-trade results.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# --- FROZEN parameters (absolute values; see spec) ---
VOL_THRESHOLD = 0.002249     # std of scaled-mid first-diffs over the 120 context (90th pct on val)
TREND_K       = 30           # decisions (~60 min): trend-filter lookback
TREND_TT_BP   = 30.0         # bp: veto fading a move larger than this
ER_KER        = 60           # decisions (~120 min): trend-efficiency lookback
ER_MIN        = 0.15         # require trend-efficiency >= this (trade only in trending regimes)
CAP           = 3            # inventory accumulation cap (units)
DECISION_LAG  = 6            # 5s buckets (~30s) — decide for bucket now-LAG for exact feature parity


def head_sign(logits3: np.ndarray) -> int:
    """One horizon's 3 class-logits -> -1 (down) / 0 (flat) / +1 (up)."""
    i = int(np.argmax(logits3))
    return 1 if i == 2 else (-1 if i == 0 else 0)


def base_signal(s0: int, s1: int, s2: int) -> int:
    """3-horizon agreement (30s/1m/2m). Trade the 2-min direction only if all agree & non-flat."""
    return s2 if (s0 == s1 == s2 and s2 != 0) else 0


def context_vol(scaled_mid_context: np.ndarray) -> float:
    """Volatility gate input: std of first-differences of the scaled mid over the 120-step context."""
    return float(np.std(np.diff(scaled_mid_context)))


def trend_return_bp(mid_decisions: np.ndarray, K: int = TREND_K) -> float:
    """Trailing return (bp) over the last K decisions of the (raw) decision-mid series."""
    if len(mid_decisions) <= K:
        return 0.0
    return (mid_decisions[-1] - mid_decisions[-1 - K]) / mid_decisions[-1 - K] * 1e4


def trend_veto(sig: int, mid_decisions: np.ndarray, K: int = TREND_K, TT: float = TREND_TT_BP) -> bool:
    """Veto fading a strong recent move: don't go long into a strong downtrend / short into uptrend."""
    r = trend_return_bp(mid_decisions, K)
    return bool((sig > 0 and r < -TT) or (sig < 0 and r > TT))


def efficiency_ratio(mid_decisions: np.ndarray, KER: int = ER_KER) -> float:
    """Kaufman-style trend efficiency over last KER decisions: |net move| / total path. 0=choppy, 1=trend."""
    if len(mid_decisions) <= KER:
        return 0.0
    seg = mid_decisions[-1 - KER:]
    net = abs(seg[-1] - seg[0])
    path = float(np.abs(np.diff(seg)).sum())
    return net / path if path > 1e-12 else 0.0


def decide_signal(s0: int, s1: int, s2: int,
                  scaled_mid_context: np.ndarray,
                  mid_decisions: np.ndarray) -> int:
    """Full FROZEN gate stack -> desired direction in {-1, 0, +1}.

    Order: 3-horizon agreement -> volatility gate -> don't-fade-trend -> trending-regime gate.
    """
    sig = base_signal(s0, s1, s2)
    if sig == 0:
        return 0
    if context_vol(scaled_mid_context) < VOL_THRESHOLD:
        return 0
    if trend_veto(sig, mid_decisions):
        return 0
    if efficiency_ratio(mid_decisions) < ER_MIN:
        return 0
    return sig


@dataclass
class Inventory:
    """Accumulating inventory engine, taker execution. PnL/costs tracked in bps on 1-unit notional."""
    cap: int = CAP
    position: float = 0.0
    realized_cost_bp: float = 0.0
    marked_pnl_bp: float = 0.0
    _last_mid: float | None = None

    def target(self, signal: int, gated: bool) -> float:
        """Desired position: accumulate toward signal (capped) when a gated signal fires; else hold."""
        if not gated or signal == 0:
            return self.position
        return float(np.clip(self.position + signal, -self.cap, self.cap))

    def on_decision(self, signal: int, gated: bool, mid: float, spread: float, fee_bp: float):
        """Process one decision: mark the held position to the new mid, then trade toward target.

        Returns (delta_units, cost_bp). Call once per decision in chronological order.
        """
        # 1) mark prior position over the interval that just elapsed
        if self._last_mid is not None and self.position != 0.0:
            self.marked_pnl_bp += self.position * (mid - self._last_mid) / self._last_mid * 1e4
        self._last_mid = mid
        # 2) trade toward target via taker
        tgt = self.target(signal, gated)
        delta = tgt - self.position
        cost = 0.0
        if abs(delta) > 1e-9:
            half_spread_bp = (spread / 2.0) / mid * 1e4
            cost = abs(delta) * (half_spread_bp + fee_bp)
            self.realized_cost_bp += cost
            self.position = tgt
        return delta, cost

    def flatten(self, mid: float, spread: float, fee_bp: float) -> tuple[float, float]:
        """Mark the held position to `mid`, then close it entirely via taker.

        Used by the trailing-stop overlay (risk.py). Additive: the FROZEN live
        path never calls this, so existing behavior is unchanged. Returns
        (delta_units, cost_bp).
        """
        if self._last_mid is not None and self.position != 0.0:
            self.marked_pnl_bp += self.position * (mid - self._last_mid) / self._last_mid * 1e4
        self._last_mid = mid
        delta = -self.position
        cost = 0.0
        if abs(delta) > 1e-9:
            half_spread_bp = (spread / 2.0) / mid * 1e4
            cost = abs(delta) * (half_spread_bp + fee_bp)
            self.realized_cost_bp += cost
            self.position = 0.0
        return delta, cost

    @property
    def net_pnl_bp(self) -> float:
        return self.marked_pnl_bp - self.realized_cost_bp
