#!/usr/bin/env python3
"""Trailing-stop risk overlay for the inventory strategy.

Position-level, volatility-scaled trailing stop. Pure functions of
(position, decision-mid series) — no model/DB dependency, unit-testable in
isolation, same discipline as strategy.py. Layered on top of the FROZEN
gate+inventory brain: entries are unchanged; the stop adds a hard flatten-all
exit that overrides the trend-veto.

See docs/superpowers/specs/2026-06-15-trailing-stop-walkforward-design.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Defaults — SWEPT by the walk-forward inner loop, not hand-tuned.
STOP_K        = 2.0       # stop distance in sigma units
STOP_L        = 12        # sigma lookback (decisions ~ 2-min bars)
STOP_COOLDOWN = 2         # re-entry cooldown (decisions) after a stop-out
STOP_DMIN     = 0.0005    # floor on stop distance (fraction = 5 bp)


def sigma_returns(mid_decisions: np.ndarray, L: int = STOP_L) -> float:
    """Stdev of log-returns over the last L decision-mids (~2-min-bar vol).
    Returns 0.0 if there is not enough history (stop stays unarmed)."""
    if len(mid_decisions) <= L:
        return 0.0
    seg = np.asarray(mid_decisions[-(L + 1):], dtype=np.float64)
    if (seg <= 0).any():
        return 0.0
    return float(np.std(np.diff(np.log(seg))))


@dataclass
class TrailingStop:
    k: float = STOP_K
    L: int = STOP_L
    cooldown_n: int = STOP_COOLDOWN
    d_min: float = STOP_DMIN
    side: int = 0                  # +1 long / -1 short / 0 flat
    hwm: float | None = None       # favorable extreme of mid since position opened
    _cooldown: int = 0

    def reset(self) -> None:
        self.side = 0
        self.hwm = None

    def update(self, position: float, mid: float) -> None:
        """Ratchet side + high-water mark. Call once per decision AFTER the
        trade (so a freshly opened position arms its hwm at the entry mid)."""
        new_side = int(np.sign(position))
        if new_side == 0:
            self.reset()
            return
        if new_side != self.side or self.hwm is None:   # opened or flipped
            self.side = new_side
            self.hwm = float(mid)
        else:
            self.hwm = max(self.hwm, mid) if self.side > 0 else min(self.hwm, mid)

    def stop_level(self, mid_decisions: np.ndarray) -> float | None:
        """Trailed stop price, or None if flat / not enough vol history."""
        if self.side == 0 or self.hwm is None:
            return None
        sig = sigma_returns(mid_decisions, self.L)
        if sig == 0.0:
            return None
        d = max(self.k * sig, self.d_min)
        return self.hwm * (1 - d) if self.side > 0 else self.hwm * (1 + d)

    def breached(self, mid: float, mid_decisions: np.ndarray) -> bool:
        lvl = self.stop_level(mid_decisions)
        if lvl is None:
            return False
        return (mid <= lvl) if self.side > 0 else (mid >= lvl)

    # --- re-entry cooldown ---
    def start_cooldown(self) -> None:
        self._cooldown = self.cooldown_n

    def tick_cooldown(self) -> None:
        if self._cooldown > 0:
            self._cooldown -= 1

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown > 0
