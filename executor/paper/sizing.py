#!/usr/bin/env python3
"""Vol-targeted, signal-decay exit overlay (v1) — trade the signal, size by risk.

Pure numpy; no model/DB. Replaces fixed-cap accumulate + bolt-on trailing stop with:
  (A) vol-targeted dynamic cap, (B) signal-decay exit, (C) wide catastrophic floor.

Walk-forward verdict (2026-06-18, 3-fold faithful OOS, vs base & rejected trailing stop):
  base +1623bp / maxDD 5657 / 2-of-3 profitable ; trailing-stop -3048 / 1-of-3 (REJECTED) ;
  THIS exit +665bp / maxDD 803 / 2-of-3  -> 7x drawdown cut, net-positive. The deployable exit.
See docs/superpowers/specs/2026-06-18-vol-target-signal-decay-exit-design.md
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from executor.paper import strategy as S
from executor.paper.strategy import Inventory

TARGET_VOL    = 0.03
CAP_MAX       = 3
STOP_FLOOR_BP = 150.0
COOLDOWN_N    = 2


def vol_target_cap(sigma: float, target_vol: float = TARGET_VOL, cap_max: int = CAP_MAX) -> int:
    """Constant-risk cap = round(target_vol / sigma), clipped to [0, cap_max].
    sigma <= 0 (no vol / insufficient history) -> 0 (stay flat)."""
    if sigma <= 0.0:
        return 0
    return int(np.clip(int(round(target_vol / sigma)), 0, cap_max))


@dataclass
class PositionTracker:
    """Tracks the open position's entry net (for the catastrophic floor) + re-entry cooldown."""
    cooldown_n: int = COOLDOWN_N
    entry_net: float | None = None
    _cooldown: int = field(default=0, init=False)

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown > 0

    def tick(self) -> None:
        if self._cooldown > 0:
            self._cooldown -= 1

    def start_cooldown(self) -> None:
        self._cooldown = self.cooldown_n


def step_with_sizing(s0: int, s1: int, s2: int,
                     scaled_mid_ctx: np.ndarray, mid_decisions: np.ndarray,
                     mid: float, spread: float, fee_bp: float,
                     inv: Inventory, tracker: PositionTracker, params: dict) -> dict:
    """One decision. Order: mark -> catastrophic floor -> cooldown -> gate -> resize 1 unit
    toward (gated_signal * vol_target_cap), decaying to flat when the signal stops confirming.
    Mutates inv + tracker. Returns {action, signal, position, net_bp}."""
    tv = params["target_vol"]; floor = abs(params.get("stop_floor_bp", STOP_FLOOR_BP))
    tracker.cooldown_n = int(params.get("cooldown_n", COOLDOWN_N))
    tracker.tick()

    inv.on_decision(0, gated=False, mid=mid, spread=spread, fee_bp=fee_bp)   # mark, no trade
    p = inv.position

    if p != 0 and tracker.entry_net is not None and (inv.net_pnl_bp - tracker.entry_net) <= -floor:
        inv.flatten(mid, spread, fee_bp)
        tracker.entry_net = None
        tracker.start_cooldown()
        return {"action": "floor", "signal": 0, "position": inv.position, "net_bp": inv.net_pnl_bp}

    if tracker.in_cooldown:
        return {"action": "cooldown", "signal": 0, "position": inv.position, "net_bp": inv.net_pnl_bp}

    g = S.decide_signal(int(s0), int(s1), int(s2), scaled_mid_ctx, mid_decisions)
    cap = vol_target_cap(S.context_vol(scaled_mid_ctx), tv)
    desired = g * cap if g != 0 else 0
    step = int(np.sign(desired - p))
    if step != 0:
        inv.on_decision(step, gated=True, mid=mid, spread=spread, fee_bp=fee_bp)

    if inv.position != 0 and tracker.entry_net is None:
        tracker.entry_net = inv.net_pnl_bp
    if inv.position == 0:
        tracker.entry_net = None

    return {"action": ("trade" if step != 0 else ("hold" if g != 0 else "flat")),
            "signal": g, "position": inv.position, "net_bp": inv.net_pnl_bp}
