# Trailing Stop + Nested Walk-Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a position-level volatility-scaled trailing stop to the inventory strategy, and a nested, model-retrained, rolling-window walk-forward harness that evaluates the *actual* traded strategy with zero leakage.

**Architecture:** A pure `TrailingStop` overlay wraps the existing FROZEN `strategy.py` brain (entries unchanged; stop adds a hard flatten-all exit that overrides the trend-veto). A walk-forward harness retrains the model per rolling fold (GPU/Colab), caches per-decision signals, then runs the cheap strategy sim + inner param sweep on a held-out tune slice and reports stitched out-of-sample results. Heavy steps (retrain, signal precompute) run on Colab; pure steps are TDD'd locally.

**Tech Stack:** Python, NumPy, PyTorch, pandas/pyarrow, pytest. Reuses `executor/paper/strategy.py`, `training/dataset.py`, `training/model_v2.py`, `training/features_v2.py`, `training/train_v2.py` helpers.

**Spec:** `docs/superpowers/specs/2026-06-15-trailing-stop-walkforward-design.md`

---

## Preamble: branch

The working tree is on `main` (diverged from origin, with prior uncommitted work). Before Task 1:

- [ ] **Create a feature branch**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai
git checkout -b feat/trailing-stop-walkforward
```

All task commits land on this branch. Run tests with the project venv: `.venv/bin/python -m pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| `executor/paper/strategy.py` (edit) | add **additive** `Inventory.flatten()` — does not change existing frozen behavior |
| `executor/paper/risk.py` (create) | `sigma_returns`, `TrailingStop`, `step_with_stop` orchestrator |
| `executor/paper/test_risk.py` (create) | unit tests for the above |
| `training/data_source.py` (edit) | parquet time-range bounds |
| `training/dataset.py` (edit) | pass `start_time`/`end_time` into the parquet branch |
| `training/walkforward.py` (create) | fold ranges, strategy sim, inner sweep, signal cache, aggregation, retrain hook, orchestration |
| `training/test_walkforward.py` (create) | unit tests for the pure harness pieces |
| `training/colab_walkforward.md` (create) | Colab runner cells + calibration-fold protocol |

---

## Task 1: Additive `Inventory.flatten()`

The stop must close the whole position. `Inventory` only accumulates toward a signal; add a flatten path. **Additive only** — the live bot never calls it, so frozen behavior is unchanged.

**Files:**
- Modify: `executor/paper/strategy.py` (add method to `Inventory`, after `on_decision`)
- Test: `executor/paper/test_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# executor/paper/test_risk.py
import sys
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from executor.paper.strategy import Inventory


def test_inventory_flatten_marks_then_closes():
    inv = Inventory(cap=3)
    # open +2 at mid 100 (two gated longs)
    inv.on_decision(1, True, 100.0, 0.02, 3.0)
    inv.on_decision(1, True, 100.0, 0.02, 3.0)
    assert inv.position == 2.0
    # price rises to 101, then flatten
    delta, cost = inv.flatten(101.0, 0.02, 3.0)
    assert inv.position == 0.0
    assert delta == -2.0
    # marked PnL: 2 units * (101-100)/100 * 1e4 = +200 bp (marking happened in flatten)
    assert abs(inv.marked_pnl_bp - 200.0) < 1e-6
    # cost: 2 units * (half_spread_bp + 3bp); half_spread = (0.02/2)/101*1e4 ≈ 0.990 bp
    assert cost > 0
    # flatten on an already-flat book is a no-op (no cost)
    d2, c2 = inv.flatten(101.0, 0.02, 3.0)
    assert d2 == 0.0 and c2 == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest executor/paper/test_risk.py::test_inventory_flatten_marks_then_closes -v`
Expected: FAIL — `AttributeError: 'Inventory' object has no attribute 'flatten'`

- [ ] **Step 3: Add the method**

In `executor/paper/strategy.py`, inside `class Inventory`, immediately after `on_decision` (before the `net_pnl_bp` property):

```python
    def flatten(self, mid: float, spread: float, fee_bp: float):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest executor/paper/test_risk.py::test_inventory_flatten_marks_then_closes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add executor/paper/strategy.py executor/paper/test_risk.py
git commit -m "feat(strategy): additive Inventory.flatten for trailing-stop exit"
```

---

## Task 2: `TrailingStop` core — sigma, HWM ratchet, stop level, breach

**Files:**
- Create: `executor/paper/risk.py`
- Test: `executor/paper/test_risk.py`

- [ ] **Step 1: Write the failing tests**

Append to `executor/paper/test_risk.py`:

```python
import numpy as np
from executor.paper import risk as R


def test_sigma_returns_needs_history():
    assert R.sigma_returns(np.array([100.0, 101.0]), L=12) == 0.0   # too short -> 0
    mids = 100.0 * np.cumprod(1 + np.full(20, 0.0))                 # flat -> zero vol
    assert R.sigma_returns(mids, L=12) == 0.0


def test_hwm_ratchets_favorable_only_long():
    st = R.TrailingStop(k=2.0, L=12)
    st.update(position=1.0, mid=100.0)     # opens long -> hwm=100
    assert st.side == 1 and st.hwm == 100.0
    st.update(position=1.0, mid=102.0)     # rises -> hwm=102
    assert st.hwm == 102.0
    st.update(position=1.0, mid=101.0)     # dips -> hwm stays 102 (ratchet up only)
    assert st.hwm == 102.0


def test_hwm_ratchets_favorable_only_short():
    st = R.TrailingStop(k=2.0, L=12)
    st.update(position=-1.0, mid=100.0)
    assert st.side == -1 and st.hwm == 100.0
    st.update(position=-1.0, mid=98.0)     # favorable for short -> hwm=98
    assert st.hwm == 98.0
    st.update(position=-1.0, mid=99.0)     # adverse -> hwm stays 98
    assert st.hwm == 98.0


def test_flat_resets():
    st = R.TrailingStop()
    st.update(1.0, 100.0)
    st.update(0.0, 100.0)
    assert st.side == 0 and st.hwm is None


def test_breach_long_fires_below_trailed_level():
    # build a noisy-but-trending mid series so sigma > 0
    rng = np.linspace(100.0, 110.0, 30) + np.sin(np.arange(30))
    st = R.TrailingStop(k=2.0, L=12, d_min=0.0005)
    st.update(1.0, rng[-1])                # long, hwm = last (a peak)
    sig = R.sigma_returns(rng, 12)
    assert sig > 0
    lvl = st.stop_level(rng)
    assert lvl is not None and lvl < st.hwm           # stop sits below the peak for a long
    assert st.breached(lvl - 1e-9, rng) is True       # just below the level -> breach
    assert st.breached(st.hwm, rng) is False          # at the peak -> no breach


def test_stop_not_armed_without_vol():
    st = R.TrailingStop()
    st.update(1.0, 100.0)
    flat = np.full(20, 100.0)
    assert st.stop_level(flat) is None                # zero vol -> unarmed
    assert st.breached(50.0, flat) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest executor/paper/test_risk.py -k "sigma or hwm or flat_resets or breach or armed" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'executor.paper.risk'`

- [ ] **Step 3: Create `executor/paper/risk.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest executor/paper/test_risk.py -k "sigma or hwm or flat_resets or breach or armed" -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add executor/paper/risk.py executor/paper/test_risk.py
git commit -m "feat(risk): TrailingStop core — vol-scaled HWM trail + breach"
```

---

## Task 3: `step_with_stop` orchestrator (stop overrides entries; cooldown; arms after trade)

**Files:**
- Modify: `executor/paper/risk.py` (add `step_with_stop` + module imports)
- Test: `executor/paper/test_risk.py`

- [ ] **Step 1: Write the failing tests**

Append to `executor/paper/test_risk.py`:

```python
from executor.paper.strategy import Inventory


def _ctx_highvol():
    # scaled-mid context whose first-diff std clears VOL_THRESHOLD (0.002249).
    # alternating +/-0.01 increments -> first-diffs std ~0.01 (a CONSTANT cumsum
    # would give zero-variance diffs and wrongly fail the vol gate).
    return np.cumsum(np.tile([0.01, -0.01], 60))


def test_step_adds_while_above_stop_then_stops_out_flattens():
    inv, st = Inventory(cap=3), R.TrailingStop(k=2.0, L=12, cooldown_n=2)
    ctx = _ctx_highvol()
    # strictly-rising but vol-bearing mids: all-positive *varying* increments -> price
    # rises monotonically (hwm==current each step, so NO intermediate stop) yet sigma>0
    # so the stop is armed for the later crash.
    incs = 0.001 + 0.0005 * np.sin(np.arange(80))
    mids = list(100.0 * np.cumprod(1 + incs))
    for j, m in enumerate(mids):
        R.step_with_stop(1, 1, 1, ctx, np.array(mids[:j + 1]), m, 0.02, 3.0, inv, st)
    assert inv.position == 3.0            # accumulated to cap while above stop
    pos_before = inv.position
    # sharp drop below the trailed level on the next decision -> flatten-all
    crash = mids + [mids[-1] * 0.95]
    out = R.step_with_stop(1, 1, 1, ctx, np.array(crash), crash[-1], 0.02, 3.0, inv, st)
    assert out["action"] == "stop"
    assert inv.position == 0.0            # whole stack flattened
    assert pos_before == 3.0


def test_step_cooldown_blocks_reentry_then_releases():
    inv, st = Inventory(cap=3), R.TrailingStop(k=2.0, L=12, cooldown_n=2)
    ctx = _ctx_highvol()
    incs = 0.001 + 0.0005 * np.sin(np.arange(40))
    mids = list(100.0 * np.cumprod(1 + incs))
    for j, m in enumerate(mids):
        R.step_with_stop(1, 1, 1, ctx, np.array(mids[:j + 1]), m, 0.02, 3.0, inv, st)
    crash = mids + [mids[-1] * 0.95]
    R.step_with_stop(1, 1, 1, ctx, np.array(crash), crash[-1], 0.02, 3.0, inv, st)  # stop fires
    assert st.in_cooldown
    # next decision: even a perfect gated long must NOT re-open (cooldown)
    series = crash + [crash[-1]]
    out = R.step_with_stop(1, 1, 1, ctx, np.array(series), series[-1], 0.02, 3.0, inv, st)
    assert out["action"] == "cooldown" and inv.position == 0.0


def test_step_stop_overrides_trend_veto():
    # In a strong uptrend the trend-veto blocks SHORTS, but a long that breaks its
    # trailing stop must still flatten. Build long, then dip below stop.
    inv, st = Inventory(cap=3), R.TrailingStop(k=1.0, L=12, cooldown_n=0)
    ctx = _ctx_highvol()
    incs = 0.001 + 0.0005 * np.sin(np.arange(40))
    mids = list(100.0 * np.cumprod(1 + incs))
    for j, m in enumerate(mids):
        R.step_with_stop(1, 1, 1, ctx, np.array(mids[:j + 1]), m, 0.02, 3.0, inv, st)
    assert inv.position > 0
    drop = mids + [mids[-1] * 0.97]
    out = R.step_with_stop(1, 1, 1, ctx, np.array(drop), drop[-1], 0.02, 3.0, inv, st)
    assert out["action"] == "stop" and inv.position == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest executor/paper/test_risk.py -k step -v`
Expected: FAIL — `AttributeError: module 'executor.paper.risk' has no attribute 'step_with_stop'`

- [ ] **Step 3: Add `step_with_stop` to `executor/paper/risk.py`**

At the top of `risk.py`, add the import:

```python
from executor.paper import strategy as S
from executor.paper.strategy import Inventory
```

Then append:

```python
def step_with_stop(s0: int, s1: int, s2: int,
                   scaled_mid_ctx: np.ndarray,
                   mid_decisions: np.ndarray,
                   mid: float, spread: float, fee_bp: float,
                   inv: Inventory, stop: TrailingStop) -> dict:
    """One decision with the trailing-stop overlay. Mutates `inv` and `stop`.

    Order per decision:
      1. tick cooldown
      2. if the trailing stop is breached -> flatten-all (overrides gates/veto)
      3. elif in re-entry cooldown -> mark + hold (no new entries)
      4. else -> normal FROZEN gate decision + inventory accumulate
      5. ratchet/arm the stop's high-water mark with the post-trade position
    Returns {"action", "signal", "position", "net_bp"}.
    """
    stop.tick_cooldown()

    if stop.breached(mid, mid_decisions):
        inv.flatten(mid, spread, fee_bp)
        stop.start_cooldown()
        stop.reset()
        action, signal = "stop", 0
    elif stop.in_cooldown:
        inv.on_decision(0, gated=False, mid=mid, spread=spread, fee_bp=fee_bp)  # mark + hold
        action, signal = "cooldown", 0
    else:
        base = S.base_signal(s0, s1, s2)
        signal = 0
        if (base != 0
                and S.context_vol(scaled_mid_ctx) >= S.VOL_THRESHOLD
                and not S.trend_veto(base, mid_decisions)
                and S.efficiency_ratio(mid_decisions) >= S.ER_MIN):
            signal = base
        inv.on_decision(signal, gated=signal != 0, mid=mid, spread=spread, fee_bp=fee_bp)
        action = "trade" if signal != 0 else "flat"

    stop.update(inv.position, mid)   # arm/ratchet hwm AFTER the trade
    return {"action": action, "signal": signal,
            "position": inv.position, "net_bp": inv.net_pnl_bp}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest executor/paper/test_risk.py -v`
Expected: PASS (all risk tests)

- [ ] **Step 5: Commit**

```bash
git add executor/paper/risk.py executor/paper/test_risk.py
git commit -m "feat(risk): step_with_stop orchestrator (stop overrides entries, cooldown, arm-after-trade)"
```

---

## Task 4: Parquet time-range bounds

The 88-day canonical data is parquet; the walk-forward needs to load a date-bounded window per fold. The DB path already supports `start_time`/`end_time`; mirror it for parquet.

**Files:**
- Modify: `training/data_source.py` (`fetch_parquet_stream` signature + filter)
- Modify: `training/dataset.py:150-152` (pass bounds into the parquet branch)
- Test: `training/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

```python
# training/test_walkforward.py
import sys, datetime as dt
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
import pandas as pd
from training.data_source import fetch_parquet_stream


def test_parquet_time_bounds(tmp_path):
    # tiny synthetic parquet with the columns _build_feature_columns(1) expects
    from training.dataset import _build_feature_columns
    cols = _build_feature_columns(1)              # 1 level -> bid/ask price+vol + mid + spread
    n = 100
    base = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    df = pd.DataFrame({c: np.arange(n, dtype=float) + 1.0 for c in cols})
    df["bucket"] = [base + dt.timedelta(seconds=5 * i) for i in range(n)]
    p = tmp_path / "binance_perp_BTC-USDT.parquet"
    df.to_parquet(p)

    # unbounded -> all rows
    feats, ts = fetch_parquet_stream(str(tmp_path), "binance_perp", "BTC-USDT", 1)
    assert len(feats) == n
    # bounded [t10, t40) -> rows 10..39
    start = base + dt.timedelta(seconds=5 * 10)
    end = base + dt.timedelta(seconds=5 * 40)
    feats_b, ts_b = fetch_parquet_stream(str(tmp_path), "binance_perp", "BTC-USDT", 1,
                                         start_time=start, end_time=end)
    assert len(feats_b) == 30
    assert ts_b[0] == start.timestamp()
    assert ts_b[-1] == (end - dt.timedelta(seconds=5)).timestamp()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest training/test_walkforward.py::test_parquet_time_bounds -v`
Expected: FAIL — `fetch_parquet_stream() got an unexpected keyword argument 'start_time'`

- [ ] **Step 3: Implement bounds**

Replace `training/data_source.py` with:

```python
"""Parquet-backed data source (Colab cannot reach the local TimescaleDB)."""
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

from training.dataset import _build_feature_columns


def fetch_parquet_stream(parquet_dir, exchange, symbol, n_levels,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None):
    """Return (features (T, 4N+2) float64, timestamps (T,) float64) from an exported parquet.

    Optional [start_time, end_time) bounds filter on the `bucket` column (half-open,
    matching the DB path's `bucket >= start AND bucket < end`)."""
    path = Path(parquet_dir) / f"{exchange}_{symbol}.parquet"
    df = pd.read_parquet(path)
    buckets = pd.to_datetime(df["bucket"], utc=True)
    if start_time is not None:
        df = df[buckets >= pd.Timestamp(start_time)]
        buckets = buckets[buckets >= pd.Timestamp(start_time)]
    if end_time is not None:
        mask = buckets < pd.Timestamp(end_time)
        df = df[mask]
    feature_cols = _build_feature_columns(n_levels)
    feats = df[feature_cols].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(df["bucket"], utc=True).astype("int64").to_numpy() / 1e9  # unix seconds
    finite_rows = np.isfinite(feats).all(axis=1)
    feats = feats[finite_rows]
    ts = ts[finite_rows]
    return feats, ts
```

Then in `training/dataset.py`, replace the parquet branch in `_fetch_stream_data` (currently lines ~150-152):

```python
    if getattr(config, "source", "db") == "parquet":
        from training.data_source import fetch_parquet_stream
        return fetch_parquet_stream(config.parquet_dir, exchange, symbol, config.lob_levels,
                                    start_time=start_time, end_time=end_time)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest training/test_walkforward.py::test_parquet_time_bounds -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/data_source.py training/dataset.py training/test_walkforward.py
git commit -m "feat(data): parquet time-range bounds for walk-forward folds"
```

---

## Task 5: Fold-range generation (rolling fixed-window) + leakage asserts

**Files:**
- Create: `training/walkforward.py`
- Test: `training/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Append to `training/test_walkforward.py`:

```python
from training import walkforward as WF


def test_fold_ranges_rolling_no_leakage():
    import datetime as dt
    t0 = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
    t1 = t0 + dt.timedelta(days=88)
    folds = WF.make_folds(t0, t1, train_days=45, test_days=8, tune_days=7)
    assert len(folds) == 5
    prev_test_start = None
    for f in folds:
        # tune ⊂ train, test strictly after train, no overlap
        assert f.train_start < f.tune_start < f.train_end == f.test_start
        assert f.tune_end == f.train_end
        assert f.test_end > f.test_start
        assert (f.train_end - f.train_start) == dt.timedelta(days=45)   # constant width
        # test windows step forward and never overlap training
        assert f.test_start >= f.train_end
        if prev_test_start is not None:
            assert f.test_start > prev_test_start
        prev_test_start = f.test_start


def test_fold_ranges_assert_disjoint():
    import datetime as dt
    t0 = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
    # too-short span yields zero folds rather than overlapping ones
    folds = WF.make_folds(t0, t0 + dt.timedelta(days=40), train_days=45, test_days=8, tune_days=7)
    assert folds == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest training/test_walkforward.py -k fold_ranges -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.walkforward'`

- [ ] **Step 3: Create `training/walkforward.py` with `Fold` + `make_folds`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest training/test_walkforward.py -k fold_ranges -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/walkforward.py training/test_walkforward.py
git commit -m "feat(wf): rolling-window fold generation with leakage asserts"
```

---

## Task 6: Strategy sim over cached signals + metrics

The cheap core: given per-decision arrays (signs, mids, spreads, vol) and a param set, run `step_with_stop` over the slice and return an equity series + metrics.

**Files:**
- Modify: `training/walkforward.py` (add `simulate`, `metrics`)
- Test: `training/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Append to `training/test_walkforward.py`:

```python
def test_simulate_runs_and_metrics_consistent():
    rng = np.random.default_rng(0)
    N = 300
    # clear up-drift (drift ~ noise) so the ER gate admits trades after warmup
    mids = 100.0 * np.cumprod(1 + rng.normal(0.0008, 0.0008, N))
    spreads = np.full(N, 0.02)
    vol = np.full(N, 0.01)                  # clears VOL_THRESHOLD
    signs = np.tile(np.array([1, 1, 1], dtype=np.int8), (N, 1))   # always-up heads
    params = dict(k=2.0, L=12, cooldown_n=2)
    net_series, n_trades = WF.simulate(signs, mids, spreads, vol, params, fee_bp=3.0)
    assert len(net_series) == N
    assert n_trades >= 1
    m = WF.metrics(net_series, n_trades)
    assert set(m) >= {"net_bp", "sharpe", "max_dd_bp", "n_trades", "n_decisions"}
    assert abs(m["net_bp"] - net_series[-1]) < 1e-9
    assert m["n_decisions"] == N


def test_simulate_flat_when_no_signal():
    N = 200
    mids = np.full(N, 100.0)
    spreads = np.full(N, 0.02)
    vol = np.zeros(N)                       # below VOL_THRESHOLD -> never trades
    signs = np.tile(np.array([1, 1, 1], dtype=np.int8), (N, 1))
    net_series, n_trades = WF.simulate(signs, mids, spreads, vol, dict(k=2.0, L=12, cooldown_n=2))
    assert n_trades == 0
    assert abs(net_series[-1]) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest training/test_walkforward.py -k "simulate" -v`
Expected: FAIL — `AttributeError: module 'training.walkforward' has no attribute 'simulate'`

- [ ] **Step 3: Add `simulate` + `metrics`**

Append to `training/walkforward.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest training/test_walkforward.py -k "simulate" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/walkforward.py training/test_walkforward.py
git commit -m "feat(wf): strategy sim over cached signals + Sharpe/DD metrics"
```

---

## Task 7: Inner parameter sweep (tune slice → best params)

**Files:**
- Modify: `training/walkforward.py` (add `param_grid`, `sweep`)
- Test: `training/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Append to `training/test_walkforward.py`:

```python
def test_sweep_picks_best_on_tune_slice():
    rng = np.random.default_rng(1)
    N = 400
    mids = 100.0 * np.cumprod(1 + rng.normal(0.0009, 0.0010, N))
    spreads = np.full(N, 0.02)
    vol = np.full(N, 0.01)
    signs = np.tile(np.array([1, 1, 1], dtype=np.int8), (N, 1))
    grid = WF.param_grid(ks=[1.0, 2.0, 3.0], Ls=[12], cooldowns=[0, 2])
    assert len(grid) == 6
    best, table = WF.sweep(signs, mids, spreads, vol, grid, fee_bp=3.0)
    assert set(best) >= {"k", "L", "cooldown_n"}
    # best must be the argmax by (net_bp, sharpe) over the table
    top = max(table, key=lambda r: (r["net_bp"], r["sharpe"]))
    assert (best["k"], best["L"], best["cooldown_n"]) == (top["k"], top["L"], top["cooldown_n"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest training/test_walkforward.py -k sweep -v`
Expected: FAIL — `AttributeError: ... has no attribute 'param_grid'`

- [ ] **Step 3: Add `param_grid` + `sweep`**

Append to `training/walkforward.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest training/test_walkforward.py -k sweep -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/walkforward.py training/test_walkforward.py
git commit -m "feat(wf): inner param sweep (net-primary, sharpe tie-break)"
```

---

## Task 8: Signal cache I/O (.npz roundtrip)

**Files:**
- Modify: `training/walkforward.py` (add `save_signals`, `load_signals`)
- Test: `training/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Append to `training/test_walkforward.py`:

```python
def test_signal_cache_roundtrip(tmp_path):
    N = 50
    sig = dict(signs=np.tile([1, 0, -1], (N, 1)).astype(np.int8),
               mids=np.linspace(100, 110, N), spreads=np.full(N, 0.02),
               vol=np.full(N, 0.01), ts=np.arange(N, dtype=float))
    p = tmp_path / "fold0_test.npz"
    WF.save_signals(str(p), **sig)
    got = WF.load_signals(str(p))
    for kk in sig:
        assert np.array_equal(got[kk], sig[kk])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest training/test_walkforward.py -k signal_cache -v`
Expected: FAIL — `AttributeError: ... has no attribute 'save_signals'`

- [ ] **Step 3: Add cache helpers**

Append to `training/walkforward.py`:

```python
def save_signals(path: str, signs, mids, spreads, vol, ts):
    np.savez(path, signs=signs, mids=mids, spreads=spreads, vol=vol, ts=ts)


def load_signals(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in ("signs", "mids", "spreads", "vol", "ts")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest training/test_walkforward.py -k signal_cache -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/walkforward.py training/test_walkforward.py
git commit -m "feat(wf): per-decision signal cache I/O"
```

---

## Task 9: Aggregation (stitch OOS folds → report)

**Files:**
- Modify: `training/walkforward.py` (add `aggregate`)
- Test: `training/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Append to `training/test_walkforward.py`:

```python
def test_aggregate_stitches_oos_curve():
    # two folds' OOS net series (each starts at its own 0); stitched curve is cumulative
    f0 = dict(fold=0, net_series=np.array([0.0, 5.0, 8.0]), n_trades=2,
              best=dict(k=2.0, L=12, cooldown_n=2))
    f1 = dict(fold=1, net_series=np.array([0.0, -3.0, 4.0]), n_trades=1,
              best=dict(k=1.5, L=12, cooldown_n=0))
    rep = WF.aggregate([f0, f1])
    # stitched final = 8 + 4 = 12
    assert abs(rep["overall"]["net_bp"] - 12.0) < 1e-9
    assert rep["overall"]["n_folds"] == 2
    assert rep["overall"]["n_profitable_folds"] == 2     # +8 and +4
    assert len(rep["per_fold"]) == 2
    assert rep["per_fold"][0]["net_bp"] == 8.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest training/test_walkforward.py -k aggregate -v`
Expected: FAIL — `AttributeError: ... has no attribute 'aggregate'`

- [ ] **Step 3: Add `aggregate`**

Append to `training/walkforward.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest training/test_walkforward.py -v`
Expected: PASS (all walk-forward pure-piece tests)

- [ ] **Step 5: Commit**

```bash
git add training/walkforward.py training/test_walkforward.py
git commit -m "feat(wf): aggregate per-fold OOS into stitched report"
```

---

## Task 10: `train_fold` — date-bounded model retrain (composes train_v2 helpers)

This runs on GPU/Colab. It composes the existing, battle-tested training helpers rather than refactoring `train_v2.py`. Smoke-tested with a tiny window/epochs; not full TDD (needs GPU + minutes).

**Files:**
- Modify: `training/walkforward.py` (add `train_fold`)
- Test: manual smoke (documented), no unit assertion on model quality

- [ ] **Step 1: Add `train_fold`**

Append to `training/walkforward.py`:

```python
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
```

- [ ] **Step 2: Smoke test (GPU/Colab or CPU with tiny epochs)**

Run (one short fold, 2 epochs, to confirm it trains + writes artifacts):

```bash
.venv/bin/python -c "
import datetime as dt, sys; sys.path.insert(0,'.')
from training.walkforward import Fold, train_fold
f = Fold(train_start=dt.datetime(2026,3,1,tzinfo=dt.timezone.utc),
         tune_start=dt.datetime(2026,4,8,tzinfo=dt.timezone.utc),
         train_end=dt.datetime(2026,4,15,tzinfo=dt.timezone.utc),
         test_start=dt.datetime(2026,4,15,tzinfo=dt.timezone.utc),
         test_end=dt.datetime(2026,4,23,tzinfo=dt.timezone.utc))
print(train_fold(f, '/tmp/wf_smoke', epochs=2))
"
```
Expected: prints `{'best_val_dir_acc': <float>, 'run_dir': '/tmp/wf_smoke'}`; `/tmp/wf_smoke/checkpoints/best.pt`, `scalers.pkl`, `config.json` exist. (On the 8GB local box, use a small window; full retrains run on Colab.)

- [ ] **Step 3: Commit**

```bash
git add training/walkforward.py
git commit -m "feat(wf): train_fold — date-bounded per-fold retrain (composes train_v2 helpers)"
```

---

## Task 11: `precompute_signals` — bounded slice → cached per-decision signals

**Files:**
- Modify: `training/walkforward.py` (add `precompute_signals`)
- Test: manual smoke (uses a trained fold dir)

- [ ] **Step 1: Add `precompute_signals`**

Append to `training/walkforward.py`:

```python
import json
import numpy as np

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
```

- [ ] **Step 2: Smoke test (after Task 10 smoke produced `/tmp/wf_smoke`)**

```bash
.venv/bin/python -c "
import datetime as dt, sys; sys.path.insert(0,'.')
from training.walkforward import precompute_signals, save_signals
sig = precompute_signals('/tmp/wf_smoke', 'binance_perp_BTC-USDT',
    dt.datetime(2026,4,15,tzinfo=dt.timezone.utc), dt.datetime(2026,4,23,tzinfo=dt.timezone.utc))
print({k: v.shape for k,v in sig.items()})
save_signals('/tmp/wf_smoke/sig_btc_test.npz', **sig)
"
```
Expected: prints non-empty shapes; `.npz` written. (Memory-light on the 8GB box because the slice is ~8 days; full runs cache per fold.)

- [ ] **Step 3: Commit**

```bash
git add training/walkforward.py
git commit -m "feat(wf): precompute_signals — bounded slice -> cached per-decision signals"
```

---

## Task 12: `run_walkforward` orchestration + calibration mode

Ties folds → retrain → signal cache → inner sweep on tune → OOS sim on test → aggregate.

**Files:**
- Modify: `training/walkforward.py` (add `run_walkforward`, `__main__`)
- Test: dry-run with `calibrate=True` (single fold) on Colab; pure pieces already covered by unit tests

- [ ] **Step 1: Add `run_walkforward` + CLI**

Append to `training/walkforward.py`:

```python
STREAMS = ["binance_perp_BTC-USDT", "binance_perp_ETH-USDT",
           "binance_perp_SOL-USDT", "binance_perp_WLD-USDT"]


def run_walkforward(data_start, data_end, out_root: str, parquet_dir: str = "lob_data",
                    train_days: int = 45, test_days: int = 8, tune_days: int = 7,
                    streams=STREAMS, epochs: int = 50, calibrate: bool = False) -> dict:
    """Full nested walk-forward. If calibrate=True, runs ONLY fold 0 and reports
    per-fold train time + best_val_dir_acc (compare to the full-history model
    before committing to all folds — spec §5.1 override check)."""
    import time, json
    from pathlib import Path
    folds = make_folds(data_start, data_end, train_days, test_days, tune_days)
    if calibrate:
        folds = folds[:1]
    Path(out_root).mkdir(parents=True, exist_ok=True)
    grid = param_grid()
    fold_results = []
    timings = []
    for fi, fold in enumerate(folds):
        fdir = f"{out_root}/fold{fi}"
        t0 = time.time()
        train_info = train_fold(fold, fdir, parquet_dir=parquet_dir, epochs=epochs)
        timings.append({"fold": fi, "train_secs": time.time() - t0, **train_info})
        # per stream: cache tune + test signals, sweep on tune, evaluate on test
        per_stream_test_nets = []
        chosen = {}
        for stream in streams:
            tune = precompute_signals(fdir, stream, fold.tune_start, fold.tune_end, parquet_dir)
            test = precompute_signals(fdir, stream, fold.test_start, fold.test_end, parquet_dir)
            save_signals(f"{fdir}/{stream}_tune.npz", **tune)
            save_signals(f"{fdir}/{stream}_test.npz", **test)
            best, table = sweep(tune["signs"], tune["mids"], tune["spreads"], tune["vol"], grid)
            chosen[stream] = best
            net, n_tr = simulate(test["signs"], test["mids"], test["spreads"], test["vol"], best)
            per_stream_test_nets.append((net, n_tr))
        # portfolio OOS for the fold = sum of per-stream net series (equal weight)
        maxlen = max(len(n) for n, _ in per_stream_test_nets)
        port = np.zeros(maxlen)
        total_trades = 0
        for net, n_tr in per_stream_test_nets:
            padded = np.concatenate([net, np.full(maxlen - len(net), net[-1] if len(net) else 0.0)])
            port += padded
            total_trades += n_tr
        fold_results.append({"fold": fi, "net_series": port, "n_trades": total_trades,
                             "best": chosen})
    report = aggregate(fold_results)
    report["timings"] = timings
    report["calibrate"] = calibrate
    Path(f"{out_root}/report.json").write_text(json.dumps(report, indent=2, default=list))
    return report


if __name__ == "__main__":
    import argparse, datetime as dt
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-start", required=True)   # ISO date, e.g. 2026-03-01
    ap.add_argument("--data-end", required=True)
    ap.add_argument("--out", default="experiments/walkforward/run")
    ap.add_argument("--parquet-dir", default="lob_data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    ds = dt.datetime.fromisoformat(a.data_start).replace(tzinfo=dt.timezone.utc)
    de = dt.datetime.fromisoformat(a.data_end).replace(tzinfo=dt.timezone.utc)
    rep = run_walkforward(ds, de, a.out, a.parquet_dir, epochs=a.epochs, calibrate=a.calibrate)
    print(json.dumps(rep["overall"], indent=2))
    print("timings:", rep["timings"])
```

- [ ] **Step 2: Determine the real data date range**

Run (to fill `--data-start/--data-end` with the parquet's actual bucket span):

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('lob_data/binance_perp_BTC-USDT.parquet', columns=['bucket'])
print('start', df['bucket'].min(), 'end', df['bucket'].max(), 'rows', len(df))
"
```
Record the span; the calibration command uses it.

- [ ] **Step 3: Commit**

```bash
git add training/walkforward.py
git commit -m "feat(wf): run_walkforward orchestration + calibration mode"
```

---

## Task 13: Colab runner doc + calibration protocol

**Files:**
- Create: `training/colab_walkforward.md`

- [ ] **Step 1: Write the runner doc**

Create `training/colab_walkforward.md` with:
- Mount Drive; `pip install` the project requirements; `git pull` the `feat/trailing-stop-walkforward` branch (or upload).
- Confirm GPU: `import torch; print(torch.cuda.get_device_name())`.
- **Calibration fold first:**
  ```bash
  python training/walkforward.py --data-start <START> --data-end <END> \
      --out experiments/walkforward/calib --epochs 50 --calibrate
  ```
  Read `report.json["timings"][0]`: `train_secs` (× K for the full run) and `best_val_dir_acc`.
  **Override check (spec §5.1):** compare `best_val_dir_acc` to the existing full-history
  `experiments/v2_balanced` val dir-acc. If materially lower (> ~3 pts abs), switch
  `make_folds` to anchored-expanding (set `train_start = data_start` for every fold) before
  the full run.
- **Full run** (after confirming timing/quality, with chosen K via train/test widths):
  ```bash
  python training/walkforward.py --data-start <START> --data-end <END> \
      --out experiments/walkforward/full --epochs 50
  ```
- Copy `experiments/walkforward/full/report.json` back; inspect `overall` (net_bp, sharpe,
  max_dd_bp, n_profitable_folds) and `per_fold`. **Go bar:** profitable in a majority of folds
  with the stop *improving* net/DD vs a stop-disabled control (`k=1e9` makes the stop never fire).

- [ ] **Step 2: Commit**

```bash
git add training/colab_walkforward.md
git commit -m "docs(wf): Colab runner + calibration/override protocol"
```

---

## Task 14: Full local test sweep + branch summary

- [ ] **Step 1: Run the complete local test suite**

Run: `.venv/bin/python -m pytest executor/paper/test_risk.py training/test_walkforward.py -v`
Expected: ALL PASS (Tasks 1-9 pure pieces). Tasks 10-12 are GPU/Colab smoke-verified.

- [ ] **Step 2: Confirm the existing strategy tests still pass (no frozen-behavior regression)**

Run: `.venv/bin/python -m pytest executor/paper/ -q`
Expected: previously-green tests still pass (the only strategy.py change was the additive `flatten`).

- [ ] **Step 3: Commit (if any test-only fixes were needed) and push the branch when ready**

```bash
# Targeted adds ONLY — never `git add -A` (executor/, lob_data/, *.zip are untracked and large).
git add executor/paper/risk.py executor/paper/test_risk.py executor/paper/strategy.py \
        training/walkforward.py training/test_walkforward.py training/data_source.py \
        training/dataset.py training/colab_walkforward.md .gitignore \
        docs/superpowers/specs/2026-06-15-trailing-stop-walkforward-design.md \
        docs/superpowers/plans/2026-06-15-trailing-stop-walkforward.md
git commit -m "test(wf): full local suite green" || echo "nothing to commit"
# Do NOT push without explicit user go-ahead (origin/main has diverged).
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** trailing stop §3 → Tasks 1-3; data bounds §4.4 → Task 4; rolling folds + leakage §4.1/§4.3 → Task 5; inner sweep §4.2 → Task 7; cost model → `simulate(fee_bp=...)` + maker bracket via re-running with lower fee; signal cache/data-flow §5 → Tasks 8/11; calibration + override §5.1 → Tasks 12/13; aggregation/report §4.1 → Task 9.
- **Type consistency:** `Fold` fields, `simulate(...)→(net_series, n_trades)`, `metrics(...)` keys, `sweep(...)→(best, table)`, signal-dict keys `signs/mids/spreads/vol/ts` are used identically across Tasks 6-12.
- **Winsorize:** intentionally skipped in `precompute_signals` to match the live path (documented in code + spec); flagged as a refinement, not a gap.
- **Maker bracket:** produced by re-running `simulate`/`sweep` with `fee_bp` lowered (e.g., 1.0); add to the report in Task 12 if desired — currently taker-only `FEE_BP=3.0` is the headline, matching the live harness.
