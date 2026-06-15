# Trailing Stop + Nested Walk-Forward Backtest — Design

- **Date:** 2026-06-15
- **Status:** Draft for review
- **Author:** Shannon Maxwell / Claude
- **Related:** `training data/FROZEN_strategy_spec_2026-06-14.md`, `executor/paper/strategy.py`, `training/backtest_v2.py`, `executor/paper/signal_pipeline.py`

## 1. Motivation

The live paper trader runs an **inventory-reversion** strategy: accumulate to ±3 units on a
gated signal, hold, and exit only when an opposing gated signal fires. It has **no protective
exit** — no stop, no take-profit — and during strong trends the trend-veto actively suppresses
the counter-signal, so a position can ride a move all the way up *and* all the way back down.

Two problems this design fixes:

1. **No downside protection.** A position that turns adverse bleeds unrealized PnL until the
   model flips. We want to lock in favorable excursions without giving up the ability to add.
2. **Backtest ↔ live exit mismatch.** The only rigorous committed backtest (`backtest_v2.py`)
   tested a *fixed 2-minute hold*, not the inventory logic the bot actually runs. We have never
   walk-forward-validated the strategy we are trading. The old inventory research was profitable
   in only **2 of 6** windows — that inconsistency is the bar to beat.

## 2. Goals / Non-Goals

**Goals**
- Add a **position-level, volatility-scaled trailing stop** that protects gains while still
  allowing the position to pyramid to the cap.
- Build a **nested, rolling-window, model-retrained walk-forward harness** that evaluates the
  *actual* inventory + trailing-stop strategy with **zero leakage**, and reports per-fold and
  aggregate out-of-sample (OOS) performance.
- Keep both components behind clean, unit-testable interfaces.

**Non-Goals**
- Not changing the **entry** gate logic (3-horizon agreement → vol → trend-veto → ER) in this
  phase. Entry gates stay FROZEN so we can isolate the stop's marginal effect.
- Not a tick-level fill simulator. Fills remain at the decision-bucket mid with a taker-cost
  bracket (consistent with the existing harness); a realistic fill model is deferred.
- Not modifying the running live bot until the walk-forward validates the stop. Once validated,
  the same `risk.py` component is ported to the live harness (a follow-up).

## 3. Component A — Trailing Stop (`executor/paper/risk.py`)

A **risk overlay** layered on top of the existing gate+inventory brain. It does not change
entries; it only adds a hard exit. Pure function of (position, decision-mid series) — no model
or DB dependency, unit-testable in isolation, the same discipline as `strategy.py`.

### 3.1 Semantics

State per stream/position:
- `side` — sign of the open position (+1 long / −1 short).
- `hwm` — favorable extreme of the **decision-mid** since the position opened
  (running `max` for a long, `min` for a short). Ratchets in the favorable direction only.
- `cooldown` — decisions remaining before a new entry is allowed after a stop-out.

Per decision, in chronological order:
1. **Volatility estimate** `σ` = standard deviation of **decision-mid log-returns** over a
   trailing lookback `L` decisions. Because decisions are ~2 min apart (`STRIDE=23` ≈ 115 s),
   this is literally **2-minute-bar volatility** — the "2-minute window" the stop is built on.
2. **Stop level** (fractional distance `d = k · σ`, floored at `d_min` to avoid a zero-width
   stop in dead-vol regimes):
   - long:  `stop = hwm · (1 − d)`
   - short: `stop = hwm · (1 + d)`
3. **Breach test:** long fires if `mid ≤ stop`; short fires if `mid ≥ stop`.
4. **On breach → flatten the entire position** (target = 0), set `cooldown = C`, reset state.
   This **overrides the trend-veto** — a stop is a hard risk control and must be able to exit
   mid-trend (exactly the case the current bot cannot).

### 3.2 Interaction with entries (the "add while not losing gains" behavior)

While a position is open and **above** its stop, the normal gated signal can still pyramid to
±`CAP`. Each add continues to push `hwm` in the favorable direction, so the locked-in floor
**rises with the position**. After a stop-out, entries are suppressed for `C` decisions
(re-entry guard) to avoid whipsaw, then normal entry resumes.

### 3.3 Decision flow (orchestrator `decide_with_stop`)

```
1. signs + gates  → desired add signal      (existing strategy.decide_signal — unchanged)
2. inventory.mark(mid); stop.update(position, mid)   (ratchet hwm)
3. if stop.breached(mid, position): target = 0; start cooldown        # EXIT overrides everything
   elif in cooldown:                target = position (hold, no add)
   else:                            apply gated add via inventory      # existing accumulation
```

`strategy.py` stays FROZEN; the overlay wraps it. The same `decide_with_stop` is used by the
walk-forward sim now and ported to the live harness later.

### 3.4 Swept parameters (selected by the inner loop, never hand-tuned)

| Param | Meaning | Grid (initial) |
|---|---|---|
| `k` | stop distance in σ units | {1.0, 1.5, 2.0, 2.5, 3.0} |
| `L` | σ lookback (decisions) | {12, 24} |
| `C` | re-entry cooldown (decisions) | {0, 2, 5} |
| `d_min` | floor on stop distance (fraction) | fixed 0.0005 (5 bp) |

Entry-gate constants (`ER_MIN`, `VOL_THRESHOLD`, `TREND_*`, `CAP`) remain frozen this phase.

## 4. Component B — Nested Walk-Forward Harness (`training/walkforward.py`)

### 4.1 Structure (rolling fixed-window, nested)

```
 rolling window: train (fixed width W) → tune (inner, last portion of train) → test (OOS)
 f1: TRAIN[ 0..45] tune[38..45] TEST[45..53]
 f2: TRAIN[ 8..53] tune[46..53] TEST[53..61]
 f3: TRAIN[16..61] tune[54..61] TEST[61..69]
 f4: TRAIN[24..69] tune[62..69] TEST[69..77]
 f5: TRAIN[32..77] tune[70..77] TEST[77..85]
 (days; ~5 folds over 88 days. step = test width.)
```

- **Outer loop (model honesty):** for each fold, **retrain the model** on the train window, then
  evaluate the strategy on the *next, unseen* test slice. Every test trade is genuinely
  model-out-of-sample. **Rolling fixed-width** (not anchored-expanding) so (a) it mirrors the
  realistic production cadence of retraining on a trailing window, and (b) train-set size is
  constant across folds, so cross-fold differences reflect regime/strategy — not data volume.
- **Inner loop (strategy honesty):** the trailing-stop params are selected on the `tune` slice
  (the tail of the train window), then applied **unchanged** to the OOS `test` slice. The test
  slice is never touched during tuning.
- **Aggregation:** concatenate the K OOS test slices into one continuous equity curve = the
  honest number. Report **per-fold** metrics (net return, Sharpe-per-trade, win rate, max DD,
  trade count, how many folds are profitable) so consistency is visible.
- **Cost model:** reuse the existing `Inventory` accounting — taker cost of `half-spread +
  FEE_BP` (3 bp) per unit traded, marked-to-mid PnL. Also report a maker-optimistic bracket
  (lower fee) alongside, so we see the result under both fill assumptions (as `backtest_v2.py`
  did). The stop counts as a taker exit (it crosses the spread to flatten).

### 4.2 Inner objective

Select params maximizing OOS-robust performance on the `tune` slice. Primary objective:
**net return after taker cost**; tie-break / regularize toward **Sharpe-per-trade** and toward
**flat regions of the param surface** (a value whose neighbors also do well) to avoid picking a
fragile spike. Record the full tune-slice surface for inspection.

### 4.3 Leakage safeguards (asserted in code)

- `assert test_start >= train_end` for every fold (no train/test overlap).
- Tune slice ⊂ train window; test slice ∩ train window = ∅.
- The fold model is the only model used for that fold's test signals.
- Feature engineering uses the **30 s decision-lag** convention so the centered-savgol window is
  causal at decision time (already proven bit-exact in `feature_parity_check.py`).

### 4.4 Required supporting change

`training/dataset.py` / `DataConfig` must accept **time-range bounds** (`start`, `end` on the
`bucket` column) so each fold trains on its window only. Today it loads the full parquet/DB.
This is a small, well-scoped extension with its own test.

## 5. Data Flow & Compute

```
 per fold (GPU / Colab):  train model on window  ──► fold checkpoint
                          run model over tune+test ─► cached signs .npz  (3-horizon, per decision)
 per fold (CPU / local):  inner sweep on tune slice (cheap, no model)  ──► best params
                          apply best params to test slice              ──► OOS trades/equity
 aggregate (local):       stitch OOS slices ─► experiments/walkforward/<ts>/report.json + per-fold
```

The **expensive** steps (retrain + signal precompute) run on **Colab GPU**; the cached signals
are the handoff artifact. The **cheap** steps (stop sim, sweep, aggregation) run anywhere —
pure NumPy, seconds. This is why the sweep can be exhaustive without more GPU.

### 5.1 Compute plan — calibration fold first

Before committing to K retrains: run **one calibration fold** to (a) measure real per-fold train
time on the Colab GPU, and (b) compare the 45-day-window model's eval metrics against the
existing full-history `v2_balanced` model. **Override check:** if the shorter-window model is
materially weaker (it would contaminate the strategy test), switch the outer loop to
anchored-expanding. Only then run the full K≈5.

## 6. File Layout / Deliverables

| File | Purpose |
|---|---|
| `executor/paper/risk.py` | `TrailingStop` + `decide_with_stop` orchestrator |
| `executor/paper/test_risk.py` | unit tests (ratchet, breach, flatten-all, cooldown, long/short, adds-while-above, veto-override) |
| `training/walkforward.py` | fold-range generation, retrain hook, signal cache, inner sweep, OOS sim, aggregation |
| `training/test_walkforward.py` | range/leakage asserts, signal-cache roundtrip, aggregation stitching |
| `training/dataset.py` (edit) | time-range bounds on `DataConfig` (+ test) |
| `training/colab_walkforward.md` | Colab runner cells (reuse `colab_v2_runner.py` pattern) |
| `experiments/walkforward/<ts>/` | `report.json`, per-fold checkpoints, cached signals, param surfaces |

## 7. Testing Strategy

- **TrailingStop unit tests:** hwm ratchets favorable-only; fires on breach long & short;
  flatten-all on breach; cooldown suppresses re-entry then releases; resets on flat; adds occur
  while above stop; stop fires even when trend-veto would block the counter-signal; `d_min` floor
  prevents zero-width stop.
- **Harness tests:** fold ranges never overlap (leakage assert); tune ⊂ train, test disjoint;
  signal-cache save/load roundtrip is exact; aggregation stitches K slices into the right
  cumulative curve; degenerate fold (no trades) handled.
- **Determinism:** fixed seeds for retrains; harness reproducible given cached signals.

## 8. Risks & Mitigations

- **Retrain cost/time** — mitigated by the calibration fold; K chosen with eyes open.
- **Short-window model quality** — calibration override check (§5.1).
- **σ cold-start / dead-vol** — `d_min` floor; require `L` decisions of history before the stop
  arms (until then, no stop — position behaves as today).
- **Stop-param overfit** — small grid, inner tuning on a held-out tune slice, prefer flat regions
  of the surface; report tune-vs-test degradation per fold.
- **Dataset date-bounding correctness** — dedicated test asserting row counts/time bounds.

## 9. Open Questions (for review)

- **K (fold count):** recommend ~5; confirm after the calibration fold's timing.
- **Window widths:** train 45 d / test ~8 d proposed; adjustable.
- **Inner objective weighting:** net-return primary, Sharpe tie-break — acceptable?
- Should phase 2 also **re-sweep entry gates** (`ER_MIN`, `VOL_THRESHOLD`), or keep them frozen
  to isolate the stop? (Default: frozen this phase.)
