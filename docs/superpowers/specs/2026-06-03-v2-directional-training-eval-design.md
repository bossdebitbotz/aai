# V2 Directional Training & Evaluation Pipeline — Design

- **Date:** 2026-06-03
- **Status:** Approved design (pre-implementation)
- **Author:** Shannon Maxwell (with Claude)
- **Supersedes/feeds:** `phase_improvement_plan.md` (Phases 1–2 implementation), `masterplan.md` (strategy direction)

---

## 1. Problem Statement

The project has ~83 days of clean, 16-stream, 40-level LOB data in TimescaleDB and a well-built, unit-tested **V2** model stack (`model_v2.py`, `features_v2.py`) that adds a direction-classification head, directional sign loss (DSL), causal temporal attention, multi-horizon labels, and momentum features. The goal of V2 was to fix the 30-day baseline's **50.5% directional accuracy** (coin-flip) despite R²=0.997 on price levels.

A readiness audit (2026-06-03, 7-agent workflow) found that **V2 cannot currently be trained or proven**, due to three hard blockers and one operational blocker:

1. **No V2 training path.** `train.py` and `dataset.py` import and run the **V1** model, loss, and features (211 features, direction-blind `CompoundAttentionModel`). The V2 modules are reachable only from the live `signal_service.py`. There is no `train_v2.py`. Training today would train the 50.5% V1 model.
2. **No held-out evaluation of directional accuracy.** `train.py` reports only MSE. The only accuracy number that exists (~57–62%) is **train-split** accuracy measured while the model was overfitting (val loss 1.10 > train 0.83), in an **uncommitted Colab notebook**. We cannot currently answer "does V2 beat 50.5% on unseen data?"
3. **Feature-engineering-before-split leakage.** Savitzky-Golay smoothing (non-causal, centered) and momentum/OFI diffs are computed over the full stream **before** the chronological split (`dataset.py:420` precedes `:427-433`), bleeding val/test info into train at the seams. `savgol_window` is also hardcoded to 21 — the over-smoothing value the improvement plan says destroys directional signal.
4. **Operational: stale export.** `lob_data/*.parquet` is dated Apr 11 and misses ~64% of available data; `export_training_data.py` has a size-based skip guard that **silently keeps stale files** on re-run.

### Strategic decision (made during design)

The audit also surfaced a strategy mismatch: the masterplan names **basis (spot-vs-perp) trading** as primary, but a 30s–2min **directional** forecast is largely orthogonal to basis/funding (a carry trade on an 8h+ horizon), and **Binance hedge mode is a position-accounting feature irrelevant to the model**. The model's actual output fits **confidence-gated directional perpetual scalping** — which the executor already partly implements.

**Decision:** commit to **directional perp scalping** as the primary strategy for this cycle. Basis/funding and hedge mode are explicitly deferred to a separate future sleeve (would require funding-rate data that is not currently collected).

---

## 2. Goal & Exit Criteria

**Goal:** Make V2 trainable on the full available data and produce an **honest, reproducible, calibrated** assessment of directional performance **and net profitability after fees** on a held-out test set.

**Exit criteria (this cycle is "done" when):**
- A `train_v2.py` run trains `CompoundAttentionModelV2` + `LOBLossV2` + `engineer_features_v2` end-to-end on the fresh ~83-day export, with no train/val/test leakage, logging **val/test directional accuracy per horizon** each epoch.
- `evaluate_v2.py` reports, on the held-out **test** split: per-horizon directional accuracy, 3×3 confusion matrix, per-class precision/recall/F1, a **calibration** report (reliability diagram + ECE, before/after temperature scaling), and a **calibrated-confidence → accuracy/coverage** curve — with the **V1 baseline re-measured on the same test split and label definition** for an apples-to-apples comparison to 50.5%.
- `backtest_v2.py` reports net PnL / win-rate / avg-edge-bps / coverage / Sharpe / max-drawdown on the test split under **both** a maker-optimistic and a taker-conservative cost model, plus a **calibrated-confidence → net-PnL** curve, per horizon and per symbol (BTC/ETH Binance perp).
- All artifacts (checkpoint with embedded metadata + per-horizon temperatures, scalers, metrics, reports) are reproducible from a committed, **resumable** Colab notebook.

**Non-goals (explicitly deferred):**
- Cross-exchange / lead-lag features (`compute_cross_exchange_features` stays uncalled this cycle).
- Basis/funding strategy and any funding-rate data collection.
- Binance hedge mode / dual-side positions.
- Live execution wiring, Freqtrade changes, or production deployment of the new model.
- Hyperparameter sweeps (a single well-configured production run + early stopping).

---

## 3. Scope Summary

| # | Component | Type | New/Changed |
|---|-----------|------|-------------|
| C1 | Data export refresh | Core | Change `export_training_data.py` |
| C2 | Dataset pipeline (leakage fix + V2 features) | Core | Change `dataset.py` |
| C3 | V2 training entry point | Core | New `train_v2.py` |
| C4 | Confidence calibration | Optional (in) | Part of C5; embedded in checkpoint |
| C5 | Evaluation harness | Core | New `evaluate_v2.py` |
| C6 | Offline PnL backtest | Optional (in) | New `backtest_v2.py` |
| C7 | Resumable Colab notebook | Core | New / committed |

---

## 4. Design Decisions (recorded)

- **D1 — V2 training lives in a new `train_v2.py`; `dataset.py` is parameterized** (feature-version switch). V1 `train.py`/`model.py` remain runnable so the **50.5% baseline can be re-measured** apples-to-apples. *(Alt rejected: edit `train.py` in place — loses the baseline runner.)*
- **D2 — Leakage fix = split first, then engineer features independently per split** (fit scaler on train, transform val/test); trim the first ~`max(savgol_window, max horizon)=~60` warm-up rows per split. `savgol_window 21 → 11`. *(Alt: causal smoothing + guard band — changes SG semantics, fiddlier.)*
- **D3 — Calibration = temperature scaling, one temperature per horizon**, fit on validation (minimize NLL), verified with reliability diagrams + ECE. Argmax (accuracy) is unchanged by construction. Temperatures embedded in the checkpoint for serve-time use.
- **D4 — Best-checkpoint selection = mean validation directional accuracy across the 3 horizons** (tie-break: val total loss). *(Was: val MSE — wrong objective for a directional model.)*
- **D5 — Train on all 16 streams** (exchange/symbol embeddings share microstructure → more data); **report accuracy per-stream**; treat **Binance perp BTC/ETH** as the tradable target. *(Alt: binance-perp-only — less data, weaker embeddings.)*
- **D6 — Offline backtest replays the held-out test split**, gated on **calibrated** confidence, with **both** maker-optimistic (~2 bps/side) and taker-conservative (~5 bps/side) cost models reported to bracket the true edge. Not wired into Freqtrade.

---

## 5. Component Design

### C1 — Data export refresh (`export_training_data.py`)

**Current behavior:** exports `bucket, bid/ask price/vol ×40, mid_price, spread` from `lob_5s` per stream; skips any stream whose parquet already exists and is >1000 bytes (`:66-69`).

**Changes:**
1. **Add a `--fresh` flag (default true for production runs)** that deletes existing `lob_data/*.parquet` and `lob_training_data.zip` before exporting; keep the skip-guard only behind an explicit `--resume`. Add a freshness assertion: compare `max(bucket)` in `lob_5s` for each stream vs. the existing parquet's last bucket; warn loudly if a kept file is >1 day stale.
2. **Add `all_valid` to the SELECT** (the `lob_5s` continuous aggregate already exposes `all_valid = bool_and(is_valid)`), and **drop rows where `all_valid = false`** at export time (or export the column and drop in C2 — decided: drop at export to keep parquet clean). This removes the ~4,787 crossed-book buckets/week from training silently entering.
3. **Spread winsorization** is applied downstream in C2 (not at export), since it depends on per-split stats.
4. **DB-contention guardrail:** the live signal service reads the same container. Keep the existing retry/recovery logic; document running the export during a low-activity window and with bounded `work_mem`.

**Output:** 16 fresh parquet files spanning the full ~83-day range + `lob_training_data.zip` for Colab upload.

### C2 — Dataset pipeline (`dataset.py`)

**Parameterize the feature pipeline.** Add to `DataConfig`:
- `feature_version: str = "v2"` (selects `engineer_features_v2` vs `engineer_features`)
- `savgol_window: int = 11`
- `n_features` computed property: V2 → `n_levels*5 + 19` (= **219** at 40 levels); V1 → `n_levels*5 + 11` (= 211).
- `warmup_trim: int` = `max(savgol_window, max(direction-feature horizons)=60)` — rows trimmed at the head of each split.

**Fix leakage — reorder `build_dataloaders` (currently `:419-443`):**

```
# OLD (leaky):
features = engineer_features(full_stream, savgol_window=21)   # full series
train/val/test = chronological_split(features)
scaler.fit_transform(train); transform(val); transform(test)

# NEW (leak-free):
train_raw, val_raw, test_raw = chronological_split(raw_features)     # split FIRST
fe = engineer_features_v2 if cfg.feature_version=="v2" else engineer_features
train_fe = fe(train_raw, savgol_window=cfg.savgol_window)[warmup_trim:]
val_fe   = fe(val_raw,   savgol_window=cfg.savgol_window)[warmup_trim:]
test_fe  = fe(test_raw,  savgol_window=cfg.savgol_window)[warmup_trim:]
scaler.fit_transform(train_fe); scaler.transform(val_fe); scaler.transform(test_fe)
```

Each split's features are now computed only from data within that split; the warm-up trim removes the zero-initialized leading rows of momentum/SG features. Scaler remains fit-on-train-only (already correct).

**Spread/outlier winsorization:** after fit, clip extreme `spread`/derived values to train-set percentile bounds (e.g. [0.1%, 99.9%]) using train stats only; apply to all splits. Guards against residual crossed-book artifacts not caught by `all_valid`.

**Direction labels in the batch.** `LOBLossV2` derives labels internally from `target` + `context_last`, so the **batch dict does not need explicit labels** — but `__getitem__` must continue to provide `context`, `target`, `exchange_id`, `symbol_id`. `train_v2.py` passes `context[:, -1, :]` as `context_last`. (No dataset change needed beyond the feature pipeline.)

**Fix `stream_info['train_windows']` bug** (`:466`): track the just-created train dataset in a local variable instead of the fragile `train_datasets[-1].exchange == ...` guard.

**Partial-depth completeness:** in `_fetch_stream_data`, require all bid/ask price levels > 0 (not just `bid_price_40`); log the fraction of rows dropped per stream.

### C3 — V2 training entry point (`train_v2.py`)

A sibling to `train.py`, importing the V2 stack:

```python
from training.model_v2 import CompoundAttentionModelV2, LOBLossV2
from training.dataset import build_dataloaders, DataConfig
# DataConfig(feature_version="v2", savgol_window=11, ...)
```

**Construction:**
- `model = CompoundAttentionModelV2(n_levels=40, n_features=219, context_length=120, prediction_length=24, ...)` — **n_features=219 must be explicit** (default is 211; the direction head is `Linear(d_model*n_features, …)` and will silently mis-size otherwise).
- `loss_fn = LOBLossV2(n_levels=40, use_feature_weights=True, mid_price_idx=160, direction_horizons=(5,11,23), scaler_means=..., scaler_stds=...)` — `mid_price_idx=160` is asserted by `test_mid_price_idx` (§8), not assumed.

**Train/eval loop contract (differs from V1):**
```python
pred, dir_logits = model(context, exchange_ids, symbol_ids)
total, forecast, structure, direction, dsl = loss_fn(pred, target, dir_logits, context[:, -1, :])
```

**Phase-1e gradient accumulation:** add `--accum-steps` (default 8). Scale loss by `1/accum_steps`; step optimizer + scheduler every `accum_steps` mini-batches. Target effective batch 128 with physical batch 16.

**Directional-accuracy metric (train + val + test):** in `evaluate()`, compute per-horizon `argmax(dir_logits_3h)` vs `LOBLossV2._direction_labels(target, context_last)`; report per-horizon and mean accuracy. Pass into `tracker.log_epoch(extra={...})`.

**Checkpoint selection (D4):** `best.pt` chosen on **mean val directional accuracy** (tie-break val total loss), not val MSE.

**Resume:** add `--resume` → `tracker.load_checkpoint(model, optimizer, "latest.pt")`, restore `start_epoch`, and **persist scheduler `step_count`** (extend `tracker.py` save/load — currently omits scheduler state).

**Checkpoint contract (embed, so serving can't silently default):** `n_features`, `context_length`, `prediction_length`, `direction_horizons`, `mid_price_idx`, `feature_version`, and (after C4) per-horizon `temperatures`.

### C4 — Confidence calibration (in `evaluate_v2.py`, applied at serve time)

The direction head emits 9 logits → reshape `(B,3,3)` → per-horizon 3-class softmax. Confidence = max softmax prob.

- **Fit:** for each horizon `h`, find temperature `T_h > 0` minimizing NLL of `softmax(logits_h / T_h)` against val labels (1-D optimization, e.g. LBFGS or scalar minimize).
- **Apply:** calibrated probs = `softmax(logits_h / T_h)`. Argmax unchanged ⇒ **accuracy unaffected**; only confidence values are corrected.
- **Verify:** reliability diagram (confidence-bucket vs empirical accuracy) + **ECE** before/after; expect post-calibration ECE materially lower and the curve near the diagonal.
- **Persist:** `T_h` embedded in the checkpoint; `signal_service.py` (future, out of scope to wire) divides logits by `T_h` before thresholding so the live `conf ≥ 0.54/0.7` gates become real probabilities.

### C5 — Evaluation harness (`evaluate_v2.py`)

Loads `best.pt` (+ scalers), builds the **test** loader from `build_dataloaders` (same chronological split), and produces a committed, reproducible report:

1. **Per-horizon directional accuracy** (30s/1m/2m ≈ steps 5/11/23), overall and **per stream**.
2. **3×3 confusion matrix** and **per-class precision/recall/F1** per horizon.
3. **Calibration (C4):** fit temperatures on val, report reliability + ECE before/after on test.
4. **Calibrated-confidence → accuracy & coverage curve** per horizon (sweep thresholds 0.4→0.95): accuracy among acted-on samples and fraction of bars acted on. Validates/sets the live thresholds (`0.54`, `0.85/0.9`).
5. **Baseline re-measure (apples-to-apples):** retrain the V1 model on the **same fresh, leak-free data/split** (via D1's preserved `train.py`/`model.py`, reading the parquet path) and measure its per-horizon directional accuracy on the **same test split** — isolating the V2 architecture's contribution. V1 has **no direction head**, so its directional accuracy is defined as `sign(predicted mid-price change over horizon h)` vs `sign(actual)`, using the **same `flat_threshold` and horizons** as `LOBLossV2._direction_labels` — this is the reproducible definition of the original 50.5% baseline. Report V1 vs V2 side-by-side per horizon.
6. Emits JSON + PNG artifacts under `experiments/<run>/eval/`.

### C6 — Offline PnL backtest (`backtest_v2.py`)

Offline replay of the **held-out test split** (out-of-sample). **Not** wired into Freqtrade; it is the single source of truth for execution assumptions, making the live service's "MATCH BACKTEST EXACTLY" claim verifiable.

- **Signal → trade:** per-bar **calibrated** direction probs; enter long/short when calibrated `conf ≥ threshold` at the primary 2-min horizon; **120s hold**, **300s cooldown**, exit at hold expiry or reversal. Same mid-price reference / label definition as training.
- **Cost/fill model (both, to bracket the edge):**
  - **Maker-optimistic:** ~2 bps/side; fills assumed when price trades through the quote.
  - **Taker-conservative:** ~5 bps/side; guaranteed fills (cross the spread).
- **Config block** mirrors `signal_config.json` (fees, hold, cooldown, gate, leverage=1x, venue=binance_perp, symbols=BTC/ETH) — one source of truth.
- **Outputs:** net PnL, win-rate, avg edge/trade (bps), trade count, **coverage** (% bars traded), Sharpe, max drawdown — per horizon and per symbol; plus a **calibrated-confidence → net-PnL** curve so the gate is chosen on money, not accuracy. Artifacts under `experiments/<run>/backtest/`.

### C7 — Resumable Colab notebook

Committed notebook (or `.py` + jupytext) that:
1. Mounts Google Drive; downloads/extracts the fresh `lob_training_data.zip` (parquet, not DB — Colab can't reach local TimescaleDB).
2. Adds a **parquet-backed fetch path** so `build_dataloaders` reads the exported parquet instead of `lob_5s` when running on Colab (new `DataConfig.source="parquet"` branch in `_fetch_stream_data`).
3. GPU check (**require high-RAM A100**; ~20 GB resident for the 219-feature ×16-stream array), runs `train_v2.py` with `--accum-steps 8 --resume`, then `evaluate_v2.py` and `backtest_v2.py`.
4. Saves checkpoint + scalers + eval/backtest artifacts back to Drive as `aai_<span>_v2_results.zip`.
5. Documents session-length risk (~3× the 30-day run ≈ 18–20h for ~40 epochs) → rely on `--resume` + early stopping.

---

## 6. Data Flow

```
TimescaleDB lob_5s (all_valid filtered)
   │  export_training_data.py --fresh        [C1]
   ▼
lob_data/*.parquet  ──zip──►  Google Drive
   │  Colab: parquet-backed _fetch_stream_data   [C7,C2]
   ▼
chronological split (raw)  →  engineer_features_v2 per split  →  warmup trim  →  scaler(fit train)   [C2 leak-free]
   ▼
LOBDataset (gap-skipped windows) ── train_v2.py ──►  CompoundAttentionModelV2 + LOBLossV2   [C3]
   │   (grad accum, val/test dir-acc, best=val dir-acc, resumable)
   ▼
best.pt (+embedded meta) + scalers
   ├── evaluate_v2.py  → accuracy / confusion / calibration(T_h) / conf-coverage / V1 baseline   [C4,C5]
   └── backtest_v2.py  → PnL/Sharpe/coverage (maker+taker) / conf-PnL curve                       [C6]
```

---

## 7. Error Handling & Correctness Requirements

- **`mid_price_idx` guard (blocker-class):** integration test asserting the mid-price column index in `engineer_features_v2` output equals `LOBLossV2.mid_price_idx` (160 expected) — prevents silently mislabeled direction targets.
- **n_features guard:** assert `model.n_features == features.shape[-1]` at train start; fail loudly rather than zero-pad.
- **Export freshness guard:** fail/warn if exported parquet max-bucket lags DB max-bucket by >1 day.
- **Leakage guard:** unit test that feature values at a split's first post-trim row do not depend on any data from the prior split (e.g. identical features whether or not the prior split is prepended).
- **Calibration sanity:** assert post-calibration accuracy == pre-calibration accuracy (argmax invariance) and ECE_after ≤ ECE_before on val.
- **Resume correctness:** restored run reproduces the same loss trajectory as an uninterrupted run for the overlapping epochs (within fp tolerance).
- **DB contention:** export tolerates the concurrently-running live signal service (existing retry/recovery retained).

---

## 8. Testing Plan

| Test | Asserts |
|------|---------|
| `test_export_fresh` | `--fresh` clears stale files; freshness warning fires on stale parquet; `all_valid=false` rows excluded |
| `test_dataset_v2_pipeline` | `feature_version="v2"` → 219 features; warmup trim length; per-split FE produces no cross-split dependence (leakage guard) |
| `test_dataset_scaler` | scaler fit on train only (existing) still holds post-reorder |
| `test_train_v2_smoke` | one step end-to-end: forward returns `(pred,dir_logits)`; 5-tuple loss; grad-accum steps optimizer every N; dir-acc computed |
| `test_mid_price_idx` | mid-price column index == `LOBLossV2.mid_price_idx` |
| `test_calibration` | argmax invariance; ECE_after ≤ ECE_before on a fixture |
| `test_evaluate_v2` | finite per-horizon accuracy; confusion matrix sums to N; conf-coverage monotonic in coverage |
| `test_backtest_v2` | maker & taker runs both produce finite PnL; coverage ∈ [0,1]; cost ordering (taker net ≤ maker net) |
| `test_resume` | resumed run matches uninterrupted loss for overlapping epochs |

Existing V1 tests (`test_dataset.py`, `test_model.py`, `test_features.py`) must continue to pass (V1 preserved for baseline).

---

## 9. Risks & Assumptions

- **Near-breakeven economics:** raw 53–58% accuracy against 4–10 bps round-trip means the tradable edge lives only in the high-confidence regime; the backtest may show **no positive net PnL under the taker model**. That is a valid, informative outcome — the cycle's job is to find out honestly, not to guarantee profit.
- **Maker fills are not guaranteed:** the maker-optimistic backtest is an upper bound; real fill rate/adverse selection is not modeled (deferred). Reported as the ceiling, with taker as the floor.
- **Overfitting risk on the direction head** (~3.8M params): mitigated by val-accuracy-gated checkpointing, early stopping, dropout; watch val vs train dir-acc divergence.
- **Time embedding assumes uniform spacing** (`forward` uses `linspace(0,1,T)`): valid because windows spanning >12.5s gaps are skipped (existing, verified).
- **83 vs 90 days:** proceeding now; pipeline is re-runnable when more data accrues.
- **Separate urgent issue (out of scope here, flagged):** real Binance API key+secret are committed in `executor/config.json` + `.env` and in git history; the executor ran live. **Rotate/revoke keys, move to untracked secrets, scrub history** — independent of this spec.

---

## 10. Deferred / Future Work

- Cross-exchange features (`compute_cross_exchange_features`) as directional inputs — requires synchronized multi-stream fetch on a common grid.
- Basis/funding sleeve (delta-neutral, hedge mode) — requires funding-rate data collection; model used only for leg timing.
- Live execution: wire calibrated thresholds + retrained checkpoint into `signal_service.py`; reconcile config drift (stake/threshold/leverage); add supervisor + liveness alerting; default `dry_run:true` gate.
- Realistic maker fill model (queue position, adverse selection) for the backtest.
