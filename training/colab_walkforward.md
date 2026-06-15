# Colab Runner — Nested Walk-Forward Backtest

Runs the trailing-stop + inventory strategy through the **nested, model-retrained,
rolling-window walk-forward** (`training/walkforward.py`). The heavy steps (per-fold
model retrain + signal precompute) need a GPU, so they run here on Colab; the cheap
steps (stop sim, param sweep, aggregation) run inline in the same process.

> The pure pieces (folds, sim, metrics, sweep, cache, aggregate) are unit-tested
> locally (`training/test_walkforward.py`). `train_fold` / `precompute_signals` /
> `run_walkforward` were validated *structurally* on the 8GB local box (signatures,
> data pipeline, model+loss forward pass, cross-function contracts) but their full
> execution happens here — this is the first true end-to-end run.

## 0. Data facts (real, verified)

- Parquet: `lob_data/<exchange>_<symbol>.parquet`, 40 levels, `bucket` stored as **strings**
  (parse to `datetime64[us]` → divide int64 by **1e6** for unix seconds; this is handled
  inside `data_source.fetch_parquet_stream`).
- Span: **2026-03-14 12:48 UTC → 2026-06-10 13:57 UTC (88 days)**.
- Tradable streams (the strategy trades these 4): `binance_perp_{BTC,ETH,SOL,WLD}-USDT`.
- Default folds (train 45d / test 8d / tune 7d, step = test): **~5 folds**.

## 1. Setup

```python
# GPU runtime (Runtime > Change runtime type > GPU, e.g. A100/T4)
import torch; print(torch.cuda.get_device_name(0))

# Get the code + data onto the box (choose one):
#  a) git clone/pull the feat/trailing-stop-walkforward branch, OR
#  b) upload the repo + lob_data/ parquet via Drive.
%cd /content/aai
!pip -q install numpy pandas pyarrow scipy torch
```

Confirm the module imports and the pure tests pass on Colab too:
```bash
python -m pytest training/test_walkforward.py executor/paper/test_risk.py -q
```

## 2. Calibration fold FIRST (do not skip — spec §5.1)

Run a single fold to (a) measure real per-fold train time and (b) check the 45-day-window
model isn't materially weaker than the existing full-history model.

```bash
python training/walkforward.py \
    --data-start 2026-03-14 --data-end 2026-06-11 \
    --out experiments/walkforward/calib --epochs 50 --calibrate
```

Then read `experiments/walkforward/calib/report.json`:
- `timings[0].train_secs` → multiply by the fold count (~5) for the full-run GPU budget.
- `timings[0].best_val_dir_acc` → the 45-day model's validation directional accuracy.

**Override check:** compare `best_val_dir_acc` to the existing full-history model
(`experiments/v2_balanced` — see its `eval/report.json` or training logs for val dir-acc).
If the 45-day model is materially weaker (more than ~3 percentage points absolute), the
short window is starving the model and would contaminate the strategy test. In that case,
switch the outer loop to **anchored-expanding**: in `training/walkforward.py::make_folds`,
set `train_start = data_start` for every fold (train window grows instead of sliding),
then re-run the calibration fold to confirm recovery before the full run.

Also sanity-check the calibration fold's `per_fold[0]` strategy result is non-degenerate
(non-zero `n_trades`, finite `net_bp`).

## 3. Full run

Only after the calibration fold's timing + quality are acceptable:

```bash
python training/walkforward.py \
    --data-start 2026-03-14 --data-end 2026-06-11 \
    --out experiments/walkforward/full --epochs 50
```

Copy `experiments/walkforward/full/report.json` back for analysis. Key fields:
- `overall`: `net_bp`, `sharpe`, `max_dd_bp`, `n_folds`, `n_profitable_folds` — the honest,
  stitched out-of-sample result across all folds.
- `per_fold[i]`: per-fold `net_bp` / `sharpe` / `max_dd_bp` / `n_trades` and the `best`
  trailing-stop params chosen on that fold's tune slice.

## 4. Does the trailing stop actually help? (stop-on vs stop-off control)

The walk-forward above tunes and applies the stop. To measure its **marginal** value, run a
control where the stop can never fire and compare. Quickest way: temporarily narrow the grid
to a single never-triggering `k` (a huge multiple makes `stop_level` sit far beyond any
real move), e.g. in a scratch cell:

```python
import training.walkforward as WF, datetime as dt
WF.param_grid = lambda *a, **k: [dict(k=1e9, L=12, cooldown_n=0)]   # stop effectively disabled
rep_off = WF.run_walkforward(dt.datetime(2026,3,14,tzinfo=dt.timezone.utc),
                             dt.datetime(2026,6,11,tzinfo=dt.timezone.utc),
                             "experiments/walkforward/full_nostop", epochs=50)
print(rep_off["overall"])
```
Compare `full` (stop tuned) vs `full_nostop` (stop disabled) on `net_bp`, `max_dd_bp`,
and `n_profitable_folds`. The stop earns its place only if it improves net and/or drawdown
across the majority of folds.

## 5. GO / NO-GO

**GO** toward the (already running) paper trader / small live capital iff, on the *stitched
OOS curve*: net-positive after taker cost, positive in a **majority** of folds, PnL
concentrated in trending (high-ER) regimes as designed, drawdown tolerable, and the stop
improves (or at least does not hurt) net/DD vs the stop-off control.

**NO-GO** if net-negative overall, profitable in only one fold (the old failure mode), or the
stop only helps via one lucky `k` (param surface is a fragile spike, not a flat region —
inspect the per-fold `best` params for consistency).

⚠️ Rotate/revoke the leaked Binance API keys before ANY real capital. Paper trading uses no keys.

## Caveats

- **Winsorize skipped:** training clipped features to train 0.1/99.9 pct before scaling; those
  bounds aren't persisted, so `precompute_signals` applies exact z-score only (matches the live
  paper path — we backtest what we trade). Affects only rare extremes.
- **Full-parquet read:** `fetch_parquet_stream` reads the whole stream then filters by date.
  Fine on Colab RAM; it's why the local box (8GB, shared with the live trader) can't run these.
- **Entry gates frozen this phase:** `ER_MIN`/`VOL_THRESHOLD`/`TREND_*`/`CAP` are fixed; only the
  trailing-stop params (`k`, `L`, `cooldown_n`) are swept, to isolate the stop's marginal effect.
