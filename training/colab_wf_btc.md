# Colab Runner — BTC-only Faithful Walk-Forward (base vs deployed sizing exit)

Validates the **deployed** BTC directional strategy with a leakage-free, per-fold model
RETRAIN. This is the GO/NO-GO gate before live. The cheap pieces are unit-tested locally
(25/25 green) and the pipeline was smoked end-to-end on CPU; the per-fold **retrain** needs
an A100, which is why it runs here.

Fixes already backported (commit 6e035db) — do NOT re-apply:
- `data_source.py` `/1e6` timestamp bug (was rejecting every training window).
- `walkforward.py precompute_signals` now runs on GPU (`get_device()`), not CPU.
- `walkforward.py` evaluates `--strategy {base,sizing}` (sizing = the deployed `to_flat` exit),
  with `--btc-only`.

## 0. Runtime + code + data
- Runtime → Change runtime type → **A100** (or any CUDA GPU). `import torch; print(torch.cuda.get_device_name(0))`.
- Code: get THIS branch onto the box (`feat/wf-btc-sizing-backports`, or local `main` @ 6e035db).
  It needs `training/` + `executor/paper/` (the sizing/strategy modules the WF imports).
- Data: `lob_data/binance_perp_BTC-USDT.parquet` (88 days, 2026-03-14 → 2026-06-10). For BTC-only
  you only need the BTC parquet. (`--btc-only` ignores the other streams.)
- `%cd /content/aai && pip -q install numpy pandas pyarrow scipy torch`
- Confirm pure tests: `python -m pytest training/test_walkforward.py executor/paper/test_risk.py -q`

## 1. Calibration fold FIRST (do not skip — measures retrain time + 45-day model quality)
```bash
python training/walkforward.py --data-start 2026-03-14 --data-end 2026-06-11 \
    --out experiments/walkforward/calib_btc --epochs 50 --calibrate --btc-only --strategy base
```
Read `experiments/walkforward/calib_btc/report.json`:
- `timings[0].train_secs` × fold-count (~5) = total A100 budget.
- `timings[0].best_val_dir_acc` vs `experiments/v2_balanced` val dir-acc. If the 45-day model is
  >~3pp weaker, switch to anchored-expanding folds (set `train_start = data_start` in `make_folds`)
  and re-calibrate. Also confirm `per_fold[0].n_trades > 0` (BTC-only can be sparse — see caveat).

## 2. Full run — base (edge control) AND sizing (deployed)
```bash
python training/walkforward.py --data-start 2026-03-14 --data-end 2026-06-11 \
    --out experiments/walkforward/full_btc_base   --epochs 50 --btc-only --strategy base
python training/walkforward.py --data-start 2026-03-14 --data-end 2026-06-11 \
    --out experiments/walkforward/full_btc_sizing --epochs 50 --btc-only --strategy sizing
```
(The retrains are identical; only the strategy overlay differs. If you want to save GPU time,
the trained fold models + cached `*_test.npz` signals from the base run can be reused — re-run
just the `simulate(..., strategy="sizing")` step on the cached test signals.)

## 3. GO / NO-GO (read each `report.json` → `overall` + `per_fold`)
**GO** toward small live capital iff, on the stitched OOS curve:
- **net-positive after taker cost** (this is the thing we have NOT yet shown for BTC),
- **profitable in a majority of folds** (not one lucky fold — the old failure mode),
- PnL concentrated in trending/high-ER regimes as designed,
- the **sizing** exit improves (or at least doesn't hurt) drawdown vs **base**.

**NO-GO** if net-negative overall, profitable in only one fold, or trades are too few to be
meaningful.

⚠️ **Caveat — BTC-only sparsity:** BTC trades rarely (≈4 trades / 3 days live). Over ~5 folds ×
8 test-days the BTC-only trade count may be small → a statistically thin verdict. If `n_trades`
is tiny, the honest read is "insufficient evidence," not "GO". Consider reporting the multi-asset
result alongside for context.

## 4. Bring results back
Copy `experiments/walkforward/full_btc_{base,sizing}/report.json` back to the repo for analysis.
Keys: `overall.{net_bp,sharpe,max_dd_bp,n_folds,n_profitable_folds}`, `per_fold[i]`.
Keys are no-key/no-capital (paper backtest) — no secrets involved.
