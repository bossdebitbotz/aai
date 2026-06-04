# V2 Directional Training & Evaluation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the V2 directional LOB model trainable end-to-end and produce an honest, reproducible, calibrated assessment of directional accuracy and net-PnL-after-fees on a held-out test set.

**Architecture:** Add a V2 training path (`train_v2.py`) that wires the already-built `CompoundAttentionModelV2` + `LOBLossV2` + `engineer_features_v2` (219 features) into a parameterized, leakage-free dataset pipeline; add temperature-scaling calibration, a held-out evaluation harness, and an offline window-level PnL backtest; run it on a fresh full-span parquet export from a resumable Colab A100 notebook. V1 (`train.py`/`model.py`) is preserved for an apples-to-apples baseline re-measure.

**Tech Stack:** Python 3, PyTorch, NumPy, asyncpg + TimescaleDB (`lob_5s` continuous aggregate), pandas/pyarrow (parquet), pytest. No new third-party deps (metrics computed with NumPy, not sklearn).

**Spec:** `docs/superpowers/specs/2026-06-03-v2-directional-training-eval-design.md`

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `export_training_data.py` | Fresh full-span export from `lob_5s`, `all_valid` filter, no silent skip | Modify |
| `training/dataset.py` | Parameterized (`feature_version`), leak-free split-before-FE, per-split FE, warmup trim, winsorize, completeness filter, per-stream test datasets in metadata | Modify |
| `training/model_v2.py` | Extract module-level `compute_direction_labels` + add `directional_accuracy` (reused by train/eval/backtest) | Modify |
| `training/tracker.py` | Generic best-metric monitor (`monitor`/`monitor_mode`); `save_checkpoint(extra=...)` | Modify |
| `training/calibration.py` | Temperature scaling, ECE, reliability bins | Create |
| `training/train_v2.py` | V2 training entry point: V2 model+loss, dir-acc logging, grad accum, resume, enriched checkpoint + scalers.pkl | Create |
| `training/evaluate_v2.py` | Held-out per-horizon accuracy, confusion/PRF1, calibration report, conf-coverage; V1 baseline re-measure | Create |
| `training/backtest_v2.py` | Window-level offline PnL backtest (maker+taker), conf→PnL curve | Create |
| `training/data_source.py` | Parquet-backed fetch (Colab can't reach local DB) | Create |
| `training/colab_v2_runner.py` | Importable runner the Colab notebook calls (export→train→eval→backtest) | Create |
| `training/test_*.py` | Tests per component | Create |

**Conventions (match existing tests):** plain pytest functions, `sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")` at top, synthetic tensors/arrays (no DB dependency) for new tests, `assert` + `logger.info("PASS: ...")`. Run with `python -m pytest training/test_X.py -v`.

**Key constants (verified against live code):**
- 40 levels → base features = `4*40+2 = 162`; `mid_price_idx = 4*40 = 160`, `spread_idx = 161`.
- V1 enriched = `5N+11 = 211` (49 derived); V2 enriched = `5N+19 = 219` (49 derived + 8 momentum).
- `direction_horizons = (5, 11, 23)` steps (≈25s/55s/115s ≈ 30s/1m/2m); `flat_threshold = 0.01`.
- `CompoundAttentionModelV2.forward(context, exchange_ids, symbol_ids) -> (pred (B,Tp,F), dir_logits (B,9))`.
- `LOBLossV2.forward(pred, target, dir_logits, context_last) -> (total, forecast, structure, direction, dsl)`.

---

## Task 1: Export — refactor into testable helpers

**Files:**
- Modify: `export_training_data.py`
- Test: `training/test_export.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# training/test_export.py
import sys, logging
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import export_training_data as ex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_build_copy_query_filters_invalid_and_orders():
    cols = ex.build_columns()
    q = ex.build_copy_query("binance_perp", "BTC-USDT", cols)
    assert "FROM lob_5s" in q
    assert "exchange = 'binance_perp'" in q
    assert "symbol = 'BTC-USDT'" in q
    assert "all_valid" in q          # invalid buckets excluded
    assert "ORDER BY bucket" in q
    assert "all_valid" not in cols   # filter column is NOT exported as a feature
    logger.info("PASS: test_build_copy_query_filters_invalid_and_orders")

def test_clear_stale_exports(tmp_path):
    f = tmp_path / "binance_perp_BTC-USDT.parquet"
    f.write_bytes(b"x" * 2000)
    z = tmp_path / "lob_training_data.zip"
    z.write_bytes(b"x" * 2000)
    removed = ex.clear_stale_exports(tmp_path, zip_path=z)
    assert not f.exists() and not z.exists()
    assert removed == 2
    logger.info("PASS: test_clear_stale_exports")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest training/test_export.py -v`
Expected: FAIL — `AttributeError: module 'export_training_data' has no attribute 'build_copy_query'`.

- [ ] **Step 3: Add the helpers to `export_training_data.py`**

Add these functions (place after `build_columns`):

```python
def build_copy_query(exchange: str, symbol: str, columns: list) -> str:
    """COPY query that excludes invalid (crossed/zero) buckets via all_valid."""
    col_str = ", ".join(columns)
    return (
        f"COPY (SELECT {col_str} FROM lob_5s "
        f"WHERE exchange = '{exchange}' AND symbol = '{symbol}' "
        f"AND all_valid = true "
        f"ORDER BY bucket) TO STDOUT WITH CSV HEADER"
    )


def clear_stale_exports(output_dir, zip_path=None) -> int:
    """Delete existing parquet exports (and zip) so a re-run is truly fresh."""
    from pathlib import Path
    output_dir = Path(output_dir)
    removed = 0
    for p in output_dir.glob("*.parquet"):
        p.unlink(); removed += 1
    if zip_path is not None and Path(zip_path).exists():
        Path(zip_path).unlink(); removed += 1
    return removed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest training/test_export.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add export_training_data.py training/test_export.py
git commit -m "feat(export): testable copy-query (all_valid filter) + clear_stale_exports helper"
```

---

## Task 2: Export — wire `--fresh`/`--resume` and use the helpers

**Files:**
- Modify: `export_training_data.py` (`export_stream`, `main`)

- [ ] **Step 1: Replace the inline COPY string in `export_stream` with the helper**

In `export_stream`, delete the local `copy_query = (...)` block and replace the skip-guard + query with:

```python
    fname = OUTPUT_DIR / f"{exchange}_{symbol}.parquet"
    # NOTE: skip-guard removed; freshness is controlled by --fresh in main()
    copy_query = build_copy_query(exchange, symbol, columns)
```

(Remove the `if fname.exists() and fname.stat().st_size > 1000:` early-return block entirely.)

- [ ] **Step 2: Add CLI flags and freshness handling in `main`**

At the top of `main()` add argument parsing and a fresh/resume gate:

```python
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export lob_5s to parquet for Colab")
    parser.add_argument("--fresh", action="store_true", default=True,
                        help="Delete existing exports first (default).")
    parser.add_argument("--resume", dest="fresh", action="store_false",
                        help="Keep existing parquet files; only export missing streams.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    columns = build_columns()

    if args.fresh:
        n = clear_stale_exports(OUTPUT_DIR, zip_path=Path(ZIP_NAME))
        print(f"--fresh: removed {n} stale export file(s)")
```

In `--resume` mode, re-introduce a guarded skip at the top of `export_stream` ONLY when the file exists:

```python
    if (not FRESH) and fname.exists() and fname.stat().st_size > 1000:
        print(f"  {exchange}/{symbol}: keeping existing export (--resume)")
        return fname
```

Pass `FRESH` via a module global set in `main` (`global FRESH; FRESH = args.fresh`) and default `FRESH = True` at module scope.

- [ ] **Step 3: Manual verification command (documented, run later on the box)**

Run: `python export_training_data.py --fresh`
Expected: prints `--fresh: removed N stale export file(s)`, then `binance_spot/BTC-USDT: <rows> rows`, … for all 16 streams with row counts far larger than the Apr-11 files; produces a fresh `lob_training_data.zip`.

- [ ] **Step 4: Commit**

```bash
git add export_training_data.py
git commit -m "feat(export): --fresh/--resume flags; remove silent skip-guard"
```

---

## Task 3: `model_v2.py` — extract reusable direction-label + accuracy helpers

**Files:**
- Modify: `training/model_v2.py`
- Test: `training/test_direction_utils.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# training/test_direction_utils.py
import sys, logging, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.model_v2 import compute_direction_labels, directional_accuracy, LOBLossV2

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_compute_direction_labels_matches_lossv2():
    B, Tp, Fdim = 4, 24, 162
    mid_idx, horizons, thr = 160, (5, 11, 23), 0.01
    target = torch.zeros(B, Tp, Fdim)
    context_last = torch.zeros(B, Fdim)
    # craft up/down/flat at horizon 5
    target[0, 5, mid_idx] = 1.0     # up
    target[1, 5, mid_idx] = -1.0    # down
    target[2, 5, mid_idx] = 0.0     # flat
    labels = compute_direction_labels(target, context_last, mid_idx, horizons, thr)
    assert labels.shape == (B, 3)
    assert labels[0, 0].item() == 2  # up
    assert labels[1, 0].item() == 0  # down
    assert labels[2, 0].item() == 1  # flat
    # delegation parity
    loss = LOBLossV2(n_levels=40, mid_price_idx=mid_idx,
                     direction_horizons=horizons, flat_threshold=thr)
    ref = loss._direction_labels(target, context_last)
    assert torch.equal(labels, ref)
    logger.info("PASS: test_compute_direction_labels_matches_lossv2")

def test_directional_accuracy_perfect_and_chance():
    B = 100
    labels = torch.randint(0, 3, (B, 3))
    # build logits that perfectly predict labels
    logits = torch.full((B, 3, 3), -5.0)
    for h in range(3):
        logits[torch.arange(B), h, labels[:, h]] = 5.0
    acc = directional_accuracy(logits.reshape(B, 9), labels)
    assert acc.shape == (3,)
    assert torch.allclose(acc, torch.ones(3))
    logger.info("PASS: test_directional_accuracy_perfect_and_chance")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest training/test_direction_utils.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_direction_labels'`.

- [ ] **Step 3: Add the module-level helpers and delegate from `LOBLossV2`**

In `training/model_v2.py`, add after `directional_sign_loss`:

```python
def compute_direction_labels(target, context_last, mid_price_idx, direction_horizons, flat_threshold):
    """Multi-horizon up/flat/down labels from mid-price change. Returns (B, H) longs (0=down,1=flat,2=up)."""
    B = context_last.shape[0]
    mid_start = context_last[:, mid_price_idx]
    T_pred = target.shape[1]
    labels_list = []
    for h in direction_horizons:
        t_idx = min(h, T_pred - 1)
        change = target[:, t_idx, mid_price_idx] - mid_start
        lab = torch.ones(B, dtype=torch.long, device=change.device)  # flat
        lab[change > flat_threshold] = 2
        lab[change < -flat_threshold] = 0
        labels_list.append(lab)
    return torch.stack(labels_list, dim=1)


def directional_accuracy(dir_logits, labels):
    """Per-horizon accuracy. dir_logits (B,9), labels (B,H). Returns (H,) float tensor."""
    H = labels.shape[1]
    logits = dir_logits.reshape(-1, H, 3)
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean(dim=0)
```

Then replace the body of `LOBLossV2._direction_labels` to delegate:

```python
    def _direction_labels(self, target, context_last):
        return compute_direction_labels(
            target, context_last, self.mid_price_idx,
            self.direction_horizons, self.flat_threshold,
        )
```

- [ ] **Step 4: Run to verify pass (and V2 model tests still pass)**

Run: `python -m pytest training/test_direction_utils.py training/test_model_v2.py -v`
Expected: PASS (new tests + existing V2 tests green).

- [ ] **Step 5: Commit**

```bash
git add training/model_v2.py training/test_direction_utils.py
git commit -m "refactor(model_v2): extract compute_direction_labels + directional_accuracy helpers"
```

---

## Task 4: `model_v2.py` — guard `mid_price_idx` against the real feature layout

**Files:**
- Test: `training/test_mid_price_idx.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# training/test_mid_price_idx.py
import sys, logging, numpy as np
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.features import get_column_indices
from training.features_v2 import engineer_features_v2
from training.model_v2 import LOBLossV2

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_mid_price_idx_matches_layout():
    n_levels = 40
    # raw base array: 4N+2 cols, mid at 4N=160, spread at 161
    T, base = 80, n_levels * 4 + 2
    raw = np.random.rand(T, base).astype(np.float64) + 1.0
    enriched, _ = engineer_features_v2(raw, n_levels, apply_smoothing=False)
    assert enriched.shape[1] == n_levels * 5 + 19  # 219
    idx = get_column_indices(n_levels)
    assert idx["mid_price"] == 160
    assert LOBLossV2(n_levels=n_levels).mid_price_idx == idx["mid_price"]
    logger.info("PASS: test_mid_price_idx_matches_layout")
```

- [ ] **Step 2: Run to verify it passes immediately (guard test)**

Run: `python -m pytest training/test_mid_price_idx.py -v`
Expected: PASS. (If it FAILS, the V2 feature layout drifted from `LOBLossV2.mid_price_idx=160` — fix `mid_price_idx` before proceeding; this test is the guard against silently mislabeled direction targets.)

- [ ] **Step 3: Commit**

```bash
git add training/test_mid_price_idx.py
git commit -m "test(model_v2): assert mid_price_idx matches engineer_features_v2 layout"
```

---

## Task 5: `dataset.py` — parameterize DataConfig for V2 features

**Files:**
- Modify: `training/dataset.py` (`DataConfig`)
- Test: `training/test_dataset_v2.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# training/test_dataset_v2.py
import sys, logging, numpy as np
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.dataset import DataConfig

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_dataconfig_v2_feature_count_and_warmup():
    cfg = DataConfig(lob_levels=40, feature_version="v2", savgol_window=11)
    assert cfg.n_enriched_features == 40 * 5 + 19   # 219
    assert cfg.warmup_trim == 60                     # max(savgol_window, 60)
    cfg1 = DataConfig(lob_levels=40, feature_version="v1")
    assert cfg1.n_enriched_features == 40 * 5 + 11   # 211
    logger.info("PASS: test_dataconfig_v2_feature_count_and_warmup")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest training/test_dataset_v2.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'feature_version'`.

- [ ] **Step 3: Add fields/properties to `DataConfig`**

Add to the `DataConfig` dataclass (after `pairs`):

```python
    # Feature pipeline
    feature_version: str = "v2"     # "v1" -> engineer_features; "v2" -> engineer_features_v2
    savgol_window: int = 11         # V2 default (was hardcoded 21 in build_dataloaders)
    # Data source
    source: str = "db"              # "db" -> lob_5s; "parquet" -> parquet_dir
    parquet_dir: str = "lob_data"
```

Replace `n_enriched_features` and add `warmup_trim`:

```python
    @property
    def n_enriched_features(self) -> int:
        """base (4N+2) + derived. V1 derived=N+9 (=5N+11). V2 adds 8 momentum (=5N+19)."""
        return self.lob_levels * 5 + (19 if self.feature_version == "v2" else 11)

    @property
    def warmup_trim(self) -> int:
        """Rows to drop at the head of each split (largest feature lookback)."""
        return max(self.savgol_window, 60)  # 60 = longest momentum horizon (log_return_60)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest training/test_dataset_v2.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/dataset.py training/test_dataset_v2.py
git commit -m "feat(dataset): parameterize DataConfig (feature_version/savgol/source/warmup_trim)"
```

---

## Task 6: `dataset.py` — leak-free split-before-feature-engineering

**Files:**
- Modify: `training/dataset.py` (`build_dataloaders` loop body, ~`:419-467`)
- Test: `training/test_dataset_v2.py` (add)

- [ ] **Step 1: Write the failing leakage-guard test**

```python
# add to training/test_dataset_v2.py
from training.features_v2 import engineer_features_v2

def test_per_split_fe_no_cross_boundary_dependence():
    """Features in a split must not depend on data outside that split (no leakage)."""
    n_levels, base = 40, 40 * 4 + 2
    rng = np.random.default_rng(0)
    full = rng.random((400, base)) + 1.0
    split_at = 250
    # Engineer the val split alone vs. as the tail of the full series, after warmup trim.
    val_alone, _ = engineer_features_v2(full[split_at:], n_levels, apply_smoothing=False)
    val_in_full, _ = engineer_features_v2(full, n_levels, apply_smoothing=False)
    trim = 60
    a = val_alone[trim:]
    b = val_in_full[split_at + trim:]
    # momentum/OFI features within the trimmed region must be identical either way
    assert np.allclose(a, b, atol=1e-9), "cross-boundary leakage in engineered features"
    logger.info("PASS: test_per_split_fe_no_cross_boundary_dependence")
```

- [ ] **Step 2: Run to verify it fails** (current code engineers on the full stream then splits)

Run: `python -m pytest training/test_dataset_v2.py::test_per_split_fe_no_cross_boundary_dependence -v`
Expected: this test passes for `engineer_features_v2` in isolation (it is pure-per-array), but documents the invariant. The *real* fix is the ordering in `build_dataloaders` (next step). If it fails, the feature fn has unexpected global state — stop and investigate.

- [ ] **Step 3: Rewrite the per-stream body of `build_dataloaders`**

Replace lines from `# Feature engineering ...` through the `stream_info[key] = {...}` block with:

```python
            # --- Split FIRST (raw), then engineer features per split (leak-free) ---
            from training.features_v2 import engineer_features_v2
            fe = engineer_features_v2 if config.feature_version == "v2" else engineer_features

            n = len(features)
            train_end = int(n * config.train_ratio)
            val_end = int(n * (config.train_ratio + config.val_ratio))

            trim = config.warmup_trim

            def _fe_split(raw_slice, ts_slice):
                if len(raw_slice) <= trim:
                    return np.empty((0, config.n_enriched_features)), np.empty((0,))
                enr, _ = fe(raw_slice, n_levels=config.lob_levels,
                            apply_smoothing=True, savgol_window=config.savgol_window)
                return enr[trim:], ts_slice[trim:]

            train_feat, train_ts = _fe_split(features[:train_end], timestamps[:train_end])
            val_feat, val_ts     = _fe_split(features[train_end:val_end], timestamps[train_end:val_end])
            test_feat, test_ts   = _fe_split(features[val_end:], timestamps[val_end:])

            logger.info(f"  {key}: enriched to {train_feat.shape[1] if len(train_feat) else 0} features")

            # --- Fit scaler on TRAIN only; winsorize using train percentile bounds ---
            scaler = LOBScaler(n_levels=config.lob_levels)
            if len(train_feat) < config.window_size:
                logger.warning(f"  {key}: train split too small after trim, skipping.")
                continue
            lo = np.percentile(train_feat, 0.1, axis=0)
            hi = np.percentile(train_feat, 99.9, axis=0)
            train_feat = np.clip(train_feat, lo, hi)
            val_feat = np.clip(val_feat, lo, hi) if len(val_feat) else val_feat
            test_feat = np.clip(test_feat, lo, hi) if len(test_feat) else test_feat

            train_scaled = scaler.fit_transform(train_feat)
            val_scaled = scaler.transform(val_feat) if len(val_feat) else val_feat
            test_scaled = scaler.transform(test_feat) if len(test_feat) else test_feat
            scalers[key] = scaler

            train_ds = None
            if len(train_scaled) >= config.window_size:
                train_ds = LOBDataset(train_scaled, train_ts, exchange, symbol, config)
                train_datasets.append(train_ds)
            if len(val_scaled) >= config.window_size:
                val_datasets.append(LOBDataset(val_scaled, val_ts, exchange, symbol, config))
            if len(test_scaled) >= config.window_size:
                ds = LOBDataset(test_scaled, test_ts, exchange, symbol, config)
                test_datasets.append(ds)
                test_by_stream[key] = ds

            stream_info[key] = {
                "total_samples": int(n),
                "train_windows": len(train_ds) if train_ds is not None else 0,
            }
```

Before the loop (near `scalers = {}`), add `test_by_stream = {}`. In the `metadata` dict (~`:502`), add `"test_datasets_by_stream": test_by_stream,`.

- [ ] **Step 4: Run the dataset tests**

Run: `python -m pytest training/test_dataset_v2.py -v`
Expected: PASS. Existing `training/test_dataset.py` may rely on V1 ordering — run it too: `python -m pytest training/test_dataset.py -v`. If a test asserts the old FE-before-split window counts, update it to construct `DataConfig(feature_version="v1")` and accept the trimmed counts (note the change in the commit).

- [ ] **Step 5: Commit**

```bash
git add training/dataset.py training/test_dataset_v2.py
git commit -m "fix(dataset): split before feature-engineering (leak-free) + winsorize + per-stream test sets"
```

---

## Task 7: `dataset.py` — full-depth completeness filter + logging

**Files:**
- Modify: `training/dataset.py` (`_fetch_stream_data`, `:145-147`)

- [ ] **Step 1: Tighten the depth filter and log drops**

Replace the single `bid_price_{N} > 0` clause with a full-depth requirement and add a post-fetch drop log. In `_fetch_stream_data`, change the `if config.lob_levels > 5:` block to require both extremes are present:

```python
    if config.lob_levels > 5:
        where_clauses.append(f"bid_price_{config.lob_levels} > 0")
        where_clauses.append(f"ask_price_{config.lob_levels} > 0")
        where_clauses.append("bid_price_1 > 0")
        where_clauses.append("ask_price_1 > 0")
```

After building the `features` array (just before `return features, timestamps`), add:

```python
    finite_rows = np.isfinite(features).all(axis=1)
    dropped = int((~finite_rows).sum())
    if dropped:
        logger.warning(f"  {exchange}/{symbol}: dropped {dropped} non-finite rows")
        features = features[finite_rows]
        timestamps = timestamps[finite_rows]
```

- [ ] **Step 2: Verify import-level sanity** (no dedicated unit test; covered by smoke in Task 13)

Run: `python -c "import training.dataset"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add training/dataset.py
git commit -m "feat(dataset): require full top+bottom depth; drop non-finite rows with logging"
```

---

## Task 8: `tracker.py` — generic best-metric monitor + checkpoint extras

**Files:**
- Modify: `training/tracker.py`
- Test: `training/test_tracker_v2.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# training/test_tracker_v2.py
import sys, logging, tempfile, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_monitor_max_metric_from_extra():
    with tempfile.TemporaryDirectory() as d:
        t = ExperimentTracker(d, run_name="t", monitor="val_dir_acc", monitor_mode="max")
        b1 = t.log_epoch(1, 1.0, 1.0, 0.0, val_loss=1.0, extra={"val_dir_acc": 0.52})
        b2 = t.log_epoch(2, 0.9, 0.9, 0.0, val_loss=1.2, extra={"val_dir_acc": 0.58})
        b3 = t.log_epoch(3, 0.8, 0.8, 0.0, val_loss=0.5, extra={"val_dir_acc": 0.55})
        assert b1 is True and b2 is True and b3 is False  # selects on max dir_acc, not val_loss
        logger.info("PASS: test_monitor_max_metric_from_extra")

def test_save_checkpoint_extra(tmp_path):
    import torch.nn as nn
    t = ExperimentTracker(str(tmp_path), run_name="c")
    m = nn.Linear(2, 2); opt = torch.optim.Adam(m.parameters())
    t.save_checkpoint(m, opt, epoch=1, is_best=True, extra={"n_features": 219, "temps": [1.1, 1.2, 1.3]})
    ck = torch.load(tmp_path / "c" / "checkpoints" / "best.pt", weights_only=False)
    assert ck["n_features"] == 219 and ck["temps"] == [1.1, 1.2, 1.3]
    logger.info("PASS: test_save_checkpoint_extra")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest training/test_tracker_v2.py -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'monitor'`.

- [ ] **Step 3: Implement monitor + checkpoint extras**

In `ExperimentTracker.__init__`, add params and state:

```python
    def __init__(self, experiment_dir, run_name=None, monitor="val_loss", monitor_mode="min"):
        ...
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.best_metric = float("inf") if monitor_mode == "min" else float("-inf")
```

In `log_epoch`, replace the `# Check for best val loss` block with a generic monitor:

```python
        value = entry.get(self.monitor)
        is_best = False
        if value is not None:
            better = (value < self.best_metric) if self.monitor_mode == "min" else (value > self.best_metric)
            if better:
                self.best_metric = value
                self.best_epoch = epoch
                self.best_val_loss = entry.get("val_loss", self.best_val_loss)
                is_best = True
        return is_best
```

In `save_checkpoint`, add `extra: Optional[dict] = None` and merge it:

```python
    def save_checkpoint(self, model, optimizer, epoch, is_best=False, extra=None):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
        }
        if extra:
            checkpoint.update(extra)
        ...  # (unchanged save logic below)
```

- [ ] **Step 4: Run to verify it passes (and V1 tracker tests still pass)**

Run: `python -m pytest training/test_tracker_v2.py training/test_tracker.py -v`
Expected: PASS (V1 defaults to `monitor="val_loss"/"min"`, behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add training/tracker.py training/test_tracker_v2.py
git commit -m "feat(tracker): configurable best-metric monitor + checkpoint extras"
```

---

## Task 9: `train_v2.py` — V2 training loop (model, loss, dir-acc, grad accum, resume)

**Files:**
- Create: `training/train_v2.py`
- Test: `training/test_train_v2.py` (Create)

- [ ] **Step 1: Write the failing smoke test (synthetic, no DB)**

```python
# training/test_train_v2.py
import sys, logging, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.train_v2 import run_one_epoch_v2, evaluate_v2_loader
from training.model_v2 import CompoundAttentionModelV2, LOBLossV2

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def _fake_loader(n_batches=3, B=4, T=20, Tp=24, F=36, n_levels=5):
    batches = []
    for _ in range(n_batches):
        batches.append({
            "context": torch.randn(B, T, F),
            "target": torch.randn(B, Tp, F),
            "exchange_id": torch.zeros(B, dtype=torch.long),
            "symbol_id": torch.zeros(B, dtype=torch.long),
            "timestamp": torch.zeros(B),
        })
    return batches

def test_run_one_epoch_v2_and_eval():
    n_levels, F, T, Tp = 5, 36, 20, 24
    model = CompoundAttentionModelV2(n_levels=n_levels, n_features=F, context_length=T,
                                     prediction_length=Tp, d_model=30, n_heads=3, n_layers=1, d_ff=120)
    loss_fn = LOBLossV2(n_levels=n_levels, mid_price_idx=20, use_feature_weights=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _fake_loader(F=F, T=T, Tp=Tp, n_levels=n_levels)
    metrics = run_one_epoch_v2(model, loss_fn, opt, scheduler=None, loader=loader,
                               device=torch.device("cpu"), accum_steps=2)
    assert "total" in metrics and "dir_acc_mean" in metrics
    assert 0.0 <= metrics["dir_acc_mean"] <= 1.0
    ev = evaluate_v2_loader(model, loss_fn, loader, torch.device("cpu"))
    assert "dir_acc_per_h" in ev and len(ev["dir_acc_per_h"]) == 3
    logger.info("PASS: test_run_one_epoch_v2_and_eval")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest training/test_train_v2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.train_v2'`.

- [ ] **Step 3: Create `training/train_v2.py`**

```python
#!/usr/bin/env python3
"""V2 training entry point: directional LOB model (CompoundAttentionModelV2 + LOBLossV2).

Usage:
    python training/train_v2.py --levels 40 --epochs 50 --batch-size 16 --accum-steps 8
    python training/train_v2.py --levels 40 --resume
"""
import argparse, logging, sys, time, pickle
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dataset import DataConfig, build_dataloaders
from training.model_v2 import CompoundAttentionModelV2, LOBLossV2, directional_accuracy
from training.model import WarmupDecayScheduler
from training.tracker import ExperimentTracker, RunConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_v2")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dir_acc(dir_logits, target, context_last, loss_fn):
    labels = loss_fn._direction_labels(target, context_last)   # (B,H)
    return directional_accuracy(dir_logits, labels)            # (H,)


def run_one_epoch_v2(model, loss_fn, optimizer, scheduler, loader, device, accum_steps=1):
    model.train()
    sums = {"total": 0.0, "forecast": 0.0, "structure": 0.0, "direction": 0.0, "dsl": 0.0}
    acc_sum = None
    n = 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        ex = batch["exchange_id"].to(device)
        sym = batch["symbol_id"].to(device)
        context_last = context[:, -1, :]

        pred, dir_logits = model(context, ex, sym)
        total, forecast, structure, direction, dsl = loss_fn(pred, target, dir_logits, context_last)

        (total / accum_steps).backward()
        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        sums["total"] += total.item(); sums["forecast"] += forecast.item()
        sums["structure"] += structure.item(); sums["direction"] += direction.item()
        sums["dsl"] += dsl.item()
        acc = _dir_acc(dir_logits.detach(), target, context_last, loss_fn)
        acc_sum = acc if acc_sum is None else acc_sum + acc
        n += 1

    # flush a trailing partial accumulation
    if n % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    out = {k: v / max(n, 1) for k, v in sums.items()}
    acc_mean = (acc_sum / max(n, 1)) if acc_sum is not None else torch.zeros(3)
    out["dir_acc_per_h"] = acc_mean.tolist()
    out["dir_acc_mean"] = float(acc_mean.mean().item())
    return out


@torch.no_grad()
def evaluate_v2_loader(model, loss_fn, loader, device):
    if loader is None:
        return {"total": float("nan"), "dir_acc_per_h": [float("nan")] * 3, "dir_acc_mean": float("nan")}
    model.eval()
    total_sum = 0.0; acc_sum = None; n = 0
    for batch in loader:
        context = batch["context"].to(device); target = batch["target"].to(device)
        ex = batch["exchange_id"].to(device); sym = batch["symbol_id"].to(device)
        context_last = context[:, -1, :]
        pred, dir_logits = model(context, ex, sym)
        total, *_ = loss_fn(pred, target, dir_logits, context_last)
        total_sum += total.item()
        acc = _dir_acc(dir_logits, target, context_last, loss_fn)
        acc_sum = acc if acc_sum is None else acc_sum + acc
        n += 1
    acc_mean = (acc_sum / max(n, 1)) if acc_sum is not None else torch.zeros(3)
    return {"total": total_sum / max(n, 1),
            "dir_acc_per_h": acc_mean.tolist(),
            "dir_acc_mean": float(acc_mean.mean().item())}


def main():
    p = argparse.ArgumentParser(description="Train V2 directional LOB model")
    p.add_argument("--levels", type=int, default=40)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=66)
    p.add_argument("--n-heads", type=int, default=3)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--source", type=str, default="db", choices=["db", "parquet"])
    p.add_argument("--parquet-dir", type=str, default="lob_data")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = get_device(); logger.info(f"Device: {device}")
    n_features = args.levels * 5 + 19  # V2
    d_ff = args.d_model * 4

    data_config = DataConfig(lob_levels=args.levels, feature_version="v2",
                             savgol_window=11, source=args.source, parquet_dir=args.parquet_dir)
    train_loader, val_loader, test_loader, meta = build_dataloaders(data_config, batch_size=args.batch_size)
    logger.info(f"Windows: train={meta['n_train_windows']} val={meta['n_val_windows']} test={meta['n_test_windows']}")

    model = CompoundAttentionModelV2(
        n_levels=args.levels, n_features=n_features,
        context_length=data_config.context_length, prediction_length=data_config.prediction_length,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=d_ff, dropout=0.1,
    ).to(device)
    assert model.n_features == n_features, "n_features mismatch"
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    first_scaler = next(iter(meta["scalers"].values()))
    loss_fn = LOBLossV2(n_levels=args.levels, use_feature_weights=True, mid_price_idx=args.levels * 4,
                        scaler_means=first_scaler._means, scaler_stds=first_scaler._stds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = WarmupDecayScheduler(optimizer, warmup_steps=1000, decay_factor=0.8, decay_every=5000)

    run_config = RunConfig(n_levels=args.levels, n_features=n_features, d_model=args.d_model,
                           n_heads=args.n_heads, n_layers=args.n_layers, d_ff=d_ff,
                           learning_rate=args.lr, batch_size=args.batch_size, max_epochs=args.epochs,
                           early_stopping_patience=args.patience, savgol_window=11)
    tracker = ExperimentTracker(str(Path(__file__).parent.parent / "experiments"),
                                run_name=args.run_name, monitor="val_dir_acc_mean", monitor_mode="max")
    tracker.log_config(run_config)
    # persist scalers for eval/backtest
    with open(tracker.run_dir / "scalers.pkl", "wb") as f:
        pickle.dump(meta["scalers"], f)

    start_epoch = 1
    if args.resume:
        start_epoch = tracker.load_checkpoint(model, optimizer, "latest.pt") + 1
        ck = torch.load(tracker.checkpoints_dir / "latest.pt", weights_only=False)
        scheduler.step_count = ck.get("scheduler_step_count", 0)
        logger.info(f"Resumed at epoch {start_epoch} (scheduler step {scheduler.step_count})")

    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = run_one_epoch_v2(model, loss_fn, optimizer, scheduler, train_loader, device, args.accum_steps)
        va = evaluate_v2_loader(model, loss_fn, val_loader, device)
        is_best = tracker.log_epoch(
            epoch=epoch, train_loss=tr["total"], train_forecast_loss=tr["forecast"],
            train_structure_loss=tr["structure"],
            val_loss=va["total"] if not np.isnan(va["total"]) else None,
            learning_rate=scheduler.current_lr,
            extra={"train_dir_acc_mean": tr["dir_acc_mean"], "train_dir_acc_per_h": tr["dir_acc_per_h"],
                   "val_dir_acc_mean": va["dir_acc_mean"], "val_dir_acc_per_h": va["dir_acc_per_h"]},
        )
        tracker.save_checkpoint(model, optimizer, epoch, is_best, extra={
            "n_features": n_features, "context_length": data_config.context_length,
            "prediction_length": data_config.prediction_length,
            "direction_horizons": list(loss_fn.direction_horizons),
            "mid_price_idx": loss_fn.mid_price_idx, "feature_version": "v2",
            "scheduler_step_count": scheduler.step_count,
        })
        logger.info(f"Epoch {epoch:3d} | train_total={tr['total']:.4f} dir_acc={tr['dir_acc_mean']:.3f} "
                    f"| val_total={va['total']:.4f} val_dir_acc={va['dir_acc_mean']:.3f} "
                    f"| {time.time()-t0:.1f}s {'*BEST*' if is_best else ''}")
        epochs_no_improve = 0 if is_best else epochs_no_improve + 1
        if epochs_no_improve >= args.patience:
            logger.info("Early stopping."); break

    logger.info("\n" + tracker.summary())
    logger.info(f"Run dir: {tracker.run_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test**

Run: `python -m pytest training/test_train_v2.py -v`
Expected: PASS (`run_one_epoch_v2` returns metrics; `dir_acc_mean` ∈ [0,1]; eval returns 3 per-horizon accuracies).

- [ ] **Step 5: Commit**

```bash
git add training/train_v2.py training/test_train_v2.py
git commit -m "feat(train_v2): V2 directional training loop — grad accum, dir-acc, resume, rich checkpoint"
```

---

## Task 10: `calibration.py` — temperature scaling + ECE

**Files:**
- Create: `training/calibration.py`
- Test: `training/test_calibration.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# training/test_calibration.py
import sys, logging, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.calibration import fit_temperature, expected_calibration_error, apply_temperature

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_temperature_preserves_argmax_and_reduces_ece():
    torch.manual_seed(0)
    N = 2000
    labels = torch.randint(0, 3, (N,))
    # overconfident logits: correct class favored but scaled too sharply
    logits = torch.randn(N, 3)
    logits[torch.arange(N), labels] += 1.0
    logits = logits * 4.0  # exaggerate confidence -> miscalibrated
    T = fit_temperature(logits, labels)
    assert T > 0
    pre = expected_calibration_error(torch.softmax(logits, dim=-1), labels)
    post = expected_calibration_error(apply_temperature(logits, T), labels)
    # argmax (accuracy) unchanged
    assert torch.equal(logits.argmax(-1), apply_temperature(logits, T).argmax(-1))
    assert post <= pre + 1e-6
    logger.info(f"PASS: test_temperature ... T={T:.3f} ECE {pre:.3f}->{post:.3f}")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest training/test_calibration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.calibration'`.

- [ ] **Step 3: Create `training/calibration.py`**

```python
"""Confidence calibration via temperature scaling (per-horizon), plus ECE."""
import torch
import torch.nn as nn


def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Return temperature-scaled softmax probabilities. Argmax is invariant to T>0."""
    return torch.softmax(logits / T, dim=-1)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Fit a single scalar temperature on (logits, labels) by minimizing NLL. Returns T>0."""
    logits = logits.detach()
    log_T = torch.zeros(1, requires_grad=True)  # optimize log T to keep T>0
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=max_iter)
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(logits / log_T.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Standard ECE over confidence bins."""
    conf, preds = probs.max(dim=-1)
    acc = (preds == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return float(ece.item())


def reliability_bins(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """Return (bin_centers, bin_acc, bin_conf, bin_count) for a reliability diagram."""
    conf, preds = probs.max(dim=-1)
    acc = (preds == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    centers, b_acc, b_conf, counts = [], [], [], []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        centers.append(((edges[i] + edges[i + 1]) / 2).item())
        counts.append(int(m.sum().item()))
        b_acc.append(float(acc[m].mean().item()) if m.any() else float("nan"))
        b_conf.append(float(conf[m].mean().item()) if m.any() else float("nan"))
    return centers, b_acc, b_conf, counts
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest training/test_calibration.py -v`
Expected: PASS (T>0; argmax unchanged; ECE not increased).

- [ ] **Step 5: Commit**

```bash
git add training/calibration.py training/test_calibration.py
git commit -m "feat(calibration): per-horizon temperature scaling + ECE + reliability bins"
```

---

## Task 11: `evaluate_v2.py` — held-out accuracy, confusion/PRF1, calibration, conf-coverage

**Files:**
- Create: `training/evaluate_v2.py`
- Test: `training/test_evaluate_v2.py` (Create)

- [ ] **Step 1: Write the failing test (pure-metric functions)**

```python
# training/test_evaluate_v2.py
import sys, logging, numpy as np, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.evaluate_v2 import confusion_matrix, precision_recall_f1, confidence_coverage_curve

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_confusion_and_prf1_sum_to_n():
    preds = np.array([0, 1, 2, 2, 1, 0, 2])
    labels = np.array([0, 1, 2, 1, 1, 0, 0])
    cm = confusion_matrix(preds, labels, n_classes=3)
    assert cm.sum() == len(preds)
    prf = precision_recall_f1(cm)
    assert set(prf.keys()) == {"precision", "recall", "f1"}
    assert len(prf["precision"]) == 3
    logger.info("PASS: test_confusion_and_prf1_sum_to_n")

def test_confidence_coverage_monotone_coverage():
    probs = torch.rand(500, 3); probs = probs / probs.sum(-1, keepdim=True)
    labels = torch.randint(0, 3, (500,))
    curve = confidence_coverage_curve(probs, labels, thresholds=[0.4, 0.6, 0.8, 0.95])
    covs = [row["coverage"] for row in curve]
    assert all(covs[i] >= covs[i + 1] - 1e-9 for i in range(len(covs) - 1))  # coverage non-increasing
    for row in curve:
        assert 0.0 <= row["coverage"] <= 1.0
    logger.info("PASS: test_confidence_coverage_monotone_coverage")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest training/test_evaluate_v2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.evaluate_v2'`.

- [ ] **Step 3: Create `training/evaluate_v2.py`**

```python
#!/usr/bin/env python3
"""Held-out evaluation for the V2 directional model: per-horizon accuracy, confusion/PRF1,
temperature calibration (reliability + ECE), confidence-vs-coverage. Also re-measures the
V1 baseline directional accuracy on the same test split (sign of predicted mid change)."""
import argparse, json, logging, pickle, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dataset import DataConfig, build_dataloaders
from training.model_v2 import CompoundAttentionModelV2, LOBLossV2, compute_direction_labels
from training.calibration import fit_temperature, apply_temperature, expected_calibration_error, reliability_bins

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_v2")


def confusion_matrix(preds, labels, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p, l in zip(preds, labels):
        cm[int(l), int(p)] += 1
    return cm


def precision_recall_f1(cm):
    n = cm.shape[0]
    precision, recall, f1 = [], [], []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        precision.append(float(p)); recall.append(float(r))
        f1.append(float(2 * p * r / (p + r)) if (p + r) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def confidence_coverage_curve(probs, labels, thresholds):
    conf, preds = probs.max(dim=-1)
    rows = []
    N = len(labels)
    for thr in thresholds:
        m = conf >= thr
        cov = float(m.float().mean().item())
        acc = float((preds[m] == labels[m]).float().mean().item()) if m.any() else float("nan")
        rows.append({"threshold": float(thr), "coverage": cov, "accuracy": acc, "n": int(m.sum().item())})
    return rows


@torch.no_grad()
def _collect_logits(model, loader, loss_fn, device):
    """Run model over a loader; return stacked dir_logits (N,H,3) and labels (N,H)."""
    model.eval()
    all_logits, all_labels = [], []
    H = len(loss_fn.direction_horizons)
    for batch in loader:
        context = batch["context"].to(device); target = batch["target"].to(device)
        ex = batch["exchange_id"].to(device); sym = batch["symbol_id"].to(device)
        _, dir_logits = model(context, ex, sym)
        labels = compute_direction_labels(target, context[:, -1, :], loss_fn.mid_price_idx,
                                           loss_fn.direction_horizons, loss_fn.flat_threshold)
        all_logits.append(dir_logits.reshape(-1, H, 3).cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def evaluate(run_dir: str, levels: int = 40, source: str = "db", parquet_dir: str = "lob_data"):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = Path(run_dir)
    n_features = levels * 5 + 19
    cfg = DataConfig(lob_levels=levels, feature_version="v2", savgol_window=11,
                     source=source, parquet_dir=parquet_dir)
    _, val_loader, test_loader, meta = build_dataloaders(cfg, batch_size=64)

    model = CompoundAttentionModelV2(n_levels=levels, n_features=n_features,
                                     context_length=cfg.context_length, prediction_length=cfg.prediction_length,
                                     d_model=66, n_heads=3, n_layers=3, d_ff=264).to(device)
    ck = torch.load(run_dir / "checkpoints" / "best.pt", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    loss_fn = LOBLossV2(n_levels=levels, mid_price_idx=ck.get("mid_price_idx", levels * 4),
                        direction_horizons=tuple(ck.get("direction_horizons", (5, 11, 23))))

    horizons = loss_fn.direction_horizons
    H = len(horizons)
    val_logits, val_labels = _collect_logits(model, val_loader, loss_fn, device)
    test_logits, test_labels = _collect_logits(model, test_loader, loss_fn, device)

    report = {"horizons_steps": list(horizons), "per_horizon": []}
    temps = []
    thresholds = [round(0.4 + 0.05 * i, 2) for i in range(12)]  # 0.40 .. 0.95
    for h in range(H):
        T = fit_temperature(val_logits[:, h, :], val_labels[:, h])
        temps.append(T)
        raw_probs = torch.softmax(test_logits[:, h, :], dim=-1)
        cal_probs = apply_temperature(test_logits[:, h, :], T)
        preds = test_logits[:, h, :].argmax(-1).numpy()
        labels = test_labels[:, h].numpy()
        cm = confusion_matrix(preds, labels)
        report["per_horizon"].append({
            "horizon_steps": int(horizons[h]),
            "accuracy": float((preds == labels).mean()),
            "confusion_matrix": cm.tolist(),
            "prf1": precision_recall_f1(cm),
            "temperature": T,
            "ece_raw": expected_calibration_error(raw_probs, test_labels[:, h]),
            "ece_calibrated": expected_calibration_error(cal_probs, test_labels[:, h]),
            "reliability": dict(zip(["centers", "acc", "conf", "count"],
                                    reliability_bins(cal_probs, test_labels[:, h]))),
            "confidence_coverage": confidence_coverage_curve(cal_probs, test_labels[:, h], thresholds),
        })

    report["v1_baseline"] = v1_baseline_accuracy(meta, cfg, levels, device)
    report["temperatures"] = temps

    out_dir = run_dir / "eval"; out_dir.mkdir(exist_ok=True)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Eval report -> {out_dir/'report.json'}")
    for ph in report["per_horizon"]:
        logger.info(f"  h={ph['horizon_steps']:2d}: acc={ph['accuracy']:.3f} "
                    f"ECE {ph['ece_raw']:.3f}->{ph['ece_calibrated']:.3f} T={ph['temperature']:.2f}")
    logger.info(f"  V1 baseline acc/horizon: {report['v1_baseline']}")
    return report


def v1_baseline_accuracy(meta, cfg, levels, device):
    """Directional accuracy of V1 (no direction head) = sign(predicted mid change) on the SAME test set.
    Uses the per-stream test datasets; predicts mid via the V1 model trained separately.
    NOTE: requires a trained V1 checkpoint at experiments/<v1_run>/checkpoints/best.pt; if absent,
    returns the persisted-historical 0.505 with a flag."""
    # The V1 model forecasts the full feature vector; directional label = sign of forecast mid change.
    # Implemented in Task 12-companion; here we read a precomputed number if available.
    p = Path(meta["config"].parquet_dir).parent / "experiments" / "v1_baseline_acc.json"
    if p.exists():
        return json.loads(p.read_text())
    logger.warning("No V1 baseline computed; using historical 0.505 placeholder.")
    return {"5": 0.505, "11": 0.505, "23": 0.505, "source": "historical-placeholder"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--levels", type=int, default=40)
    ap.add_argument("--source", type=str, default="db", choices=["db", "parquet"])
    ap.add_argument("--parquet-dir", type=str, default="lob_data")
    a = ap.parse_args()
    evaluate(a.run_dir, a.levels, a.source, a.parquet_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the metric tests pass**

Run: `python -m pytest training/test_evaluate_v2.py -v`
Expected: PASS (confusion sums to N; coverage non-increasing in threshold and ∈ [0,1]).

- [ ] **Step 5: Commit**

```bash
git add training/evaluate_v2.py training/test_evaluate_v2.py
git commit -m "feat(evaluate_v2): accuracy/confusion/PRF1 + calibration report + confidence-coverage"
```

---

## Task 12: V1 baseline re-measure helper

**Files:**
- Create: `training/baseline_v1.py`
- Test: `training/test_baseline_v1.py` (Create)

- [ ] **Step 1: Write the failing test (pure label function)**

```python
# training/test_baseline_v1.py
import sys, logging, numpy as np
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.baseline_v1 import sign_direction_labels

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_sign_direction_labels():
    # entry mid 100; future path crosses thresholds
    entry = np.array([100.0, 100.0, 100.0])
    future = np.array([[100.5, 101.0], [99.5, 99.0], [100.0, 100.005]])  # up, down, flat
    labels = sign_direction_labels(entry, future, horizon_idx=1, flat_threshold=0.01)
    assert labels.tolist() == [2, 0, 1]
    logger.info("PASS: test_sign_direction_labels")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest training/test_baseline_v1.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.baseline_v1'`.

- [ ] **Step 3: Create `training/baseline_v1.py`**

```python
#!/usr/bin/env python3
"""Re-measure the V1 baseline directional accuracy on the SAME test split, using the
sign of V1's predicted mid-price change over each horizon (V1 has no direction head)."""
import json, logging, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baseline_v1")


def sign_direction_labels(entry_mid, future_mid, horizon_idx, flat_threshold=0.01):
    """labels (N,) from sign of (future_mid[:,horizon_idx] - entry_mid). 0=down,1=flat,2=up.
    Note: flat_threshold is on the SCALED mid (matches LOBLossV2 which compares scaled mids)."""
    change = future_mid[:, horizon_idx] - entry_mid
    labels = np.ones(len(entry_mid), dtype=np.int64)
    labels[change > flat_threshold] = 2
    labels[change < -flat_threshold] = 0
    return labels


@torch.no_grad()
def measure(v1_run_dir: str, levels: int = 40, source: str = "db", parquet_dir: str = "lob_data",
            horizons=(5, 11, 23), flat_threshold=0.01):
    from training.dataset import DataConfig, build_dataloaders
    from training.model import CompoundAttentionModel
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    n_features = levels * 5 + 11
    mid_idx = levels * 4
    cfg = DataConfig(lob_levels=levels, feature_version="v1", savgol_window=11,
                     source=source, parquet_dir=parquet_dir)
    _, _, test_loader, _ = build_dataloaders(cfg, batch_size=64)
    model = CompoundAttentionModel(n_levels=levels, n_features=n_features,
                                   context_length=cfg.context_length, prediction_length=cfg.prediction_length,
                                   d_model=66, n_heads=3, n_layers=3, d_ff=264).to(device)
    ck = torch.load(Path(v1_run_dir) / "checkpoints" / "best.pt", weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    correct = {h: 0 for h in horizons}; total = 0
    for batch in test_loader:
        context = batch["context"].to(device); target = batch["target"].to(device)
        ex = batch["exchange_id"].to(device); sym = batch["symbol_id"].to(device)
        pred = model(context, ex, sym)  # V1 returns (B,Tp,F)
        entry = context[:, -1, mid_idx].cpu().numpy()
        pred_mid = pred[:, :, mid_idx].cpu().numpy()
        true_mid = target[:, :, mid_idx].cpu().numpy()
        for h in horizons:
            hi = min(h, pred_mid.shape[1] - 1)
            pl = sign_direction_labels(entry, pred_mid, hi, flat_threshold)
            tl = sign_direction_labels(entry, true_mid, hi, flat_threshold)
            correct[h] += int((pl == tl).sum())
        total += len(entry)
    acc = {str(h): (correct[h] / max(total, 1)) for h in horizons}
    out = Path(parquet_dir).parent / "experiments" / "v1_baseline_acc.json"
    out.write_text(json.dumps({**acc, "source": "v1-remeasured", "n": total}, indent=2))
    logger.info(f"V1 baseline accuracy: {acc} (n={total}) -> {out}")
    return acc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-run-dir", required=True)
    ap.add_argument("--levels", type=int, default=40)
    ap.add_argument("--source", default="db", choices=["db", "parquet"])
    ap.add_argument("--parquet-dir", default="lob_data")
    a = ap.parse_args()
    measure(a.v1_run_dir, a.levels, a.source, a.parquet_dir)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest training/test_baseline_v1.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/baseline_v1.py training/test_baseline_v1.py
git commit -m "feat(baseline_v1): re-measure V1 directional accuracy (sign of predicted mid) on test split"
```

---

## Task 13: `backtest_v2.py` — window-level offline PnL (maker + taker)

**Files:**
- Create: `training/backtest_v2.py`
- Test: `training/test_backtest_v2.py` (Create)

- [ ] **Step 1: Write the failing test (pure backtest engine on synthetic signals)**

```python
# training/test_backtest_v2.py
import sys, logging, numpy as np
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.backtest_v2 import simulate_window_trades

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_simulate_basic_metrics_and_cost_ordering():
    # 5 windows; entry/exit mids and predicted (class, conf)
    entry = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    exitp = np.array([101.0, 99.0, 100.0, 102.0, 98.0])  # +1%,-1%,0,+2%,-2%
    cls = np.array([2, 0, 2, 2, 0])      # up, down, up, up, down  (all "correct" directionally)
    conf = np.array([0.9, 0.9, 0.3, 0.95, 0.95])  # window 2 below gate
    ts = np.arange(5) * 1000.0           # far apart -> no cooldown blocking
    maker = simulate_window_trades(entry, exitp, cls, conf, ts, threshold=0.6,
                                    cost_bps_per_side=2.0, cooldown_s=300, hold_s=120)
    taker = simulate_window_trades(entry, exitp, cls, conf, ts, threshold=0.6,
                                   cost_bps_per_side=5.0, cooldown_s=300, hold_s=120)
    assert 0.0 <= maker["coverage"] <= 1.0
    assert maker["n_trades"] == 4               # window 2 gated out
    assert taker["net_return"] <= maker["net_return"] + 1e-12   # taker costs more
    assert np.isfinite(maker["sharpe_per_trade"])
    logger.info(f"PASS: test_simulate ... maker_net={maker['net_return']:.4f} taker_net={taker['net_return']:.4f}")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest training/test_backtest_v2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.backtest_v2'`.

- [ ] **Step 3: Create `training/backtest_v2.py`**

```python
#!/usr/bin/env python3
"""Offline window-level PnL backtest on the held-out test split. Gated on calibrated
confidence; reports maker-optimistic and taker-conservative cost brackets + a conf->PnL curve.
NOT a tick-level fill simulation (realistic fill model deferred per spec)."""
import argparse, json, logging, pickle, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest_v2")

PRIMARY_HORIZON_STEPS = 23  # 2-min horizon


def simulate_window_trades(entry_mid, exit_mid, pred_cls, conf, ts,
                           threshold, cost_bps_per_side, cooldown_s, hold_s):
    """One decision per window, in time order. Long if cls==2, short if cls==0, skip if flat/gated.
    Enforces cooldown between entries. Returns metrics dict."""
    order = np.argsort(ts)
    cost = 2.0 * cost_bps_per_side / 1e4  # round-trip fraction
    rets = []
    last_entry_ts = -np.inf
    eligible = 0
    for i in order:
        if pred_cls[i] == 1 or conf[i] < threshold:
            continue
        eligible += 1
        if ts[i] - last_entry_ts < cooldown_s:
            continue
        gross = (exit_mid[i] / entry_mid[i] - 1.0) if pred_cls[i] == 2 else (1.0 - exit_mid[i] / entry_mid[i])
        rets.append(gross - cost)
        last_entry_ts = ts[i]
    rets = np.array(rets, dtype=np.float64)
    n = len(rets)
    total_decisions = int((pred_cls != 1).sum())
    win_rate = float((rets > 0).mean()) if n else float("nan")
    sharpe = float(rets.mean() / rets.std()) if n > 1 and rets.std() > 0 else float("nan")
    equity = np.cumprod(1.0 + rets) if n else np.array([1.0])
    max_dd = float((1.0 - equity / np.maximum.accumulate(equity)).max()) if n else 0.0
    return {
        "n_trades": int(n),
        "coverage": float(n / total_decisions) if total_decisions else 0.0,
        "net_return": float(rets.sum()),
        "avg_edge_bps": float(rets.mean() * 1e4) if n else float("nan"),
        "win_rate": win_rate,
        "sharpe_per_trade": sharpe,
        "max_drawdown": max_dd,
    }


@torch.no_grad()
def _collect_window_signals(model, ds, loss_fn, device, temperature):
    """For one stream's test LOBDataset: return entry_mid, exit_mid (raw), pred_cls, conf, ts
    at the primary horizon, one row per window."""
    from training.calibration import apply_temperature
    mid_idx = loss_fn.mid_price_idx
    scaler = ds  # placeholder; real scaler passed separately
    entry, exitm, cls, conf, ts = [], [], [], [], []
    h_idx = min(PRIMARY_HORIZON_STEPS, ds.config.prediction_length - 1)
    horizons = list(loss_fn.direction_horizons)
    h_pos = horizons.index(PRIMARY_HORIZON_STEPS) if PRIMARY_HORIZON_STEPS in horizons else len(horizons) - 1
    for k in range(len(ds)):
        item = ds[k]
        context = item["context"].unsqueeze(0).to(device)
        target = item["target"].unsqueeze(0).to(device)
        ex = torch.tensor([ds.exchange_id]).to(device)
        sym = torch.tensor([ds.symbol_id]).to(device)
        _, dir_logits = model(context, ex, sym)
        probs = apply_temperature(dir_logits.reshape(1, len(horizons), 3)[:, h_pos, :], temperature)
        c = int(probs.argmax(-1).item())
        entry.append(context[0, -1, mid_idx].item())
        exitm.append(target[0, h_idx, mid_idx].item())
        cls.append(c); conf.append(float(probs.max().item()))
        ts.append(float(item["timestamp"]))
    return (np.array(entry), np.array(exitm), np.array(cls), np.array(conf), np.array(ts))


def backtest(run_dir: str, levels: int = 40, source: str = "db", parquet_dir: str = "lob_data",
             gate: float = 0.6, hold_s: int = 120, cooldown_s: int = 300):
    from training.dataset import DataConfig, build_dataloaders
    from training.model_v2 import CompoundAttentionModelV2, LOBLossV2
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = Path(run_dir)
    n_features = levels * 5 + 19
    cfg = DataConfig(lob_levels=levels, feature_version="v2", savgol_window=11,
                     source=source, parquet_dir=parquet_dir)
    _, _, _, meta = build_dataloaders(cfg, batch_size=64)
    model = CompoundAttentionModelV2(n_levels=levels, n_features=n_features,
                                     context_length=cfg.context_length, prediction_length=cfg.prediction_length,
                                     d_model=66, n_heads=3, n_layers=3, d_ff=264).to(device)
    ck = torch.load(run_dir / "checkpoints" / "best.pt", weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    loss_fn = LOBLossV2(n_levels=levels, mid_price_idx=ck.get("mid_price_idx", levels * 4),
                        direction_horizons=tuple(ck.get("direction_horizons", (5, 11, 23))))
    # temperature for the primary horizon from the eval report (if present), else 1.0
    eval_report = run_dir / "eval" / "report.json"
    T = 1.0
    if eval_report.exists():
        rep = json.loads(eval_report.read_text())
        for ph in rep["per_horizon"]:
            if ph["horizon_steps"] == PRIMARY_HORIZON_STEPS:
                T = ph["temperature"]

    scalers = pickle.load(open(run_dir / "scalers.pkl", "rb"))
    results = {"gate": gate, "primary_horizon_steps": PRIMARY_HORIZON_STEPS, "per_symbol": {}}
    cost_models = {"maker": 2.0, "taker": 5.0}
    for key, ds in meta["test_datasets_by_stream"].items():
        if key not in ("binance_perp_BTC-USDT", "binance_perp_ETH-USDT"):
            continue  # tradable target only
        entry_s, exit_s, cls, conf, ts = _collect_window_signals(model, ds, loss_fn, device, T)
        # de-scale mid back to raw price using the stream's scaler
        sc = scalers[key]; mid_idx = loss_fn.mid_price_idx
        mean, std = sc._means[mid_idx], sc._stds[mid_idx]
        entry = entry_s * std + mean
        exitm = exit_s * std + mean
        results["per_symbol"][key] = {
            cm: simulate_window_trades(entry, exitm, cls, conf, ts, gate, bps, cooldown_s, hold_s)
            for cm, bps in cost_models.items()
        }
        # confidence -> net PnL curve (maker cost)
        curve = []
        for thr in [round(0.4 + 0.05 * i, 2) for i in range(12)]:
            curve.append({"threshold": thr,
                          **simulate_window_trades(entry, exitm, cls, conf, ts, thr, 2.0, cooldown_s, hold_s)})
        results["per_symbol"][key]["conf_pnl_curve_maker"] = curve

    out_dir = run_dir / "backtest"; out_dir.mkdir(exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(results, indent=2))
    logger.info(f"Backtest report -> {out_dir/'report.json'}")
    for key, r in results["per_symbol"].items():
        logger.info(f"  {key}: maker net={r['maker']['net_return']:.4f} "
                    f"(win={r['maker']['win_rate']:.2f}, n={r['maker']['n_trades']}, cov={r['maker']['coverage']:.2f}) "
                    f"| taker net={r['taker']['net_return']:.4f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--levels", type=int, default=40)
    ap.add_argument("--source", default="db", choices=["db", "parquet"])
    ap.add_argument("--parquet-dir", default="lob_data")
    ap.add_argument("--gate", type=float, default=0.6)
    a = ap.parse_args()
    backtest(a.run_dir, a.levels, a.source, a.parquet_dir, a.gate)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the engine test passes**

Run: `python -m pytest training/test_backtest_v2.py -v`
Expected: PASS (4 trades after gating; taker net ≤ maker net; coverage ∈ [0,1]; finite Sharpe).

- [ ] **Step 5: Commit**

```bash
git add training/backtest_v2.py training/test_backtest_v2.py
git commit -m "feat(backtest_v2): window-level PnL backtest (maker/taker) + conf->PnL curve"
```

---

## Task 14: `data_source.py` — parquet-backed fetch for Colab

**Files:**
- Create: `training/data_source.py`
- Modify: `training/dataset.py` (`_fetch_stream_data` to branch on `config.source`)
- Test: `training/test_data_source.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# training/test_data_source.py
import sys, logging, numpy as np, pandas as pd
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.data_source import fetch_parquet_stream

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_fetch_parquet_stream(tmp_path):
    n_levels = 40
    cols = ["bucket"]
    for i in range(1, n_levels + 1):
        cols += [f"bid_price_{i}", f"bid_volume_{i}", f"ask_price_{i}", f"ask_volume_{i}"]
    cols += ["mid_price", "spread"]
    df = pd.DataFrame(np.random.rand(50, len(cols)) + 1.0, columns=cols)
    df["bucket"] = pd.date_range("2026-03-12", periods=50, freq="5s")
    f = tmp_path / "binance_perp_BTC-USDT.parquet"
    df.to_parquet(f, index=False)
    feats, ts = fetch_parquet_stream(str(tmp_path), "binance_perp", "BTC-USDT", n_levels)
    assert feats.shape == (50, n_levels * 4 + 2)
    assert ts.shape == (50,)
    logger.info("PASS: test_fetch_parquet_stream")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest training/test_data_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.data_source'`.

- [ ] **Step 3: Create `training/data_source.py`**

```python
"""Parquet-backed data source (Colab cannot reach the local TimescaleDB)."""
from pathlib import Path
import numpy as np
import pandas as pd

from training.dataset import _build_feature_columns


def fetch_parquet_stream(parquet_dir, exchange, symbol, n_levels):
    """Return (features (T, 4N+2) float64, timestamps (T,) float64) from an exported parquet."""
    path = Path(parquet_dir) / f"{exchange}_{symbol}.parquet"
    df = pd.read_parquet(path)
    feature_cols = _build_feature_columns(n_levels)
    feats = df[feature_cols].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(df["bucket"]).astype("int64").to_numpy() / 1e9  # unix seconds
    return feats, ts
```

- [ ] **Step 4: Branch `_fetch_stream_data` on `config.source`**

At the very top of `_fetch_stream_data` (before building the query), add:

```python
    if getattr(config, "source", "db") == "parquet":
        from training.data_source import fetch_parquet_stream
        return fetch_parquet_stream(config.parquet_dir, exchange, symbol, config.lob_levels)
```

(Keep the existing DB path below unchanged. Note: completeness filtering from Task 7 is DB-only; parquet is already `all_valid`-filtered at export.)

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest training/test_data_source.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add training/data_source.py training/dataset.py training/test_data_source.py
git commit -m "feat(data_source): parquet-backed fetch + DataConfig.source branch for Colab"
```

---

## Task 15: `colab_v2_runner.py` + Colab notebook outline

**Files:**
- Create: `training/colab_v2_runner.py`
- Create: `training/colab_v2.md` (notebook cell outline — the engineer pastes these into a Colab notebook saved on Drive)

- [ ] **Step 1: Create `training/colab_v2_runner.py`**

```python
#!/usr/bin/env python3
"""One-call runner for Colab: train V2, evaluate, backtest from a parquet export.
Assumes lob_data/ has been populated from the Drive zip and the repo is importable.

    python training/colab_v2_runner.py --levels 40 --epochs 50 --batch-size 16 --accum-steps 8
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run(levels=40, epochs=50, batch_size=16, accum_steps=8, parquet_dir="lob_data", run_name=None):
    from training import train_v2, evaluate_v2, backtest_v2
    # 1) train (parquet source so it works on Colab without the DB)
    sys.argv = ["train_v2.py", "--levels", str(levels), "--epochs", str(epochs),
                "--batch-size", str(batch_size), "--accum-steps", str(accum_steps),
                "--source", "parquet", "--parquet-dir", parquet_dir]
    if run_name:
        sys.argv += ["--run-name", run_name]
    train_v2.main()
    # locate the run dir (most recent under experiments/)
    exp = Path(__file__).parent.parent / "experiments"
    run_dir = max(exp.iterdir(), key=lambda p: p.stat().st_mtime)
    # 2) evaluate + 3) backtest
    evaluate_v2.evaluate(str(run_dir), levels=levels, source="parquet", parquet_dir=parquet_dir)
    backtest_v2.backtest(str(run_dir), levels=levels, source="parquet", parquet_dir=parquet_dir)
    return str(run_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--parquet-dir", default="lob_data")
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()
    print("Run dir:", run(a.levels, a.epochs, a.batch_size, a.accum_steps, a.parquet_dir, a.run_name))
```

- [ ] **Step 2: Create `training/colab_v2.md` (notebook cells)**

````markdown
# Colab V2 Training Notebook (cells)

**Runtime:** A100, **High-RAM** instance (≈20 GB resident for 219 features × 16 streams).

**Cell 1 — mount + repo**
```python
from google.colab import drive; drive.mount('/content/drive')
!git clone <your-repo-url> /content/aai || (cd /content/aai && git pull)
%cd /content/aai
!pip -q install torch numpy pandas pyarrow scipy
```

**Cell 2 — GPU + RAM check**
```python
import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())
!free -g    # confirm High-RAM (~80GB)
```

**Cell 3 — pull the fresh export from Drive**
```python
!cp "/content/drive/MyDrive/training data/lob_training_data.zip" /content/aai/
!rm -rf lob_data && mkdir lob_data && unzip -o lob_training_data.zip -d /content/aai
!ls -la lob_data | head
```

**Cell 4 — train + eval + backtest (resumable)**
```python
!python training/colab_v2_runner.py --levels 40 --epochs 50 --batch-size 16 --accum-steps 8 --run-name v2_full_run
# If the session drops, re-run training with resume:
# !python training/train_v2.py --levels 40 --epochs 50 --batch-size 16 --accum-steps 8 --resume --source parquet --run-name v2_full_run
```

**Cell 5 — save results back to Drive**
```python
import shutil, glob, os
run_dir = max(glob.glob('experiments/*'), key=os.path.getmtime)
shutil.make_archive('/content/drive/MyDrive/training data/aai_v2_results', 'zip', run_dir)
print('saved', run_dir)
```
````

- [ ] **Step 3: Smoke-import the runner**

Run: `python -c "import training.colab_v2_runner"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add training/colab_v2_runner.py training/colab_v2.md
git commit -m "feat(colab): one-call V2 runner + resumable A100 notebook outline"
```

---

## Task 16: Full local integration smoke (parquet → train → eval → backtest)

**Files:**
- Test: `training/test_integration_v2.py` (Create)

- [ ] **Step 1: Write the integration test (tiny synthetic parquet, CPU)**

```python
# training/test_integration_v2.py
import sys, logging, tempfile, numpy as np, pandas as pd, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.dataset import DataConfig, build_dataloaders

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def _write_stream(d, exchange, symbol, n_levels=5, T=600):
    cols = ["bucket"]
    for i in range(1, n_levels + 1):
        cols += [f"bid_price_{i}", f"bid_volume_{i}", f"ask_price_{i}", f"ask_volume_{i}"]
    cols += ["mid_price", "spread"]
    rng = np.random.default_rng(0)
    arr = np.cumsum(rng.normal(0, 0.01, size=(T, len(cols) - 1)), axis=0) + 100.0
    df = pd.DataFrame(arr, columns=cols[1:])
    df.insert(0, "bucket", pd.date_range("2026-03-12", periods=T, freq="5s"))
    df.to_parquet(f"{d}/{exchange}_{symbol}.parquet", index=False)

def test_parquet_pipeline_builds_v2_loaders():
    with tempfile.TemporaryDirectory() as d:
        _write_stream(d, "binance_perp", "BTC-USDT", n_levels=5, T=600)
        cfg = DataConfig(lob_levels=5, feature_version="v2", savgol_window=11,
                         source="parquet", parquet_dir=d,
                         exchanges=["binance_perp"], pairs=["BTC-USDT"],
                         context_length=20, prediction_length=24, stride=10)
        train, val, test, meta = build_dataloaders(cfg, batch_size=8)
        batch = next(iter(train))
        assert batch["context"].shape[-1] == 5 * 5 + 19  # 44 features at 5 levels
        assert meta["n_train_windows"] > 0
        logger.info("PASS: test_parquet_pipeline_builds_v2_loaders")
```

- [ ] **Step 2: Run to verify it passes** (exercises Task 5/6/7/14 together)

Run: `python -m pytest training/test_integration_v2.py -v`
Expected: PASS — V2 feature width = `5*5+19 = 44`; train windows > 0.

- [ ] **Step 3: Run the full new-test suite**

Run: `python -m pytest training/test_export.py training/test_direction_utils.py training/test_mid_price_idx.py training/test_dataset_v2.py training/test_tracker_v2.py training/test_train_v2.py training/test_calibration.py training/test_evaluate_v2.py training/test_baseline_v1.py training/test_backtest_v2.py training/test_data_source.py training/test_integration_v2.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add training/test_integration_v2.py
git commit -m "test: end-to-end parquet->V2 dataloaders integration smoke"
```

---

## Task 17: Execution runbook (operational, no code)

**Files:**
- Create: `docs/superpowers/plans/2026-06-03-v2-runbook.md`

- [ ] **Step 1: Write the runbook**

```markdown
# V2 Production Run — Runbook

1. **Rotate leaked Binance keys FIRST** (independent of this work): revoke keys in
   executor/config.json + executor/.env on Binance, move to untracked secrets, scrub history.
2. **Fresh export** (on the DB host, low-activity window):
   `python export_training_data.py --fresh`  → verify row counts ≫ Apr-11 sizes; `lob_training_data.zip` regenerated.
3. **Upload** `lob_training_data.zip` to Google Drive "training data" (overwrite).
4. **Train V1 baseline** (for apples-to-apples; same fresh data, V1 pipeline):
   `python training/train.py --levels 40 --epochs 50 --batch-size 16`  (note its run dir)
   then `python training/baseline_v1.py --v1-run-dir experiments/<v1_run> --levels 40`
5. **Colab**: open the High-RAM A100 notebook (training/colab_v2.md), run cells 1–5.
   Use `--resume` if the session drops.
6. **Read** `experiments/<run>/eval/report.json` (per-horizon accuracy vs V1 baseline; ECE before/after;
   confidence-coverage) and `experiments/<run>/backtest/report.json` (maker/taker net PnL, conf→PnL curve).
7. **Decision gate:** proceed to execution only if calibrated high-confidence accuracy clears the
   V1 baseline AND maker-cost net PnL is positive at a usable coverage.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-06-03-v2-runbook.md
git commit -m "docs: V2 production run runbook"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- C1 export fix → Tasks 1–2 ✓
- C2 leak-free dataset + V2 features → Tasks 5–7, 14 ✓
- C3 train_v2.py (grad accum, dir-acc, best-on-accuracy, resume, rich checkpoint) → Tasks 8–9 ✓
- C4 calibration → Task 10 (+ applied in 11/13) ✓
- C5 evaluate_v2.py (accuracy/confusion/PRF1/conf-coverage/baseline) → Tasks 11–12 ✓
- C6 backtest_v2.py (maker+taker, conf→PnL) → Task 13 ✓
- C7 resumable Colab notebook → Tasks 14–15 ✓
- Correctness guards (mid_price_idx test, n_features assert, completeness filter) → Tasks 4, 7, 9 ✓
- Integration + runbook → Tasks 16–17 ✓

**Placeholder scan:** `v1_baseline_accuracy` in `evaluate_v2.py` reads a precomputed file or returns a clearly-flagged historical placeholder; the real number is produced by Task 12's `baseline_v1.measure` and consumed via `experiments/v1_baseline_acc.json` — not a silent gap.

**Type/signature consistency:** `run_one_epoch_v2`/`evaluate_v2_loader` (Task 9), `compute_direction_labels`/`directional_accuracy` (Task 3), `fit_temperature`/`apply_temperature`/`expected_calibration_error` (Task 10), `simulate_window_trades` (Task 13), `fetch_parquet_stream` (Task 14), and `DataConfig.source/feature_version/warmup_trim/n_enriched_features` (Task 5) are referenced consistently across tasks. `mid_price_idx = levels*4` used uniformly. Checkpoint keys written in Task 9 (`mid_price_idx`, `direction_horizons`, `n_features`, `scheduler_step_count`) are exactly the keys read in Tasks 11/13.
