# AAI Master Plan — Multi-Exchange LOB Forecasting & Trading System

## Project Overview

A multi-exchange Limit Order Book (LOB) forecasting system that collects real-time 40-level orderbook data from 4 exchanges, trains a Compound Attention model to predict future LOB states, and ultimately drives a profitable trading bot focused on basis trading (spot vs perp arbitrage).

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Infrastructure | RUNNING | TimescaleDB + 40-level collector live |
| Training Pipeline | COMPLETE | 5 steps built & tested, dry run passed |
| 40-Level Data Collection | IN PROGRESS | Need ~3 months for production training |
| Production Model Training | BLOCKED | Waiting on sufficient 40-level data |
| Trading Execution | NOT STARTED | Post-model milestone |

---

## Architecture

```
Exchanges (WS)          TimescaleDB           Training Pipeline         Model
─────────────────       ──────────────        ──────────────────       ──────────
Binance Spot  ──┐       lob_snapshots         dataset.py               Compound
Binance Perp  ──┤──→    (40-level hyper-  ──→ features.py          ──→ Attention
Bybit Spot    ──┤       table, 1hr chunks)    model.py                 Model
KuCoin Spot   ──┘       lob_5s (5s cont.      tracker.py               (arXiv:
                         aggregate)            train.py                  2409.02277)
```

---

## Infrastructure

### Docker — TimescaleDB (`docker-compose.yml`)
- Single `timescale/timescaledb:latest-pg17` instance
- Port 5432, DB: `lob_data`, User: `lob_user`
- 8GB shared_buffers, 256MB work_mem

### Database Schema (`db/init.sql`)
- **`lob_snapshots`** — Primary hypertable, 40-level columnar storage (160 bid/ask price/volume columns + mid_price, spread, volume_imbalance, is_valid, validation_flags)
- **`data_quality_log`** — Tracks gaps, crossed books, zero prices, stale data
- **`lob_5s`** — Continuous aggregate using `last(value, time)` for 5-second resampled snapshots
- Hypertable chunked by 1-hour intervals, compression after 1 day
- Migration from 5→40 levels: `db/migrate_40_levels.sql`

### MCP Server (`.mcp.json`)
- `crystaldba/postgres-mcp` for direct SQL access to TimescaleDB from Claude
- Unrestricted access mode for ad-hoc data quality queries

### Data Collector (`lob_collector.py`)
- Async Python with `asyncpg` connection pool (5-10 connections)
- Batched inserts (1-second buffer flush)
- Proper OrderBook class with delta application for each exchange
- Inline validation on every snapshot (crossed books, zero prices, level inversions)
- Infinite retry with exponential backoff (1s→60s cap)
- Health monitoring (stale stream detection, throughput tracking)
- Dynamic SQL generation based on configurable `lob_levels` (default: 40)

**Exchange WebSocket feeds:**
| Exchange | Feed | Levels |
|----------|------|--------|
| Binance Spot | `@depth` (diff stream) | 40 (from 1000+ via REST) |
| Binance Perp | `@depth` (diff stream) | 40 (from 1000+ via REST) |
| Bybit Spot | `orderbook.200` | 40 (from 200) |
| KuCoin Spot | `level2Depth50` | 40 (from 50 full snapshots) |

**Symbols:** BTC-USDT, ETH-USDT, SOL-USDT, WLD-USDT (16 streams total)

---

## Training Pipeline

### Step 1: DataLoader (`training/dataset.py`)
- `DataConfig` — DB connection, LOB params (levels=40, context=120 steps, prediction=24 steps, stride=60)
- `_fetch_stream_data()` — Async fetch from `lob_5s` continuous aggregate
- `LOBScaler` — Z-score normalization (per-column mean/std), fitted on train only, with inverse_transform for structure loss. Save/load support.
- `build_dataloaders()` — Full pipeline: fetch → engineer_features() → split (60/20/20 chronological) → z-score scale → DataLoaders
- `LOBDataset` — Sliding window PyTorch Dataset returning context/target/metadata dicts
- `MultiStreamLOBDataset` — Combines multiple stream datasets with cumulative index mapping
- Integer encodings for exchange, symbol, market type (for compound attribute embeddings)
- **Tests:** `training/test_dataset.py` (10 tests)

### Step 2: Model Architecture (`training/model.py`)
- **`Time2Vec`** — Learnable temporal embedding (linear + periodic sine components)
- **`CompoundAttributeEmbedding`** — Encodes: level position (0..N-1), side (bid/ask/both), feature type (price/volume/flow), exchange, symbol, context vs target. Pre-computes attribute index buffers for base + derived features.
- **`FeatureProjection`** — Scalar feature → d_model projection + attribute embedding
- **`DualAttentionBlock`** — Factored attention: feature attention (across F per timestep) + temporal attention (across T per feature) + FFN. Complexity O(TF² + FT²) vs O((TF)²) for flat attention. 77x reduction at 40 levels.
- **`TransformerBlock`** — Legacy flat attention block (kept for backward compatibility)
- **`CompoundAttentionModel`** — Feature projection → attribute embedding → Time2Vec → DualAttention encoder (4D tensor throughout) → per-feature prediction head
  - Default config: d_model=66, n_heads=3, n_layers=3, d_ff=264, dropout=0.1
  - 5-level: ~34K params; 40-level: ~249K params
- **`LOBLoss`** — MSE forecasting loss + structure-preserving loss with inverse-transform. Accepts scaler means/stds to check ordering in raw price space. Normalizes by reference price for dimensionless loss.
- **`WarmupDecayScheduler`** — Linear warmup (1000 steps) then multiplicative decay (factor=0.8 every 5000 steps)
- Based on arXiv:2409.02277 (Compound Attention) + arXiv:2502.15757 (TLOB dual attention)
- **Tests:** `training/test_model.py` (15 tests, including DualAttentionBlock, scaler-aware loss, overfit sanity check)

### Step 3: Feature Engineering (`training/features.py`)
- **Savitzky-Golay smoothing** — Window=21, polyorder=3 on price columns only. Highest-impact preprocessing step per arXiv:2506.05764.
- **Multi-level OFI** — Order Flow Imbalance per level: measures net buying/selling pressure from LOB changes. Reduces RMSE by 68-74% (Kolm & Turiel, 2023).
- **Aggregate OFI** — Sum across all levels.
- **Volume features** — Cumulative bid/ask volume, volume ratio, total volume imbalance.
- **Price features** — Price imbalance, spread ratio, depth bid/ask.
- **Cross-exchange features** — Mid-price differential (raw + bps), spread ratio between exchanges.
- Output: 5-level = 22→36 features; 40-level = 162→211 features.
- **Tests:** `training/test_features.py` (11 tests, including real DB data)

### Step 4: Experiment Tracking (`training/tracker.py`)
- `RunConfig` — Dataclass with all hyperparameters (model, training, data, feature engineering)
- `ExperimentTracker` — File-based tracking:
  - Config logging (JSON)
  - Per-epoch metrics (JSON array)
  - Checkpoint save/load: `best.pt`, `latest.pt`, periodic every 10 epochs
  - Run summary
- Experiments saved to `experiments/` directory
- Designed to be replaceable with W&B/MLflow later
- **Tests:** `training/test_tracker.py` (3 tests)

### Step 5: Training Script (`training/train.py`)
- CLI: `--levels`, `--epochs`, `--batch-size`, `--lr`, `--d-model`, `--n-heads`, `--n-layers`, `--patience`, `--exchanges`, `--pairs`, `--dry`, `--run-name`
- `train_one_epoch()` — Forward + backward + gradient clipping (max_norm=1.0) + scheduler step
- `evaluate()` — No-grad validation/test evaluation
- Full loop: build dataloaders → create model → train with early stopping → load best → final test eval
- MPS GPU (Apple Silicon) supported
- **Dry run verified:** 3 epochs on binance_spot BTC-USDT 5-level enriched data (36 features). Train loss: 1.008→0.895, val: 0.637, test: 0.874. Structure loss: ~0.001 (meaningful, operating on inverse-transformed prices).

---

## Data Requirements for 40-Level Training

| Timeframe | 5s Rows/Stream | Total (16 streams) | Use Case |
|-----------|----------------|---------------------|----------|
| 1 week | ~120K | ~2M | Smoke test |
| 1 month | ~520K | ~8.3M | Initial training |
| 3 months | ~1.6M | ~25M | Production model |
| 6 months | ~3.1M | ~50M | Multiple regime coverage |

**Minimum for production:** 3 months covers bull, bear, and sideways regimes across multiple market cycles. Currently accumulating since 40-level migration.

---

## Model Training Parameters

| Parameter | Dry Run (5-level) | Production (40-level) |
|-----------|--------------------|-----------------------|
| Features | 36 (5*5+11 enriched) | 211 (40*5+11 enriched) |
| Context window | 120 steps (10 min) | 120 steps (10 min) |
| Prediction horizon | 24 steps (2 min) | 24 steps (2 min) |
| Stride | 60 (50% overlap) | 60 (50% overlap) |
| d_model | 66 | 66 |
| n_heads | 3 | 3 |
| n_layers | 3 | 3 |
| Attention | DualAttention (factored) | DualAttention (factored) |
| Batch size | 64 | 16-32 (memory limited) |
| Learning rate | 1e-3 | 1e-3 |
| Early stopping | patience=10 | patience=10 |
| Max epochs | 50 | 100 |
| Structure loss weight | 0.01 | 0.01 |
| Normalization | Z-score (per-column) | Z-score (per-column) |
| Parameters | ~34K | ~249K |

---

## Future Ensemble Plan

1. **Compound Attention Model** (primary) — Full LOB state prediction
2. **XGBoost on OFI** — Gradient boosted trees on multi-level OFI features for direction signal
3. **TLOB** (arXiv:2502.15757) — Direction classifier for mid-price movement
4. **Mamba Regime Classifier** — Identifies bull/bear/sideways regimes to switch strategy parameters

---

## Trading Strategy

**Primary alpha source:** Basis trading (spot vs perpetual futures arbitrage)
- Highest-confidence, lowest-risk strategy identified in research
- Model predicts LOB state across both spot and perp → identifies basis divergences
- Execution: market-make on wider side, take on tighter side

**Secondary:** Cross-exchange arbitrage using multi-exchange LOB predictions

---

## File Index

```
aai/
├── docker-compose.yml          # TimescaleDB container
├── docker-compose-backtest.yml # Legacy (unused)
├── lob_collector.py            # Main data collector (40-level, 4 exchanges)
├── multi_exchange_lob_collector.py  # Legacy collector (superseded)
├── .env.example                # Environment variable template
├── .mcp.json                   # MCP server config (Postgres)
├── requirements.txt            # Python dependencies
├── masterplan.md               # This file
├── db/
│   ├── init.sql                # Full schema (40-level, hypertables, continuous agg)
│   └── migrate_40_levels.sql   # Migration from 5→40 levels
├── training/
│   ├── __init__.py
│   ├── dataset.py              # Step 1: DataLoader, scaler, multi-stream dataset
│   ├── model.py                # Step 2: Compound Attention model + loss + scheduler
│   ├── features.py             # Step 3: Feature engineering pipeline
│   ├── tracker.py              # Step 4: Experiment tracker
│   ├── train.py                # Step 5: Training script
│   ├── test_dataset.py         # 8 tests
│   ├── test_model.py           # 12 tests
│   ├── test_features.py        # 11 tests
│   └── test_tracker.py         # 3 tests
├── research/
│   ├── sota-lob-architectures-2024-2026.md
│   └── lob-trading-strategies-profitability-2024-2026.md
├── experiments/                 # Training run outputs (config, metrics, checkpoints)
└── old-scripts/                 # Legacy order lifecycle trackers
```

---

## Test Summary

All 38 tests passing across 4 test files:
- `training/test_dataset.py` — 10 tests (z-score scaler, enriched features, unit + integration with live DB)
- `training/test_model.py` — 15 tests (DualAttentionBlock, expanded embeddings, scaler-aware loss, shapes, scheduler, overfit sanity)
- `training/test_features.py` — 11 tests (OFI, smoothing, volume, price, cross-exchange, pipeline)
- `training/test_tracker.py` — 3 tests (config, checkpoints, summary)

---

## Next Steps

1. **Accumulate 40-level data** — Collector running, target 3 months minimum
2. **Weekly data quality checks** — Via MCP SQL queries against TimescaleDB
3. **Train on 40-level data** — Once sufficient data accumulated
4. **Implement ensemble models** — XGBoost/TLOB/Mamba after primary model validated
5. **Build execution layer** — Trading bot with basis trading strategy
6. **Backtest** — Historical simulation of predicted LOB states → trade signals → PnL
