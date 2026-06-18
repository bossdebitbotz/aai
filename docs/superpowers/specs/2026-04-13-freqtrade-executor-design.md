# AAI Trading Executor — Design Spec

## Overview

Hybrid trading executor: custom 5-second signal service generates trade signals from the Phase 2 LOB forecasting model, pushes them to Freqtrade which handles position management, paper/live trading, and Telegram monitoring on Binance USDT-M perpetual futures.

## Architecture

```
TimescaleDB (lob_5s)
       ↓ (5s poll)
Signal Service (signal_service.py)
  - Loads CompoundAttentionModelV2 checkpoint
  - Queries last 120 rows of lob_5s per pair
  - Runs feature_v2 engineering → model inference → softmax
  - Applies confidence threshold (0.48 for 30s)
  - Pushes /forcelong, /forceshort, /forceexit to Freqtrade REST API
       ↓ (REST API)
Freqtrade (1min candles, dry-run or live)
  - Receives forced trades
  - Manages positions, stoploss, PnL tracking
  - Telegram bot for monitoring
  - Paper ↔ live via config flag
       ↓
Binance Futures (USDT-M perps)
  - BTC/USDT:USDT, ETH/USDT:USDT
```

## Components

### 1. Signal Service (`executor/signal_service.py`)

**Responsibilities:**
- 5-second async loop querying TimescaleDB lob_5s
- Maintains rolling 120-step feature window per pair
- Runs V2 feature engineering (219 features, SG window=11)
- Loads model checkpoint, runs inference on MPS/CPU
- Applies per-horizon confidence thresholds from backtest optimization
- Tracks current position state to avoid duplicate signals
- Pushes trade commands to Freqtrade REST API
- Logs all signals, predictions, confidence levels

**Data flow per tick:**
1. Query `SELECT * FROM lob_5s WHERE exchange='binance_perp' AND symbol=$1 AND bucket > now() - interval '12 minutes' ORDER BY bucket` for each pair
2. Extract 162 base features, run `engineer_features_v2()` → 219 features
3. Z-score normalize using saved scaler from training
4. Take last 120 rows as context window
5. Model forward pass → `(pred, dir_logits)`
6. Reshape dir_logits to (3 horizons, 3 classes), softmax, get confidence + predicted class
7. If 30s horizon confidence >= 0.48 AND predicted class != flat AND no existing same-direction position:
   - predicted=UP → POST /api/v1/forcelong
   - predicted=DOWN → POST /api/v1/forceshort
8. If position open AND (signal flips direction OR confidence drops below threshold):
   - POST /api/v1/forceexit

**Position management rules:**
- One position per pair at a time
- 30s horizon is primary signal (Sharpe 6.0, 77% win rate)
- Exit after 30 seconds OR on signal reversal, whichever comes first
- Stoploss: 0.5% (handled by Freqtrade)

**Dependencies:**
- asyncpg (TimescaleDB connection)
- torch (model inference)
- aiohttp (Freqtrade REST API calls)
- training.model_v2, training.features_v2, training.dataset (LOBScaler)

### 2. Freqtrade Configuration (`executor/config.json`)

```json
{
    "mode": "paper",
    "dry_run": true,
    "dry_run_wallet": 10000,
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "stake_currency": "USDT",
    "stake_amount": 1000,
    "max_open_trades": 2,
    "exchange": {
        "name": "binance",
        "key": "${BINANCE_API_KEY}",
        "secret": "${BINANCE_API_SECRET}",
        "pair_whitelist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        "pair_blacklist": []
    },
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "username": "freqtrade",
        "password": "supersecret",
        "jwt_secret_key": "random_secret_key"
    },
    "telegram": {
        "enabled": true,
        "token": "${TELEGRAM_BOT_TOKEN}",
        "chat_id": "${TELEGRAM_CHAT_ID}"
    },
    "stoploss": -0.005,
    "minimal_roi": {"0": 0.003},
    "order_types": {
        "entry": "market",
        "exit": "market",
        "stoploss": "market"
    }
}
```

**Paper→Live transition:** Change `"mode": "paper"` to `"mode": "live"`, which sets `"dry_run": false`. Config reads API keys and Telegram tokens from environment variables.

### 3. Freqtrade Strategy (`executor/strategies/AAIStrategy.py`)

Minimal strategy — Freqtrade needs a strategy class even when trades are forced via API. This strategy:
- Sets `can_short = True` for futures
- Sets `timeframe = '1m'`
- Sets `stoploss = -0.005` (0.5%)
- `populate_indicators/entry/exit` are no-ops (signals come from signal service)
- `confirm_trade_entry` callback logs the forced trade

### 4. Launcher (`executor/run.sh`)

Starts both processes:
1. Freqtrade in background: `freqtrade trade --config config.json --strategy AAIStrategy`
2. Signal service in foreground: `python signal_service.py`

Ctrl+C stops both.

## File Structure

```
executor/
├── config.json              # Freqtrade config (mode, exchange, telegram)
├── signal_service.py        # 5s inference loop + REST API push
├── run.sh                   # Launcher for both processes
├── strategies/
│   └── AAIStrategy.py       # Minimal Freqtrade strategy (no-op, accepts forced trades)
├── .env.example             # Template for API keys
└── .env                     # Actual keys (gitignored)
```

Model checkpoint and scalers loaded from `experiments/` or Google Drive download.

## Configuration

All tunable parameters in `config.json`:

| Parameter | Paper Default | Notes |
|-----------|--------------|-------|
| mode | paper | paper or live |
| dry_run_wallet | 10000 | USDT starting balance |
| stake_amount | 1000 | Per-trade notional |
| max_open_trades | 2 | One per pair |
| stoploss | -0.005 | 0.5% |
| minimal_roi | 0.003 | 0.3% take-profit |
| confidence_threshold_30s | 0.48 | From backtest optimization |
| confidence_threshold_1min | 0.46 | From backtest optimization |
| confidence_threshold_2min | 0.46 | From backtest optimization |
| primary_horizon | 30s | Which horizon drives trades |
| pairs | BTC/USDT:USDT, ETH/USDT:USDT | Tradeable pairs |

## Fee Model (from research)

| Pair | Round-trip (fee + slippage) |
|------|---------------------------|
| BTC-USDT perp | 0.20% |
| ETH-USDT perp | 0.20% |
| Funding rate | 0.01% per 8h |

## Error Handling

- **DB connection lost:** Signal service retries with exponential backoff (same pattern as lob_collector.py)
- **Freqtrade API unreachable:** Log warning, skip signal, retry next tick
- **Model inference fails:** Log error, skip tick, continue
- **Stale data (>30s since last lob_5s row):** Skip signal generation, log warning
- **Position conflict:** If Freqtrade already has a position and signal says enter same direction, skip

## Monitoring

Via Telegram bot:
- `/status` — current positions
- `/profit` — daily/weekly PnL
- `/balance` — wallet balance
- `/performance` — per-pair performance

Signal service logs to `executor/signal_service.log` with timestamps, predictions, confidences, and trade decisions.

## Success Criteria

Paper trading is considered validated when:
1. Signal service runs for 7+ days without crashes
2. Freqtrade correctly tracks all forced trades
3. Paper PnL is directionally consistent with backtest expectations
4. Telegram alerts fire on every trade entry/exit
5. No missed signals due to latency or errors

## Out of Scope

- Ensemble models (post-90-day)
- Cross-exchange arbitrage
- Position sizing optimization (Optuna, later)
- Multiple timeframe voting (later)
- VPS deployment (after paper validation)
