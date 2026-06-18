# AAI Trading Executor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid trading executor: custom 5s signal service + Freqtrade position manager, paper trading BTC-USDT and ETH-USDT on Binance Futures.

**Architecture:** Signal service queries TimescaleDB every 5s, runs model inference with confidence thresholding, pushes /forcelong /forceshort /forceexit to Freqtrade REST API. Freqtrade handles position management, stoploss, PnL, Telegram alerts in dry-run mode.

**Tech Stack:** Python 3.12, Freqtrade (git clone + stable branch), asyncpg, torch, aiohttp

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `executor/config.json` | Create | Freqtrade config: futures, dry-run, API server, Telegram |
| `executor/signal_config.json` | Create | Signal service config: thresholds, pairs, model path, DB connection |
| `executor/strategies/AAIStrategy.py` | Create | Minimal Freqtrade strategy (no-op, accepts forced trades) |
| `executor/signal_service.py` | Create | 5s async loop: DB query → features → inference → REST API push |
| `executor/test_signal_service.py` | Create | Tests for signal generation logic |
| `executor/run.sh` | Create | Launcher script for both processes |

Freqtrade installed via git clone into `executor/freqtrade/`.

---

### Task 1: Install Freqtrade + Create Config

**Files:**
- Create: `executor/config.json`
- Create: `executor/signal_config.json`

- [ ] **Step 1: Clone and install Freqtrade**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai/executor
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
git checkout stable
pip install -e '.[all]'
```

Verify: `freqtrade --version`

- [ ] **Step 2: Create Freqtrade user directory**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai/executor
freqtrade create-userdir --userdir user_data
```

- [ ] **Step 3: Create executor/config.json**

```json
{
    "tradingMode": "futures",
    "marginMode": "isolated",
    "max_open_trades": 2,
    "stake_currency": "USDT",
    "stake_amount": 1000,
    "dry_run": true,
    "dry_run_wallet": 10000,
    "cancel_open_orders_on_exit": true,
    "exchange": {
        "name": "binance",
        "key": "",
        "secret": "",
        "pair_whitelist": [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT"
        ],
        "pair_blacklist": []
    },
    "entry_pricing": {
        "price_side": "other",
        "use_order_book": true,
        "order_book_top": 1
    },
    "exit_pricing": {
        "price_side": "other",
        "use_order_book": true,
        "order_book_top": 1
    },
    "order_types": {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": false
    },
    "stoploss": -0.005,
    "minimal_roi": {
        "0": 0.003
    },
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "verbosity": "error",
        "enable_openapi": false,
        "jwt_secret_key": "aai_jwt_secret_change_me",
        "CORS_origins": [],
        "username": "aai",
        "password": "aai_paper_trade"
    },
    "telegram": {
        "enabled": false,
        "token": "",
        "chat_id": ""
    },
    "bot_name": "AAI_LOB_Trader",
    "initial_state": "running",
    "internals": {
        "process_throttle_secs": 5
    },
    "strategy": "AAIStrategy",
    "strategy_path": "strategies/"
}
```

- [ ] **Step 4: Create executor/signal_config.json**

```json
{
    "db_host": "localhost",
    "db_port": 5432,
    "db_user": "lob_user",
    "db_password": "lob_password",
    "db_name": "lob_data",
    "model_checkpoint": "../checkpoints/best.pt",
    "scaler_path": "../checkpoints/scalers.pkl",
    "lob_levels": 40,
    "context_length": 120,
    "prediction_length": 24,
    "n_features": 219,
    "pairs": {
        "BTC-USDT": "BTC/USDT:USDT",
        "ETH-USDT": "ETH/USDT:USDT"
    },
    "primary_horizon": "30s",
    "thresholds": {
        "30s": 0.48,
        "1min": 0.46,
        "2min": 0.46
    },
    "horizon_steps": {
        "30s": 5,
        "1min": 11,
        "2min": 23
    },
    "freqtrade_api": {
        "url": "http://127.0.0.1:8080",
        "username": "aai",
        "password": "aai_paper_trade"
    },
    "tick_interval_seconds": 5,
    "stale_data_threshold_seconds": 30,
    "log_file": "signal_service.log"
}
```

- [ ] **Step 5: Verify config files parse correctly**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai/executor
python -c "import json; json.load(open('config.json')); json.load(open('signal_config.json')); print('✓ Both configs valid')"
```

---

### Task 2: Freqtrade Strategy (AAIStrategy)

**Files:**
- Create: `executor/strategies/AAIStrategy.py`

- [ ] **Step 1: Create strategies directory**

```bash
mkdir -p /Volumes/Docker-SSD/projects/aaiwdbback/aai/executor/strategies
```

- [ ] **Step 2: Create AAIStrategy.py**

```python
"""
AAI LOB Trading Strategy for Freqtrade.

This is a minimal no-op strategy. All trade signals come from the
external signal_service.py via Freqtrade's REST API (/forcelong,
/forceshort, /forceexit). This strategy only provides the required
interface and logging.
"""

import logging
from datetime import datetime
from typing import Optional

from freqtrade.strategy import IStrategy
from pandas import DataFrame

logger = logging.getLogger(__name__)


class AAIStrategy(IStrategy):
    """Minimal strategy that accepts forced trades from the signal service."""

    INTERFACE_VERSION = 3

    # Required settings
    timeframe = "1m"
    can_short = True
    stoploss = -0.005        # 0.5% stoploss
    minimal_roi = {"0": 0.003}  # 0.3% take-profit
    startup_candle_count = 0  # no warmup needed (signals come externally)

    # Disable Freqtrade's own signal generation
    process_only_new_candles = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """No-op — indicators computed externally by signal service."""
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """No-op — entries forced via REST API."""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """No-op — exits forced via REST API or handled by stoploss/ROI."""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """Log forced trade entries from signal service."""
        logger.info(
            f"AAI ENTRY: {side.upper()} {pair} @ {rate:.2f} "
            f"(amount={amount:.4f}, tag={entry_tag})"
        )
        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """Log trade exits."""
        profit = trade.calc_profit_ratio(rate)
        logger.info(
            f"AAI EXIT: {pair} @ {rate:.2f} "
            f"(reason={exit_reason}, profit={profit:.4f})"
        )
        return True
```

- [ ] **Step 3: Verify Freqtrade loads the strategy**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai/executor
freqtrade list-strategies --strategy-path strategies/
```

Expected: `AAIStrategy` appears in the list.

---

### Task 3: Signal Service

**Files:**
- Create: `executor/signal_service.py`
- Create: `executor/test_signal_service.py`

- [ ] **Step 1: Write tests for signal generation logic**

Create `executor/test_signal_service.py`:

```python
"""Tests for the signal service inference and decision logic."""

import sys
import json
import numpy as np
import torch

sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")

from executor.signal_service import (
    SignalEngine,
    TradeDecision,
)


def test_signal_engine_init():
    """SignalEngine loads config and initializes state."""
    config = {
        "lob_levels": 5,
        "context_length": 20,
        "prediction_length": 4,
        "n_features": 36,
        "primary_horizon": "30s",
        "thresholds": {"30s": 0.48, "1min": 0.46, "2min": 0.46},
        "horizon_steps": {"30s": 5, "1min": 11, "2min": 23},
        "pairs": {"BTC-USDT": "BTC/USDT:USDT"},
    }
    engine = SignalEngine(config, model=None, scalers={})
    assert engine.primary_horizon == "30s"
    assert engine.thresholds["30s"] == 0.48
    print("PASS: test_signal_engine_init")


def test_trade_decision_from_logits():
    """Trade decisions correctly classify up/down/flat with confidence."""
    config = {
        "lob_levels": 5,
        "context_length": 20,
        "prediction_length": 4,
        "n_features": 36,
        "primary_horizon": "30s",
        "thresholds": {"30s": 0.48, "1min": 0.46, "2min": 0.46},
        "horizon_steps": {"30s": 5, "1min": 11, "2min": 23},
        "pairs": {"BTC-USDT": "BTC/USDT:USDT"},
    }
    engine = SignalEngine(config, model=None, scalers={})

    # Logits strongly favoring "up" for all horizons
    # Shape: (1, 9) = 3 horizons * 3 classes
    # [down, flat, up] per horizon
    logits = torch.tensor([[
        -5.0, -5.0, 10.0,  # 30s: strong up
        -5.0, -5.0, 10.0,  # 1min: strong up
        -5.0, -5.0, 10.0,  # 2min: strong up
    ]])

    decisions = engine.logits_to_decisions(logits)
    assert decisions["30s"].direction == "long"
    assert decisions["30s"].confidence > 0.99
    assert decisions["30s"].should_trade is True
    print("PASS: test_trade_decision_from_logits")


def test_low_confidence_no_trade():
    """Low confidence should result in no trade."""
    config = {
        "lob_levels": 5,
        "context_length": 20,
        "prediction_length": 4,
        "n_features": 36,
        "primary_horizon": "30s",
        "thresholds": {"30s": 0.90, "1min": 0.90, "2min": 0.90},
        "horizon_steps": {"30s": 5, "1min": 11, "2min": 23},
        "pairs": {"BTC-USDT": "BTC/USDT:USDT"},
    }
    engine = SignalEngine(config, model=None, scalers={})

    # Uncertain logits (roughly equal)
    logits = torch.tensor([[
        0.1, 0.0, 0.2,  # 30s: slightly up but low confidence
        0.1, 0.0, 0.2,
        0.1, 0.0, 0.2,
    ]])

    decisions = engine.logits_to_decisions(logits)
    assert decisions["30s"].should_trade is False
    print("PASS: test_low_confidence_no_trade")


def test_flat_prediction_no_trade():
    """Flat prediction should not trade regardless of confidence."""
    config = {
        "lob_levels": 5,
        "context_length": 20,
        "prediction_length": 4,
        "n_features": 36,
        "primary_horizon": "30s",
        "thresholds": {"30s": 0.40, "1min": 0.40, "2min": 0.40},
        "horizon_steps": {"30s": 5, "1min": 11, "2min": 23},
        "pairs": {"BTC-USDT": "BTC/USDT:USDT"},
    }
    engine = SignalEngine(config, model=None, scalers={})

    # Strong flat signal
    logits = torch.tensor([[
        -5.0, 10.0, -5.0,  # 30s: strong flat
        -5.0, 10.0, -5.0,
        -5.0, 10.0, -5.0,
    ]])

    decisions = engine.logits_to_decisions(logits)
    assert decisions["30s"].should_trade is False
    assert decisions["30s"].direction == "flat"
    print("PASS: test_flat_prediction_no_trade")


def test_position_tracking():
    """Position tracker prevents duplicate entries."""
    config = {
        "lob_levels": 5,
        "context_length": 20,
        "prediction_length": 4,
        "n_features": 36,
        "primary_horizon": "30s",
        "thresholds": {"30s": 0.48, "1min": 0.46, "2min": 0.46},
        "horizon_steps": {"30s": 5, "1min": 11, "2min": 23},
        "pairs": {"BTC-USDT": "BTC/USDT:USDT"},
    }
    engine = SignalEngine(config, model=None, scalers={})

    # No position → should allow entry
    assert engine.should_enter("BTC-USDT", "long") is True

    # Record position
    engine.record_position("BTC-USDT", "long")

    # Same direction → should NOT allow duplicate entry
    assert engine.should_enter("BTC-USDT", "long") is False

    # Opposite direction → should allow (signal reversal triggers exit first)
    assert engine.should_enter("BTC-USDT", "short") is True

    # Clear position
    engine.clear_position("BTC-USDT")
    assert engine.should_enter("BTC-USDT", "long") is True

    print("PASS: test_position_tracking")


if __name__ == "__main__":
    test_signal_engine_init()
    test_trade_decision_from_logits()
    test_low_confidence_no_trade()
    test_flat_prediction_no_trade()
    test_position_tracking()
    print("\nAll signal service tests passed!")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Volumes/Docker-SSD/projects/aaiwdbback/aai && .venv/bin/python executor/test_signal_service.py`
Expected: `ImportError: cannot import name 'SignalEngine' from 'executor.signal_service'`

- [ ] **Step 3: Implement signal_service.py**

Create `executor/signal_service.py`:

```python
#!/usr/bin/env python3
"""
AAI Signal Service — 5-second inference loop for LOB trading.

Queries TimescaleDB lob_5s, runs Phase 2 model inference with
confidence thresholding, and pushes trade signals to Freqtrade
via REST API.

Usage:
    python executor/signal_service.py
    python executor/signal_service.py --config executor/signal_config.json
"""

import asyncio
import json
import logging
import os
import pickle
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import asyncpg
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.model_v2 import CompoundAttentionModelV2
from training.features_v2 import engineer_features_v2
from training.dataset import LOBScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("signal_service")

# Column list for lob_5s query (must match training pipeline)
LOB_COLUMNS = ["bucket"]
for i in range(1, 41):
    LOB_COLUMNS.extend([
        f"bid_price_{i}", f"bid_volume_{i}",
        f"ask_price_{i}", f"ask_volume_{i}",
    ])
LOB_COLUMNS.extend(["mid_price", "spread"])


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    """Result of model inference for a single horizon."""
    horizon: str
    predicted_class: int  # 0=down, 1=flat, 2=up
    direction: str        # "long", "short", "flat"
    confidence: float
    should_trade: bool

    def __repr__(self):
        return f"TradeDecision({self.horizon}: {self.direction} conf={self.confidence:.3f} trade={self.should_trade})"


# ---------------------------------------------------------------------------
# Signal Engine — model inference + decision logic
# ---------------------------------------------------------------------------

class SignalEngine:
    """Runs model inference and produces trade decisions."""

    CLASS_TO_DIRECTION = {0: "short", 1: "flat", 2: "long"}

    def __init__(self, config: dict, model, scalers: dict):
        self.config = config
        self.model = model
        self.scalers = scalers
        self.primary_horizon = config["primary_horizon"]
        self.thresholds = config["thresholds"]
        self.n_levels = config["lob_levels"]
        self.context_length = config["context_length"]
        self.n_features = config["n_features"]

        # Position tracking: pair -> current direction ("long"/"short"/None)
        self.positions: dict[str, Optional[str]] = {}

    def logits_to_decisions(self, dir_logits: torch.Tensor) -> dict[str, TradeDecision]:
        """Convert model direction logits to trade decisions.

        Args:
            dir_logits: (1, 9) tensor — 3 horizons × 3 classes

        Returns:
            dict mapping horizon name to TradeDecision
        """
        logits = dir_logits.reshape(1, 3, 3)  # (1, horizons, classes)
        probs = F.softmax(logits, dim=2)       # (1, 3, 3)
        confidences, predictions = probs.max(dim=2)  # (1, 3) each

        decisions = {}
        for h_idx, h_name in enumerate(["30s", "1min", "2min"]):
            pred_class = predictions[0, h_idx].item()
            conf = confidences[0, h_idx].item()
            direction = self.CLASS_TO_DIRECTION[pred_class]
            threshold = self.thresholds.get(h_name, 0.5)

            should_trade = (
                direction != "flat"
                and conf >= threshold
            )

            decisions[h_name] = TradeDecision(
                horizon=h_name,
                predicted_class=pred_class,
                direction=direction,
                confidence=conf,
                should_trade=should_trade,
            )

        return decisions

    def should_enter(self, pair: str, direction: str) -> bool:
        """Check if we should enter a new position."""
        current = self.positions.get(pair)
        if current is None:
            return True
        if current == direction:
            return False  # already in same direction
        return True  # opposite direction = signal reversal

    def should_exit(self, pair: str, new_direction: str) -> bool:
        """Check if we should exit the current position."""
        current = self.positions.get(pair)
        if current is None:
            return False
        # Exit if signal reversed or went flat
        return current != new_direction

    def record_position(self, pair: str, direction: str):
        self.positions[pair] = direction

    def clear_position(self, pair: str):
        self.positions.pop(pair, None)

    def prepare_features(self, raw_rows: list[dict]) -> Optional[np.ndarray]:
        """Convert DB rows to model-ready feature tensor.

        Args:
            raw_rows: list of asyncpg Record objects from lob_5s

        Returns:
            (1, context_length, n_features) numpy array, or None if insufficient data
        """
        if len(raw_rows) < self.context_length:
            return None

        # Extract base features (162 columns for 40 levels)
        n_base = self.n_levels * 4 + 2
        data = np.zeros((len(raw_rows), n_base), dtype=np.float32)

        for i, row in enumerate(raw_rows):
            col_idx = 0
            for c in LOB_COLUMNS[1:]:  # skip 'bucket'
                val = row[c]
                data[i, col_idx] = float(val) if val is not None else 0.0
                col_idx += 1

        # Forward-fill any NaN/zero
        for col in range(n_base):
            for i in range(1, len(data)):
                if data[i, col] == 0.0 and data[i - 1, col] != 0.0:
                    data[i, col] = data[i - 1, col]

        # Feature engineering
        features, _ = engineer_features_v2(data, n_levels=self.n_levels)

        # Take last context_length rows
        features = features[-self.context_length:]

        if features.shape[0] < self.context_length:
            return None

        return features

    def run_inference(
        self, features: np.ndarray, scaler: LOBScaler, device: torch.device
    ) -> Optional[dict[str, TradeDecision]]:
        """Run model inference on prepared features.

        Args:
            features: (context_length, n_features_raw) numpy array
            scaler: LOBScaler for z-score normalization
            device: torch device

        Returns:
            dict of horizon -> TradeDecision, or None on error
        """
        try:
            scaled = scaler.transform(features)
            context = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)

            # Dummy exchange/symbol IDs (binance_perp=1)
            eid = torch.tensor([1], dtype=torch.long).to(device)
            sid = torch.tensor([0], dtype=torch.long).to(device)

            with torch.no_grad():
                pred, dir_logits = self.model(context, eid, sid)

            return self.logits_to_decisions(dir_logits.cpu())

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None


# ---------------------------------------------------------------------------
# Freqtrade API Client
# ---------------------------------------------------------------------------

class FreqtradeClient:
    """Async client for Freqtrade REST API."""

    def __init__(self, url: str, username: str, password: str):
        self.url = url.rstrip("/")
        self.username = username
        self.password = password
        self._session: Optional[aiohttp.ClientSession] = None
        self._token: Optional[str] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def _login(self):
        await self._ensure_session()
        async with self._session.post(
            f"{self.url}/api/v1/token/login",
            json={"username": self.username, "password": self.password},
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                self._token = data.get("access_token")
                logger.info("Freqtrade API login successful")
            else:
                logger.error(f"Freqtrade login failed: {resp.status}")

    async def _headers(self) -> dict:
        if self._token is None:
            await self._login()
        return {"Authorization": f"Bearer {self._token}"}

    async def _post(self, endpoint: str, payload: dict = None) -> Optional[dict]:
        await self._ensure_session()
        headers = await self._headers()
        try:
            async with self._session.post(
                f"{self.url}{endpoint}",
                json=payload or {},
                headers=headers,
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 401:
                    self._token = None  # force re-login
                    headers = await self._headers()
                    async with self._session.post(
                        f"{self.url}{endpoint}",
                        json=payload or {},
                        headers=headers,
                    ) as retry:
                        return await retry.json() if retry.status == 200 else None
                else:
                    text = await resp.text()
                    logger.warning(f"Freqtrade API {endpoint}: {resp.status} {text}")
                    return None
        except aiohttp.ClientError as e:
            logger.warning(f"Freqtrade API error: {e}")
            return None

    async def force_entry(self, pair: str, side: str, stake: float = None) -> Optional[dict]:
        """Force a trade entry. side = 'long' or 'short'."""
        payload = {"pair": pair, "side": side}
        if stake:
            payload["stakeamount"] = stake
        result = await self._post("/api/v1/forceentry", payload)
        if result:
            logger.info(f"Force {side} {pair}: {result.get('id', 'ok')}")
        return result

    async def force_exit(self, trade_id: int) -> Optional[dict]:
        """Force exit an open trade by ID."""
        result = await self._post("/api/v1/forceexit", {"tradeid": str(trade_id)})
        if result:
            logger.info(f"Force exit trade {trade_id}")
        return result

    async def get_trades(self) -> list:
        """Get open trades."""
        await self._ensure_session()
        headers = await self._headers()
        try:
            async with self._session.get(
                f"{self.url}/api/v1/status",
                headers=headers,
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
        except aiohttp.ClientError:
            return []

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

async def main():
    # Load config
    config_path = Path(__file__).parent / "signal_config.json"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    with open(config_path) as f:
        config = json.load(f)

    # Setup file logging
    file_handler = logging.FileHandler(
        Path(__file__).parent / config.get("log_file", "signal_service.log")
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model_path = Path(__file__).parent / config["model_checkpoint"]
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        return

    model = CompoundAttentionModelV2(
        n_levels=config["lob_levels"],
        n_features=config["n_features"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        d_model=66, n_heads=3, n_layers=3, d_ff=264, dropout=0.2,
    ).to(device)

    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Model loaded from {model_path} (epoch {checkpoint.get('epoch', '?')})")

    # Load scalers
    scaler_path = Path(__file__).parent / config["scaler_path"]
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
        logger.info(f"Loaded {len(scalers)} scalers")
    else:
        logger.error(f"Scalers not found: {scaler_path}")
        return

    # Initialize engine
    engine = SignalEngine(config, model, scalers)

    # Connect to DB
    pool = await asyncpg.create_pool(
        host=config["db_host"],
        port=config["db_port"],
        user=config["db_user"],
        password=config["db_password"],
        database=config["db_name"],
        min_size=1,
        max_size=3,
    )
    logger.info("Database connection pool established")

    # Connect to Freqtrade API
    ft_config = config["freqtrade_api"]
    ft_client = FreqtradeClient(ft_config["url"], ft_config["username"], ft_config["password"])

    # Build column select string
    col_select = ", ".join(LOB_COLUMNS)

    # Graceful shutdown
    running = True
    def handle_signal(signum, frame):
        nonlocal running
        running = False
        logger.info("Shutdown signal received")
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Signal service started — entering main loop")
    tick_interval = config.get("tick_interval_seconds", 5)
    stale_threshold = config.get("stale_data_threshold_seconds", 30)

    while running:
        tick_start = time.time()

        for pair_db, pair_ft in config["pairs"].items():
            try:
                # Query lob_5s
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        f"SELECT {col_select} FROM lob_5s "
                        f"WHERE exchange = 'binance_perp' AND symbol = $1 "
                        f"AND bucket > now() - interval '12 minutes' "
                        f"ORDER BY bucket",
                        pair_db,
                    )

                if not rows:
                    logger.warning(f"{pair_db}: no data from lob_5s")
                    continue

                # Check for stale data
                last_ts = rows[-1]["bucket"]
                age = (datetime.now(timezone.utc) - last_ts).total_seconds()
                if age > stale_threshold:
                    logger.warning(f"{pair_db}: stale data ({age:.0f}s old)")
                    continue

                # Prepare features
                features = engine.prepare_features(rows)
                if features is None:
                    logger.debug(f"{pair_db}: insufficient data ({len(rows)} rows)")
                    continue

                # Get scaler for this pair
                scaler_key = f"binance_perp/{pair_db}"
                scaler = scalers.get(scaler_key)
                if scaler is None:
                    # Fallback: try first available scaler
                    scaler = next(iter(scalers.values()))

                # Run inference
                decisions = engine.run_inference(features, scaler, device)
                if decisions is None:
                    continue

                # Primary horizon decision
                primary = decisions[engine.primary_horizon]
                logger.info(
                    f"{pair_db}: {primary.direction} "
                    f"conf={primary.confidence:.3f} "
                    f"trade={primary.should_trade}"
                )

                # Check if we need to exit current position
                if engine.should_exit(pair_db, primary.direction):
                    trades = await ft_client.get_trades()
                    for t in trades:
                        if t.get("pair") == pair_ft:
                            await ft_client.force_exit(t["trade_id"])
                            engine.clear_position(pair_db)
                            logger.info(f"{pair_db}: exited position (signal reversal)")

                # Check if we should enter
                if primary.should_trade and engine.should_enter(pair_db, primary.direction):
                    result = await ft_client.force_entry(pair_ft, primary.direction)
                    if result:
                        engine.record_position(pair_db, primary.direction)

            except Exception as e:
                logger.error(f"{pair_db}: tick error: {e}", exc_info=True)

        # Sleep until next tick
        elapsed = time.time() - tick_start
        sleep_time = max(0, tick_interval - elapsed)
        if running:
            await asyncio.sleep(sleep_time)

    # Cleanup
    await ft_client.close()
    await pool.close()
    logger.info("Signal service stopped")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run tests**

Run: `cd /Volumes/Docker-SSD/projects/aaiwdbback/aai && .venv/bin/python executor/test_signal_service.py`
Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai
git add executor/signal_service.py executor/test_signal_service.py
git commit -m "feat: signal service — 5s inference loop with Freqtrade REST API client"
```

---

### Task 4: Launcher Script + Model Checkpoint Setup

**Files:**
- Create: `executor/run.sh`

- [ ] **Step 1: Download Phase 2 checkpoint from Drive to local**

The Phase 2 results zip (`aai_30day_phase2_results.zip`) on Google Drive contains `checkpoints/best.pt` and `checkpoints/scalers.pkl`. Download and extract to `executor/`:

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai/executor
# Manual: copy aai_30day_phase2_results.zip from Google Drive to executor/
unzip aai_30day_phase2_results.zip -d .
ls checkpoints/best.pt checkpoints/scalers.pkl
```

- [ ] **Step 2: Create executor/run.sh**

```bash
#!/bin/bash
# AAI Trading Executor Launcher
# Starts Freqtrade (position manager) and Signal Service (5s inference)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# Check prerequisites
if ! command -v freqtrade &> /dev/null; then
    echo "ERROR: freqtrade not found. Install: cd executor/freqtrade && pip install -e '.[all]'"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/checkpoints/best.pt" ]; then
    echo "ERROR: Model checkpoint not found at executor/checkpoints/best.pt"
    echo "Download aai_30day_phase2_results.zip from Google Drive and extract to executor/"
    exit 1
fi

echo "Starting AAI Trading Executor..."
echo "  Mode: $(python3 -c "import json; print(json.load(open('$SCRIPT_DIR/config.json')).get('dry_run', True) and 'PAPER' or 'LIVE')")"

# Start Freqtrade in background
echo "Starting Freqtrade..."
cd "$SCRIPT_DIR"
freqtrade trade \
    --config config.json \
    --strategy AAIStrategy \
    --strategy-path strategies/ \
    --db-url sqlite:///tradesv3.sqlite \
    &
FT_PID=$!
echo "  Freqtrade PID: $FT_PID"

# Wait for API server to be ready
echo "Waiting for Freqtrade API..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8080/api/v1/ping > /dev/null 2>&1; then
        echo "  Freqtrade API ready"
        break
    fi
    sleep 1
done

# Start signal service in foreground
echo "Starting Signal Service..."
cd "$PROJECT_DIR"
python3 -m executor.signal_service &
SS_PID=$!
echo "  Signal Service PID: $SS_PID"

# Wait for either process to exit
echo ""
echo "Both processes running. Press Ctrl+C to stop."

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $SS_PID 2>/dev/null || true
    kill $FT_PID 2>/dev/null || true
    wait $SS_PID 2>/dev/null || true
    wait $FT_PID 2>/dev/null || true
    echo "Done."
}

trap cleanup SIGINT SIGTERM
wait -n $FT_PID $SS_PID 2>/dev/null
cleanup
```

- [ ] **Step 3: Make launcher executable**

```bash
chmod +x /Volumes/Docker-SSD/projects/aaiwdbback/aai/executor/run.sh
```

- [ ] **Step 4: Create executor/__init__.py for module imports**

```bash
touch /Volumes/Docker-SSD/projects/aaiwdbback/aai/executor/__init__.py
```

- [ ] **Step 5: Commit**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai
git add executor/run.sh executor/__init__.py executor/config.json executor/signal_config.json executor/strategies/AAIStrategy.py executor/.env.example
git commit -m "feat: trading executor — Freqtrade config, strategy, launcher"
```

---

### Task 5: Integration Test (End-to-End)

- [ ] **Step 1: Install Freqtrade if not already done**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai/executor
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
git checkout stable
pip install -e '.[all]'
cd ..
```

- [ ] **Step 2: Start Freqtrade alone and verify API**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai/executor
freqtrade trade --config config.json --strategy AAIStrategy --strategy-path strategies/ &
sleep 10
curl -X POST http://127.0.0.1:8080/api/v1/token/login \
  -H "Content-Type: application/json" \
  -d '{"username":"aai","password":"aai_paper_trade"}'
```

Expected: JSON response with `access_token`.

- [ ] **Step 3: Run signal service tests**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai
.venv/bin/python executor/test_signal_service.py
```

Expected: All 5 tests pass.

- [ ] **Step 4: Run signal service for 1 minute, check logs**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai
timeout 60 .venv/bin/python -m executor.signal_service || true
cat executor/signal_service.log | tail -20
```

Expected: Log lines showing DB queries, inference results, and confidence levels for BTC-USDT and ETH-USDT every 5 seconds.

- [ ] **Step 5: Kill Freqtrade background process**

```bash
kill %1 2>/dev/null || pkill -f "freqtrade trade"
```
