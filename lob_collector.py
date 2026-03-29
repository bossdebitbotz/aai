#!/usr/bin/env python3
"""
LOB Data Collector — Multi-Exchange Limit Order Book collector with TimescaleDB.

Collects N-level LOB data (default 40) from Binance Spot, Binance Perp,
Bybit Spot, and KuCoin Spot. Properly manages order book state with delta
application, validates data inline, and writes to TimescaleDB via asyncpg
with batched inserts.
"""

import asyncio
import json
import os
import signal
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

import asyncpg
import websockets
import aiohttp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "lob_user"
    db_password: str = "lob_password"
    db_name: str = "lob_data"
    lob_levels: int = 40
    batch_interval: float = 1.0       # seconds between batch flushes
    health_check_interval: float = 30  # seconds between health checks
    log_level: str = "INFO"
    trading_pairs: list = field(default_factory=lambda: [
        "BTC-USDT", "ETH-USDT", "SOL-USDT", "WLD-USDT"
    ])
    exchanges: list = field(default_factory=lambda: [
        "binance_spot", "binance_perp", "bybit_spot", "kucoin_spot"
    ])

    @classmethod
    def from_env(cls):
        return cls(
            db_host=os.getenv("DB_HOST", "localhost"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            db_user=os.getenv("DB_USER", "lob_user"),
            db_password=os.getenv("DB_PASSWORD", "lob_password"),
            db_name=os.getenv("DB_NAME", "lob_data"),
            lob_levels=int(os.getenv("LOB_LEVELS", "40")),
            batch_interval=float(os.getenv("BATCH_INTERVAL", "1.0")),
            health_check_interval=float(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            trading_pairs=os.getenv("TRADING_PAIRS", "BTC-USDT,ETH-USDT,SOL-USDT,WLD-USDT").split(","),
            exchanges=os.getenv("EXCHANGES", "binance_spot,binance_perp,bybit_spot,kucoin_spot").split(","),
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("lob_collector")


# ---------------------------------------------------------------------------
# WebSocket URLs
# ---------------------------------------------------------------------------

BINANCE_SPOT_WS = "wss://stream.binance.com:9443/ws"
BINANCE_PERP_WS = "wss://fstream.binance.com/ws"
BYBIT_SPOT_WS   = "wss://stream.bybit.com/v5/public/spot"


# ---------------------------------------------------------------------------
# OrderBook — maintains correct state with delta application
# ---------------------------------------------------------------------------

class OrderBook:
    """Maintains a single order book with proper delta/snapshot handling."""

    def __init__(self, levels: int = 5):
        self.levels = levels
        # Full book state: price -> volume (sorted dicts)
        self._bids: dict[float, float] = {}  # descending by price
        self._asks: dict[float, float] = {}  # ascending by price
        self.last_update_id: int = 0
        self.last_update_time: float = 0
        self._initialized = False
        # Buffer for Binance diff updates received before snapshot
        self._pending_updates: list = []

    @property
    def initialized(self) -> bool:
        return self._initialized

    def apply_snapshot(self, bids: list, asks: list, last_update_id: int = 0):
        """Replace entire book state from a REST snapshot."""
        self._bids = {}
        self._asks = {}
        for price_str, vol_str in bids:
            p, v = float(price_str), float(vol_str)
            if v > 0:
                self._bids[p] = v
        for price_str, vol_str in asks:
            p, v = float(price_str), float(vol_str)
            if v > 0:
                self._asks[p] = v
        self.last_update_id = last_update_id
        self._initialized = True
        self.last_update_time = time.time()

    def apply_delta(self, bid_updates: list, ask_updates: list,
                    first_update_id: int = 0, final_update_id: int = 0):
        """Apply incremental updates to existing book state.

        For Binance: only apply if first_update_id <= last_update_id + 1 <= final_update_id
        For Bybit: always apply (set first/final_update_id = 0 to skip the check)
        """
        # Binance sequence validation
        if first_update_id > 0 and final_update_id > 0:
            if final_update_id <= self.last_update_id:
                return  # Already applied
            if first_update_id > self.last_update_id + 1:
                pass  # Normal for Binance @depth batched stream

        for price_str, vol_str in bid_updates:
            p, v = float(price_str), float(vol_str)
            if v == 0:
                self._bids.pop(p, None)
            else:
                self._bids[p] = v

        for price_str, vol_str in ask_updates:
            p, v = float(price_str), float(vol_str)
            if v == 0:
                self._asks.pop(p, None)
            else:
                self._asks[p] = v

        if final_update_id > 0:
            self.last_update_id = final_update_id
        self.last_update_time = time.time()

    def get_top_levels(self, n: int = 5) -> dict:
        """Return top N bid/ask levels as flat dict for DB insertion."""
        sorted_bids = sorted(self._bids.items(), reverse=True)[:n]
        sorted_asks = sorted(self._asks.items())[:n]

        result = {}
        for i in range(n):
            if i < len(sorted_bids):
                result[f"bid_price_{i+1}"] = sorted_bids[i][0]
                result[f"bid_volume_{i+1}"] = sorted_bids[i][1]
            else:
                result[f"bid_price_{i+1}"] = 0.0
                result[f"bid_volume_{i+1}"] = 0.0

            if i < len(sorted_asks):
                result[f"ask_price_{i+1}"] = sorted_asks[i][0]
                result[f"ask_volume_{i+1}"] = sorted_asks[i][1]
            else:
                result[f"ask_price_{i+1}"] = 0.0
                result[f"ask_volume_{i+1}"] = 0.0

        # Derived metrics
        bp1 = result["bid_price_1"]
        ap1 = result["ask_price_1"]
        if bp1 > 0 and ap1 > 0:
            result["mid_price"] = (bp1 + ap1) / 2
            result["spread"] = ap1 - bp1
        else:
            result["mid_price"] = 0.0
            result["spread"] = 0.0

        # Volume imbalance across all returned levels
        total_bid_vol = sum(result[f"bid_volume_{i+1}"] for i in range(n))
        total_ask_vol = sum(result[f"ask_volume_{i+1}"] for i in range(n))
        denom = total_bid_vol + total_ask_vol
        result["volume_imbalance"] = (total_bid_vol - total_ask_vol) / denom if denom > 0 else 0.0

        return result

    def validate(self) -> tuple[bool, list[str]]:
        """Validate current book state. Returns (is_valid, list_of_issues)."""
        issues = []
        top = self.get_top_levels(self.levels)

        bp1 = top["bid_price_1"]
        ap1 = top["ask_price_1"]

        if bp1 <= 0 or ap1 <= 0:
            issues.append("missing_top_level")
        elif bp1 >= ap1:
            issues.append("crossed_book")

        # Check for zero prices in populated levels
        for i in range(1, self.levels + 1):
            bp = top[f"bid_price_{i}"]
            ap = top[f"ask_price_{i}"]
            if (bp > 0 and top[f"bid_volume_{i}"] <= 0):
                issues.append(f"zero_volume_bid_{i}")
            if (ap > 0 and top[f"ask_volume_{i}"] <= 0):
                issues.append(f"zero_volume_ask_{i}")

        # Check level ordering (bids should be descending, asks ascending)
        for i in range(1, self.levels):
            bp_cur = top[f"bid_price_{i}"]
            bp_next = top[f"bid_price_{i+1}"]
            if bp_cur > 0 and bp_next > 0 and bp_cur < bp_next:
                issues.append("bid_level_inversion")
                break

        for i in range(1, self.levels):
            ap_cur = top[f"ask_price_{i}"]
            ap_next = top[f"ask_price_{i+1}"]
            if ap_cur > 0 and ap_next > 0 and ap_cur > ap_next:
                issues.append("ask_level_inversion")
                break

        return (len(issues) == 0, issues)


# ---------------------------------------------------------------------------
# DBWriter — asyncpg pool + batched inserts
# ---------------------------------------------------------------------------

class DBWriter:
    """Manages asyncpg connection pool and batched writes to TimescaleDB."""

    QUALITY_LOG_SQL = """
        INSERT INTO data_quality_log (time, exchange, symbol, issue_type, details)
        VALUES ($1, $2, $3, $4, $5)
    """

    @staticmethod
    def _build_insert_sql(n_levels: int) -> str:
        """Build INSERT SQL dynamically for N LOB levels."""
        level_cols = []
        for i in range(1, n_levels + 1):
            level_cols.extend([f"bid_price_{i}", f"bid_volume_{i}"])
        for i in range(1, n_levels + 1):
            level_cols.extend([f"ask_price_{i}", f"ask_volume_{i}"])
        all_cols = ["time", "exchange", "symbol"] + level_cols + [
            "mid_price", "spread", "volume_imbalance",
            "is_valid", "validation_flags",
        ]
        col_names = ", ".join(all_cols)
        placeholders = ", ".join(f"${i+1}" for i in range(len(all_cols)))
        return f"INSERT INTO lob_snapshots ({col_names}) VALUES ({placeholders})"

    def __init__(self, config: Config):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self._buffer: list[tuple] = []
        self._quality_buffer: list[tuple] = []
        self._lock = asyncio.Lock()
        self.INSERT_SQL = self._build_insert_sql(config.lob_levels)

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            host=self.config.db_host,
            port=self.config.db_port,
            user=self.config.db_user,
            password=self.config.db_password,
            database=self.config.db_name,
            min_size=2,
            max_size=10,
        )
        logger.info("Database connection pool established")

    async def close(self):
        if self.pool:
            await self.flush()
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def enqueue(self, exchange: str, symbol: str, ts: datetime,
                      top_levels: dict, is_valid: bool, issues: list[str]):
        """Add a snapshot to the write buffer."""
        n = self.config.lob_levels
        row_parts = [ts, exchange, symbol]
        for i in range(1, n + 1):
            row_parts.append(top_levels[f"bid_price_{i}"])
            row_parts.append(top_levels[f"bid_volume_{i}"])
        for i in range(1, n + 1):
            row_parts.append(top_levels[f"ask_price_{i}"])
            row_parts.append(top_levels[f"ask_volume_{i}"])
        row_parts.extend([
            top_levels["mid_price"], top_levels["spread"],
            top_levels["volume_imbalance"],
            is_valid, issues,
        ])
        row = tuple(row_parts)
        async with self._lock:
            self._buffer.append(row)

        # Log quality issues
        if issues:
            now = datetime.now(timezone.utc)
            for issue in issues:
                async with self._lock:
                    self._quality_buffer.append(
                        (now, exchange, symbol, issue, None)
                    )

    async def flush(self):
        """Flush buffered rows to the database in a single batch."""
        async with self._lock:
            snapshot_rows = self._buffer[:]
            quality_rows = self._quality_buffer[:]
            self._buffer.clear()
            self._quality_buffer.clear()

        if not snapshot_rows and not quality_rows:
            return

        try:
            async with self.pool.acquire() as conn:
                if snapshot_rows:
                    await conn.executemany(self.INSERT_SQL, snapshot_rows)
                if quality_rows:
                    await conn.executemany(self.QUALITY_LOG_SQL, quality_rows)
            if snapshot_rows:
                logger.debug(f"Flushed {len(snapshot_rows)} snapshots, {len(quality_rows)} quality logs")
        except Exception as e:
            logger.error(f"DB flush error: {e}")
            # Re-enqueue failed rows
            async with self._lock:
                self._buffer = snapshot_rows + self._buffer
                self._quality_buffer = quality_rows + self._quality_buffer

    async def flush_loop(self, stop_event: asyncio.Event):
        """Periodically flush the buffer."""
        while not stop_event.is_set():
            await asyncio.sleep(self.config.batch_interval)
            await self.flush()
        # Final flush
        await self.flush()

    async def log_quality_issue(self, exchange: str, symbol: str,
                                issue_type: str, details: str = None):
        """Directly log a quality issue (e.g., gap, stale, reconnect)."""
        async with self._lock:
            self._quality_buffer.append(
                (datetime.now(timezone.utc), exchange, symbol, issue_type, details)
            )


# ---------------------------------------------------------------------------
# Helper: symbol formatting
# ---------------------------------------------------------------------------

def to_exchange_symbol(pair: str) -> str:
    """Convert 'BTC-USDT' to 'BTCUSDT' for exchange APIs."""
    return pair.replace("-", "")


# ---------------------------------------------------------------------------
# WebSocket managers — one per exchange, infinite retry with exp backoff
# ---------------------------------------------------------------------------

class BinanceSpotWS:
    def __init__(self, config: Config, db: DBWriter, books: dict[str, OrderBook],
                 stop_event: asyncio.Event):
        self.config = config
        self.db = db
        self.books = books
        self.stop = stop_event
        self.exchange = "binance_spot"
        self.snapshot_counts: dict[str, int] = defaultdict(int)

    async def _fetch_snapshot(self, session: aiohttp.ClientSession, pair: str) -> Optional[dict]:
        symbol = to_exchange_symbol(pair)
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=1000"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error(f"[{self.exchange}] Snapshot HTTP {resp.status} for {pair}")
        except Exception as e:
            logger.error(f"[{self.exchange}] Snapshot fetch error for {pair}: {e}")
        return None

    async def _process_update(self, pair: str, data: dict):
        book = self.books[f"{self.exchange}_{pair}"]
        ts = datetime.fromtimestamp(data.get("E", time.time() * 1000) / 1000, tz=timezone.utc)

        if not book.initialized:
            # Buffer the update for later
            book._pending_updates.append(data)
            return

        book.apply_delta(
            bid_updates=data.get("b", []),
            ask_updates=data.get("a", []),
            first_update_id=data.get("U", 0),
            final_update_id=data.get("u", 0),
        )

        top = book.get_top_levels(self.config.lob_levels)
        is_valid, issues = book.validate()
        await self.db.enqueue(self.exchange, pair, ts, top, is_valid, issues)
        self.snapshot_counts[pair] += 1

    async def _sync_book(self, http_session: aiohttp.ClientSession, pair: str):
        """Binance diff depth sync: fetch REST snapshot, replay buffered WS updates."""
        key = f"{self.exchange}_{pair}"
        book = self.books[key]
        snap = await self._fetch_snapshot(http_session, pair)
        if not snap:
            return
        snapshot_uid = snap.get("lastUpdateId", 0)
        book.apply_snapshot(snap.get("bids", []), snap.get("asks", []), snapshot_uid)
        # Replay buffered updates: drop where u <= lastUpdateId,
        # first applied must have U <= lastUpdateId+1 <= u
        applied = False
        for pending in book._pending_updates:
            u = pending.get("u", 0)
            U = pending.get("U", 0)
            if u <= snapshot_uid:
                continue  # Already reflected in snapshot
            if not applied:
                if U > snapshot_uid + 1:
                    # Gap between snapshot and first WS update — re-buffer
                    logger.warning(f"[{self.exchange}] {pair} gap after snapshot, will retry")
                    book._initialized = False
                    book._pending_updates.clear()
                    return
                applied = True
            book.apply_delta(pending.get("b", []), pending.get("a", []), U, u)
        book._pending_updates.clear()
        logger.info(f"[{self.exchange}] Snapshot synced for {pair} (lastUpdateId={snapshot_uid})")

    async def run(self):
        backoff = 1
        async with aiohttp.ClientSession() as http_session:
            while not self.stop.is_set():
                try:
                    # Initialize book objects
                    for pair in self.config.trading_pairs:
                        key = f"{self.exchange}_{pair}"
                        self.books[key] = OrderBook(self.config.lob_levels)

                    # Step 1: Connect WebSocket FIRST and start buffering
                    streams = [f"{to_exchange_symbol(p).lower()}@depth" for p in self.config.trading_pairs]
                    payload = {"method": "SUBSCRIBE", "params": streams, "id": 1}

                    logger.info(f"[{self.exchange}] Connecting WebSocket...")
                    async with websockets.connect(BINANCE_SPOT_WS, ping_interval=20, ping_timeout=10) as ws:
                        await ws.send(json.dumps(payload))
                        backoff = 1
                        logger.info(f"[{self.exchange}] Connected, buffering updates...")

                        # Buffer a few updates before fetching snapshots
                        buffer_deadline = time.time() + 2
                        while time.time() < buffer_deadline:
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=1)
                                data = json.loads(msg)
                                if "result" in data:
                                    continue
                                if data.get("e") == "depthUpdate" and "s" in data:
                                    raw_sym = data["s"]
                                    pair = next(
                                        (p for p in self.config.trading_pairs
                                         if to_exchange_symbol(p) == raw_sym), None)
                                    if pair:
                                        self.books[f"{self.exchange}_{pair}"]._pending_updates.append(data)
                            except asyncio.TimeoutError:
                                continue

                        # Step 2: Fetch REST snapshots and replay buffered updates
                        for pair in self.config.trading_pairs:
                            await self._sync_book(http_session, pair)

                        logger.info(f"[{self.exchange}] All pairs synced, streaming")

                        # Step 3: Process live updates
                        while not self.stop.is_set():
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                                data = json.loads(msg)
                                if "result" in data:
                                    continue
                                if data.get("e") == "depthUpdate" and "s" in data:
                                    raw_sym = data["s"]
                                    pair = next(
                                        (p for p in self.config.trading_pairs
                                         if to_exchange_symbol(p) == raw_sym), None)
                                    if pair:
                                        await self._process_update(pair, data)
                            except asyncio.TimeoutError:
                                continue
                            except websockets.ConnectionClosed:
                                logger.warning(f"[{self.exchange}] Connection closed, reconnecting")
                                break

                except Exception as e:
                    logger.error(f"[{self.exchange}] Error: {e}")

                if not self.stop.is_set():
                    for pair in self.config.trading_pairs:
                        await self.db.log_quality_issue(self.exchange, pair, "reconnect",
                                                        f"Reconnecting after error, backoff={backoff}s")
                    await asyncio.sleep(min(backoff, 60))
                    backoff = min(backoff * 2, 60)


class BinancePerpWS:
    def __init__(self, config: Config, db: DBWriter, books: dict[str, OrderBook],
                 stop_event: asyncio.Event):
        self.config = config
        self.db = db
        self.books = books
        self.stop = stop_event
        self.exchange = "binance_perp"
        self.snapshot_counts: dict[str, int] = defaultdict(int)

    async def _fetch_snapshot(self, session: aiohttp.ClientSession, pair: str) -> Optional[dict]:
        symbol = to_exchange_symbol(pair)
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit=1000"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error(f"[{self.exchange}] Snapshot HTTP {resp.status} for {pair}")
        except Exception as e:
            logger.error(f"[{self.exchange}] Snapshot fetch error for {pair}: {e}")
        return None

    async def _process_update(self, pair: str, data: dict):
        book = self.books[f"{self.exchange}_{pair}"]
        ts = datetime.fromtimestamp(data.get("E", time.time() * 1000) / 1000, tz=timezone.utc)

        if not book.initialized:
            book._pending_updates.append(data)
            return

        book.apply_delta(
            bid_updates=data.get("b", []),
            ask_updates=data.get("a", []),
            first_update_id=data.get("U", 0),
            final_update_id=data.get("u", 0),
        )

        if not book.initialized:
            await self.db.log_quality_issue(self.exchange, pair, "gap",
                                            "Sequence gap detected, re-syncing")
            return

        top = book.get_top_levels(self.config.lob_levels)
        is_valid, issues = book.validate()
        await self.db.enqueue(self.exchange, pair, ts, top, is_valid, issues)
        self.snapshot_counts[pair] += 1

    async def _sync_book(self, http_session: aiohttp.ClientSession, pair: str):
        """Binance diff depth sync: fetch REST snapshot, replay buffered WS updates."""
        key = f"{self.exchange}_{pair}"
        book = self.books[key]
        snap = await self._fetch_snapshot(http_session, pair)
        if not snap:
            return
        snapshot_uid = snap.get("lastUpdateId", 0)
        book.apply_snapshot(snap.get("bids", []), snap.get("asks", []), snapshot_uid)
        applied = False
        for pending in book._pending_updates:
            u = pending.get("u", 0)
            U = pending.get("U", 0)
            if u <= snapshot_uid:
                continue
            if not applied:
                if U > snapshot_uid + 1:
                    logger.warning(f"[{self.exchange}] {pair} gap after snapshot, will retry")
                    book._initialized = False
                    book._pending_updates.clear()
                    return
                applied = True
            book.apply_delta(pending.get("b", []), pending.get("a", []), U, u)
        book._pending_updates.clear()
        logger.info(f"[{self.exchange}] Snapshot synced for {pair} (lastUpdateId={snapshot_uid})")

    async def run(self):
        backoff = 1
        async with aiohttp.ClientSession() as http_session:
            while not self.stop.is_set():
                try:
                    for pair in self.config.trading_pairs:
                        key = f"{self.exchange}_{pair}"
                        self.books[key] = OrderBook(self.config.lob_levels)

                    streams = [f"{to_exchange_symbol(p).lower()}@depth" for p in self.config.trading_pairs]
                    payload = {"method": "SUBSCRIBE", "params": streams, "id": 1}

                    logger.info(f"[{self.exchange}] Connecting WebSocket...")
                    async with websockets.connect(BINANCE_PERP_WS, ping_interval=20, ping_timeout=10) as ws:
                        await ws.send(json.dumps(payload))
                        backoff = 1
                        logger.info(f"[{self.exchange}] Connected, buffering updates...")

                        # Buffer a few updates before fetching snapshots
                        buffer_deadline = time.time() + 2
                        while time.time() < buffer_deadline:
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=1)
                                data = json.loads(msg)
                                if "result" in data:
                                    continue
                                if data.get("e") == "depthUpdate" and "s" in data:
                                    raw_sym = data["s"]
                                    pair = next(
                                        (p for p in self.config.trading_pairs
                                         if to_exchange_symbol(p) == raw_sym), None)
                                    if pair:
                                        self.books[f"{self.exchange}_{pair}"]._pending_updates.append(data)
                            except asyncio.TimeoutError:
                                continue

                        for pair in self.config.trading_pairs:
                            await self._sync_book(http_session, pair)

                        logger.info(f"[{self.exchange}] All pairs synced, streaming")

                        while not self.stop.is_set():
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                                data = json.loads(msg)
                                if "result" in data:
                                    continue
                                if data.get("e") == "depthUpdate" and "s" in data:
                                    raw_sym = data["s"]
                                    pair = next(
                                        (p for p in self.config.trading_pairs
                                         if to_exchange_symbol(p) == raw_sym), None)
                                    if pair:
                                        book = self.books[f"{self.exchange}_{pair}"]
                                        if not book.initialized:
                                            book._pending_updates.append(data)
                                            if len(book._pending_updates) >= 5:
                                                await self._sync_book(http_session, pair)
                                        else:
                                            await self._process_update(pair, data)
                            except asyncio.TimeoutError:
                                continue
                            except websockets.ConnectionClosed:
                                logger.warning(f"[{self.exchange}] Connection closed, reconnecting")
                                break

                except Exception as e:
                    logger.error(f"[{self.exchange}] Error: {e}")

                if not self.stop.is_set():
                    for pair in self.config.trading_pairs:
                        await self.db.log_quality_issue(self.exchange, pair, "reconnect",
                                                        f"Reconnecting, backoff={backoff}s")
                    await asyncio.sleep(min(backoff, 60))
                    backoff = min(backoff * 2, 60)


class BybitSpotWS:
    def __init__(self, config: Config, db: DBWriter, books: dict[str, OrderBook],
                 stop_event: asyncio.Event):
        self.config = config
        self.db = db
        self.books = books
        self.stop = stop_event
        self.exchange = "bybit_spot"
        self.snapshot_counts: dict[str, int] = defaultdict(int)

    async def _fetch_snapshot(self, session: aiohttp.ClientSession, pair: str) -> Optional[dict]:
        symbol = to_exchange_symbol(pair)
        url = f"https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}&limit=200"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("retCode") == 0 and "result" in data:
                        return data["result"]
                    logger.error(f"[{self.exchange}] Unexpected response: {data.get('retCode')}")
                else:
                    logger.error(f"[{self.exchange}] Snapshot HTTP {resp.status} for {pair}")
        except Exception as e:
            logger.error(f"[{self.exchange}] Snapshot fetch error for {pair}: {e}")
        return None

    async def _process_message(self, pair: str, msg_type: str, data: dict):
        book = self.books[f"{self.exchange}_{pair}"]
        ts = datetime.fromtimestamp(int(data.get("ts", time.time() * 1000)) / 1000, tz=timezone.utc)

        if msg_type == "snapshot":
            book.apply_snapshot(
                data.get("b", []),
                data.get("a", []),
                int(data.get("u", 0)),
            )
        elif msg_type == "delta":
            if not book.initialized:
                return
            book.apply_delta(
                bid_updates=data.get("b", []),
                ask_updates=data.get("a", []),
            )

        if not book.initialized:
            return

        top = book.get_top_levels(self.config.lob_levels)
        is_valid, issues = book.validate()
        await self.db.enqueue(self.exchange, pair, ts, top, is_valid, issues)
        self.snapshot_counts[pair] += 1

    async def run(self):
        backoff = 1
        async with aiohttp.ClientSession() as http_session:
            while not self.stop.is_set():
                try:
                    for pair in self.config.trading_pairs:
                        key = f"{self.exchange}_{pair}"
                        if key not in self.books:
                            self.books[key] = OrderBook(self.config.lob_levels)

                        snap = await self._fetch_snapshot(http_session, pair)
                        if snap:
                            self.books[key].apply_snapshot(
                                snap.get("b", []),
                                snap.get("a", []),
                                int(snap.get("u", 0)),
                            )
                            logger.info(f"[{self.exchange}] Snapshot loaded for {pair}")

                    args = [f"orderbook.200.{to_exchange_symbol(p)}" for p in self.config.trading_pairs]
                    payload = {"op": "subscribe", "args": args}

                    logger.info(f"[{self.exchange}] Connecting WebSocket...")
                    async with websockets.connect(BYBIT_SPOT_WS, ping_interval=20, ping_timeout=10) as ws:
                        await ws.send(json.dumps(payload))
                        backoff = 1
                        logger.info(f"[{self.exchange}] Connected, subscribed to {len(args)} streams")

                        # Bybit requires heartbeat
                        last_ping = time.time()

                        while not self.stop.is_set():
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                                data = json.loads(msg)

                                # Subscription confirmation
                                if data.get("op") == "subscribe":
                                    if data.get("success"):
                                        logger.info(f"[{self.exchange}] Subscription confirmed")
                                    else:
                                        logger.error(f"[{self.exchange}] Subscription failed: {data}")
                                    continue

                                # Pong response
                                if data.get("op") == "pong":
                                    continue

                                # Orderbook data
                                topic = data.get("topic", "")
                                if topic.startswith("orderbook."):
                                    parts = topic.split(".")
                                    if len(parts) == 3:
                                        raw_sym = parts[2]
                                        pair = next(
                                            (p for p in self.config.trading_pairs
                                             if to_exchange_symbol(p) == raw_sym),
                                            None,
                                        )
                                        if pair and "data" in data:
                                            await self._process_message(
                                                pair, data.get("type", ""), data["data"]
                                            )

                                # Send heartbeat every 20 seconds
                                if time.time() - last_ping > 20:
                                    await ws.send(json.dumps({"op": "ping"}))
                                    last_ping = time.time()

                            except asyncio.TimeoutError:
                                # Send heartbeat on timeout too
                                await ws.send(json.dumps({"op": "ping"}))
                                last_ping = time.time()
                            except websockets.ConnectionClosed:
                                logger.warning(f"[{self.exchange}] Connection closed, reconnecting")
                                break

                except Exception as e:
                    logger.error(f"[{self.exchange}] Error: {e}")

                if not self.stop.is_set():
                    for pair in self.config.trading_pairs:
                        await self.db.log_quality_issue(self.exchange, pair, "reconnect",
                                                        f"Reconnecting, backoff={backoff}s")
                    await asyncio.sleep(min(backoff, 60))
                    backoff = min(backoff * 2, 60)


# ---------------------------------------------------------------------------
# KuCoin Spot — uses level2Depth50 (full 50-level snapshots every 100ms)
# ---------------------------------------------------------------------------

class KuCoinSpotWS:
    def __init__(self, config: Config, db: DBWriter, books: dict[str, OrderBook],
                 stop_event: asyncio.Event):
        self.config = config
        self.db = db
        self.books = books
        self.stop = stop_event
        self.exchange = "kucoin_spot"
        self.snapshot_counts: dict[str, int] = defaultdict(int)

    async def _get_ws_token(self, session: aiohttp.ClientSession) -> Optional[dict]:
        """Get WebSocket connection token from KuCoin REST API."""
        try:
            async with session.post("https://api.kucoin.com/api/v1/bullet-public") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("code") == "200000" and "data" in data:
                        return data["data"]
                logger.error(f"[{self.exchange}] Token request failed: HTTP {resp.status}")
        except Exception as e:
            logger.error(f"[{self.exchange}] Token fetch error: {e}")
        return None

    async def _process_snapshot(self, pair: str, data: dict):
        key = f"{self.exchange}_{pair}"
        if key not in self.books:
            self.books[key] = OrderBook(self.config.lob_levels)

        book = self.books[key]
        ts = datetime.fromtimestamp(
            int(data.get("timestamp", time.time() * 1000)) / 1000, tz=timezone.utc
        )

        # level2Depth50 gives full snapshots — just apply directly
        book.apply_snapshot(data.get("bids", []), data.get("asks", []))

        top = book.get_top_levels(self.config.lob_levels)
        is_valid, issues = book.validate()
        await self.db.enqueue(self.exchange, pair, ts, top, is_valid, issues)
        self.snapshot_counts[pair] += 1

    async def run(self):
        backoff = 1
        async with aiohttp.ClientSession() as http_session:
            while not self.stop.is_set():
                try:
                    # Step 1: Get WS token
                    token_data = await self._get_ws_token(http_session)
                    if not token_data:
                        raise Exception("Failed to get WebSocket token")

                    token = token_data["token"]
                    server = token_data["instanceServers"][0]
                    endpoint = server["endpoint"]
                    ping_interval = server.get("pingInterval", 30000) / 1000  # to seconds

                    # Initialize books
                    for pair in self.config.trading_pairs:
                        key = f"{self.exchange}_{pair}"
                        self.books[key] = OrderBook(self.config.lob_levels)

                    # Step 2: Connect with token
                    ws_url = f"{endpoint}?token={token}&connectId={int(time.time() * 1000)}"
                    logger.info(f"[{self.exchange}] Connecting WebSocket...")

                    async with websockets.connect(ws_url, ping_interval=None) as ws:
                        # Wait for welcome message
                        welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                        if welcome.get("type") != "welcome":
                            raise Exception(f"Unexpected welcome: {welcome}")

                        # Subscribe to level2Depth50 for all pairs
                        # KuCoin uses hyphens natively: BTC-USDT
                        symbols = ",".join(self.config.trading_pairs)
                        sub_id = str(int(time.time() * 1000))
                        await ws.send(json.dumps({
                            "id": sub_id,
                            "type": "subscribe",
                            "topic": f"/spotMarket/level2Depth50:{symbols}",
                            "response": True,
                        }))

                        backoff = 1
                        logger.info(f"[{self.exchange}] Connected, subscribed to {len(self.config.trading_pairs)} streams")

                        last_ping = time.time()

                        while not self.stop.is_set():
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=10)
                                data = json.loads(msg)

                                msg_type = data.get("type", "")

                                if msg_type == "pong":
                                    continue

                                if msg_type == "ack":
                                    logger.info(f"[{self.exchange}] Subscription confirmed")
                                    continue

                                if msg_type == "message":
                                    topic = data.get("topic", "")
                                    if topic.startswith("/spotMarket/level2Depth50:"):
                                        pair = topic.split(":")[-1]
                                        if pair in self.config.trading_pairs and "data" in data:
                                            await self._process_snapshot(pair, data["data"])

                                # Send ping on schedule
                                if time.time() - last_ping > ping_interval * 0.8:
                                    await ws.send(json.dumps({
                                        "id": str(int(time.time() * 1000)),
                                        "type": "ping",
                                    }))
                                    last_ping = time.time()

                            except asyncio.TimeoutError:
                                # Send ping on timeout
                                await ws.send(json.dumps({
                                    "id": str(int(time.time() * 1000)),
                                    "type": "ping",
                                }))
                                last_ping = time.time()
                            except websockets.ConnectionClosed:
                                logger.warning(f"[{self.exchange}] Connection closed, reconnecting")
                                break

                except Exception as e:
                    logger.error(f"[{self.exchange}] Error: {e}")

                if not self.stop.is_set():
                    for pair in self.config.trading_pairs:
                        await self.db.log_quality_issue(self.exchange, pair, "reconnect",
                                                        f"Reconnecting, backoff={backoff}s")
                    await asyncio.sleep(min(backoff, 60))
                    backoff = min(backoff * 2, 60)


# ---------------------------------------------------------------------------
# Health Monitor
# ---------------------------------------------------------------------------

class HealthMonitor:
    def __init__(self, config: Config, db: DBWriter, books: dict[str, OrderBook],
                 ws_managers: list, stop_event: asyncio.Event):
        self.config = config
        self.db = db
        self.books = books
        self.ws_managers = ws_managers
        self.stop = stop_event
        self._last_counts: dict[str, int] = defaultdict(int)

    async def run(self):
        await asyncio.sleep(10)  # Let streams initialize
        while not self.stop.is_set():
            try:
                lines = ["--- Health Check ---"]
                now = time.time()

                for mgr in self.ws_managers:
                    exchange = mgr.exchange
                    for pair in self.config.trading_pairs:
                        key = f"{exchange}_{pair}"
                        book = self.books.get(key)
                        count = mgr.snapshot_counts.get(pair, 0)
                        prev = self._last_counts.get(key, 0)
                        rate = (count - prev) / self.config.health_check_interval

                        if book:
                            age = now - book.last_update_time if book.last_update_time > 0 else -1
                            status = "OK" if age >= 0 and age < 30 else "STALE"
                            if age >= 0 and age >= 30:
                                await self.db.log_quality_issue(
                                    exchange, pair, "stale",
                                    f"No update for {age:.0f}s"
                                )
                            lines.append(
                                f"  {exchange}/{pair}: {status} "
                                f"(age={age:.1f}s, rate={rate:.1f}/s, total={count})"
                            )
                        else:
                            lines.append(f"  {exchange}/{pair}: NOT INITIALIZED")

                        self._last_counts[key] = count

                logger.info("\n".join(lines))

            except Exception as e:
                logger.error(f"Health check error: {e}")

            await asyncio.sleep(self.config.health_check_interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    config = Config.from_env()
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    logger.info("Starting LOB Data Collector")
    logger.info(f"  Exchanges: {config.exchanges}")
    logger.info(f"  Pairs: {config.trading_pairs}")
    logger.info(f"  DB: {config.db_host}:{config.db_port}/{config.db_name}")

    # Shared state
    stop_event = asyncio.Event()
    books: dict[str, OrderBook] = {}

    # Handle shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    # Database writer
    db = DBWriter(config)
    await db.connect()

    # WebSocket managers
    ws_managers = []
    if "binance_spot" in config.exchanges:
        ws_managers.append(BinanceSpotWS(config, db, books, stop_event))
    if "binance_perp" in config.exchanges:
        ws_managers.append(BinancePerpWS(config, db, books, stop_event))
    if "bybit_spot" in config.exchanges:
        ws_managers.append(BybitSpotWS(config, db, books, stop_event))
    if "kucoin_spot" in config.exchanges:
        ws_managers.append(KuCoinSpotWS(config, db, books, stop_event))

    # Health monitor
    health = HealthMonitor(config, db, books, ws_managers, stop_event)

    # Launch all tasks
    tasks = [asyncio.create_task(mgr.run()) for mgr in ws_managers]
    tasks.append(asyncio.create_task(db.flush_loop(stop_event)))
    tasks.append(asyncio.create_task(health.run()))

    logger.info(f"Launched {len(ws_managers)} exchange streams + flush loop + health monitor")
    logger.info("Press Ctrl+C to stop")

    # Wait for stop signal
    await stop_event.wait()
    logger.info("Shutdown signal received, stopping...")

    # Cancel all tasks and wait
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Clean up
    await db.close()
    logger.info("LOB Data Collector stopped")


if __name__ == "__main__":
    asyncio.run(main())
