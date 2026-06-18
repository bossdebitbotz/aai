"""
Signal Service -- THIN dispatcher.

Delegates ALL decision logic to the parity-proven paper signal path
(executor.paper.signal_generator.generate_from_buffer, which is bit-exact to
training: feature_parity_check Delta~1e-14, test_signal_parity 10/10). It then
maintains the SSOT inventory (executor.paper.strategy.Inventory in a PaperLedger)
and PROJECTS the inventory sign onto Freqtrade via REST.

There is intentionally NO model loading, feature engineering, scaling, gate logic,
or thresholds in this file -- that bespoke re-implementation is exactly how the
service previously diverged from training. The single source of truth is
generate_from_buffer; freqtrade is a dumb executor.

Components:
    service_decide      - pure SSOT decision for one buffer (delegates to generate_from_buffer)
    position_to_orders  - map an inventory sign change to freqtrade entry/exit actions
    FreqtradeClient     - async REST client (unchanged from the prior version)
    main()              - STRIDE-cadence loop: DB buffer -> service_decide -> Inventory -> freqtrade
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any, Optional

import aiohttp
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from executor.paper import signal_generator as SG       # SSOT signal path (bit-exact to training)
from executor.paper import db_source as DBS
from executor.paper.harness import FEE_BP, STRIDE
from executor.paper.ledger import PaperLedger

logger = logging.getLogger(__name__)


def ft_pair(symbol: str) -> str:
    """('BTC-USDT') -> freqtrade futures pair 'BTC/USDT:USDT'."""
    base, quote = symbol.split("-")
    return f"{base}/{quote}:USDT"


def service_decide(raw_buffer: np.ndarray, exchange: str, symbol: str, buckets=None) -> dict:
    """SSOT decision for one stream buffer -- delegates ENTIRELY to the parity-proven
    paper signal generator (decision bucket LAG back from 'now', frozen scaler, FROZEN
    gate stack). No independent logic. Returns the generate_from_buffer dict."""
    return SG.generate_from_buffer(raw_buffer, exchange, symbol, decision_offset=SG.LAG, buckets=buckets)


def position_to_orders(prev_pos: float, new_pos: float) -> list[dict]:
    """Project an inventory position change onto Freqtrade (1 trade/pair: it tracks only
    the SIGN; the service ledger holds the true +/-CAP inventory = accounting SSOT).
    Returns an ordered list of {'action':'exit'} / {'action':'entry','side':...}."""
    ps, ns = int(np.sign(prev_pos)), int(np.sign(new_pos))
    if ps == ns:
        return []                                   # no sign change -> freqtrade unchanged
    orders: list[dict] = []
    if ps != 0:
        orders.append({"action": "exit"})           # close the existing trade first
    if ns != 0:
        orders.append({"action": "entry", "side": "long" if ns > 0 else "short"})
    return orders


# ---------------------------------------------------------------------------
# FreqtradeClient (async) -- unchanged REST contract
# ---------------------------------------------------------------------------

class FreqtradeClient:
    """Async HTTP client for Freqtrade's REST API."""

    def __init__(self, url: str, username: str, password: str):
        self.base_url = url.rstrip("/")
        self.username = username
        self.password = password
        self._session: Optional[Any] = None
        self._token: Optional[str] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def _login(self) -> None:
        await self._ensure_session()
        auth = aiohttp.BasicAuth(self.username, self.password)
        resp = await self._session.post(f"{self.base_url}/api/v1/token/login", auth=auth)
        resp.raise_for_status()
        data = await resp.json()
        self._token = data["access_token"]

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    async def force_entry(self, pair: str, side: str, stake: float, price: float = None) -> dict:
        await self._ensure_session()
        if not self._token:
            await self._login()
        try:
            payload = {"pair": pair, "side": side, "stakeamount": stake, "ordertype": "limit"}
            if price:
                payload["price"] = str(price)
            resp = await self._session.post(f"{self.base_url}/api/v1/forceenter",
                                            json=payload, headers=self._auth_headers())
            data = await resp.json()
            if resp.status != 200:
                logger.warning("force_entry %s %s failed: %s", pair, side, data.get("error", resp.status))
                return {}
            logger.info("TRADE OPENED: %s %s trade_id=%s", pair, side, data.get("trade_id"))
            return data
        except Exception as e:
            logger.error("force_entry exception: %s", e)
            return {}

    async def force_exit(self, trade_id: int) -> dict:
        await self._ensure_session()
        if not self._token:
            await self._login()
        try:
            resp = await self._session.post(f"{self.base_url}/api/v1/forceexit",
                                            json={"tradeid": str(trade_id)}, headers=self._auth_headers())
            data = await resp.json()
            if resp.status != 200:
                logger.warning("force_exit trade %s failed: %s", trade_id, data.get("error", resp.status))
                return {}
            logger.info("TRADE CLOSED: trade_id=%s", trade_id)
            return data
        except Exception as e:
            logger.error("force_exit exception: %s", e)
            return {}

    async def get_trades(self) -> list[dict]:
        await self._ensure_session()
        if not self._token:
            await self._login()
        try:
            resp = await self._session.get(f"{self.base_url}/api/v1/status", headers=self._auth_headers())
            if resp.status == 401:
                self._token = None
                await self._login()
                resp = await self._session.get(f"{self.base_url}/api/v1/status", headers=self._auth_headers())
            return await resp.json() if resp.status == 200 else []
        except Exception as e:
            logger.error("get_trades exception: %s", e)
            return []

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


async def _apply_orders(ft: FreqtradeClient, pair_ft: str, orders: list[dict],
                        mid: float, spread: float, stake: float) -> None:
    """Translate projected orders into freqtrade REST calls (maker limit prices)."""
    for o in orders:
        if o["action"] == "exit":
            for t in await ft.get_trades():
                if t.get("pair") == pair_ft:
                    await ft.force_exit(t["trade_id"])
        else:
            side = o["side"]
            limit = mid - spread / 2 if side == "long" else mid + spread / 2
            await ft.force_entry(pair_ft, side, stake, price=limit)


async def main() -> None:
    """STRIDE-cadence loop: live DB buffer -> SSOT decision -> SSOT inventory -> freqtrade.

    Sync DB fetch + model inference run in a worker thread (asyncio.to_thread) so the
    aiohttp REST client stays async without re-implementing anything.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    cfg = json.load(open(os.path.join(_THIS_DIR, "signal_config.json")))
    ft_cfg = cfg.get("freqtrade_api", {})
    stake = cfg.get("stake_amount", 100)
    db_path = os.path.join(_THIS_DIR, "paper", "service_ledger.db")
    ledger = PaperLedger(db_path)
    ft = FreqtradeClient(ft_cfg.get("url", "http://127.0.0.1:8080"),
                         ft_cfg.get("username", "aai"), ft_cfg.get("password", "aai_paper_trade"))
    pairs = list(cfg.get("pairs", {}).keys()) or ["BTC-USDT", "ETH-USDT"]
    streams = [("binance_perp", p) for p in pairs]

    stop = asyncio.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.set())

    last_bucket: dict = {}
    logger.info("Signal service (SSOT delegate) starting: streams=%s ledger=%s", streams, db_path)
    try:
        while not stop.is_set():
            for exch, sym in streams:
                stream = f"{exch}_{sym}"
                try:
                    buckets, raw = await asyncio.to_thread(DBS.fetch_buffer, exch, sym, SG.MIN_BUFFER + 200)
                except Exception as e:
                    logger.warning("%s db error: %s", stream, e); continue
                if buckets is None or len(raw) < SG.MIN_BUFFER:
                    continue
                now_b = buckets[-1]
                prev = last_bucket.get(stream)
                if prev is not None and (now_b - prev).total_seconds() < (STRIDE - 1) * 5:
                    continue
                sig = await asyncio.to_thread(service_decide, raw, exch, sym, buckets)
                last_bucket[stream] = now_b
                if sig.get("reason"):
                    continue
                inv = ledger.load_inventory(stream)
                prev_pos, prev_cost = inv.position, inv.realized_cost_bp
                inv.on_decision(sig["signal"], sig["signal"] != 0, sig["mid"], sig["spread"], FEE_BP)
                ledger.record(now_b.isoformat(), stream, sig, inv,
                              inv.position - prev_pos, inv.realized_cost_bp - prev_cost)
                orders = position_to_orders(prev_pos, inv.position)
                if orders:
                    await _apply_orders(ft, ft_pair(sym), orders, sig["mid"], sig["spread"], stake)
                    logger.info("%s sig=%+d pos=%+.0f -> %s", stream, sig["signal"], inv.position,
                                [o["action"] for o in orders])
            try:
                await asyncio.wait_for(stop.wait(), timeout=float((STRIDE - 1) * 5))
            except asyncio.TimeoutError:
                pass
    finally:
        await ft.close()
        ledger.close()
        logger.info("Signal service shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())
