#!/usr/bin/env python3
"""Live data source: trailing buffer of lob_5s rows from TimescaleDB, in the EXACT
training/parquet column order (reuses export_training_data.build_columns).

Returns the raw (T, 162) feature matrix [interleaved bid/ask per level + mid + spread]
that feeds engineer_features_v2 — identical layout to the parquet the model trained on.
"""
import sys, asyncio
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
import asyncpg
import export_training_data as ex   # reuse build_columns() => guaranteed parity with parquet

DSN = "postgresql://lob_user:lob_password@localhost:5432/lob_data"
COLS = ex.build_columns()           # ['bucket', bid_price_1, bid_volume_1, ask_price_1, ask_volume_1, ..., mid_price, spread]
FEATURE_COLS = COLS[1:]             # drop 'bucket' -> 162 raw feature columns


async def _fetch(exchange: str, symbol: str, n: int, end_time):
    conn = await asyncpg.connect(DSN)
    try:
        await conn.execute("SET max_parallel_workers_per_gather = 0")
        collist = ", ".join(COLS)
        end_clause = "AND bucket <= $3" if end_time is not None else ""
        params = [exchange, symbol] + ([end_time, n] if end_time is not None else [n])
        nparam = "$4" if end_time is not None else "$3"
        q = (f"SELECT {collist} FROM ("
             f"  SELECT {collist} FROM lob_5s "
             f"  WHERE exchange = $1 AND symbol = $2 AND all_valid = true {end_clause} "
             f"  ORDER BY bucket DESC LIMIT {nparam}"
             f") sub ORDER BY bucket ASC")
        rows = await conn.fetch(q, *params)
        return rows
    finally:
        await conn.close()


def fetch_buffer(exchange: str, symbol: str, n: int = 400, end_time=None):
    """Return (buckets: list[datetime], raw: np.ndarray (T,162)) — the most recent
    n valid lob_5s buckets for the stream (optionally up to end_time), chronological."""
    rows = asyncio.run(_fetch(exchange, symbol, n, end_time))
    if not rows:
        return [], np.empty((0, len(FEATURE_COLS)))
    buckets = [r["bucket"] for r in rows]
    raw = np.array([[float(r[c]) for c in FEATURE_COLS] for r in rows], dtype=np.float64)
    return buckets, raw


if __name__ == "__main__":
    b, raw = fetch_buffer("binance_perp", "BTC-USDT", n=5)
    print("rows:", len(b), "shape:", raw.shape)
    if b:
        print("latest bucket:", b[-1], "| mid(idx160):", raw[-1, 160], "| spread(161):", raw[-1, 161])
