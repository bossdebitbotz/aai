#!/usr/bin/env python3
"""
Export lob_5s data from TimescaleDB to parquet files for Colab training.

Uses psql COPY with retries and recovery waits to handle DB memory limits.
Exports one parquet per exchange/symbol, then zips for Colab upload.

Usage:
    python export_training_data.py
"""

import os
import subprocess
import time
import zipfile
from pathlib import Path
from io import StringIO

import pandas as pd

LOB_LEVELS = 40
OUTPUT_DIR = Path("lob_data")
ZIP_NAME = "lob_training_data.zip"

EXCHANGES = ["binance_spot", "binance_perp", "bybit_spot", "kucoin_spot"]
PAIRS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "WLD-USDT"]

CONTAINER = "lob_timescaledb"
DB_USER = "lob_user"
DB_NAME = "lob_data"

MAX_RETRIES = 3
RETRY_WAIT = 30  # seconds to wait for DB recovery

FRESH = True  # set by main(); --resume flips to False to keep existing exports


def build_columns():
    cols = ["bucket"]
    for i in range(1, LOB_LEVELS + 1):
        cols.extend([
            f"bid_price_{i}", f"bid_volume_{i}",
            f"ask_price_{i}", f"ask_volume_{i}",
        ])
    cols.extend(["mid_price", "spread"])
    return cols


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


def wait_for_db():
    """Wait until the DB is accepting connections again."""
    for i in range(30):
        result = subprocess.run(
            ["docker", "exec", CONTAINER, "pg_isready", "-U", DB_USER, "-d", DB_NAME],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return True
        time.sleep(2)
    return False


def export_stream(exchange, symbol, columns):
    """Export one stream via psql COPY, with retries."""
    fname = OUTPUT_DIR / f"{exchange}_{symbol}.parquet"

    if (not FRESH) and fname.exists() and fname.stat().st_size > 1000:
        print(f"  {exchange}/{symbol}: keeping existing export (--resume)")
        return fname

    # NOTE: skip-guard removed; freshness is controlled by --fresh in main()
    copy_query = build_copy_query(exchange, symbol, columns)

    for attempt in range(1, MAX_RETRIES + 1):
        result = subprocess.run(
            ["docker", "exec", CONTAINER, "psql", "-U", DB_USER, "-d", DB_NAME,
             "-c", "SET work_mem = '64MB'", "-c", copy_query],
            capture_output=True, text=True, timeout=600,
        )

        if result.returncode == 0:
            csv_data = result.stdout
            # Strip "SET\n" prefix from work_mem command
            if csv_data.startswith("SET\n"):
                csv_data = csv_data[4:]
            if not csv_data.strip() or csv_data.count('\n') <= 1:
                print(f"  {exchange}/{symbol}: no data, skipping")
                return None

            df = pd.read_csv(StringIO(csv_data))
            df.to_parquet(fname, index=False)

            size_mb = fname.stat().st_size / 1024 / 1024
            print(f"  {exchange}/{symbol}: {len(df)} rows ({size_mb:.1f} MB)")
            return fname

        # Failed — wait for recovery
        print(f"  {exchange}/{symbol}: attempt {attempt}/{MAX_RETRIES} failed, waiting {RETRY_WAIT}s for DB recovery...")
        time.sleep(RETRY_WAIT)
        if not wait_for_db():
            print(f"  {exchange}/{symbol}: DB not recovering, giving up")
            return None

    print(f"  {exchange}/{symbol}: all retries exhausted")
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export lob_5s to parquet for Colab")
    parser.add_argument("--fresh", action="store_true", default=True,
                        help="Delete existing exports first (default).")
    parser.add_argument("--resume", dest="fresh", action="store_false",
                        help="Keep existing parquet files; only export missing streams.")
    args = parser.parse_args()

    global FRESH
    FRESH = args.fresh

    OUTPUT_DIR.mkdir(exist_ok=True)
    columns = build_columns()

    if args.fresh:
        n = clear_stale_exports(OUTPUT_DIR, zip_path=Path(ZIP_NAME))
        print(f"--fresh: removed {n} stale export file(s)")

    # Make sure DB is ready
    print("Waiting for DB...")
    if not wait_for_db():
        print("DB not available!")
        return

    print(f"Exporting {len(EXCHANGES) * len(PAIRS)} streams from lob_5s...\n")
    files = []

    for exchange in EXCHANGES:
        for symbol in PAIRS:
            f = export_stream(exchange, symbol, columns)
            if f:
                files.append(f)
            # Brief pause between streams to let DB breathe
            time.sleep(5)

    if not files:
        print("No data exported!")
        return

    # Zip
    print(f"\nZipping {len(files)} files...")
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f"lob_data/{f.name}")

    zip_size = Path(ZIP_NAME).stat().st_size / 1024 / 1024
    print(f"\n✓ Created {ZIP_NAME} ({zip_size:.1f} MB)")
    print(f"  Upload this file to the Colab notebook.")


if __name__ == "__main__":
    main()
