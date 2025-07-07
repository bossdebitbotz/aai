#!/usr/bin/env python3
import psycopg2
import sys

DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

try:
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    print("Checking total record count...")
    cursor.execute("SELECT COUNT(*) FROM lob_snapshots;")
    total = cursor.fetchone()[0]
    print(f"Total records: {total:,}")
    
    print("Checking date range...")
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM lob_snapshots;")
    min_date, max_date = cursor.fetchone()
    print(f"Date range: {min_date} to {max_date}")
    
    print("Checking distinct pairs...")
    cursor.execute("SELECT exchange, trading_pair, COUNT(*) FROM lob_snapshots GROUP BY exchange, trading_pair ORDER BY COUNT(*) DESC;")
    pairs = cursor.fetchall()
    print(f"Found {len(pairs)} exchange/pair combinations:")
    for exchange, pair, count in pairs:
        print(f"  {exchange} {pair}: {count:,} records")
    
    conn.close()
    print("Done!")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1) 