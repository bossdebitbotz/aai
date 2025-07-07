#!/usr/bin/env python3
"""
Add Database Indexes for LOB Analysis

This script adds indexes to the lob_snapshots table to dramatically improve
query performance for gap analysis and data extraction.
"""

import psycopg2
import sys

DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

def add_indexes():
    try:
        print("Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Set autocommit for CONCURRENTLY operations
        conn.autocommit = True
        
        print("Adding composite index for (exchange, trading_pair, timestamp)...")
        cursor.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lob_exchange_pair_timestamp 
            ON lob_snapshots (exchange, trading_pair, timestamp);
        """)
        
        print("Adding index for timestamp only...")
        cursor.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lob_timestamp 
            ON lob_snapshots (timestamp);
        """)
        print("✅ Indexes added successfully!")
        
        # Check existing indexes
        print("\nCurrent indexes on lob_snapshots:")
        cursor.execute("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'lob_snapshots';
        """)
        indexes = cursor.fetchall()
        for name, definition in indexes:
            print(f"  {name}: {definition}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error adding indexes: {e}")
        sys.exit(1)

if __name__ == "__main__":
    add_indexes() 