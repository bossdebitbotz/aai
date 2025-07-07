#!/usr/bin/env python3
"""
Database Data Gap Identifier

This script connects to the PostgreSQL database and analyzes the `lob_snapshots`
table to identify and report any time gaps in the collected data for each
exchange and trading pair.
"""

import os
import logging
import pandas as pd
import psycopg2

# --- Configuration ---

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Define the threshold for detecting a gap.
# If the time between two consecutive data points exceeds this, it's a gap.
GAP_THRESHOLD_SECONDS = 10  # e.g., 10 seconds

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_gap_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_gap_analyzer")


def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Successfully connected to the database.")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def get_distinct_pairs(conn):
    """Get all distinct exchange and trading_pair combinations."""
    logger.info("Fetching distinct exchange/trading_pair combinations...")
    query = "SELECT DISTINCT exchange, trading_pair FROM lob_snapshots;"
    try:
        df = pd.read_sql(query, conn)
        logger.info(f"Found {len(df)} distinct pairs to analyze.")
        return df.to_records(index=False)
    except Exception as e:
        logger.error(f"Failed to fetch distinct pairs: {e}")
        return []

def analyze_gaps_for_pair(conn, exchange, trading_pair):
    """Fetches timestamps for a given pair and identifies gaps using an efficient SQL query."""
    logger.info(f"Analyzing {exchange} / {trading_pair} for data gaps...")
    
    query = """
    WITH lagged_timestamps AS (
        SELECT
            timestamp,
            LAG(timestamp, 1) OVER (ORDER BY timestamp) as prev_timestamp
        FROM lob_snapshots
        WHERE exchange = %(exchange)s AND trading_pair = %(trading_pair)s
    )
    SELECT
        prev_timestamp as gap_start,
        timestamp as gap_end,
        (timestamp - prev_timestamp) as duration
    FROM lagged_timestamps
    WHERE (timestamp - prev_timestamp) > %(threshold)s;
    """
    
    try:
        # Pass the threshold as a parameter to the query
        threshold_interval = f"{GAP_THRESHOLD_SECONDS} seconds"
        df_gaps = pd.read_sql(query, conn, params={
            'exchange': exchange, 
            'trading_pair': trading_pair,
            'threshold': threshold_interval
        })
        
        if df_gaps.empty:
            logger.info(f"No significant gaps found for {exchange} / {trading_pair}.")
            return []

        logger.warning(f"Found {len(df_gaps)} data gaps for {exchange} / {trading_pair}!")

        gap_list = []
        for index, row in df_gaps.iterrows():
            # The first row will have a NULL prev_timestamp, so we skip it
            if pd.isna(row['gap_start']):
                continue
            
            gap_list.append({
                "exchange": exchange,
                "trading_pair": trading_pair,
                "gap_start": row['gap_start'],
                "gap_end": row['gap_end'],
                "duration_seconds": row['duration'].total_seconds()
            })
        
        return gap_list

    except Exception as e:
        logger.error(f"An error occurred while analyzing {exchange} / {trading_pair}: {e}", exc_info=True)
        return []


def main():
    """Main function to run the gap analysis."""
    logger.info(f"Starting data gap analysis with a threshold of {GAP_THRESHOLD_SECONDS} seconds.")
    
    conn = get_db_connection()
    if not conn:
        return

    try:
        pairs = get_distinct_pairs(conn)
        all_gaps = []

        for exchange, trading_pair in pairs:
            gaps = analyze_gaps_for_pair(conn, exchange, trading_pair)
            all_gaps.extend(gaps)

        if not all_gaps:
            logger.info("\n--- Overall Result ---")
            logger.info("No data gaps found across all instruments based on the threshold.")
            print("\n✅ No significant data gaps were found.")
        else:
            logger.info("\n--- Gap Analysis Summary ---")
            print("\n🚨 The following data gaps were identified:")
            print("-" * 80)
            print(f"{'Exchange':<15} {'Trading Pair':<15} {'Gap Start (UTC)':<30} {'Gap End (UTC)':<30} {'Duration (s)':<15}")
            print("-" * 80)
            
            # Sort gaps by start time for cleaner reporting
            all_gaps.sort(key=lambda x: x['gap_start'])
            
            for gap in all_gaps:
                print(
                    f"{gap['exchange']:<15} "
                    f"{gap['trading_pair']:<15} "
                    f"{str(gap['gap_start']):<30} "
                    f"{str(gap['gap_end']):<30} "
                    f"{gap['duration_seconds']:<15.2f}"
                )
            print("-" * 80)


    finally:
        conn.close()
        logger.info("Database connection closed.")
        
if __name__ == "__main__":
    main() 