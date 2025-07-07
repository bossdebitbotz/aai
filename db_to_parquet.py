#!/usr/bin/env python3
"""
Database to Parquet Data Exporter

This script extracts limit order book (LOB) data from a PostgreSQL database
and exports it into Parquet files, maintaining the same structure as the
live data collection script. This is useful for regenerating Parquet datasets
from the database archive.
"""

import os
import json
import logging
import pandas as pd
import psycopg2
from collections import defaultdict

# --- Configuration ---

# Database configuration (should match your setup)
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Storage path for Parquet files
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_from_db") # Saving to a new dir to avoid conflicts

# Number of LOB levels to process
LOB_LEVELS = 5

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("db_to_parquet.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("db_to_parquet")


def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Successfully connected to the database.")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def fetch_lob_snapshots(conn):
    """Fetch all LOB snapshots from the database using a server-side cursor."""
    logger.info("Starting to fetch LOB snapshots from the database...")
    
    # Use a named cursor for efficient, streaming-like fetching
    with conn.cursor('lob_snapshot_cursor') as cursor:
        cursor.itersize = 1000  # Fetch 1000 rows at a time from the backend
        cursor.execute("SELECT exchange, trading_pair, timestamp, data FROM lob_snapshots ORDER BY timestamp")
        
        count = 0
        for record in cursor:
            yield record
            count += 1
            if count % 10000 == 0:
                logger.info(f"Fetched {count} records so far...")
    
    logger.info(f"Finished fetching. Total records: {count}.")

def process_and_save_data(records):
    """Process database records and save them to hourly Parquet files."""
    logger.info("Processing records and buffering for Parquet export...")
    
    # Buffer to hold data rows for each hourly parquet file
    # Key: (exchange, trading_pair, date_str, hour_str)
    # Value: list of data rows (dicts)
    data_buffer = defaultdict(list)

    for exchange, trading_pair, timestamp, data in records:
        # Generate the key for the buffer
        date_str = timestamp.strftime("%Y-%m-%d")
        hour_str = timestamp.strftime("%H")
        buffer_key = (exchange, trading_pair, date_str, hour_str)

        # Prepare a single row of data for the DataFrame
        ts_ms = int(timestamp.timestamp() * 1000)
        
        lob_row = {
            'timestamp': ts_ms,
            'exchange': exchange,
            'trading_pair': trading_pair,
        }

        # Initialize columns for LOB levels
        for i in range(1, LOB_LEVELS + 1):
            lob_row[f'bid_price_{i}'] = 0.0
            lob_row[f'bid_volume_{i}'] = 0.0
            lob_row[f'ask_price_{i}'] = 0.0
            lob_row[f'ask_volume_{i}'] = 0.0

        # Fill in bid data
        bids = data.get('bids', [])
        for i, (price, qty) in enumerate(bids[:LOB_LEVELS], 1):
            lob_row[f'bid_price_{i}'] = float(price)
            lob_row[f'bid_volume_{i}'] = float(qty)

        # Fill in ask data
        asks = data.get('asks', [])
        for i, (price, qty) in enumerate(asks[:LOB_LEVELS], 1):
            lob_row[f'ask_price_{i}'] = float(price)
            lob_row[f'ask_volume_{i}'] = float(qty)
            
        data_buffer[buffer_key].append(lob_row)

    logger.info(f"Data buffered into {len(data_buffer)} hourly chunks. Now writing to Parquet files.")

    # Write buffered data to Parquet files
    for key, rows in data_buffer.items():
        exchange, trading_pair, date_str, hour_str = key
        
        # Create directory path
        dir_path = os.path.join(RAW_DATA_DIR, exchange, trading_pair, date_str)
        os.makedirs(dir_path, exist_ok=True)
        
        # Create file path
        file_path = os.path.join(dir_path, f"{exchange}_{trading_pair}_{date_str}_{hour_str}.parquet")
        
        # Convert list of rows to DataFrame and save
        df = pd.DataFrame(rows)
        df.to_parquet(file_path, engine='pyarrow', compression='snappy', index=False)

    logger.info(f"Successfully exported {len(data_buffer)} Parquet files to '{RAW_DATA_DIR}'.")


def main():
    """Main function to run the database to Parquet export."""
    logger.info("Starting database to Parquet exporter.")
    
    # Ensure the main output directory exists
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        logger.info(f"Created output directory: {RAW_DATA_DIR}")

    conn = get_db_connection()
    if conn:
        try:
            records_iterator = fetch_lob_snapshots(conn)
            process_and_save_data(records_iterator)
        except Exception as e:
            logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
        finally:
            conn.close()
            logger.info("Database connection closed.")
            
if __name__ == "__main__":
    main() 