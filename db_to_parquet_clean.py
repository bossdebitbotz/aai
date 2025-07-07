#!/usr/bin/env python3
"""
Clean Database to Parquet Data Exporter

This script extracts limit order book (LOB) data from PostgreSQL database
and exports it into Parquet files, EXCLUDING days with hour-plus gaps
to create clean, research-quality datasets.
"""

import os
import json
import logging
import pandas as pd
import psycopg2
from collections import defaultdict
from datetime import datetime, date
import pyarrow as pa
import pyarrow.parquet as pq

# --- Configuration ---

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Storage path for Parquet files
DATA_DIR = "data"
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "clean_parquet")

# Number of LOB levels to process
LOB_LEVELS = 5

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clean_parquet_export.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_bad_days():
    """Load the list of days with hour-plus gaps to exclude."""
    try:
        bad_days_df = pd.read_csv('bad_days_hour_plus_gaps.csv')
        bad_days_set = set()
        
        for _, row in bad_days_df.iterrows():
            # Convert string date to date object for comparison
            trading_date = pd.to_datetime(row['trading_date']).date()
            bad_days_set.add((row['exchange'], row['trading_pair'], trading_date))
        
        logger.info(f"📋 Loaded {len(bad_days_df)} bad day records covering {len(set(bad_days_df['trading_date']))} unique dates")
        return bad_days_set
        
    except FileNotFoundError:
        logger.error("❌ bad_days_hour_plus_gaps.csv not found. Please run clean analysis first.")
        return set()
    except Exception as e:
        logger.error(f"❌ Error loading bad days: {e}")
        return set()

def get_clean_data_summary(bad_days_set):
    """Get summary of available clean data."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Get overall data range and volume
        query = """
        SELECT 
            MIN(DATE(timestamp)) as min_date,
            MAX(DATE(timestamp)) as max_date,
            COUNT(*) as total_records,
            COUNT(DISTINCT CONCAT(exchange, '|', trading_pair)) as unique_pairs
        FROM lob_snapshots;
        """
        
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        min_date = result['min_date'].iloc[0]
        max_date = result['max_date'].iloc[0]
        total_records = result['total_records'].iloc[0]
        unique_pairs = result['unique_pairs'].iloc[0]
        
        total_days = (max_date - min_date).days + 1
        
        logger.info(f"📊 Data Summary:")
        logger.info(f"   • Date range: {min_date} to {max_date} ({total_days} days)")
        logger.info(f"   • Total records: {total_records:,}")
        logger.info(f"   • Trading pairs: {unique_pairs}")
        logger.info(f"   • Days to exclude: {len(set([date for _, _, date in bad_days_set]))}")
        
        return min_date, max_date, total_records
        
    except Exception as e:
        logger.error(f"❌ Error getting data summary: {e}")
        return None, None, 0

def extract_lob_data(lob_data, levels=5):
    """Extract bid/ask data from LOB JSONB into a flat dictionary."""
    try:
        data = json.loads(lob_data) if isinstance(lob_data, str) else lob_data
        
        result = {}
        
        # Extract bids (sorted by price descending - highest first)
        if 'bids' in data and isinstance(data['bids'], list):
            bids = sorted(data['bids'], key=lambda x: float(x[0]), reverse=True)[:levels]
            for i, (price, quantity) in enumerate(bids):
                result[f'bid_price_{i+1}'] = float(price)
                result[f'bid_quantity_{i+1}'] = float(quantity)
        
        # Extract asks (sorted by price ascending - lowest first)
        if 'asks' in data and isinstance(data['asks'], list):
            asks = sorted(data['asks'], key=lambda x: float(x[0]))[:levels]
            for i, (price, quantity) in enumerate(asks):
                result[f'ask_price_{i+1}'] = float(price)
                result[f'ask_quantity_{i+1}'] = float(quantity)
        
        # Fill missing levels with NaN
        for i in range(1, levels + 1):
            for side in ['bid', 'ask']:
                for field in ['price', 'quantity']:
                    key = f'{side}_{field}_{i}'
                    if key not in result:
                        result[key] = None
        
        return result
        
    except Exception as e:
        logger.warning(f"⚠️ Error parsing LOB data: {e}")
        return {}

def export_pair_to_parquet(exchange, trading_pair, bad_days_set):
    """Export clean data for a specific exchange/trading pair to Parquet."""
    logger.info(f"📤 Exporting {exchange}/{trading_pair}...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Query to get clean data (excluding bad days)
        query = """
        SELECT 
            timestamp,
            exchange,
            trading_pair,
            data
        FROM lob_snapshots 
        WHERE exchange = %s AND trading_pair = %s
        ORDER BY timestamp;
        """
        
        # Use chunked reading for memory efficiency
        chunk_size = 50000
        all_data = []
        
        logger.info(f"   📥 Loading data in chunks...")
        
        for chunk in pd.read_sql_query(query, conn, params=[exchange, trading_pair], chunksize=chunk_size):
            # Filter out bad days
            chunk['date'] = chunk['timestamp'].dt.date
            
            clean_chunk = []
            for _, row in chunk.iterrows():
                trading_date = row['date']
                if (exchange, trading_pair, trading_date) not in bad_days_set:
                    clean_chunk.append(row)
            
            if clean_chunk:
                clean_df = pd.DataFrame(clean_chunk)
                all_data.append(clean_df)
                logger.info(f"   ✅ Processed chunk: {len(clean_chunk):,} clean records")
        
        conn.close()
        
        if not all_data:
            logger.warning(f"   ⚠️ No clean data found for {exchange}/{trading_pair}")
            return False
        
        # Combine all chunks
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"   📊 Total clean records: {len(df):,}")
        
        # Process LOB data
        logger.info(f"   🔄 Processing LOB data...")
        lob_records = []
        
        for _, row in df.iterrows():
            lob_dict = extract_lob_data(row['data'], LOB_LEVELS)
            
            record = {
                'timestamp': row['timestamp'],
                'exchange': row['exchange'],
                'trading_pair': row['trading_pair'],
                **lob_dict
            }
            lob_records.append(record)
        
        # Create final DataFrame
        final_df = pd.DataFrame(lob_records)
        
        # Create directory structure
        pair_dir = os.path.join(CLEAN_DATA_DIR, exchange)
        os.makedirs(pair_dir, exist_ok=True)
        
        # Save to Parquet with compression
        filename = f"{trading_pair.replace('/', '_').replace('-', '_')}_clean.parquet"
        filepath = os.path.join(pair_dir, filename)
        
        logger.info(f"   💾 Saving to {filepath}...")
        
        # Use efficient Parquet settings
        final_df.to_parquet(
            filepath,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"   ✅ Saved: {file_size_mb:.1f} MB ({len(final_df):,} records)")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Error exporting {exchange}/{trading_pair}: {e}")
        return False

def get_trading_pairs():
    """Get list of all trading pairs."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        query = """
        SELECT DISTINCT exchange, trading_pair 
        FROM lob_snapshots 
        ORDER BY exchange, trading_pair;
        """
        
        pairs_df = pd.read_sql_query(query, conn)
        conn.close()
        
        return [(row['exchange'], row['trading_pair']) for _, row in pairs_df.iterrows()]
        
    except Exception as e:
        logger.error(f"❌ Error fetching trading pairs: {e}")
        return []

def main():
    logger.info("🚀 Starting Clean Parquet Export")
    logger.info("=" * 60)
    
    # Load bad days to exclude
    bad_days_set = load_bad_days()
    if not bad_days_set:
        logger.error("❌ Could not load bad days list. Exiting.")
        return
    
    # Get data summary
    min_date, max_date, total_records = get_clean_data_summary(bad_days_set)
    if not min_date:
        logger.error("❌ Could not get data summary. Exiting.")
        return
    
    # Create output directory
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    
    # Get trading pairs
    trading_pairs = get_trading_pairs()
    if not trading_pairs:
        logger.error("❌ No trading pairs found. Exiting.")
        return
    
    logger.info(f"📋 Found {len(trading_pairs)} trading pairs to process")
    
    # Process each pair
    successful_exports = 0
    total_size_mb = 0
    
    for i, (exchange, trading_pair) in enumerate(trading_pairs, 1):
        logger.info(f"\n[{i}/{len(trading_pairs)}] Processing {exchange}/{trading_pair}")
        
        if export_pair_to_parquet(exchange, trading_pair, bad_days_set):
            successful_exports += 1
            
            # Get file size
            pair_dir = os.path.join(CLEAN_DATA_DIR, exchange)
            filename = f"{trading_pair.replace('/', '_').replace('-', '_')}_clean.parquet"
            filepath = os.path.join(pair_dir, filename)
            
            if os.path.exists(filepath):
                file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                total_size_mb += file_size_mb
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 EXPORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Successfully exported: {successful_exports}/{len(trading_pairs)} pairs")
    logger.info(f"💾 Total Parquet files size: {total_size_mb:.1f} MB")
    logger.info(f"📁 Output directory: {CLEAN_DATA_DIR}")
    logger.info(f"🧹 Clean data only (excluded {len(set([date for _, _, date in bad_days_set]))} problematic days)")
    
    if successful_exports > 0:
        logger.info("\n🎉 Clean Parquet export completed successfully!")
        logger.info("💡 These files are now ready for LOB forecasting research.")
    else:
        logger.error("\n❌ Export failed. Check logs for details.")

if __name__ == "__main__":
    main() 