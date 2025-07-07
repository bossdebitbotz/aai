#!/usr/bin/env python3
"""
Simplified Clean Database to Parquet Data Exporter
"""

import os
import json
import logging
import pandas as pd
import psycopg2
from datetime import datetime, date
import pyarrow as pa
import pyarrow.parquet as pq

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Storage path
CLEAN_DATA_DIR = "data/clean_parquet"
LOB_LEVELS = 5

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_export.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_bad_days():
    """Load bad days to exclude."""
    try:
        bad_days_df = pd.read_csv('bad_days_hour_plus_gaps.csv')
        bad_days_set = set()
        
        for _, row in bad_days_df.iterrows():
            trading_date = pd.to_datetime(row['trading_date']).date()
            bad_days_set.add((row['exchange'], row['trading_pair'], trading_date))
        
        logger.info(f"📋 Loaded {len(bad_days_df)} bad day records")
        return bad_days_set
        
    except FileNotFoundError:
        logger.error("❌ bad_days_hour_plus_gaps.csv not found")
        return set()

def extract_lob_data(lob_data, levels=5):
    """Extract LOB data into flat structure."""
    try:
        data = json.loads(lob_data) if isinstance(lob_data, str) else lob_data
        result = {}
        
        # Extract bids
        if 'bids' in data and isinstance(data['bids'], list):
            bids = sorted(data['bids'], key=lambda x: float(x[0]), reverse=True)[:levels]
            for i, (price, quantity) in enumerate(bids):
                result[f'bid_price_{i+1}'] = float(price)
                result[f'bid_quantity_{i+1}'] = float(quantity)
        
        # Extract asks
        if 'asks' in data and isinstance(data['asks'], list):
            asks = sorted(data['asks'], key=lambda x: float(x[0]))[:levels]
            for i, (price, quantity) in enumerate(asks):
                result[f'ask_price_{i+1}'] = float(price)
                result[f'ask_quantity_{i+1}'] = float(quantity)
        
        # Fill missing levels
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

def export_single_pair(exchange, trading_pair, bad_days_set):
    """Export a single trading pair."""
    logger.info(f"🔄 Processing {exchange}/{trading_pair}")
    
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Get total count first
        count_query = """
        SELECT COUNT(*) 
        FROM lob_snapshots 
        WHERE exchange = %s AND trading_pair = %s
        """
        cur.execute(count_query, (exchange, trading_pair))
        total_records = cur.fetchone()[0]
        logger.info(f"   📊 Total records in DB: {total_records:,}")
        
        # Query data in manageable chunks
        query = """
        SELECT timestamp, exchange, trading_pair, data
        FROM lob_snapshots 
        WHERE exchange = %s AND trading_pair = %s
        ORDER BY timestamp
        LIMIT %s OFFSET %s
        """
        
        chunk_size = 10000
        offset = 0
        all_clean_records = []
        
        while True:
            cur.execute(query, (exchange, trading_pair, chunk_size, offset))
            rows = cur.fetchall()
            
            if not rows:
                break
                
            logger.info(f"   📥 Processing chunk: {len(rows)} records (offset: {offset:,})")
            
            # Filter out bad days
            clean_chunk = []
            for row in rows:
                timestamp, exch, pair, data = row
                trading_date = timestamp.date()
                
                if (exch, pair, trading_date) not in bad_days_set:
                    # Extract LOB data
                    lob_dict = extract_lob_data(data, LOB_LEVELS)
                    
                    record = {
                        'timestamp': timestamp,
                        'exchange': exch,
                        'trading_pair': pair,
                        **lob_dict
                    }
                    clean_chunk.append(record)
            
            if clean_chunk:
                all_clean_records.extend(clean_chunk)
                logger.info(f"   ✅ Added {len(clean_chunk)} clean records")
            
            offset += chunk_size
            
            # Progress indicator
            if offset % 50000 == 0:
                logger.info(f"   📈 Progress: {offset:,}/{total_records:,} ({offset/total_records*100:.1f}%)")
        
        conn.close()
        
        if not all_clean_records:
            logger.warning(f"   ⚠️ No clean data found for {exchange}/{trading_pair}")
            return False
        
        logger.info(f"   📊 Total clean records: {len(all_clean_records):,}")
        
        # Create DataFrame and save
        df = pd.DataFrame(all_clean_records)
        
        # Create directory
        pair_dir = os.path.join(CLEAN_DATA_DIR, exchange)
        os.makedirs(pair_dir, exist_ok=True)
        
        # Save to Parquet
        filename = f"{trading_pair.replace('/', '_').replace('-', '_')}_clean.parquet"
        filepath = os.path.join(pair_dir, filename)
        
        logger.info(f"   💾 Saving to {filepath}...")
        
        df.to_parquet(
            filepath,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"   ✅ Saved: {file_size_mb:.1f} MB ({len(df):,} records)")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Error exporting {exchange}/{trading_pair}: {e}")
        return False

def main():
    logger.info("🚀 Starting Simple Clean Parquet Export")
    logger.info("=" * 50)
    
    # Load bad days
    bad_days_set = load_bad_days()
    if not bad_days_set:
        logger.error("❌ Could not load bad days")
        return
    
    # Create output directory
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    
    # Get trading pairs
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT exchange, trading_pair FROM lob_snapshots ORDER BY exchange, trading_pair")
        trading_pairs = cur.fetchall()
        conn.close()
        
        logger.info(f"📋 Found {len(trading_pairs)} trading pairs")
        
        # Process each pair
        successful = 0
        for i, (exchange, trading_pair) in enumerate(trading_pairs, 1):
            logger.info(f"\n[{i}/{len(trading_pairs)}] {exchange}/{trading_pair}")
            
            if export_single_pair(exchange, trading_pair, bad_days_set):
                successful += 1
        
        logger.info(f"\n✅ Completed: {successful}/{len(trading_pairs)} pairs exported")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 