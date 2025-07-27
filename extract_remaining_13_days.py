#!/usr/bin/env python3
"""
FINAL EXTRACTION - Remaining 13 July Days
All remaining days are July 2025 - the highest value data!
This completes our 60-day maximum dataset.
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import json
import gc
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Configuration
OUTPUT_DIR = "data/full_lob_data"
LOB_LEVELS = 5
BATCH_SIZE = 1000000
MAX_GAP_MINUTES = 30
MIN_SESSION_HOURS = 2
TARGET_EXCHANGES = ['binance_spot', 'binance_perp', 'bybit_spot']
TARGET_PAIRS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT']

# ONLY THE REMAINING 13 JULY DAYS (ALL HIGH VALUE!)
REMAINING_JULY_DAYS = [
    '2025-07-04', '2025-07-13', '2025-07-14', '2025-07-15', '2025-07-16',
    '2025-07-17', '2025-07-18', '2025-07-19', '2025-07-20', '2025-07-21',
    '2025-07-22', '2025-07-23', '2025-07-24'
]

def get_db_connection():
    """Create database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def extract_lob_data(data_dict: dict, levels: int = 5) -> dict:
    """Extract LOB data from JSONB into flat structure."""
    try:
        result = {}
        
        # Extract bids (sorted by price descending)
        if 'bids' in data_dict and isinstance(data_dict['bids'], list):
            bids = sorted(data_dict['bids'], key=lambda x: float(x[0]), reverse=True)[:levels]
            for i, (price, quantity) in enumerate(bids):
                result[f'bid_price_{i+1}'] = float(price)
                result[f'bid_quantity_{i+1}'] = float(quantity)
        
        # Extract asks (sorted by price ascending)
        if 'asks' in data_dict and isinstance(data_dict['asks'], list):
            asks = sorted(data_dict['asks'], key=lambda x: float(x[0]))[:levels]
            for i, (price, quantity) in enumerate(asks):
                result[f'ask_price_{i+1}'] = float(price)
                result[f'ask_quantity_{i+1}'] = float(quantity)
        
        # Fill missing levels with None
        for i in range(1, levels + 1):
            for side in ['bid', 'ask']:
                for field in ['price', 'quantity']:
                    key = f'{side}_{field}_{i}'
                    if key not in result:
                        result[key] = None
        
        return result
        
    except Exception as e:
        logger.warning(f"Error parsing LOB data: {e}")
        return {}

def detect_gaps(df: pd.DataFrame, max_gap_minutes: int = 30) -> List[Tuple[datetime, datetime]]:
    """Detect gaps in time series data."""
    gaps = []
    
    if len(df) < 2:
        return gaps
    
    time_diffs = df['timestamp'].diff()
    gap_threshold = timedelta(minutes=max_gap_minutes)
    large_gaps = time_diffs[time_diffs > gap_threshold]
    
    for idx in large_gaps.index:
        gap_start = df.loc[idx-1, 'timestamp']
        gap_end = df.loc[idx, 'timestamp']
        gaps.append((gap_start, gap_end))
    
    return gaps

def split_into_sessions(df: pd.DataFrame, gaps: List[Tuple[datetime, datetime]], 
                       min_session_hours: int = 2) -> List[pd.DataFrame]:
    """Split data into continuous sessions based on gaps."""
    if not gaps:
        return [df] if len(df) > 0 else []
    
    sessions = []
    start_idx = 0
    
    for gap_start, gap_end in gaps:
        session_end_idx = df[df['timestamp'] <= gap_start].index[-1]
        session_df = df.iloc[start_idx:session_end_idx + 1].copy()
        
        if len(session_df) > 0:
            duration = session_df['timestamp'].iloc[-1] - session_df['timestamp'].iloc[0]
            if duration >= timedelta(hours=min_session_hours):
                sessions.append(session_df)
        
        start_idx = df[df['timestamp'] >= gap_end].index[0] if len(df[df['timestamp'] >= gap_end]) > 0 else len(df)
    
    if start_idx < len(df):
        final_session = df.iloc[start_idx:].copy()
        if len(final_session) > 0:
            duration = final_session['timestamp'].iloc[-1] - final_session['timestamp'].iloc[0]
            if duration >= timedelta(hours=min_session_hours):
                sessions.append(final_session)
    
    return sessions

def get_day_record_count(date_str: str) -> int:
    """Get approximate record count for a day."""
    try:
        conn = get_db_connection()
        if not conn:
            return 0
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM lob_snapshots WHERE DATE(timestamp) = %s", [date_str])
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except:
        return 0

def process_single_day(date_str: str, day_num: int, total_days: int) -> bool:
    """Process a single day of data with progress tracking."""
    
    # Get expected record count for progress tracking
    expected_records = get_day_record_count(date_str)
    logger.info(f"🔄 Processing July day {day_num}/{total_days}: {date_str}")
    logger.info(f"📊 Expected records: {expected_records:,}")
    
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        # Create output directory for this date
        date_dir = os.path.join(OUTPUT_DIR, date_str)
        os.makedirs(date_dir, exist_ok=True)
        
        # Query data for this date
        query = """
        SELECT timestamp, exchange, trading_pair, data
        FROM lob_snapshots 
        WHERE DATE(timestamp) = %s
        ORDER BY timestamp, exchange, trading_pair
        """
        
        logger.info(f"  📥 Starting database query...")
        
        # Use chunked reading for memory efficiency
        chunk_size = BATCH_SIZE
        processed_records = 0
        chunk_count = 0
        
        for chunk in pd.read_sql_query(query, conn, params=[date_str], chunksize=chunk_size):
            chunk_count += 1
            progress = (processed_records / expected_records * 100) if expected_records > 0 else 0
            logger.info(f"  ⚡ Processing chunk {chunk_count}: {len(chunk)} records ({progress:.1f}% complete)")
            
            # Process by exchange/pair
            for exchange in TARGET_EXCHANGES:
                for pair in TARGET_PAIRS:
                    pair_data = chunk[
                        (chunk['exchange'] == exchange) & 
                        (chunk['trading_pair'] == pair)
                    ].copy()
                    
                    if len(pair_data) == 0:
                        continue
                    
                    # Extract LOB data
                    lob_records = []
                    for _, row in pair_data.iterrows():
                        lob_dict = extract_lob_data(row['data'], LOB_LEVELS)
                        if lob_dict:
                            record = {
                                'timestamp': row['timestamp'],
                                'exchange': exchange,
                                'trading_pair': pair,
                                **lob_dict
                            }
                            lob_records.append(record)
                    
                    if lob_records:
                        # Convert to DataFrame
                        pair_df = pd.DataFrame(lob_records)
                        pair_df['timestamp'] = pd.to_datetime(pair_df['timestamp'])
                        pair_df = pair_df.sort_values('timestamp')
                        
                        # Detect gaps and split into sessions
                        gaps = detect_gaps(pair_df, MAX_GAP_MINUTES)
                        sessions = split_into_sessions(pair_df, gaps, MIN_SESSION_HOURS)
                        
                        # Save each session
                        for session_idx, session_df in enumerate(sessions):
                            if len(session_df) > 0:
                                session_file = os.path.join(
                                    date_dir, 
                                    f"{exchange}_{pair}_{date_str}_session_{session_idx}.parquet"
                                )
                                session_df.to_parquet(session_file, compression='snappy', index=False)
            
            processed_records += len(chunk)
            del chunk
            gc.collect()
        
        conn.close()
        
        logger.info(f"  ✅ {date_str} COMPLETED: {processed_records:,} records processed")
        
        # Save daily summary
        summary = {
            'date': date_str,
            'total_records': processed_records,
            'processing_time': datetime.now().isoformat(),
            'exchanges': TARGET_EXCHANGES,
            'trading_pairs': TARGET_PAIRS
        }
        
        summary_file = os.path.join(date_dir, f"{date_str}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing {date_str}: {e}")
        return False

def main():
    """Extract the final 13 July days to complete the maximum dataset."""
    start_time = datetime.now()
    
    logger.info("🚀 FINAL EXTRACTION - COMPLETING MAXIMUM DATASET!")
    logger.info(f"💎 Processing {len(REMAINING_JULY_DAYS)} remaining July days")
    logger.info(f"🎯 Target: Complete 60-day dataset → 50,000+ sequences")
    logger.info(f"🔥 ALL remaining days are July (highest value!)")
    logger.info("⏱️ ETA: 3-4 hours for completion")
    logger.info("")
    
    successful_days = 0
    failed_days = []
    
    for i, date_str in enumerate(REMAINING_JULY_DAYS, 1):
        day_start = datetime.now()
        
        if process_single_day(date_str, i, len(REMAINING_JULY_DAYS)):
            successful_days += 1
            elapsed = datetime.now() - day_start
            total_elapsed = datetime.now() - start_time
            
            logger.info(f"✅ {date_str} SUCCESS ({i}/{len(REMAINING_JULY_DAYS)})")
            logger.info(f"⏱️ Day time: {elapsed.total_seconds()/60:.1f}m, Total: {total_elapsed.total_seconds()/60:.1f}m")
            
            # ETA calculation
            if i < len(REMAINING_JULY_DAYS):
                avg_time_per_day = total_elapsed.total_seconds() / i
                remaining_time = avg_time_per_day * (len(REMAINING_JULY_DAYS) - i)
                eta_minutes = remaining_time / 60
                logger.info(f"📊 ETA for completion: {eta_minutes:.0f} minutes")
            
            logger.info("")
        else:
            failed_days.append(date_str)
            logger.warning(f"❌ {date_str} FAILED")
    
    # Final completion
    total_time = datetime.now() - start_time
    logger.info("🎉 FINAL EXTRACTION COMPLETED!")
    logger.info(f"✅ Successful: {successful_days}/{len(REMAINING_JULY_DAYS)} July days")
    logger.info(f"⏱️ Total time: {total_time.total_seconds()/60:.1f} minutes")
    
    if failed_days:
        logger.warning(f"❌ Failed days: {failed_days}")
    
    # Create resampled data for new days
    logger.info("🔄 Resampling new July days to 5-second intervals...")
    resampled_dir = os.path.join(OUTPUT_DIR, "resampled_5s")
    os.makedirs(resampled_dir, exist_ok=True)
    
    for date_str in REMAINING_JULY_DAYS:
        if date_str not in failed_days:
            date_path = os.path.join(OUTPUT_DIR, date_str)
            if os.path.exists(date_path):
                session_files = [f for f in os.listdir(date_path) if f.endswith('.parquet')]
                
                for session_file in session_files:
                    session_path = os.path.join(date_path, session_file)
                    
                    try:
                        df = pd.read_parquet(session_path)
                        if len(df) == 0:
                            continue
                        
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                        df_resampled = df.resample('5s').last()
                        df_resampled = df_resampled.fillna(method='ffill', limit=6)
                        df_resampled = df_resampled.dropna()
                        
                        if len(df_resampled) > 0:
                            output_file = os.path.join(resampled_dir, f"{session_file.replace('.parquet', '_resampled_5s.parquet')}")
                            df_resampled.to_parquet(output_file, compression='snappy')
                    
                    except Exception as e:
                        logger.error(f"Error resampling {session_file}: {e}")
    
    # Final summary
    final_day_count = len([d for d in os.listdir(OUTPUT_DIR) if d.startswith('2025-')])
    final_parquet_count = len([f for f in os.listdir(resampled_dir) if f.endswith('.parquet')])
    
    summary = {
        'extraction_date': datetime.now().isoformat(),
        'final_status': 'MAXIMUM DATASET COMPLETE',
        'total_days_extracted': final_day_count,
        'july_days_completed': successful_days,
        'failed_days': failed_days,
        'total_parquet_files': final_parquet_count,
        'extraction_time_minutes': total_time.total_seconds() / 60,
        'next_step': 'Run prepare_attention_model_data_v5.py for 50,000+ sequences'
    }
    
    summary_file = os.path.join(OUTPUT_DIR, "final_extraction_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("🎯 MAXIMUM DATASET ACHIEVED!")
    logger.info(f"📊 Final stats: {final_day_count} days, {final_parquet_count} parquet files")
    logger.info("🚀 Ready for V5 data preparation → 50,000+ sequences!")
    logger.info(f"📂 Summary: {summary_file}")

if __name__ == "__main__":
    main() 