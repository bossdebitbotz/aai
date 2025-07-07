#!/usr/bin/env python3
"""
Clean Data Filter: Exclude Days with Hour-Plus Gaps

This script identifies and excludes trading days that contain gaps of 1 hour or more,
creating clean datasets suitable for LOB forecasting research.
"""

import os
import logging
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import json

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Gap threshold: 1 hour = 3600 seconds
HOUR_THRESHOLD_SECONDS = 3600

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clean_data_filter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_days_with_hour_plus_gaps():
    """Identify days that contain gaps of 1 hour or more."""
    logger.info("🔍 Identifying days with hour-plus gaps...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Query to find days with gaps >= 1 hour
        query = """
        WITH daily_gaps AS (
            SELECT
                exchange,
                trading_pair,
                DATE(timestamp) as trading_date,
                MAX(EXTRACT(EPOCH FROM (timestamp - prev_timestamp))) as max_gap_seconds
            FROM (
                SELECT
                    exchange,
                    trading_pair,
                    timestamp,
                    LAG(timestamp, 1) OVER (
                        PARTITION BY exchange, trading_pair, DATE(timestamp) 
                        ORDER BY timestamp
                    ) as prev_timestamp
                FROM lob_snapshots
            ) t
            WHERE prev_timestamp IS NOT NULL
            GROUP BY exchange, trading_pair, DATE(timestamp)
        )
        SELECT 
            exchange,
            trading_pair,
            trading_date,
            max_gap_seconds,
            max_gap_seconds / 3600.0 as max_gap_hours
        FROM daily_gaps
        WHERE max_gap_seconds >= %s
        ORDER BY exchange, trading_pair, trading_date;
        """
        
        bad_days_df = pd.read_sql_query(query, conn, params=[HOUR_THRESHOLD_SECONDS])
        conn.close()
        
        logger.info(f"📊 Found {len(bad_days_df)} trading days with hour-plus gaps")
        
        return bad_days_df
        
    except Exception as e:
        logger.error(f"❌ Error identifying bad days: {e}")
        return pd.DataFrame()

def get_clean_days_summary():
    """Get summary of clean days (without hour-plus gaps) by exchange and pair."""
    logger.info("📈 Analyzing clean days availability...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Query to get all trading days and identify clean ones
        query = """
        WITH daily_max_gaps AS (
            SELECT
                exchange,
                trading_pair,
                DATE(timestamp) as trading_date,
                MAX(EXTRACT(EPOCH FROM (timestamp - prev_timestamp))) as max_gap_seconds
            FROM (
                SELECT
                    exchange,
                    trading_pair,
                    timestamp,
                    LAG(timestamp, 1) OVER (
                        PARTITION BY exchange, trading_pair, DATE(timestamp) 
                        ORDER BY timestamp
                    ) as prev_timestamp
                FROM lob_snapshots
            ) t
            WHERE prev_timestamp IS NOT NULL
            GROUP BY exchange, trading_pair, DATE(timestamp)
        ),
        day_classification AS (
            SELECT 
                exchange,
                trading_pair,
                trading_date,
                max_gap_seconds,
                CASE 
                    WHEN max_gap_seconds >= %s THEN 'bad_day'
                    ELSE 'clean_day'
                END as day_quality
            FROM daily_max_gaps
        )
        SELECT 
            exchange,
            trading_pair,
            day_quality,
            COUNT(*) as day_count,
            MIN(trading_date) as earliest_date,
            MAX(trading_date) as latest_date
        FROM day_classification
        GROUP BY exchange, trading_pair, day_quality
        ORDER BY exchange, trading_pair, day_quality;
        """
        
        summary_df = pd.read_sql_query(query, conn, params=[HOUR_THRESHOLD_SECONDS])
        conn.close()
        
        return summary_df
        
    except Exception as e:
        logger.error(f"❌ Error getting clean days summary: {e}")
        return pd.DataFrame()

def get_clean_consecutive_periods():
    """Find consecutive periods of clean days for each exchange/pair."""
    logger.info("🔗 Finding consecutive clean periods...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Get all clean days
        query = """
        WITH daily_max_gaps AS (
            SELECT
                exchange,
                trading_pair,
                DATE(timestamp) as trading_date,
                MAX(EXTRACT(EPOCH FROM (timestamp - prev_timestamp))) as max_gap_seconds
            FROM (
                SELECT
                    exchange,
                    trading_pair,
                    timestamp,
                    LAG(timestamp, 1) OVER (
                        PARTITION BY exchange, trading_pair, DATE(timestamp) 
                        ORDER BY timestamp
                    ) as prev_timestamp
                FROM lob_snapshots
            ) t
            WHERE prev_timestamp IS NOT NULL
            GROUP BY exchange, trading_pair, DATE(timestamp)
        )
        SELECT 
            exchange,
            trading_pair,
            trading_date
        FROM daily_max_gaps
        WHERE max_gap_seconds < %s
        ORDER BY exchange, trading_pair, trading_date;
        """
        
        clean_days_df = pd.read_sql_query(query, conn, params=[HOUR_THRESHOLD_SECONDS])
        conn.close()
        
        # Find consecutive periods
        consecutive_periods = []
        
        for (exchange, pair), group in clean_days_df.groupby(['exchange', 'trading_pair']):
            dates = pd.to_datetime(group['trading_date']).sort_values()
            
            if len(dates) == 0:
                continue
                
            # Find consecutive sequences
            current_start = dates.iloc[0]
            current_end = dates.iloc[0]
            
            for i in range(1, len(dates)):
                if (dates.iloc[i] - current_end).days <= 1:  # Allow for weekends
                    current_end = dates.iloc[i]
                else:
                    # End of consecutive period
                    period_length = (current_end - current_start).days + 1
                    consecutive_periods.append({
                        'exchange': exchange,
                        'trading_pair': pair,
                        'start_date': current_start.date(),
                        'end_date': current_end.date(),
                        'period_length_days': period_length,
                        'clean_days_count': len(dates[(dates >= current_start) & (dates <= current_end)])
                    })
                    current_start = dates.iloc[i]
                    current_end = dates.iloc[i]
            
            # Don't forget the last period
            period_length = (current_end - current_start).days + 1
            consecutive_periods.append({
                'exchange': exchange,
                'trading_pair': pair,
                'start_date': current_start.date(),
                'end_date': current_end.date(),
                'period_length_days': period_length,
                'clean_days_count': len(dates[(dates >= current_start) & (dates <= current_end)])
            })
        
        return pd.DataFrame(consecutive_periods)
        
    except Exception as e:
        logger.error(f"❌ Error finding consecutive periods: {e}")
        return pd.DataFrame()

def main():
    logger.info("🧹 Starting Clean Data Filter Analysis")
    logger.info(f"📏 Threshold: {HOUR_THRESHOLD_SECONDS} seconds ({HOUR_THRESHOLD_SECONDS/3600:.1f} hours)")
    
    # 1. Identify bad days
    print("\n" + "="*60)
    print("🚫 DAYS WITH HOUR-PLUS GAPS")
    print("="*60)
    
    bad_days = get_days_with_hour_plus_gaps()
    if not bad_days.empty:
        print(f"\nFound {len(bad_days)} problematic days:")
        print("\nWorst offenders (top 10):")
        worst_days = bad_days.nlargest(10, 'max_gap_hours')
        for _, row in worst_days.iterrows():
            print(f"  📅 {row['trading_date']} | {row['exchange']:<15} | {row['trading_pair']:<12} | "
                  f"Max gap: {row['max_gap_hours']:.1f} hours")
        
        # Save bad days to file
        bad_days.to_csv('bad_days_hour_plus_gaps.csv', index=False)
        print(f"\n💾 Saved bad days list to: bad_days_hour_plus_gaps.csv")
    
    # 2. Get clean days summary
    print("\n" + "="*60)
    print("✅ CLEAN DAYS SUMMARY")
    print("="*60)
    
    clean_summary = get_clean_days_summary()
    if not clean_summary.empty:
        print("\nData quality by exchange and trading pair:")
        for (exchange, pair), group in clean_summary.groupby(['exchange', 'trading_pair']):
            clean_days = group[group['day_quality'] == 'clean_day']['day_count'].sum()
            bad_days_count = group[group['day_quality'] == 'bad_day']['day_count'].sum()
            total_days = clean_days + bad_days_count
            clean_pct = (clean_days / total_days * 100) if total_days > 0 else 0
            
            print(f"  📊 {exchange:<15} | {pair:<12} | "
                  f"Clean: {clean_days:3d} days ({clean_pct:5.1f}%) | "
                  f"Bad: {bad_days_count:3d} days")
    
    # 3. Find consecutive clean periods
    print("\n" + "="*60)
    print("🔗 CONSECUTIVE CLEAN PERIODS")
    print("="*60)
    
    consecutive_periods = get_clean_consecutive_periods()
    if not consecutive_periods.empty:
        print("\nLongest consecutive clean periods (top 15):")
        longest_periods = consecutive_periods.nlargest(15, 'clean_days_count')
        
        for _, period in longest_periods.iterrows():
            print(f"  📅 {period['start_date']} to {period['end_date']} | "
                  f"{period['exchange']:<15} | {period['trading_pair']:<12} | "
                  f"{period['clean_days_count']:3d} clean days")
        
        # Save consecutive periods
        consecutive_periods.to_csv('consecutive_clean_periods.csv', index=False)
        print(f"\n💾 Saved consecutive periods to: consecutive_clean_periods.csv")
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   • Total consecutive periods found: {len(consecutive_periods)}")
        print(f"   • Longest clean period: {consecutive_periods['clean_days_count'].max()} days")
        print(f"   • Average clean period: {consecutive_periods['clean_days_count'].mean():.1f} days")
        print(f"   • Periods with ≥30 days: {len(consecutive_periods[consecutive_periods['clean_days_count'] >= 30])}")
        print(f"   • Periods with ≥60 days: {len(consecutive_periods[consecutive_periods['clean_days_count'] >= 60])}")
    
    # 4. Recommendations
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    if not consecutive_periods.empty:
        # Best pairs for research
        best_pairs = consecutive_periods.groupby(['exchange', 'trading_pair'])['clean_days_count'].max().sort_values(ascending=False)
        
        print("\n🏆 BEST TRADING PAIRS FOR RESEARCH:")
        print("   (Based on longest consecutive clean periods)")
        for i, ((exchange, pair), max_days) in enumerate(best_pairs.head(8).items(), 1):
            print(f"   {i}. {exchange:<15} | {pair:<12} | {max_days:3d} consecutive clean days")
        
        # Recommended training periods
        training_candidates = consecutive_periods[consecutive_periods['clean_days_count'] >= 45]  # Need ~45 days for training
        if not training_candidates.empty:
            print(f"\n🎯 RECOMMENDED TRAINING PERIODS:")
            print(f"   Found {len(training_candidates)} periods with ≥45 consecutive clean days")
            
            for _, period in training_candidates.nlargest(5, 'clean_days_count').iterrows():
                print(f"   📅 {period['start_date']} to {period['end_date']} | "
                      f"{period['exchange']:<15} | {period['trading_pair']:<12} | "
                      f"{period['clean_days_count']} days")
    
    print(f"\n✅ Analysis complete! Check the log file: clean_data_filter.log")

if __name__ == "__main__":
    main() 