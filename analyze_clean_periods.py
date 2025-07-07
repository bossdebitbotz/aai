#!/usr/bin/env python3
"""
Analyze Clean Periods from Existing Gap Data

This script analyzes the existing detailed gap CSV file to identify
consecutive clean periods suitable for LOB forecasting research.
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Configuration
GAP_CSV_FILE = "data_gaps_detailed_20250630_001405.csv"
HOUR_THRESHOLD_SECONDS = 3600  # 1 hour

def load_and_analyze_gaps():
    """Load the detailed gap data and analyze clean periods."""
    print("📊 Loading existing gap analysis data...")
    
    try:
        # Load the gap data
        gaps_df = pd.read_csv(GAP_CSV_FILE)
        gaps_df['gap_start'] = pd.to_datetime(gaps_df['gap_start'], format='mixed')
        gaps_df['gap_end'] = pd.to_datetime(gaps_df['gap_end'], format='mixed')
        
        print(f"✅ Loaded {len(gaps_df):,} gaps from {GAP_CSV_FILE}")
        
        return gaps_df
        
    except Exception as e:
        print(f"❌ Error loading gap data: {e}")
        return None

def identify_bad_days(gaps_df):
    """Identify days with hour-plus gaps."""
    print("🔍 Identifying days with hour-plus gaps...")
    
    # Filter gaps >= 1 hour
    hour_plus_gaps = gaps_df[gaps_df['duration_seconds'] >= HOUR_THRESHOLD_SECONDS].copy()
    
    # Extract the date from gap_start for each bad day
    hour_plus_gaps['trading_date'] = hour_plus_gaps['gap_start'].dt.date
    
    # Get unique bad days
    bad_days = hour_plus_gaps[['exchange', 'trading_pair', 'trading_date', 'duration_seconds']].copy()
    bad_days['max_gap_hours'] = bad_days['duration_seconds'] / 3600.0
    
    # Group by day to get the worst gap per day
    bad_days_summary = bad_days.groupby(['exchange', 'trading_pair', 'trading_date']).agg({
        'max_gap_hours': 'max',
        'duration_seconds': 'max'
    }).reset_index()
    
    print(f"🚫 Found {len(bad_days_summary)} days with hour-plus gaps")
    
    return bad_days_summary

def get_data_date_range():
    """Get the overall date range of data from the database quickly."""
    print("📅 Getting data date range...")
    
    try:
        import psycopg2
        
        DB_CONFIG = {
            'host': 'localhost',
            'port': 5433,
            'user': 'backtest_user',
            'password': 'backtest_password',
            'database': 'backtest_db'
        }
        
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Quick query for date range
        cursor.execute("SELECT MIN(DATE(timestamp)), MAX(DATE(timestamp)) FROM lob_snapshots;")
        min_date, max_date = cursor.fetchone()
        
        conn.close()
        
        return min_date, max_date
        
    except Exception as e:
        print(f"⚠️ Could not get date range from DB: {e}")
        return None, None

def find_clean_periods(gaps_df, bad_days_summary):
    """Find consecutive clean periods by analyzing gaps."""
    print("🔗 Finding consecutive clean periods...")
    
    # Get date range
    min_date, max_date = get_data_date_range()
    if not min_date or not max_date:
        print("❌ Could not determine data date range")
        return pd.DataFrame()
    
    print(f"📊 Data spans from {min_date} to {max_date}")
    
    # Create a set of bad days for quick lookup
    bad_days_set = set()
    for _, row in bad_days_summary.iterrows():
        bad_days_set.add((row['exchange'], row['trading_pair'], row['trading_date']))
    
    # Get all unique exchange/pair combinations
    all_pairs = gaps_df[['exchange', 'trading_pair']].drop_duplicates()
    
    consecutive_periods = []
    
    for _, pair_row in all_pairs.iterrows():
        exchange = pair_row['exchange']
        trading_pair = pair_row['trading_pair']
        
        print(f"  Analyzing {exchange} / {trading_pair}")
        
        # Generate all dates in range
        current_date = min_date
        clean_periods = []
        current_period_start = None
        current_period_days = 0
        
        while current_date <= max_date:
            is_bad_day = (exchange, trading_pair, current_date) in bad_days_set
            
            if not is_bad_day:  # Clean day
                if current_period_start is None:
                    current_period_start = current_date
                    current_period_days = 1
                else:
                    current_period_days += 1
            else:  # Bad day - end current period if exists
                if current_period_start is not None and current_period_days >= 3:  # At least 3 clean days
                    clean_periods.append({
                        'exchange': exchange,
                        'trading_pair': trading_pair,
                        'start_date': current_period_start,
                        'end_date': current_date - timedelta(days=1),
                        'clean_days_count': current_period_days
                    })
                current_period_start = None
                current_period_days = 0
            
            current_date += timedelta(days=1)
        
        # Don't forget the last period
        if current_period_start is not None and current_period_days >= 3:
            clean_periods.append({
                'exchange': exchange,
                'trading_pair': trading_pair,
                'start_date': current_period_start,
                'end_date': current_date - timedelta(days=1),
                'clean_days_count': current_period_days
            })
        
        consecutive_periods.extend(clean_periods)
    
    return pd.DataFrame(consecutive_periods)

def main():
    print("🧹 CLEAN PERIODS ANALYSIS")
    print("=" * 60)
    
    # Load gap data
    gaps_df = load_and_analyze_gaps()
    if gaps_df is None:
        return
    
    # Identify bad days
    print("\n" + "=" * 60)
    print("🚫 DAYS WITH HOUR-PLUS GAPS")
    print("=" * 60)
    
    bad_days = identify_bad_days(gaps_df)
    
    if not bad_days.empty:
        print(f"\nFound {len(bad_days)} problematic days")
        print("\nWorst offenders (top 10):")
        worst_days = bad_days.nlargest(10, 'max_gap_hours')
        for _, row in worst_days.iterrows():
            print(f"  📅 {row['trading_date']} | {row['exchange']:<15} | {row['trading_pair']:<12} | "
                  f"Max gap: {row['max_gap_hours']:.1f} hours")
        
        # Save bad days
        bad_days.to_csv('bad_days_hour_plus_gaps.csv', index=False)
        print(f"\n💾 Saved bad days to: bad_days_hour_plus_gaps.csv")
        
        # Summary by pair
        print(f"\n📊 BAD DAYS BY TRADING PAIR:")
        bad_days_summary = bad_days.groupby(['exchange', 'trading_pair']).agg({
            'trading_date': 'count',
            'max_gap_hours': 'max'
        }).rename(columns={'trading_date': 'bad_days_count'}).sort_values('bad_days_count', ascending=False)
        
        for (exchange, pair), row in bad_days_summary.iterrows():
            print(f"  📊 {exchange:<15} | {pair:<12} | "
                  f"Bad days: {int(row['bad_days_count']):3d} | "
                  f"Worst gap: {row['max_gap_hours']:5.1f}h")
    
    # Find clean periods
    print("\n" + "=" * 60)
    print("🔗 CONSECUTIVE CLEAN PERIODS")
    print("=" * 60)
    
    clean_periods = find_clean_periods(gaps_df, bad_days)
    
    if not clean_periods.empty:
        print(f"\nFound {len(clean_periods)} consecutive clean periods")
        print("\nLongest consecutive clean periods (top 15):")
        
        longest_periods = clean_periods.nlargest(15, 'clean_days_count')
        for _, period in longest_periods.iterrows():
            print(f"  📅 {period['start_date']} to {period['end_date']} | "
                  f"{period['exchange']:<15} | {period['trading_pair']:<12} | "
                  f"{period['clean_days_count']:3d} clean days")
        
        # Save results
        clean_periods.to_csv('consecutive_clean_periods.csv', index=False)
        print(f"\n💾 Saved consecutive periods to: consecutive_clean_periods.csv")
        
        # Statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   • Total consecutive periods: {len(clean_periods)}")
        print(f"   • Longest clean period: {clean_periods['clean_days_count'].max()} days")
        print(f"   • Average clean period: {clean_periods['clean_days_count'].mean():.1f} days")
        print(f"   • Periods with ≥30 days: {len(clean_periods[clean_periods['clean_days_count'] >= 30])}")
        print(f"   • Periods with ≥60 days: {len(clean_periods[clean_periods['clean_days_count'] >= 60])}")
        
        # Best pairs for research
        best_pairs = clean_periods.groupby(['exchange', 'trading_pair'])['clean_days_count'].max().sort_values(ascending=False)
        
        print(f"\n🏆 BEST TRADING PAIRS FOR RESEARCH:")
        print("   (Based on longest consecutive clean periods)")
        for i, ((exchange, pair), max_days) in enumerate(best_pairs.head(8).items(), 1):
            print(f"   {i}. {exchange:<15} | {pair:<12} | {max_days:3d} consecutive clean days")
        
        # Training candidates
        training_candidates = clean_periods[clean_periods['clean_days_count'] >= 30]
        if not training_candidates.empty:
            print(f"\n🎯 RECOMMENDED TRAINING PERIODS (≥30 days):")
            for _, period in training_candidates.nlargest(8, 'clean_days_count').iterrows():
                print(f"   📅 {period['start_date']} to {period['end_date']} | "
                      f"{period['exchange']:<15} | {period['trading_pair']:<12} | "
                      f"{period['clean_days_count']:3d} days")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 