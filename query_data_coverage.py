#!/usr/bin/env python3
"""
Data Coverage Query Tool

This script provides detailed analysis of data gaps from the database,
saving results to both console and CSV for further analysis.
"""

import os
import logging
import pandas as pd
import psycopg2
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

GAP_THRESHOLD_SECONDS = 10

def get_detailed_gaps():
    """Get detailed gap information for all trading pairs."""
    print("🔍 Extracting detailed gap information from database...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Get all gaps with detailed information
        query = """
        WITH lagged_timestamps AS (
            SELECT
                exchange,
                trading_pair,
                timestamp,
                LAG(timestamp, 1) OVER (PARTITION BY exchange, trading_pair ORDER BY timestamp) as prev_timestamp
            FROM lob_snapshots
        )
        SELECT
            exchange,
            trading_pair,
            prev_timestamp as gap_start,
            timestamp as gap_end,
            EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) as duration_seconds
        FROM lagged_timestamps
        WHERE (timestamp - prev_timestamp) > interval '%s seconds'
        AND prev_timestamp IS NOT NULL
        ORDER BY exchange, trading_pair, gap_start;
        """ % GAP_THRESHOLD_SECONDS
        
        df_gaps = pd.read_sql(query, conn)
        conn.close()
        
        if df_gaps.empty:
            print("✅ No significant gaps found!")
            return df_gaps
            
        # Save to CSV
        csv_filename = f"data_gaps_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_gaps.to_csv(csv_filename, index=False)
        print(f"📊 Detailed gaps saved to: {csv_filename}")
        
        # Display summary statistics
        print("\n📈 Gap Analysis Summary:")
        print("=" * 80)
        
        summary = df_gaps.groupby(['exchange', 'trading_pair']).agg({
            'duration_seconds': ['count', 'mean', 'max', 'sum']
        }).round(2)
        
        summary.columns = ['Gap_Count', 'Avg_Duration_s', 'Max_Duration_s', 'Total_Gap_Time_s']
        print(summary)
        
        # Show worst gaps (longest duration)
        print("\n🚨 Top 10 Longest Data Gaps:")
        print("-" * 100)
        top_gaps = df_gaps.nlargest(10, 'duration_seconds')
        for _, gap in top_gaps.iterrows():
            print(f"{gap['exchange']:<15} {gap['trading_pair']:<12} "
                  f"{gap['gap_start']:%Y-%m-%d %H:%M:%S} → {gap['gap_end']:%Y-%m-%d %H:%M:%S} "
                  f"({gap['duration_seconds']:.0f}s)")
        
        # Show gap patterns by hour of day
        print("\n⏰ Gap Distribution by Hour of Day:")
        df_gaps['hour'] = pd.to_datetime(df_gaps['gap_start']).dt.hour
        hourly_gaps = df_gaps.groupby('hour')['duration_seconds'].count().sort_index()
        
        for hour, count in hourly_gaps.items():
            bar = "█" * min(int(count / max(hourly_gaps) * 20), 20)
            print(f"{hour:02d}:00 │{bar:<20}│ {count} gaps")
            
        return df_gaps
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

def analyze_coverage_quality():
    """Analyze overall data coverage quality."""
    print("\n🎯 Data Coverage Quality Analysis:")
    print("=" * 50)
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Get data coverage statistics
        coverage_query = """
        SELECT 
            exchange,
            trading_pair,
            COUNT(*) as total_snapshots,
            MIN(timestamp) as first_snapshot,
            MAX(timestamp) as last_snapshot,
            EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) as total_duration_seconds
        FROM lob_snapshots
        GROUP BY exchange, trading_pair
        ORDER BY exchange, trading_pair;
        """
        
        df_coverage = pd.read_sql(coverage_query, conn)
        conn.close()
        
        # Calculate expected vs actual snapshots
        df_coverage['expected_snapshots'] = df_coverage['total_duration_seconds'] / 1  # Assuming 1-second intervals
        df_coverage['coverage_ratio'] = df_coverage['total_snapshots'] / df_coverage['expected_snapshots']
        df_coverage['missing_snapshots'] = df_coverage['expected_snapshots'] - df_coverage['total_snapshots']
        
        print("\n📊 Coverage Statistics:")
        for _, row in df_coverage.iterrows():
            print(f"\n{row['exchange']} / {row['trading_pair']}:")
            print(f"  📅 Period: {row['first_snapshot']} → {row['last_snapshot']}")
            print(f"  📈 Snapshots: {row['total_snapshots']:,.0f} / {row['expected_snapshots']:,.0f} expected")
            print(f"  📊 Coverage: {row['coverage_ratio']:.1%}")
            print(f"  ❌ Missing: {row['missing_snapshots']:,.0f} snapshots")
            
        return df_coverage
        
    except Exception as e:
        print(f"❌ Error analyzing coverage: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print("🚀 Starting Data Coverage Analysis")
    print("=" * 50)
    
    # Get detailed gap information
    gaps_df = get_detailed_gaps()
    
    # Analyze overall coverage quality
    coverage_df = analyze_coverage_quality()
    
    print("\n✅ Analysis complete!")
    if not gaps_df.empty:
        print(f"📁 Detailed gap data saved to CSV file for further analysis.") 