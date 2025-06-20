#!/usr/bin/env python3
"""
Query script to check data coverage in the LOB database.
This script will show how many days of data we have accumulated
across different exchanges and trading pairs.
"""

import psycopg2
from datetime import datetime, timezone
import pandas as pd

# Database configuration (matching the collector script)
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {type(e).__name__} - {str(e)}")
        return None

def query_data_coverage():
    """Query and display data coverage statistics"""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return
    
    try:
        with conn.cursor() as cursor:
            print("=== LOB Data Coverage Summary ===\n")
            
            # Check if tables exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('lob_snapshots', 'lob_metrics')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                print("No LOB tables found in the database.")
                return
            
            print(f"Found tables: {', '.join(tables)}\n")
            
            # For each table, get data coverage
            for table in tables:
                print(f"=== {table.upper()} TABLE ===")
                
                # Overall date range
                cursor.execute(f"""
                    SELECT 
                        MIN(timestamp) as earliest,
                        MAX(timestamp) as latest,
                        COUNT(*) as total_records
                    FROM {table}
                """)
                result = cursor.fetchone()
                
                if result and result[0]:
                    earliest, latest, total_records = result
                    duration = latest - earliest
                    days = duration.days + (duration.seconds / 86400)  # Include partial days
                    
                    print(f"  Date Range: {earliest} to {latest}")
                    print(f"  Duration: {days:.2f} days")
                    print(f"  Total Records: {total_records:,}")
                    print(f"  Average Records per Day: {total_records / max(days, 1):,.0f}")
                else:
                    print(f"  No data found in {table}")
                    continue
                
                # Breakdown by exchange and trading pair
                cursor.execute(f"""
                    SELECT 
                        exchange,
                        trading_pair,
                        MIN(timestamp) as earliest,
                        MAX(timestamp) as latest,
                        COUNT(*) as records,
                        COUNT(DISTINCT DATE(timestamp)) as unique_days
                    FROM {table}
                    GROUP BY exchange, trading_pair
                    ORDER BY exchange, trading_pair
                """)
                
                results = cursor.fetchall()
                if results:
                    print(f"\n  Breakdown by Exchange and Trading Pair:")
                    print(f"  {'Exchange':<15} {'Pair':<15} {'Start Date':<12} {'End Date':<12} {'Days':<6} {'Records':<10}")
                    print(f"  {'-'*15} {'-'*15} {'-'*12} {'-'*12} {'-'*6} {'-'*10}")
                    
                    for row in results:
                        exchange, pair, start, end, records, unique_days = row
                        start_str = start.strftime('%Y-%m-%d')
                        end_str = end.strftime('%Y-%m-%d')
                        print(f"  {exchange:<15} {pair:<15} {start_str:<12} {end_str:<12} {unique_days:<6} {records:<10,}")
                
                # Daily breakdown for the last 7 days
                cursor.execute(f"""
                    SELECT 
                        DATE(timestamp) as date,
                        exchange,
                        trading_pair,
                        COUNT(*) as records
                    FROM {table}
                    WHERE timestamp >= NOW() - INTERVAL '7 days'
                    GROUP BY DATE(timestamp), exchange, trading_pair
                    ORDER BY date DESC, exchange, trading_pair
                """)
                
                daily_results = cursor.fetchall()
                if daily_results:
                    print(f"\n  Last 7 Days Activity:")
                    print(f"  {'Date':<12} {'Exchange':<15} {'Pair':<15} {'Records':<10}")
                    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*10}")
                    
                    for row in daily_results:
                        date, exchange, pair, records = row
                        print(f"  {date:<12} {exchange:<15} {pair:<15} {records:<10,}")
                
                print("\n")
            
            # Data quality checks
            print("=== DATA QUALITY CHECKS ===")
            
            if 'lob_snapshots' in tables:
                # Check for missing data gaps
                cursor.execute("""
                    WITH time_gaps AS (
                        SELECT 
                            exchange,
                            trading_pair,
                            timestamp,
                            LAG(timestamp) OVER (
                                PARTITION BY exchange, trading_pair 
                                ORDER BY timestamp
                            ) as prev_timestamp
                        FROM lob_snapshots
                    ),
                    large_gaps AS (
                        SELECT 
                            exchange,
                            trading_pair,
                            timestamp - prev_timestamp as gap_duration
                        FROM time_gaps
                        WHERE prev_timestamp IS NOT NULL
                        AND timestamp - prev_timestamp > INTERVAL '1 minute'
                    )
                    SELECT 
                        exchange,
                        trading_pair,
                        COUNT(*) as gap_count,
                        AVG(EXTRACT(EPOCH FROM gap_duration)/60) as avg_gap_minutes
                    FROM large_gaps
                    GROUP BY exchange, trading_pair
                    ORDER BY gap_count DESC
                """)
                
                gap_results = cursor.fetchall()
                if gap_results:
                    print("Data gaps > 1 minute:")
                    print(f"{'Exchange':<15} {'Pair':<15} {'Gap Count':<12} {'Avg Gap (min)':<15}")
                    print(f"{'-'*15} {'-'*15} {'-'*12} {'-'*15}")
                    for row in gap_results:
                        exchange, pair, gap_count, avg_gap = row
                        print(f"{exchange:<15} {pair:<15} {gap_count:<12} {avg_gap:<15.1f}")
                else:
                    print("No significant data gaps found (>1 minute)")
            
    except Exception as e:
        print(f"Error querying database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    query_data_coverage() 