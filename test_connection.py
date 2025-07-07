#!/usr/bin/env python3
import psycopg2
import signal
import sys

def timeout_handler(signum, frame):
    print("❌ Connection test timed out after 10 seconds")
    sys.exit(1)

# Set up timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10 second timeout

try:
    print("Testing database connection...")
    conn = psycopg2.connect(
        host='localhost', 
        port=5433, 
        user='backtest_user', 
        password='backtest_password', 
        database='backtest_db',
        connect_timeout=5
    )
    print('✅ Connection successful!')
    
    cursor = conn.cursor()
    cursor.execute('SELECT 1;')
    result = cursor.fetchone()
    print(f'✅ Basic query works! Result: {result}')
    
    # Try to check if the table exists
    cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'lob_snapshots');")
    table_exists = cursor.fetchone()[0]
    print(f'✅ lob_snapshots table exists: {table_exists}')
    
    conn.close()
    print('✅ All tests passed!')
    
except Exception as e:
    print(f'❌ Connection failed: {e}')
    sys.exit(1)
finally:
    signal.alarm(0)  # Disable the alarm 