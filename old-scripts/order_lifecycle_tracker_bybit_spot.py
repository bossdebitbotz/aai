#!/usr/bin/env python3
"""
Order Lifecycle Tracker for WLD-USDT (Bybit Spot)

This script tracks the lifecycle of orders on the WLD-USDT market on Bybit Spot
by monitoring the order book in real-time using WebSockets.
"""

import asyncio
import json
import time
import signal
import sys
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import websockets
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("order_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("order_tracker")

# Bybit Spot WebSocket URL (v5)
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/spot"

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5439,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Global flag for controlling continuous execution
running = True

# Global order book state
current_order_book = {
    'bids': {},  # price -> {quantity, orders: [order_id, ...]}
    'asks': {}   # price -> {quantity, orders: [order_id, ...]}
}

# Last update timestamp for Bybit
last_update_ts = 0

def handle_exit_signal(sig, frame):
    """Handle exit signals gracefully"""
    global running
    logger.info("Shutting down gracefully... (This may take a few seconds)")
    running = False

# Register signal handlers
signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)

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
        logger.error(f"Error connecting to database: {type(e).__name__} - {str(e)}")
        return None

def setup_database():
    """Set up the database tables for order lifecycle tracking."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Create order_events table to track lifecycle events
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_events (
                    id SERIAL PRIMARY KEY,
                    trading_pair VARCHAR(20) NOT NULL,
                    synthetic_order_id VARCHAR(64) NOT NULL,
                    event_type VARCHAR(20) NOT NULL,  -- 'NEW', 'MODIFY', 'CANCEL', 'FILL'
                    timestamp TIMESTAMP NOT NULL,
                    price DECIMAL(18, 8) NOT NULL,
                    quantity DECIMAL(18, 8) NOT NULL,
                    side VARCHAR(4) NOT NULL,  -- 'BID' or 'ASK'
                    remaining_quantity DECIMAL(18, 8)
                );
            ''')
            
            # Create index on order_events
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS order_events_order_id_idx 
                ON order_events (synthetic_order_id);
            ''')
            
            # Create index on timestamp
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS order_events_timestamp_idx 
                ON order_events (timestamp);
            ''')
            
            # Create active_orders table to track currently active orders
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_orders (
                    synthetic_order_id VARCHAR(64) PRIMARY KEY,
                    trading_pair VARCHAR(20) NOT NULL,
                    first_seen TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP NOT NULL,
                    price DECIMAL(18, 8) NOT NULL,
                    quantity DECIMAL(18, 8) NOT NULL,
                    side VARCHAR(4) NOT NULL,  -- 'BID' or 'ASK'
                    status VARCHAR(20) NOT NULL  -- 'ACTIVE', 'CANCELED', 'FILLED', 'PARTIALLY_FILLED'
                );
            ''')
            
            # Create index on active_orders
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS active_orders_pair_price_idx 
                ON active_orders (trading_pair, price);
            ''')
            
        conn.commit()
        logger.info("Database setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def generate_synthetic_order_id(timestamp, price, quantity, side):
    """
    Generate a synthetic order ID based on available information.
    This is not a real order ID but a best-effort attempt at uniquely identifying an order.
    """
    id_string = f"{timestamp}_{price}_{quantity}_{side}_{np.random.randint(0, 1000000)}"
    return hashlib.md5(id_string.encode()).hexdigest()

def save_order_events(events):
    """Save order lifecycle events to the database."""
    if not events:
        return True
        
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Insert order events in bulk
            values = []
            for event in events:
                values.append((
                    event['trading_pair'],
                    event['synthetic_order_id'],
                    event['event_type'],
                    event['timestamp'],
                    event['price'],
                    event['quantity'],
                    event['side'],
                    event.get('remaining_quantity')
                ))
            
            # Execute bulk insert
            execute_values(
                cursor,
                """
                INSERT INTO order_events 
                (trading_pair, synthetic_order_id, event_type, timestamp, price, quantity, side, remaining_quantity)
                VALUES %s
                """,
                values
            )
            
            # Update active_orders based on events
            for event in events:
                if event['event_type'] == 'NEW':
                    # Insert new order
                    cursor.execute("""
                        INSERT INTO active_orders
                        (synthetic_order_id, trading_pair, first_seen, last_updated, price, quantity, side, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (synthetic_order_id) DO NOTHING
                    """, (
                        event['synthetic_order_id'],
                        event['trading_pair'],
                        event['timestamp'],
                        event['timestamp'],
                        event['price'],
                        event['quantity'],
                        event['side'],
                        'ACTIVE'
                    ))
                
                elif event['event_type'] == 'MODIFY':
                    # Update existing order
                    cursor.execute("""
                        UPDATE active_orders
                        SET last_updated = %s, quantity = %s, status = 'ACTIVE'
                        WHERE synthetic_order_id = %s
                    """, (
                        event['timestamp'],
                        event['quantity'],
                        event['synthetic_order_id']
                    ))
                
                elif event['event_type'] in ('CANCEL', 'FILL'):
                    # Update existing order status
                    status = 'CANCELED' if event['event_type'] == 'CANCEL' else 'FILLED'
                    cursor.execute("""
                        UPDATE active_orders
                        SET last_updated = %s, status = %s
                        WHERE synthetic_order_id = %s
                    """, (
                        event['timestamp'],
                        status,
                        event['synthetic_order_id']
                    ))
            
            conn.commit()
            logger.info(f"Saved {len(events)} order events to database")
            return True
            
    except Exception as e:
        logger.error(f"Error saving order events to database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def detect_order_changes(old_book, new_book, timestamp, trading_pair):
    """
    Detect changes between two order book snapshots and generate order events.
    
    This function tries to infer order lifecycle events by comparing order book states:
    - New price levels are considered new orders
    - Increased quantity at a price level may be new orders or modified orders
    - Decreased quantity at a price level may be cancellations or fills
    - Disappeared price levels may be cancellations or fills
    
    Returns a list of inferred order events.
    """
    events = []
    
    # Process both sides (bids and asks)
    for side in ['bids', 'asks']:
        side_str = 'BID' if side == 'bids' else 'ASK'
        old_levels = old_book.get(side, {})
        new_levels = new_book.get(side, {})
        
        # Check for new and modified orders
        for price, new_data in new_levels.items():
            new_qty = float(new_data['quantity'])
            
            if price not in old_levels:
                # New price level - add as new order
                synthetic_id = generate_synthetic_order_id(timestamp, price, new_qty, side_str)
                events.append({
                    'trading_pair': trading_pair,
                    'synthetic_order_id': synthetic_id,
                    'event_type': 'NEW',
                    'timestamp': timestamp,
                    'price': float(price),
                    'quantity': new_qty,
                    'side': side_str,
                    'remaining_quantity': new_qty
                })
            else:
                old_qty = float(old_levels[price]['quantity'])
                
                if new_qty > old_qty:
                    # Quantity increased - consider as new order
                    delta_qty = new_qty - old_qty
                    synthetic_id = generate_synthetic_order_id(timestamp, price, delta_qty, side_str)
                    events.append({
                        'trading_pair': trading_pair,
                        'synthetic_order_id': synthetic_id,
                        'event_type': 'NEW',
                        'timestamp': timestamp,
                        'price': float(price),
                        'quantity': delta_qty,
                        'side': side_str,
                        'remaining_quantity': delta_qty
                    })
                elif new_qty < old_qty:
                    # Quantity decreased - could be cancel or fill
                    # We can't know for sure if it was filled or canceled without matching trades
                    # For simplicity, we'll mark it as cancel here
                    delta_qty = old_qty - new_qty
                    
                    # Get existing orders at this price level that are still active
                    conn = get_db_connection()
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT synthetic_order_id, quantity
                            FROM active_orders
                            WHERE trading_pair = %s AND price = %s AND side = %s AND status = 'ACTIVE'
                            ORDER BY first_seen ASC
                        """, (trading_pair, float(price), side_str))
                        
                        active_orders = cursor.fetchall()
                    
                    if conn:
                        conn.close()
                    
                    # If we have active orders to attribute this change to
                    remaining_delta = float(delta_qty)
                    for order_id, order_qty in active_orders:
                        if remaining_delta <= 0:
                            break
                            
                        # Convert order_qty to float if it's a Decimal
                        order_qty_float = float(order_qty)
                        
                        if remaining_delta >= order_qty_float:
                            # Full cancellation of this order
                            events.append({
                                'trading_pair': trading_pair,
                                'synthetic_order_id': order_id,
                                'event_type': 'CANCEL',
                                'timestamp': timestamp,
                                'price': float(price),
                                'quantity': order_qty_float,
                                'side': side_str,
                                'remaining_quantity': 0
                            })
                            remaining_delta -= order_qty_float
                        else:
                            # Partial cancellation
                            events.append({
                                'trading_pair': trading_pair,
                                'synthetic_order_id': order_id,
                                'event_type': 'MODIFY',
                                'timestamp': timestamp,
                                'price': float(price),
                                'quantity': order_qty_float - remaining_delta,
                                'side': side_str,
                                'remaining_quantity': order_qty_float - remaining_delta
                            })
                            remaining_delta = 0
        
        # Check for completely removed price levels
        for price, old_data in old_levels.items():
            if price not in new_levels:
                # Price level disappeared - all orders canceled or filled
                conn = get_db_connection()
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT synthetic_order_id, quantity
                        FROM active_orders
                        WHERE trading_pair = %s AND price = %s AND side = %s AND status = 'ACTIVE'
                    """, (trading_pair, float(price), side_str))
                    
                    active_orders = cursor.fetchall()
                
                if conn:
                    conn.close()
                
                # Mark all active orders at this price as canceled
                for order_id, order_qty in active_orders:
                    events.append({
                        'trading_pair': trading_pair,
                        'synthetic_order_id': order_id,
                        'event_type': 'CANCEL',
                        'timestamp': timestamp,
                        'price': float(price),
                        'quantity': float(order_qty),
                        'side': side_str,
                        'remaining_quantity': 0
                    })
    
    return events

# Bybit-specific: parse Bybit REST snapshot format
def parse_bybit_snapshot(snapshot):
    # snapshot: {'s': 'WLDUSDT', 'b': [[price, qty], ...], 'a': [[price, qty], ...]}
    book = {'bids': {}, 'asks': {}}
    for price, qty in snapshot.get('b', []):
        price_f = float(price)
        qty_f = float(qty)
        if qty_f > 0:
            book['bids'][str(price_f)] = {'quantity': qty_f}
    for price, qty in snapshot.get('a', []):
        price_f = float(price)
        qty_f = float(qty)
        if qty_f > 0:
            book['asks'][str(price_f)] = {'quantity': qty_f}
    return book

# Bybit-specific: update order book with snapshot or delta
def update_order_book_bybit(data, trading_pair, is_snapshot):
    global current_order_book, last_update_ts
    old_book = {
        'bids': current_order_book['bids'].copy(),
        'asks': current_order_book['asks'].copy()
    }
    if is_snapshot:
        # Full snapshot
        current_order_book = parse_bybit_snapshot(data)
    else:
        # Delta update: only update changed price levels
        for price, qty in data.get('b', []):
            price_f = float(price)
            qty_f = float(qty)
            price_str = str(price_f)
            if qty_f == 0:
                current_order_book['bids'].pop(price_str, None)
            else:
                current_order_book['bids'][price_str] = {'quantity': qty_f}
        for price, qty in data.get('a', []):
            price_f = float(price)
            qty_f = float(qty)
            price_str = str(price_f)
            if qty_f == 0:
                current_order_book['asks'].pop(price_str, None)
            else:
                current_order_book['asks'][price_str] = {'quantity': qty_f}
    # Bybit provides 'ts' (timestamp in ms)
    if 'ts' in data:
        last_update_ts = data['ts']
    timestamp = datetime.now()
    events = detect_order_changes(old_book, current_order_book, timestamp, trading_pair)
    if events:
        save_order_events(events)
    return len(events)

async def get_initial_order_book(trading_pair, depth=200):
    """Get initial order book depth from REST API (Bybit Spot v5)."""
    symbol = trading_pair.replace('-', '')  # e.g., BTCUSDT
    url = f"https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}&limit={depth}"
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error response (status {response.status}) from Bybit Spot v5")
                    return None
                data = await response.json()
        # Bybit v5 REST returns {'retCode':0, 'result':{'a': [...], 'b': [...], ...}}
        if data.get('retCode') == 0 and 'result' in data:
            result = data['result']
            # v5 returns 'a' and 'b' arrays directly in result
            return {'a': result.get('a', []), 'b': result.get('b', [])}
        else:
            logger.error(f"Unexpected Bybit v5 REST response: {data}")
            return None
    except Exception as e:
        logger.error(f"Error fetching initial order book: {type(e).__name__} - {str(e)}")
        return None

async def websocket_manager(trading_pair):
    global running
    symbol = trading_pair.replace('-', '')  # e.g., BTCUSDT
    payload = {
        "op": "subscribe",
        "args": [f"orderbook.200.{symbol}"]
    }
    retry_count = 0
    while running and retry_count < 5:
        try:
            logger.info(f"Connecting to Bybit WebSocket for {trading_pair}...")
            # Fetch initial order book
            logger.info("Fetching initial order book...")
            snapshot = await get_initial_order_book(trading_pair)
            if snapshot:
                update_order_book_bybit(snapshot, trading_pair, is_snapshot=True)
                logger.info(f"Order book initialized with {len(current_order_book['bids'])} bids and {len(current_order_book['asks'])} asks")
            else:
                logger.error("Failed to initialize order book, retrying...")
                await asyncio.sleep(5)
                retry_count += 1
                continue
            async with websockets.connect(BYBIT_WS_URL) as websocket:
                await websocket.send(json.dumps(payload))
                events_processed = 0
                last_status_time = time.time()
                while running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=10)
                        data = json.loads(message)
                        # Log v5 subscription confirmation
                        if data.get("op") == "subscribe" and data.get("success") is True:
                            logger.info(f"WebSocket subscription confirmed: {data}")
                            continue
                        # Bybit v5 sends order book updates with topic 'orderbook.200.SYMBOL'
                        if data.get("topic", "").startswith("orderbook."):
                            if data.get("type") == "snapshot":
                                events = update_order_book_bybit(data["data"], trading_pair, is_snapshot=True)
                                events_processed += events
                            elif data.get("type") == "delta":
                                events = update_order_book_bybit(data["data"], trading_pair, is_snapshot=False)
                                events_processed += events
                            # Log status periodically
                            current_time = time.time()
                            if current_time - last_status_time >= 60:
                                logger.info(f"Processed {events_processed} order events in the last minute")
                                events_processed = 0
                                last_status_time = current_time
                                bid_count = len(current_order_book['bids'])
                                ask_count = len(current_order_book['asks'])
                                logger.info(f"Current order book: {bid_count} bids, {ask_count} asks")
                    except asyncio.TimeoutError:
                        logger.warning("WebSocket receive timeout, checking connection...")
                        try:
                            pong = await websocket.ping()
                            await asyncio.wait_for(pong, timeout=5)
                            logger.info("Connection still alive")
                        except:
                            logger.warning("WebSocket connection lost, reconnecting...")
                            break
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {type(e).__name__} - {str(e)}")
                        await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"WebSocket connection error: {type(e).__name__} - {str(e)}")
            retry_count += 1
            await asyncio.sleep(5)
    logger.info("WebSocket manager stopped")

def generate_order_lifecycle_report():
    """Generate a report of order lifecycles from the database."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Get statistics on order events
            cursor.execute("""
                SELECT event_type, COUNT(*) as event_count
                FROM order_events
                GROUP BY event_type
                ORDER BY event_type
            """)
            
            event_stats = cursor.fetchall()
            
            logger.info("Order Event Statistics:")
            for event_type, count in event_stats:
                logger.info(f"{event_type}: {count}")
            
            # Get average lifetime of orders
            cursor.execute("""
                SELECT AVG(EXTRACT(EPOCH FROM (last_updated - first_seen))) as avg_lifetime_seconds
                FROM active_orders
                WHERE status IN ('CANCELED', 'FILLED')
            """)
            
            avg_lifetime = cursor.fetchone()[0]
            if avg_lifetime:
                logger.info(f"Average order lifetime: {avg_lifetime:.2f} seconds")
            
            # Get distribution of order statuses
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM active_orders
                GROUP BY status
            """)
            
            status_counts = cursor.fetchall()
            
            logger.info("Order Status Distribution:")
            for status, count in status_counts:
                logger.info(f"{status}: {count}")
            
            # Save report to CSV
            cursor.execute("""
                SELECT 
                    synthetic_order_id,
                    side,
                    price,
                    first_seen,
                    last_updated,
                    EXTRACT(EPOCH FROM (last_updated - first_seen)) as lifetime_seconds,
                    status
                FROM active_orders
                ORDER BY first_seen DESC
                LIMIT 1000
            """)
            
            columns = [desc[0] for desc in cursor.description]
            orders_data = cursor.fetchall()
            
            report_df = pd.DataFrame(orders_data, columns=columns)
            report_df.to_csv("order_lifecycle_report.csv", index=False)
            logger.info("Order lifecycle report saved to order_lifecycle_report.csv")
            
    except Exception as e:
        logger.error(f"Error generating order lifecycle report: {e}")
    finally:
        if conn:
            conn.close()

async def main():
    """Main entry point"""
    try:
        # Set up the database
        setup_successful = setup_database()
        if not setup_successful:
            logger.error("Failed to set up database. Exiting.")
            return
        
        logger.info("Starting order lifecycle tracking for WLD-USDT on Bybit Spot")
        logger.info("Press Ctrl+C to stop the script")
        
        # Main task is the WebSocket manager
        trading_pair = "WLDUSDT"
        await websocket_manager(trading_pair)
        
    except Exception as e:
        logger.error(f"Error in main function: {type(e).__name__} - {str(e)}")
    finally:
        # Generate a report before exiting
        generate_order_lifecycle_report()
        logger.info("Program exited.")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 