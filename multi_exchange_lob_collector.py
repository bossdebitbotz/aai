#!/usr/bin/env python3
"""
Multi-Exchange LOB Data Collector

This script collects and processes limit order book (LOB) data from:
- Binance Spot
- Binance Perpetual
- Bybit Spot

The collected data follows the structure required for training the 
attention-based LOB forecasting model as specified in the documentation.
"""

import asyncio
import json
import time
import signal
import sys
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import websockets
import logging
import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq
import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lob_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lob_collector")

# WebSocket URLs
BINANCE_SPOT_WS_URL = "wss://stream.binance.com:9443/ws"
BINANCE_PERP_WS_URL = "wss://fstream.binance.com/ws"
BYBIT_SPOT_WS_URL = "wss://stream.bybit.com/v5/public/spot"

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Global flag for controlling continuous execution
running = True

# Storage paths
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Exchange-specific directories
for exchange in ["binance_spot", "binance_perp", "bybit_spot"]:
    exchange_dir = os.path.join(RAW_DATA_DIR, exchange)
    if not os.path.exists(exchange_dir):
        os.makedirs(exchange_dir)

# Global order book state for each exchange and trading pair
order_books = {}

# Define number of order book levels to collect
LOB_LEVELS = 5

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

def get_current_timestamp():
    """Get current UTC timestamp"""
    return datetime.now(timezone.utc)

def setup_database():
    """Set up the database tables for LOB data."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Create lob_snapshots table with TIMESTAMPTZ
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS lob_snapshots (
                    id SERIAL PRIMARY KEY,
                    exchange VARCHAR(20) NOT NULL,
                    trading_pair VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    data JSONB NOT NULL,
                    UNIQUE (exchange, trading_pair, timestamp)
                );
            ''')
            
            # Create index on lob_snapshots
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS lob_snapshots_idx 
                ON lob_snapshots (exchange, trading_pair, timestamp);
            ''')
            
            # Create lob_metrics table with TIMESTAMPTZ
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS lob_metrics (
                    id SERIAL PRIMARY KEY,
                    exchange VARCHAR(20) NOT NULL,
                    trading_pair VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    mid_price DECIMAL(18, 8) NOT NULL,
                    spread DECIMAL(18, 8) NOT NULL,
                    bid_volume_total DECIMAL(18, 8) NOT NULL,
                    ask_volume_total DECIMAL(18, 8) NOT NULL,
                    volume_imbalance DECIMAL(18, 8) NOT NULL,
                    price_imbalance DECIMAL(18, 8) NOT NULL,
                    UNIQUE (exchange, trading_pair, timestamp)
                );
            ''')
            
            # Create index on lob_metrics
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS lob_metrics_idx 
                ON lob_metrics (exchange, trading_pair, timestamp);
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

def save_lob_snapshot(exchange, trading_pair, timestamp, data):
    """Save LOB snapshot to the database."""
    try:
        # Ensure timestamp is UTC
        if not timestamp.tzinfo:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO lob_snapshots (exchange, trading_pair, timestamp, data)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (exchange, trading_pair, timestamp) DO UPDATE 
                SET data = EXCLUDED.data
            """, (
                exchange,
                trading_pair,
                timestamp,
                json.dumps(data)
            ))
            
            # Calculate and save metrics
            if 'bids' in data and 'asks' in data and len(data['bids']) > 0 and len(data['asks']) > 0:
                # Get best bid and ask
                best_bid = float(data['bids'][0][0])
                best_ask = float(data['asks'][0][0])
                
                # Calculate metrics
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                
                # Sum volumes
                bid_volume = sum(float(bid[1]) for bid in data['bids'])
                ask_volume = sum(float(ask[1]) for ask in data['asks'])
                
                # Calculate imbalances
                volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
                price_imbalance = (best_ask - best_bid) / (best_ask + best_bid) if (best_ask + best_bid) > 0 else 0
                
                # Save metrics
                cursor.execute("""
                    INSERT INTO lob_metrics 
                    (exchange, trading_pair, timestamp, mid_price, spread, 
                     bid_volume_total, ask_volume_total, volume_imbalance, price_imbalance)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (exchange, trading_pair, timestamp) DO UPDATE 
                    SET mid_price = EXCLUDED.mid_price,
                        spread = EXCLUDED.spread,
                        bid_volume_total = EXCLUDED.bid_volume_total,
                        ask_volume_total = EXCLUDED.ask_volume_total,
                        volume_imbalance = EXCLUDED.volume_imbalance,
                        price_imbalance = EXCLUDED.price_imbalance
                """, (
                    exchange,
                    trading_pair,
                    timestamp,
                    mid_price,
                    spread,
                    bid_volume,
                    ask_volume,
                    volume_imbalance,
                    price_imbalance
                ))
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"Error saving LOB snapshot to database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_lob_to_parquet(exchange, trading_pair, timestamp, data):
    """Save LOB data to Parquet file with appropriate partitioning."""
    try:
        # Ensure timestamp is UTC and convert to milliseconds
        if not timestamp.tzinfo:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        ts_ms = int(timestamp.timestamp() * 1000)
        
        # Create date-based directory structure: exchange/symbol/YYYY-MM-DD/
        date_str = timestamp.strftime("%Y-%m-%d")
        hour_str = timestamp.strftime("%H")
        dir_path = os.path.join(RAW_DATA_DIR, exchange, trading_pair, date_str)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        # Prepare data for Parquet
        lob_data = {
            'timestamp': [ts_ms],  # UTC milliseconds
            'exchange': [exchange],
            'trading_pair': [trading_pair]
        }
        
        # Add bid data with default values
        for i in range(1, LOB_LEVELS + 1):
            lob_data[f'bid_price_{i}'] = [0.0]
            lob_data[f'bid_volume_{i}'] = [0.0]
            
        # Add ask data with default values
        for i in range(1, LOB_LEVELS + 1):
            lob_data[f'ask_price_{i}'] = [0.0]
            lob_data[f'ask_volume_{i}'] = [0.0]
        
        # Fill in actual bid data
        for i, (price, qty) in enumerate(data.get('bids', [])[:LOB_LEVELS], 1):
            lob_data[f'bid_price_{i}'][0] = float(price)
            lob_data[f'bid_volume_{i}'][0] = float(qty)
            
        # Fill in actual ask data
        for i, (price, qty) in enumerate(data.get('asks', [])[:LOB_LEVELS], 1):
            lob_data[f'ask_price_{i}'][0] = float(price)
            lob_data[f'ask_volume_{i}'][0] = float(qty)
            
        # Convert to pandas DataFrame
        df = pd.DataFrame(lob_data)
        
        # Define file path
        file_path = os.path.join(dir_path, f"{exchange}_{trading_pair}_{date_str}_{hour_str}.parquet")
        temp_file_path = file_path + '.tmp'
        
        # Write to temporary file first
        if os.path.exists(file_path):
            # Read existing file
            try:
                existing_df = pd.read_parquet(file_path)
                # Concatenate and sort by timestamp
                df = pd.concat([existing_df, df]).sort_values('timestamp').reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Could not read existing Parquet file {file_path}, creating new file. Error: {e}")
        
        # Write to temporary file
        df.to_parquet(temp_file_path, engine='pyarrow', compression='snappy', index=False)
        
        # Atomic rename
        os.replace(temp_file_path, file_path)
            
        return True
    except Exception as e:
        logger.error(f"Error saving LOB data to Parquet: {e}")
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        return False

def standardize_trading_pair(exchange, trading_pair):
    """Standardize trading pair format across exchanges."""
    if exchange == 'binance_spot' or exchange == 'binance_perp':
        # Binance format: BTCUSDT
        if '-' in trading_pair:
            return trading_pair.replace('-', '')
        return trading_pair
    elif exchange == 'bybit_spot':
        # Bybit format: BTCUSDT
        if '-' in trading_pair:
            return trading_pair.replace('-', '')
        return trading_pair
    return trading_pair

# Exchange-specific functions for Binance Spot
async def get_binance_spot_orderbook(trading_pair, depth=1000):
    """Get initial order book from Binance Spot REST API."""
    symbol = standardize_trading_pair('binance_spot', trading_pair)
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={depth}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error response (status {response.status}) from Binance Spot")
                    return None
                
                data = await response.json()
        
        logger.info(f"Initial Binance Spot order book fetched for {trading_pair}")
        return data
    
    except Exception as e:
        logger.error(f"Error fetching Binance Spot order book: {type(e).__name__} - {str(e)}")
        return None

def process_binance_spot_orderbook(data, trading_pair):
    """Process Binance Spot order book data."""
    # Use exchange timestamp if available, otherwise use current UTC time
    timestamp = datetime.fromtimestamp(data.get('E', time.time() * 1000) / 1000, tz=timezone.utc)
    
    # Initialize or get existing order book for this trading pair
    if f"binance_spot_{trading_pair}" not in order_books:
        order_books[f"binance_spot_{trading_pair}"] = {
            'lastUpdateId': 0,
            'bids': [],
            'asks': []
        }
    
    # Update last update ID
    if 'lastUpdateId' in data:
        order_books[f"binance_spot_{trading_pair}"]['lastUpdateId'] = data['lastUpdateId']
    
    # Update bids and asks
    order_books[f"binance_spot_{trading_pair}"]['bids'] = data.get('bids', [])[:LOB_LEVELS]
    order_books[f"binance_spot_{trading_pair}"]['asks'] = data.get('asks', [])[:LOB_LEVELS]
    
    # Save to database and Parquet
    save_lob_snapshot('binance_spot', trading_pair, timestamp, order_books[f"binance_spot_{trading_pair}"])
    save_lob_to_parquet('binance_spot', trading_pair, timestamp, order_books[f"binance_spot_{trading_pair}"])
    
    return True

async def binance_spot_websocket_manager(trading_pairs):
    """Manage WebSocket connection for Binance Spot."""
    global running
    
    # Prepare subscription payload
    streams = []
    for pair in trading_pairs:
        symbol = standardize_trading_pair('binance_spot', pair).lower()
        streams.append(f"{symbol}@depth")
    
    payload = {
        "method": "SUBSCRIBE",
        "params": streams,
        "id": 1
    }
    
    retry_count = 0
    max_retries = 10
    base_delay = 5
    max_delay = 300  # 5 minutes
    
    while running and retry_count < max_retries:
        try:
            logger.info(f"Connecting to Binance Spot WebSocket for {', '.join(trading_pairs)}...")
            
            # Initialize order books
            for pair in trading_pairs:
                logger.info(f"Initializing order book for Binance Spot {pair}...")
                snapshot = await get_binance_spot_orderbook(pair)
                if snapshot:
                    process_binance_spot_orderbook(snapshot, pair)
                else:
                    logger.error(f"Failed to initialize order book for Binance Spot {pair}")
            
            # Connect to WebSocket
            async with websockets.connect(
                BINANCE_SPOT_WS_URL,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=30,   # Wait 30 seconds for pong response
                close_timeout=10   # Wait 10 seconds for close response
            ) as websocket:
                # Reset retry count on successful connection
                retry_count = 0
                
                # Subscribe to streams
                await websocket.send(json.dumps(payload))
                
                # Process responses
                while running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)
                        
                        # Handle subscription confirmation
                        if 'result' in data:
                            continue
                        
                        # Handle depth update
                        if 'e' in data and data['e'] == 'depthUpdate':
                            # Extract trading pair from stream name
                            if 's' in data:
                                pair = data['s']
                                # Find matching trading pair from our list
                                matching_pair = next((p for p in trading_pairs if standardize_trading_pair('binance_spot', p) == pair), None)
                                
                                if matching_pair:
                                    # Create snapshot for the update
                                    update = {
                                        'lastUpdateId': data['u'],
                                        'bids': data['b'],
                                        'asks': data['a']
                                    }
                                    process_binance_spot_orderbook(update, matching_pair)
                            
                    except asyncio.TimeoutError:
                        # Send ping to check connection
                        try:
                            pong = await websocket.ping()
                            await asyncio.wait_for(pong, timeout=10)
                            logger.info("Binance Spot WebSocket connection alive")
                        except:
                            logger.warning("Binance Spot WebSocket connection lost, reconnecting...")
                            break
                    except websockets.exceptions.ConnectionClosedError as e:
                        logger.warning(f"Binance Spot WebSocket connection closed: {e}")
                        # Exit the inner loop to trigger reconnection
                        break
                    except Exception as e:
                        logger.error(f"Error processing Binance Spot WebSocket message: {type(e).__name__} - {str(e)}")
                        if isinstance(e, websockets.exceptions.WebSocketException):
                            break
        
        except Exception as e:
            logger.error(f"Binance Spot WebSocket connection error: {type(e).__name__} - {str(e)}")
            retry_count += 1
            # Calculate exponential backoff delay with jitter
            delay = min(max_delay, base_delay * (2 ** (retry_count - 1))) * (0.8 + 0.4 * random.random())
            logger.info(f"Reconnecting in {delay:.2f} seconds (retry {retry_count}/{max_retries})...")
            await asyncio.sleep(delay)
    
    if retry_count >= max_retries:
        logger.error(f"Max retries reached for Binance Spot WebSocket. Giving up.")
    
    logger.info("Binance Spot WebSocket manager stopped")

# Exchange-specific functions for Binance Perpetual
async def get_binance_perp_orderbook(trading_pair, depth=1000):
    """Get initial order book from Binance Perpetual REST API."""
    symbol = standardize_trading_pair('binance_perp', trading_pair)
    url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit={depth}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error response (status {response.status}) from Binance Perpetual")
                    return None
                
                data = await response.json()
        
        logger.info(f"Initial Binance Perpetual order book fetched for {trading_pair}")
        return data
    
    except Exception as e:
        logger.error(f"Error fetching Binance Perpetual order book: {type(e).__name__} - {str(e)}")
        return None

def process_binance_perp_orderbook(data, trading_pair):
    """Process Binance Perpetual order book data."""
    # Use exchange timestamp if available, otherwise use current UTC time
    timestamp = datetime.fromtimestamp(data.get('E', time.time() * 1000) / 1000, tz=timezone.utc)
    
    # Initialize or get existing order book for this trading pair
    if f"binance_perp_{trading_pair}" not in order_books:
        order_books[f"binance_perp_{trading_pair}"] = {
            'lastUpdateId': 0,
            'bids': [],
            'asks': []
        }
    
    # Update last update ID
    if 'lastUpdateId' in data:
        order_books[f"binance_perp_{trading_pair}"]['lastUpdateId'] = data['lastUpdateId']
    
    # Update bids and asks
    order_books[f"binance_perp_{trading_pair}"]['bids'] = data.get('bids', [])[:LOB_LEVELS]
    order_books[f"binance_perp_{trading_pair}"]['asks'] = data.get('asks', [])[:LOB_LEVELS]
    
    # Save to database and Parquet
    save_lob_snapshot('binance_perp', trading_pair, timestamp, order_books[f"binance_perp_{trading_pair}"])
    save_lob_to_parquet('binance_perp', trading_pair, timestamp, order_books[f"binance_perp_{trading_pair}"])
    
    return True

async def binance_perp_websocket_manager(trading_pairs):
    """Manage WebSocket connection for Binance Perpetual."""
    global running
    
    # Prepare subscription payload
    streams = []
    for pair in trading_pairs:
        symbol = standardize_trading_pair('binance_perp', pair).lower()
        streams.append(f"{symbol}@depth")
    
    payload = {
        "method": "SUBSCRIBE",
        "params": streams,
        "id": 1
    }
    
    retry_count = 0
    max_retries = 10
    base_delay = 5
    max_delay = 300  # 5 minutes
    
    while running and retry_count < max_retries:
        try:
            logger.info(f"Connecting to Binance Perpetual WebSocket for {', '.join(trading_pairs)}...")
            
            # Initialize order books
            for pair in trading_pairs:
                logger.info(f"Initializing order book for Binance Perpetual {pair}...")
                snapshot = await get_binance_perp_orderbook(pair)
                if snapshot:
                    process_binance_perp_orderbook(snapshot, pair)
                else:
                    logger.error(f"Failed to initialize order book for Binance Perpetual {pair}")
            
            # Connect to WebSocket
            async with websockets.connect(
                BINANCE_PERP_WS_URL,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=30,   # Wait 30 seconds for pong response
                close_timeout=10   # Wait 10 seconds for close response
            ) as websocket:
                # Reset retry count on successful connection
                retry_count = 0
                
                # Subscribe to streams
                await websocket.send(json.dumps(payload))
                
                # Process responses
                while running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)
                        
                        # Handle subscription confirmation
                        if 'result' in data:
                            continue
                        
                        # Handle depth update
                        if 'e' in data and data['e'] == 'depthUpdate':
                            # Extract trading pair from stream name
                            if 's' in data:
                                pair = data['s']
                                # Find matching trading pair from our list
                                matching_pair = next((p for p in trading_pairs if standardize_trading_pair('binance_perp', p) == pair), None)
                                
                                if matching_pair:
                                    # Create snapshot for the update
                                    update = {
                                        'lastUpdateId': data['u'],
                                        'bids': data['b'],
                                        'asks': data['a']
                                    }
                                    process_binance_perp_orderbook(update, matching_pair)
                            
                    except asyncio.TimeoutError:
                        # Send ping to check connection
                        try:
                            pong = await websocket.ping()
                            await asyncio.wait_for(pong, timeout=10)
                            logger.info("Binance Perpetual WebSocket connection alive")
                        except:
                            logger.warning("Binance Perpetual WebSocket connection lost, reconnecting...")
                            break
                    except websockets.exceptions.ConnectionClosedError as e:
                        logger.warning(f"Binance Perpetual WebSocket connection closed: {e}")
                        # Exit the inner loop to trigger reconnection
                        break
                    except Exception as e:
                        logger.error(f"Error processing Binance Perpetual WebSocket message: {type(e).__name__} - {str(e)}")
                        if isinstance(e, websockets.exceptions.WebSocketException):
                            break
        
        except Exception as e:
            logger.error(f"Binance Perpetual WebSocket connection error: {type(e).__name__} - {str(e)}")
            retry_count += 1
            # Calculate exponential backoff delay with jitter
            delay = min(max_delay, base_delay * (2 ** (retry_count - 1))) * (0.8 + 0.4 * random.random())
            logger.info(f"Reconnecting in {delay:.2f} seconds (retry {retry_count}/{max_retries})...")
            await asyncio.sleep(delay)
    
    if retry_count >= max_retries:
        logger.error(f"Max retries reached for Binance Perpetual WebSocket. Giving up.")
    
    logger.info("Binance Perpetual WebSocket manager stopped")

# Exchange-specific functions for Bybit Spot
async def get_bybit_spot_orderbook(trading_pair, depth=1000):
    """Get initial order book from Bybit Spot REST API."""
    symbol = standardize_trading_pair('bybit_spot', trading_pair)
    url = f"https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}&limit={depth}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error response (status {response.status}) from Bybit Spot")
                    return None
                
                data = await response.json()
        
        # Bybit returns {'retCode':0, 'result':{'a': [...], 'b': [...], ...}}
        if data.get('retCode') == 0 and 'result' in data:
            result = data['result']
            logger.info(f"Initial Bybit Spot order book fetched for {trading_pair}")
            return result
        else:
            logger.error(f"Unexpected Bybit response: {data}")
            return None
    
    except Exception as e:
        logger.error(f"Error fetching Bybit Spot order book: {type(e).__name__} - {str(e)}")
        return None

def process_bybit_spot_orderbook(data, trading_pair):
    """Process Bybit Spot order book data."""
    # Use exchange timestamp if available, otherwise use current UTC time
    timestamp = datetime.fromtimestamp(data.get('ts', time.time() * 1000) / 1000, tz=timezone.utc)
    
    # Initialize or get existing order book for this trading pair
    if f"bybit_spot_{trading_pair}" not in order_books:
        order_books[f"bybit_spot_{trading_pair}"] = {
            'ts': 0,
            'bids': [],
            'asks': []
        }
    
    # Update timestamp
    if 'ts' in data:
        order_books[f"bybit_spot_{trading_pair}"]['ts'] = data['ts']
    
    # Update bids and asks
    order_books[f"bybit_spot_{trading_pair}"]['bids'] = data.get('b', [])[:LOB_LEVELS]
    order_books[f"bybit_spot_{trading_pair}"]['asks'] = data.get('a', [])[:LOB_LEVELS]
    
    # Save to database and Parquet
    save_lob_snapshot('bybit_spot', trading_pair, timestamp, order_books[f"bybit_spot_{trading_pair}"])
    save_lob_to_parquet('bybit_spot', trading_pair, timestamp, order_books[f"bybit_spot_{trading_pair}"])
    
    return True

async def bybit_spot_websocket_manager(trading_pairs):
    """Manage WebSocket connection for Bybit Spot."""
    global running
    
    # Prepare subscription payload
    args = []
    for pair in trading_pairs:
        symbol = standardize_trading_pair('bybit_spot', pair)
        args.append(f"orderbook.200.{symbol}")
    
    payload = {
        "op": "subscribe",
        "args": args
    }
    
    retry_count = 0
    while running and retry_count < 5:
        try:
            logger.info(f"Connecting to Bybit Spot WebSocket for {', '.join(trading_pairs)}...")
            
            # Initialize order books
            for pair in trading_pairs:
                logger.info(f"Initializing order book for Bybit Spot {pair}...")
                snapshot = await get_bybit_spot_orderbook(pair)
                if snapshot:
                    process_bybit_spot_orderbook(snapshot, pair)
                else:
                    logger.error(f"Failed to initialize order book for Bybit Spot {pair}")
            
            # Connect to WebSocket with ping_interval and ping_timeout
            async with websockets.connect(
                BYBIT_SPOT_WS_URL,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,   # Wait 10 seconds for pong response
                close_timeout=10   # Wait 10 seconds for close response
            ) as websocket:
                # Subscribe to streams
                await websocket.send(json.dumps(payload))
                
                # Process responses
                while running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)
                        
                        # Handle subscription confirmation
                        if data.get("op") == "subscribe" and data.get("success") is True:
                            logger.info(f"Successfully subscribed to Bybit Spot streams: {data}")
                            continue
                        
                        # Handle depth update
                        if data.get("topic", "").startswith("orderbook."):
                            # Extract trading pair from topic
                            topic_parts = data.get("topic", "").split(".")
                            if len(topic_parts) == 3:
                                pair = topic_parts[2]
                                # Find matching trading pair from our list
                                matching_pair = next((p for p in trading_pairs if standardize_trading_pair('bybit_spot', p) == pair), None)
                                
                                if matching_pair and "data" in data:
                                    if data.get("type") == "snapshot":
                                        process_bybit_spot_orderbook(data["data"], matching_pair)
                                    elif data.get("type") == "delta":
                                        process_bybit_spot_orderbook(data["data"], matching_pair)
                            
                    except asyncio.TimeoutError:
                        # The websockets library will handle ping/pong automatically
                        logger.debug("Bybit Spot WebSocket timeout, waiting for next message...")
                        continue
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"Bybit Spot WebSocket connection closed: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Error processing Bybit Spot WebSocket message: {type(e).__name__} - {str(e)}")
                        if isinstance(e, websockets.exceptions.WebSocketException):
                            break
        
        except Exception as e:
            logger.error(f"Bybit Spot WebSocket connection error: {type(e).__name__} - {str(e)}")
            retry_count += 1
            await asyncio.sleep(5)
    
    logger.info("Bybit Spot WebSocket manager stopped")

def resample_data(exchange, trading_pair, interval='5s'):
    """Resample raw LOB data to the specified interval."""
    try:
        # Get current date in UTC
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dir_path = os.path.join(RAW_DATA_DIR, exchange, trading_pair, today)
        
        if not os.path.exists(dir_path):
            logger.warning(f"No data found for {exchange} {trading_pair} on {today}")
            return False
        
        # List all parquet files for today
        parquet_files = [f for f in os.listdir(dir_path) if f.endswith('.parquet')]
        
        if not parquet_files:
            logger.warning(f"No parquet files found for {exchange} {trading_pair} on {today}")
            return False
        
        # Read and combine all parquet files
        dfs = []
        for file in parquet_files:
            file_path = os.path.join(dir_path, file)
            df = pd.read_parquet(file_path)
            # Convert millisecond timestamps to UTC datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            dfs.append(df)
        
        if not dfs:
            return False
            
        combined_df = pd.concat(dfs)
        combined_df.set_index('datetime', inplace=True)
        
        # Resample data using UTC timestamps
        resampled = combined_df.resample(interval, origin='start_day').last()
        
        # Ensure all required fields are present with forward fill
        resampled.ffill(inplace=True)
        
        # Save resampled data
        processed_dir = os.path.join(PROCESSED_DATA_DIR, exchange, trading_pair)
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
            
        output_file = os.path.join(processed_dir, f"{exchange}_{trading_pair}_{today}_resampled_{interval}.parquet")
        
        # Convert back to millisecond timestamps before saving
        resampled['timestamp'] = resampled.index.astype(np.int64) // 10**6
        resampled.to_parquet(output_file, compression='snappy')
        
        logger.info(f"Resampled data saved for {exchange} {trading_pair} to {interval}")
        return True
        
    except Exception as e:
        logger.error(f"Error resampling data for {exchange} {trading_pair}: {e}")
        return False

async def data_processor():
    """Process and resample collected data periodically."""
    global running
    
    while running:
        try:
            # Wait for 5 minutes before processing
            await asyncio.sleep(300)
            
            logger.info("Starting data processing and resampling...")
            
            # Process for each exchange and trading pair
            for key in order_books.keys():
                parts = key.split('_')
                if len(parts) >= 3:
                    exchange = parts[0] + '_' + parts[1]  # e.g., binance_spot
                    trading_pair = '_'.join(parts[2:])    # e.g., BTC_USDT
                    
                    # Resample to 5-second intervals
                    resample_data(exchange, trading_pair, '5s')
            
            logger.info("Data processing and resampling completed")
            
        except Exception as e:
            logger.error(f"Error in data processor: {e}")
            await asyncio.sleep(60)  # Wait a bit before retrying

async def main():
    """Main entry point for the multi-exchange LOB data collector."""
    try:
        # Set up the database
        setup_successful = setup_database()
        if not setup_successful:
            logger.error("Failed to set up database. Exiting.")
            return
        
        logger.info("Starting multi-exchange LOB data collector")
        logger.info("Press Ctrl+C to stop")
        
        # Define trading pairs to collect
        # Format: [exchange_type, [trading_pairs]]
        exchanges_config = [
            ('binance_spot', ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT']),
            ('binance_perp', ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT']),
            ('bybit_spot', ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT'])
        ]
        
        # Start websocket managers for each exchange
        tasks = []
        
        for exchange, trading_pairs in exchanges_config:
            if exchange == 'binance_spot':
                tasks.append(asyncio.create_task(binance_spot_websocket_manager(trading_pairs)))
            elif exchange == 'binance_perp':
                tasks.append(asyncio.create_task(binance_perp_websocket_manager(trading_pairs)))
            elif exchange == 'bybit_spot':
                tasks.append(asyncio.create_task(bybit_spot_websocket_manager(trading_pairs)))
        
        # Start data processor
        tasks.append(asyncio.create_task(data_processor()))
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
    except Exception as e:
        logger.error(f"Error in main function: {type(e).__name__} - {str(e)}")
    finally:
        logger.info("Data collector stopped")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 