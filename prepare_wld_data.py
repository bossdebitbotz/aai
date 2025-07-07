#!/usr/bin/env python3
"""
LOB Data Preparation for WLD-USDT Forecasting (Many-to-One)

This script prepares the data to train a model that uses the entire market
context (all exchanges, all pairs) to forecast only the WLD-USDT order books.

1.  Loads the 5-second resampled Parquet data for all instruments.
2.  Combines them into a single multivariate time series for input features.
3.  Generates overlapping sequences:
    - Context (X): 10 minutes (120 steps) of all 240 features.
    - Target (y): 5 minutes (60 steps) of only WLD-USDT features.
4.  Splits the data and saves it to `data/final_wld/`.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("wld_data_preparation")

# Configuration
PROCESSED_DATA_DIR = "data/processed"
FINAL_DATA_DIR = "data/final_wld" # New directory for this specific dataset
CONTEXT_LENGTH = 120  # 120 steps * 5s = 10 minutes
TARGET_LENGTH = 60    # 60 steps * 5s = 5 minutes

# Exchanges and trading pairs to process for the input features
exchanges_config = {
    'binance_spot': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT'],
    'binance_perp': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT'],
    'bybit_spot': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT']
}

def load_resampled_data(exchange, trading_pair):
    """Load all 5s resampled parquet files for a given pair."""
    pair_dir = os.path.join(PROCESSED_DATA_DIR, exchange, trading_pair)
    if not os.path.exists(pair_dir):
        return None
    
    all_files = [os.path.join(root, file) 
                 for root, _, files in os.walk(pair_dir) 
                 for file in files if file.endswith("_resampled_5s.parquet")]
    
    if not all_files:
        return None
    
    df = pd.concat((pd.read_parquet(f) for f in all_files))
    df = df.sort_values('timestamp').drop_duplicates('timestamp')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('datetime').drop(columns=['timestamp'])
    return df

def process_single_pair(exchange, trading_pair):
    """Load and prepare data for a single trading pair."""
    logger.info(f"Loading raw data for {exchange} {trading_pair}...")
    df = load_resampled_data(exchange, trading_pair)
    if df is None:
        return None
    
    # Do not apply transformations yet. This will be done on the combined frame.
    rename_dict = {c: f"{exchange}_{trading_pair}_{c}" for c in df.columns}
    df = df.rename(columns=rename_dict)
    return df[list(rename_dict.values())]

def main():
    """Main function to orchestrate the data preparation process."""
    logger.info("Starting WLD-specific data preparation...")
    
    if not os.path.exists(FINAL_DATA_DIR):
        os.makedirs(FINAL_DATA_DIR)

    # --- 1. Load and process data in parallel ---
    all_dfs = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_single_pair, ex, tp): (ex, tp) 
            for ex, tps in exchanges_config.items() for tp in tps
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Pair Data"):
            result = future.result()
            if result is not None:
                all_dfs.append(result)

    if not all_dfs:
        logger.error("No data could be processed. Exiting.")
        return

    # --- 2. Combine, resample, and transform ---
    logger.info("Combining all datasets into a single time series...")
    combined_df = pd.concat(all_dfs, axis=1)
    
    full_range = pd.date_range(
        start=combined_df.index.min(), 
        end=combined_df.index.max(), 
        freq='5S'
    )
    combined_df = combined_df.reindex(full_range).ffill().dropna()
    
    logger.info(f"Combined shape before transforms: {combined_df.shape}")
    
    # Identify price and volume columns for transformation
    price_cols = [c for c in combined_df.columns if 'price' in c]
    
    # Stationary Transformation (Percent Change) for prices
    logger.info("Applying percent-change transformation to price columns...")
    for col in tqdm(price_cols, desc="Applying PCT"):
        combined_df[col] = combined_df[col].pct_change()

    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df.dropna(inplace=True)

    # --- 3. Scale Data and Save Scalers ---
    logger.info("Scaling data with Min-Max scalers...")
    scalers = {}
    scaled_data = np.empty_like(combined_df.values, dtype=np.float32)

    for i, col in enumerate(tqdm(combined_df.columns, desc="Scaling Columns")):
        scaler = MinMaxScaler()
        scaled_col = scaler.fit_transform(combined_df[[col]])
        scaled_data[:, i] = scaled_col.flatten()
        scalers[col] = scaler
        
    joblib.dump(scalers, os.path.join(FINAL_DATA_DIR, "scalers.gz"))
    joblib.dump(list(combined_df.columns), os.path.join(FINAL_DATA_DIR, "columns_x.gz"))
    logger.info("Scalers and X columns saved.")
    
    # --- 4. Identify Target (y) Columns for binance perpetuals ---
    all_columns = list(combined_df.columns)
    wld_cols = [c for c in all_columns if 'binance_perp_WLD-USDT' in c]
    wld_indices = [all_columns.index(c) for c in wld_cols]
    
    if not wld_cols:
        logger.error("No 'binance_perp_WLD-USDT' columns found in the dataset. Exiting.")
        return
        
    joblib.dump(wld_cols, os.path.join(FINAL_DATA_DIR, "columns_y.gz"))
    logger.info(f"Identified {len(wld_cols)} Binance Perpetual WLD-USDT columns as target.")
    logger.info(f"Target columns: {wld_cols}")
    
    # --- 5. Generate Sequences ---
    logger.info("Generating input/target sequences...")
    n_samples, n_features = scaled_data.shape
    n_sequences = n_samples - CONTEXT_LENGTH - TARGET_LENGTH + 1
    
    # Using stride_tricks for memory-efficient sequence generation
    from numpy.lib.stride_tricks import as_strided
    
    shape_x = (n_sequences, CONTEXT_LENGTH, n_features)
    strides_x = (scaled_data.strides[0], scaled_data.strides[0], scaled_data.strides[1])
    contexts = as_strided(scaled_data, shape=shape_x, strides=strides_x)

    # For targets, we first create a view of the target time window, then select columns
    target_window_data = as_strided(
        scaled_data[CONTEXT_LENGTH:],
        shape=(n_sequences, TARGET_LENGTH, n_features),
        strides=(scaled_data.strides[0], scaled_data.strides[0], scaled_data.strides[1])
    )
    targets = target_window_data[:, :, wld_indices]

    logger.info(f"Created {n_sequences} sequences.")
    logger.info(f"Context (X) shape: {contexts.shape}")
    logger.info(f"Target (y) shape: {targets.shape}")

    # --- 6. Split and Save Data ---
    logger.info("Splitting data into training, validation, and test sets...")
    train_end = int(n_sequences * 0.7)
    val_end = int(n_sequences * 0.9)
    
    logger.info("Saving train set...")
    np.savez_compressed(os.path.join(FINAL_DATA_DIR, "train.npz"), 
                        x=contexts[:train_end], y=targets[:train_end])
    
    logger.info("Saving validation set...")
    np.savez_compressed(os.path.join(FINAL_DATA_DIR, "validation.npz"), 
                        x=contexts[train_end:val_end], y=targets[train_end:val_end])
    
    logger.info("Saving test set...")
    np.savez_compressed(os.path.join(FINAL_DATA_DIR, "test.npz"), 
                        x=contexts[val_end:], y=targets[val_end:])
    
    logger.info(f"Data preparation complete. Final data saved to '{FINAL_DATA_DIR}'.")

if __name__ == "__main__":
    main() 