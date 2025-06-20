#!/usr/bin/env python3
"""
LOB Data Preparation for Attention Model

This script performs the following steps to prepare the collected LOB data 
for training the attention-based forecasting model:

1.  Loads the 5-second resampled Parquet data.
2.  Combines data from multiple exchanges and trading pairs into a single 
    multivariate time series.
3.  Applies stationary transformations (percent-change) to price data.
4.  Scales all price and volume data using Min-Max scaling.
5.  Generates overlapping sequences of context (input) and target (output) data.
6.  Splits the sequences into training, validation, and testing sets.
7.  Saves the processed data and scalers for use in the training script.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_preparation")

# Configuration
PROCESSED_DATA_DIR = "data/processed"
FINAL_DATA_DIR = "data/final"
CONTEXT_LENGTH = 120  # 120 steps * 5s = 10 minutes
TARGET_LENGTH = 24   # 24 steps * 5s = 2 minutes

# Exchanges and trading pairs to process
exchanges_config = {
    'binance_spot': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT'],
    'binance_perp': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT'],
    'bybit_spot': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT']
}

def load_resampled_data(exchange, trading_pair):
    """Load all 5s resampled parquet files for a given pair."""
    pair_dir = os.path.join(PROCESSED_DATA_DIR, exchange, trading_pair)
    if not os.path.exists(pair_dir):
        logger.warning(f"No processed data found for {exchange} {trading_pair}")
        return None
    
    all_files = []
    for root, _, files in os.walk(pair_dir):
        for file in files:
            if file.endswith("_resampled_5s.parquet"):
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        logger.warning(f"No 5s resampled files for {exchange} {trading_pair}")
        return None
    
    df = pd.concat((pd.read_parquet(f) for f in all_files))
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('datetime')
    return df

def process_single_pair(exchange, trading_pair):
    """Load, process, and prepare data for a single trading pair."""
    logger.info(f"Processing {exchange} {trading_pair}...")
    
    df = load_resampled_data(exchange, trading_pair)
    if df is None:
        return None
    
    # Identify price and volume columns
    price_cols = [c for c in df.columns if 'price' in c]
    volume_cols = [c for c in df.columns if 'volume' in c]
    
    # Stationary Transformation (Percent Change) for prices
    for col in price_cols:
        df[col] = df[col].pct_change().fillna(0)
    
    # Drop rows with any remaining NaNs/infinities after transformation
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Rename columns to be unique
    rename_dict = {c: f"{exchange}_{trading_pair}_{c}" for c in price_cols + volume_cols}
    df = df.rename(columns=rename_dict)
    
    return df[list(rename_dict.values())]

def main():
    """Main function to orchestrate the data preparation process."""
    logger.info("Starting data preparation for the attention model...")
    
    if not os.path.exists(FINAL_DATA_DIR):
        os.makedirs(FINAL_DATA_DIR)

    # --- 1. Load and process data in parallel ---
    all_dfs = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_single_pair, ex, tp): (ex, tp) 
            for ex, tps in exchanges_config.items() for tp in tps
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_dfs.append(result)

    if not all_dfs:
        logger.error("No data could be processed. Exiting.")
        return

    # --- 2. Combine into a single multivariate time series ---
    logger.info("Combining all datasets into a single time series...")
    combined_df = pd.concat(all_dfs, axis=1)
    
    # Re-index to a perfect 5-second interval and forward-fill gaps
    full_range = pd.date_range(
        start=combined_df.index.min(), 
        end=combined_df.index.max(), 
        freq='5S'
    )
    combined_df = combined_df.reindex(full_range).ffill().dropna()
    
    logger.info(f"Final combined shape: {combined_df.shape}")
    
    # --- 3. Scale Data and Save Scalers ---
    logger.info("Scaling data with Min-Max scalers...")
    scalers = {}
    scaled_data_dict = {}
    
    for col in combined_df.columns:
        scaler = MinMaxScaler()
        # Reshape data for scaler
        scaled_col = scaler.fit_transform(combined_df[[col]])
        scaled_data_dict[col] = scaled_col.flatten()
        scalers[col] = scaler
        
    scaled_df = pd.DataFrame(scaled_data_dict, index=combined_df.index)
    
    joblib.dump(scalers, os.path.join(FINAL_DATA_DIR, "scalers.gz"))
    logger.info("Scalers saved.")

    # --- 4. Generate Sequences using Memory-Mapping to conserve RAM ---
    logger.info("Generating input/target sequences using memory-mapping...")
    
    # Save the scaled data to a temporary numpy binary file
    temp_data_path = os.path.join(FINAL_DATA_DIR, "scaled_data.npy")
    np.save(temp_data_path, scaled_df.values.astype('float32'))
    del scaled_df # Free up memory
    del combined_df
    
    # Load the data as a memory-mapped array
    data = np.load(temp_data_path, mmap_mode='r')
    
    n_samples, n_features = data.shape
    
    # Create memory-mapped files for contexts and targets
    context_path = os.path.join(FINAL_DATA_DIR, "contexts.mmap")
    target_path = os.path.join(FINAL_DATA_DIR, "targets.mmap")
    
    n_sequences = n_samples - CONTEXT_LENGTH - TARGET_LENGTH + 1
    
    # Ensure the directory exists
    if not os.path.exists(FINAL_DATA_DIR):
        os.makedirs(FINAL_DATA_DIR)

    contexts = np.memmap(context_path, dtype='float32', mode='w+', shape=(n_sequences, CONTEXT_LENGTH, n_features))
    targets = np.memmap(target_path, dtype='float32', mode='w+', shape=(n_sequences, TARGET_LENGTH, n_features))
    
    # Populate the memory-mapped arrays in chunks to keep RAM usage low
    chunk_size = 100000  # Process 100,000 sequences at a time
    for i in range(0, n_sequences, chunk_size):
        end = min(i + chunk_size, n_sequences)
        logger.info(f"  ... processing sequences {i} to {end}")
        
        # Efficiently create indices for the chunk
        sequence_starts = np.arange(i, end)
        context_indices = sequence_starts[:, None] + np.arange(CONTEXT_LENGTH)
        target_indices = (sequence_starts + CONTEXT_LENGTH)[:, None] + np.arange(TARGET_LENGTH)
        
        contexts[i:end] = data[context_indices].astype('float32')
        targets[i:end] = data[target_indices].astype('float32')
        
    # The data is now on disk in the .mmap files
    logger.info(f"Created {n_sequences} sequences on disk.")

    # --- 5. Split and Save Data ---
    logger.info("Splitting data into training, validation, and test sets...")
    train_end = int(n_sequences * 0.6)
    val_end = int(n_sequences * 0.8)
    
    # Load from memmap and save to npz
    logger.info("Saving train set...")
    np.savez_compressed(os.path.join(FINAL_DATA_DIR, "train.npz"), 
                        x=contexts[:train_end], y=targets[:train_end])
    
    logger.info("Saving validation set...")
    np.savez_compressed(os.path.join(FINAL_DATA_DIR, "validation.npz"), 
                        x=contexts[train_end:val_end], y=targets[train_end:val_end])
    
    logger.info("Saving test set...")
    np.savez_compressed(os.path.join(FINAL_DATA_DIR, "test.npz"), 
                        x=contexts[val_end:], y=targets[val_end:])

    # Clean up memory-mapped files
    del contexts
    del targets
    os.remove(context_path)
    os.remove(target_path)
    os.remove(temp_data_path) # Clean up the npy file
    
    # Save column order for reference
    column_list = list(scaled_data_dict.keys())
    joblib.dump(column_list, os.path.join(FINAL_DATA_DIR, "columns.gz"))
    
    logger.info("Data preparation complete. Final data saved to 'data/final'.")
    train_x_shape = (train_end, CONTEXT_LENGTH, n_features)
    train_y_shape = (train_end, TARGET_LENGTH, n_features)
    val_x_shape = (val_end - train_end, CONTEXT_LENGTH, n_features)
    val_y_shape = (val_end - train_end, TARGET_LENGTH, n_features)
    test_x_shape = (n_sequences - val_end, CONTEXT_LENGTH, n_features)
    test_y_shape = (n_sequences - val_end, TARGET_LENGTH, n_features)

    logger.info(f"Train set shape: x={train_x_shape}, y={train_y_shape}")
    logger.info(f"Validation set shape: x={val_x_shape}, y={val_y_shape}")
    logger.info(f"Test set shape: x={test_x_shape}, y={test_y_shape}")

if __name__ == "__main__":
    main() 