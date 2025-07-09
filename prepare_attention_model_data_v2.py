#!/usr/bin/env python3
"""
Attention-Based LOB Data Preparation Script (Version 2)

This script prepares limit order book (LOB) data for the attention-based forecasting model
following the methodology described in "Attention-Based Reading, Highlighting, and 
Forecasting of the Limit Order Book" paper.

Key Updates in V2:
1. Loads full LOB data from data/full_lob_data/resampled_5s (39+ days)
2. Handles session-aware processing with proper gap management
3. Applies percent-change transformation to prices for stationarity
4. Implements compound multivariate embedding structure
5. Creates sequences with proper session boundaries
6. Scales to handle 100M+ records efficiently

Key features:
1. Loads full 39-day resampled LOB data
2. Applies percent-change transformation to prices
3. Applies min-max scaling to all variables
4. Creates sequences with context length of 120 and target length of 24
5. Implements compound multivariate embedding structure
6. Splits data into train/validation/test sets (60%/20%/20%)
7. Saves processed data for model training
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime
import pickle
import json
from typing import Dict, List, Tuple, Optional
import glob
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration following the paper's methodology
CONFIG = {
    'data_path': 'data/full_lob_data/resampled_5s',  # Updated to use full dataset
    'output_path': 'data/final_attention',
    'context_length': 240,  # 240 steps * 5s = 20 minutes (matches paper)
    'target_length': 24,    # 24 steps * 5s = 2 minutes
    'lob_levels': 5,
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'exchanges': ['binance_spot', 'binance_perp', 'bybit_spot'],
    'trading_pairs': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT'],
    'min_session_length': 300,  # Minimum 300 steps (25 minutes) for valid sequences with 20min context
    'overlap_ratio': 0.5  # 50% overlap between sequences
}

def load_resampled_data() -> Dict[str, pd.DataFrame]:
    """Load resampled LOB data from the full dataset."""
    logger.info("Loading full resampled LOB data...")
    
    data_path = CONFIG['data_path']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Find all resampled parquet files
    parquet_files = glob.glob(os.path.join(data_path, "*_resampled_5s.parquet"))
    logger.info(f"Found {len(parquet_files)} resampled parquet files")
    
    data = {}
    total_records = 0
    
    for file_path in parquet_files:
        try:
            # Extract exchange and trading pair from filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            
            if len(parts) < 6:
                logger.warning(f"Skipping file with unexpected format: {filename}")
                continue
            
            exchange = parts[0] + '_' + parts[1]  # e.g., 'binance_spot'
            trading_pair = parts[2]  # e.g., 'BTC-USDT'
            
            # Check if this exchange/pair combination is in our target list
            if exchange not in CONFIG['exchanges'] or trading_pair not in CONFIG['trading_pairs']:
                continue
            
            # Load the parquet file
            df = pd.read_parquet(file_path)
            
            # Reset index to make timestamp a column
            if df.index.name == 'timestamp':
                df = df.reset_index()
            
            # Create unique key for this exchange/pair combination
            key = f"{exchange}_{trading_pair}"
            
            # Accumulate data for this key
            if key not in data:
                data[key] = []
            
            data[key].append(df)
            total_records += len(df)
            
            logger.info(f"Loaded {len(df):,} records from {filename}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    # Concatenate all data for each exchange/pair
    combined_data = {}
    for key, df_list in data.items():
        if len(df_list) > 0:
            combined_df = pd.concat(df_list, ignore_index=True)
            # Sort by timestamp
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            combined_data[key] = combined_df
            logger.info(f"Combined {key}: {len(combined_df):,} records")
    
    logger.info(f"Total records loaded: {total_records:,} across {len(combined_data)} exchange/pair combinations")
    return combined_data

def apply_percent_change_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply percent-change transformation to price columns for stationarity."""
    logger.info("Applying percent-change transformation to prices...")
    
    df_transformed = df.copy()
    
    # Identify price columns
    price_columns = [col for col in df.columns if 'price' in col.lower()]
    
    for col in price_columns:
        # Calculate percent change
        df_transformed[col] = df[col].pct_change()
        
        # Handle infinite values and division by zero
        df_transformed[col] = df_transformed[col].replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values
        df_transformed[col] = df_transformed[col].fillna(method='ffill')
        
        # Fill remaining NaN values with 0
        df_transformed[col] = df_transformed[col].fillna(0)
    
    logger.info(f"Applied percent-change transformation to {len(price_columns)} price columns")
    return df_transformed

def detect_sessions(df: pd.DataFrame, max_gap_minutes: int = 30) -> List[Tuple[int, int]]:
    """Detect continuous sessions based on time gaps."""
    if len(df) == 0:
        return []
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate time differences
    time_diffs = df['timestamp'].diff()
    
    # Find gaps larger than max_gap_minutes
    gap_threshold = pd.Timedelta(minutes=max_gap_minutes)
    large_gaps = time_diffs > gap_threshold
    
    # Find session boundaries
    session_starts = [0]  # First row is always a session start
    session_starts.extend(df.index[large_gaps].tolist())
    
    sessions = []
    for i in range(len(session_starts)):
        start_idx = session_starts[i]
        end_idx = session_starts[i + 1] - 1 if i + 1 < len(session_starts) else len(df) - 1
        
        # Only include sessions that meet minimum length requirement
        if end_idx - start_idx + 1 >= CONFIG['min_session_length']:
            sessions.append((start_idx, end_idx))
    
    return sessions

def create_compound_multivariate_embedding_structure(data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, Dict]:
    """Create compound multivariate embedding structure for the full dataset."""
    logger.info("Creating compound multivariate embedding structure...")
    
    # Initialize embedding metadata
    embedding_metadata = {
        'exchanges': [],
        'trading_pairs': [],
        'order_types': [],
        'features': [],
        'levels': [],
        'column_mapping': {},
        'unique_attributes': {
            'exchanges': sorted(CONFIG['exchanges']),
            'trading_pairs': sorted(CONFIG['trading_pairs']),
            'order_types': ['bid', 'ask'],
            'features': ['price', 'volume'],
            'levels': list(range(1, CONFIG['lob_levels'] + 1))
        }
    }
    
    # Build column mapping and all_columns list from actual processed data
    column_idx = 0
    all_columns = []
    
    # Find common time index across all dataframes
    common_indices = []
    for key, df in data.items():
        if len(df) > 0:
            common_indices.append(df.index.tolist())
    
    if not common_indices:
        raise ValueError("No data found to process")
    
    # Create combined dataset
    combined_data_list = []
    
    for key, df in data.items():
        if len(df) == 0:
            continue
            
        exchange, trading_pair = key.split('_', 1)
        
        # Create prefixed columns  
        df_prefixed = df.copy()
        df_prefixed = df_prefixed.drop('timestamp', axis=1)  # Remove timestamp column
        
        # Remove string columns that shouldn't be in the final dataset
        string_columns_to_drop = ['exchange', 'trading_pair']
        for col in string_columns_to_drop:
            if col in df_prefixed.columns:
                df_prefixed = df_prefixed.drop(col, axis=1)
        
        # Rename remaining columns with prefix
        column_mapping = {}
        for col in df_prefixed.columns:
            new_col_name = f"{exchange}_{trading_pair}_{col}"
            column_mapping[col] = new_col_name
        
        df_prefixed = df_prefixed.rename(columns=column_mapping)
        df_prefixed['timestamp'] = df['timestamp']
        
        combined_data_list.append(df_prefixed)
    
    # Merge all dataframes on timestamp
    if len(combined_data_list) == 1:
        combined_df = combined_data_list[0]
    else:
        combined_df = combined_data_list[0]
        for df in combined_data_list[1:]:
            combined_df = pd.merge(combined_df, df, on='timestamp', how='outer')
    
    # Forward fill missing values
    combined_df = combined_df.fillna(method='ffill')
    
    # Drop rows with any remaining NaN values
    combined_df = combined_df.dropna()
    
    # Remove timestamp and string columns for final array
    final_df = combined_df.drop(['timestamp'], axis=1, errors='ignore')
    
    # Build all_columns list and metadata from the actual final columns
    final_columns = [col for col in final_df.columns]
    all_columns = final_columns
    
    # Build embedding metadata from final columns
    for col in final_columns:
        # Parse column name to extract attributes
        parts = col.split('_')
        if len(parts) >= 3:
            exchange = parts[0] + '_' + parts[1]  # e.g., 'binance_spot'
            trading_pair = parts[2]  # e.g., 'BTC-USDT'
            feature_col = '_'.join(parts[3:])  # e.g., 'bid_price_1'
            
            # Parse feature column
            if 'bid' in feature_col:
                order_type = 'bid'
            elif 'ask' in feature_col:
                order_type = 'ask'
            else:
                order_type = 'other'
            
            if 'price' in feature_col:
                feature_type = 'price'
            elif 'quantity' in feature_col or 'volume' in feature_col:
                feature_type = 'volume'
            else:
                feature_type = 'other'
            
            # Extract level number
            level = 1
            for i in range(1, CONFIG['lob_levels'] + 1):
                if f'_{i}' in feature_col:
                    level = i
                    break
            
            # Store embedding metadata
            embedding_metadata['exchanges'].append(exchange)
            embedding_metadata['trading_pairs'].append(trading_pair)
            embedding_metadata['order_types'].append(order_type)
            embedding_metadata['features'].append(feature_type)
            embedding_metadata['levels'].append(level)
            embedding_metadata['column_mapping'][col] = {
                'exchange': exchange,
                'trading_pair': trading_pair,
                'order_type': order_type,
                'feature_type': feature_type,
                'level': level,
                'column_idx': column_idx
            }
            
            column_idx += 1
    
    # Convert to numpy array
    combined_array = final_df.values
    
    logger.info(f"Created combined dataset with shape: {combined_array.shape}")
    logger.info(f"Total features: {len(all_columns)}")
    
    # Update embedding metadata with final column list
    embedding_metadata['columns'] = all_columns
    embedding_metadata['num_features'] = len(all_columns)
    
    return combined_array, embedding_metadata

def apply_min_max_scaling(data: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """Apply min-max scaling to all features."""
    logger.info("Applying min-max scaling...")
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    logger.info(f"Applied min-max scaling to {data.shape[1]} features")
    return scaled_data, scaler

def create_sequences_with_sessions(data: np.ndarray, timestamps: np.ndarray, 
                                 context_length: int, target_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences while respecting session boundaries."""
    logger.info("Creating sequences with session awareness...")
    
    # Create a dummy dataframe to detect sessions
    dummy_df = pd.DataFrame({'timestamp': timestamps})
    sessions = detect_sessions(dummy_df)
    
    contexts = []
    targets = []
    
    sequence_length = context_length + target_length
    step_size = int(context_length * (1 - CONFIG['overlap_ratio']))
    
    for start_idx, end_idx in sessions:
        session_length = end_idx - start_idx + 1
        
        logger.info(f"Processing session: {start_idx}-{end_idx} (length: {session_length})")
        
        # Create sequences within this session
        for i in range(start_idx, end_idx - sequence_length + 1, step_size):
            context = data[i:i + context_length]
            target = data[i + context_length:i + sequence_length]
            
            contexts.append(context)
            targets.append(target)
    
    if len(contexts) == 0:
        raise ValueError("No valid sequences created. Check session lengths and parameters.")
    
    contexts = np.array(contexts)
    targets = np.array(targets)
    
    logger.info(f"Created {len(contexts)} sequences from {len(sessions)} sessions")
    logger.info(f"Context shape: {contexts.shape}")
    logger.info(f"Target shape: {targets.shape}")
    
    return contexts, targets

def split_data(contexts: np.ndarray, targets: np.ndarray, 
               train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple:
    """Split data into train, validation, and test sets."""
    logger.info("Splitting data into train/validation/test sets...")
    
    total_sequences = len(contexts)
    train_size = int(total_sequences * train_ratio)
    val_size = int(total_sequences * val_ratio)
    test_size = total_sequences - train_size - val_size
    
    # Split data chronologically
    train_contexts = contexts[:train_size]
    train_targets = targets[:train_size]
    
    val_contexts = contexts[train_size:train_size + val_size]
    val_targets = targets[train_size:train_size + val_size]
    
    test_contexts = contexts[train_size + val_size:]
    test_targets = targets[train_size + val_size:]
    
    logger.info(f"Train: {len(train_contexts)} sequences")
    logger.info(f"Validation: {len(val_contexts)} sequences")
    logger.info(f"Test: {len(test_contexts)} sequences")
    
    return train_contexts, train_targets, val_contexts, val_targets, test_contexts, test_targets

def save_processed_data(data_splits: Tuple, scaler: MinMaxScaler, 
                       embedding_metadata: Dict, output_path: str):
    """Save all processed data and metadata."""
    logger.info(f"Saving processed data to {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Unpack data splits
    train_contexts, train_targets, val_contexts, val_targets, test_contexts, test_targets = data_splits
    
    # Save data splits as compressed NPZ files
    np.savez_compressed(
        os.path.join(output_path, 'train.npz'),
        x=train_contexts,
        y=train_targets
    )
    
    np.savez_compressed(
        os.path.join(output_path, 'validation.npz'),
        x=val_contexts,
        y=val_targets
    )
    
    np.savez_compressed(
        os.path.join(output_path, 'test.npz'),
        x=test_contexts,
        y=test_targets
    )
    
    # Save scaler
    with open(os.path.join(output_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save embedding metadata
    with open(os.path.join(output_path, 'embedding_metadata.json'), 'w') as f:
        json.dump(embedding_metadata, f, indent=2)
    
    # Save column names for model training
    import joblib
    joblib.dump(embedding_metadata['columns'], os.path.join(output_path, 'columns.gz'))
    
    # Save configuration
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Save dataset statistics
    stats = {
        'total_sequences': len(train_contexts) + len(val_contexts) + len(test_contexts),
        'train_sequences': len(train_contexts),
        'val_sequences': len(val_contexts),
        'test_sequences': len(test_contexts),
        'context_length': CONFIG['context_length'],
        'target_length': CONFIG['target_length'],
        'num_features': train_contexts.shape[-1],
        'data_shape': {
            'train_contexts': train_contexts.shape,
            'train_targets': train_targets.shape,
            'val_contexts': val_contexts.shape,
            'val_targets': val_targets.shape,
            'test_contexts': test_contexts.shape,
            'test_targets': test_targets.shape
        }
    }
    
    with open(os.path.join(output_path, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("All data saved successfully!")
    logger.info(f"Dataset statistics: {stats}")

def main():
    """Main data preparation pipeline."""
    logger.info("Starting attention-based LOB data preparation (Version 2)...")
    logger.info(f"Configuration: {CONFIG}")
    
    try:
        # Step 1: Load full resampled data
        raw_data = load_resampled_data()
        
        if not raw_data:
            logger.error("No data loaded. Exiting.")
            return
        
        # Step 2: Apply percent-change transformation
        logger.info("Applying percent-change transformation...")
        transformed_data = {}
        all_timestamps = []
        
        for key, df in raw_data.items():
            try:
                transformed_df = apply_percent_change_transformation(df)
                transformed_data[key] = transformed_df
                all_timestamps.extend(transformed_df['timestamp'].tolist())
                logger.info(f"Transformed {key}: {len(transformed_df)} records")
            except Exception as e:
                logger.error(f"Error transforming {key}: {e}")
                continue
        
        # Step 3: Create compound multivariate embedding structure
        combined_data, embedding_metadata = create_compound_multivariate_embedding_structure(transformed_data)
        
        # Step 4: Apply min-max scaling
        scaled_data, scaler = apply_min_max_scaling(combined_data)
        
        # Step 5: Create sequences with session awareness
        # Get timestamps for session detection
        timestamps = np.array(sorted(set(all_timestamps)))
        
        contexts, targets = create_sequences_with_sessions(
            scaled_data, 
            timestamps[:len(scaled_data)],  # Match length with scaled_data
            CONFIG['context_length'], 
            CONFIG['target_length']
        )
        
        # Step 6: Split data
        data_splits = split_data(
            contexts, 
            targets, 
            CONFIG['train_ratio'], 
            CONFIG['val_ratio'], 
            CONFIG['test_ratio']
        )
        
        # Step 7: Save processed data
        save_processed_data(data_splits, scaler, embedding_metadata, CONFIG['output_path'])
        
        logger.info("Data preparation completed successfully!")
        logger.info(f"Final dataset: {len(contexts)} sequences with {contexts.shape[-1]} features")
        
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 