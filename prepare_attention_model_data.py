#!/usr/bin/env python3
"""
Attention-Based LOB Data Preparation Script

This script prepares limit order book (LOB) data for the attention-based forecasting model
following the methodology described in "Attention-Based Reading, Highlighting, and 
Forecasting of the Limit Order Book" paper.

Key features:
1. Loads clean LOB data (excluding days with hour-plus gaps)
2. Resamples to 5-second intervals 
3. Applies percent-change transformation to prices
4. Applies min-max scaling to all variables
5. Creates sequences with context length of 120 and target length of 24
6. Implements compound multivariate embedding structure
7. Splits data into train/validation/test sets (60%/20%/20%)
8. Saves processed data for model training
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
    'data_path': 'data/clean_parquet',
    'output_path': 'data/final',
    'resample_frequency': '5s',
    'context_length': 120,  # 120 steps * 5s = 10 minutes
    'target_length': 24,    # 24 steps * 5s = 2 minutes
    'lob_levels': 5,
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'exchanges': ['binance_spot', 'bybit_spot'],
    'trading_pairs': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT']
}

def load_clean_data() -> Dict[str, pd.DataFrame]:
    """Load clean LOB data from parquet files."""
    logger.info("Loading clean LOB data...")
    
    data = {}
    total_records = 0
    
    for exchange in CONFIG['exchanges']:
        exchange_path = os.path.join(CONFIG['data_path'], exchange)
        
        if not os.path.exists(exchange_path):
            logger.warning(f"Exchange directory not found: {exchange_path}")
            continue
            
        for trading_pair in CONFIG['trading_pairs']:
            # Convert trading pair format for filename
            pair_filename = trading_pair.replace('-', '_')
            parquet_file = os.path.join(exchange_path, f"{pair_filename}_clean.parquet")
            
            if not os.path.exists(parquet_file):
                logger.warning(f"File not found: {parquet_file}")
                continue
                
            try:
                df = pd.read_parquet(parquet_file)
                key = f"{exchange}_{trading_pair}"
                data[key] = df
                total_records += len(df)
                logger.info(f"Loaded {len(df):,} records for {key}")
                
            except Exception as e:
                logger.error(f"Error loading {parquet_file}: {e}")
                continue
    
    logger.info(f"Total records loaded: {total_records:,} across {len(data)} exchange/pair combinations")
    return data

def resample_to_5s(df: pd.DataFrame) -> pd.DataFrame:
    """Resample LOB data to 5-second intervals."""
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column")
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    # Resample to 5-second intervals using the last value in each interval
    df_resampled = df.resample(CONFIG['resample_frequency']).last()
    
    # Forward fill missing values
    df_resampled = df_resampled.fillna(method='ffill')
    
    # Drop rows with any remaining NaN values
    df_resampled = df_resampled.dropna()
    
    return df_resampled

def apply_percent_change_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply percent-change transformation to price columns."""
    logger.info("Applying percent-change transformation to prices...")
    
    df_transformed = df.copy()
    
    # Identify price columns
    price_columns = [col for col in df.columns if 'price' in col.lower()]
    
    for col in price_columns:
        # Apply percent change transformation: (p_t - p_{t-1}) / p_{t-1}
        df_transformed[col] = df[col].pct_change()
    
    # Drop the first row which will have NaN after pct_change
    df_transformed = df_transformed.dropna()
    
    logger.info(f"Applied percent-change transformation to {len(price_columns)} price columns")
    return df_transformed

def create_compound_multivariate_embedding_structure(data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, Dict]:
    """
    Create compound multivariate embedding structure following the paper's methodology.
    
    Returns:
        - Combined multivariate time series
        - Metadata for embedding reconstruction
    """
    logger.info("Creating compound multivariate embedding structure...")
    
    # Combine all exchange/pair data into single multivariate time series
    combined_data = []
    embedding_metadata = {
        'exchanges': [],
        'trading_pairs': [],
        'features': [],
        'levels': [],
        'column_mapping': {}
    }
    
    column_idx = 0
    
    for key, df in data.items():
        exchange, trading_pair = key.split('_', 1)
        
        # Process each column
        for col in df.columns:
            if col in ['exchange', 'trading_pair']:
                continue
                
            # Parse column name to extract attributes
            if 'bid' in col:
                order_type = 'bid'
            elif 'ask' in col:
                order_type = 'ask'
            else:
                order_type = 'other'
            
            if 'price' in col:
                feature_type = 'price'
            elif 'quantity' in col or 'volume' in col:
                feature_type = 'volume'
            else:
                feature_type = 'other'
            
            # Extract level number
            level = 1
            for i in range(1, CONFIG['lob_levels'] + 1):
                if f'_{i}' in col:
                    level = i
                    break
            
            # Create unique column name
            new_col_name = f"{exchange}_{trading_pair}_{col}"
            
            # Store embedding metadata
            embedding_metadata['exchanges'].append(exchange)
            embedding_metadata['trading_pairs'].append(trading_pair)
            embedding_metadata['features'].append(feature_type)
            embedding_metadata['levels'].append(level)
            embedding_metadata['column_mapping'][new_col_name] = {
                'exchange': exchange,
                'trading_pair': trading_pair,
                'order_type': order_type,
                'feature_type': feature_type,
                'level': level,
                'column_idx': column_idx
            }
            
            column_idx += 1
    
    # Find common time index across all dataframes
    common_index = None
    for df in data.values():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    
    if len(common_index) == 0:
        raise ValueError("No common time index found across all dataframes")
    
    logger.info(f"Common time index length: {len(common_index)}")
    
    # Create combined multivariate time series
    combined_df = pd.DataFrame(index=common_index)
    
    for key, df in data.items():
        exchange, trading_pair = key.split('_', 1)
        
        # Align to common index
        df_aligned = df.reindex(common_index, method='ffill')
        
        # Add columns with unique names
        for col in df_aligned.columns:
            if col not in ['exchange', 'trading_pair']:
                new_col_name = f"{exchange}_{trading_pair}_{col}"
                combined_df[new_col_name] = df_aligned[col]
    
    # Convert to numpy array
    combined_array = combined_df.values
    
    logger.info(f"Combined multivariate time series shape: {combined_array.shape}")
    logger.info(f"Total features: {len(embedding_metadata['column_mapping'])}")
    
    return combined_array, embedding_metadata

def apply_min_max_scaling(data: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """Apply min-max scaling to all variables."""
    logger.info("Applying min-max scaling...")
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    logger.info(f"Scaled data shape: {scaled_data.shape}")
    return scaled_data, scaler

def create_sequences(data: np.ndarray, context_length: int, target_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping sequences for attention model training.
    
    Returns:
        - Context sequences (input)
        - Target sequences (output to predict)
    """
    logger.info(f"Creating sequences with context_length={context_length}, target_length={target_length}")
    
    sequence_length = context_length + target_length
    
    if len(data) < sequence_length:
        raise ValueError(f"Data length {len(data)} is less than required sequence length {sequence_length}")
    
    contexts = []
    targets = []
    
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length]
        
        context = sequence[:context_length]
        target = sequence[context_length:]
        
        contexts.append(context)
        targets.append(target)
    
    contexts = np.array(contexts)
    targets = np.array(targets)
    
    logger.info(f"Created {len(contexts)} sequences")
    logger.info(f"Context shape: {contexts.shape}")
    logger.info(f"Target shape: {targets.shape}")
    
    return contexts, targets

def split_data(contexts: np.ndarray, targets: np.ndarray, 
               train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple:
    """Split data into train/validation/test sets chronologically."""
    logger.info(f"Splitting data: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    n_samples = len(contexts)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_contexts = contexts[:train_end]
    train_targets = targets[:train_end]
    
    val_contexts = contexts[train_end:val_end]
    val_targets = targets[train_end:val_end]
    
    test_contexts = contexts[val_end:]
    test_targets = targets[val_end:]
    
    logger.info(f"Train samples: {len(train_contexts)}")
    logger.info(f"Validation samples: {len(val_contexts)}")
    logger.info(f"Test samples: {len(test_contexts)}")
    
    return (train_contexts, train_targets, val_contexts, val_targets, test_contexts, test_targets)

def save_processed_data(data_splits: Tuple, scaler: MinMaxScaler, 
                       embedding_metadata: Dict, output_path: str):
    """Save all processed data and metadata."""
    logger.info(f"Saving processed data to {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Unpack data splits
    train_contexts, train_targets, val_contexts, val_targets, test_contexts, test_targets = data_splits
    
    # Save data splits
    np.save(os.path.join(output_path, 'train_contexts.npy'), train_contexts)
    np.save(os.path.join(output_path, 'train_targets.npy'), train_targets)
    np.save(os.path.join(output_path, 'val_contexts.npy'), val_contexts)
    np.save(os.path.join(output_path, 'val_targets.npy'), val_targets)
    np.save(os.path.join(output_path, 'test_contexts.npy'), test_contexts)
    np.save(os.path.join(output_path, 'test_targets.npy'), test_targets)
    
    # Save scaler
    with open(os.path.join(output_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save embedding metadata
    with open(os.path.join(output_path, 'embedding_metadata.json'), 'w') as f:
        json.dump(embedding_metadata, f, indent=2)
    
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
    logger.info("Starting attention-based LOB data preparation...")
    logger.info(f"Configuration: {CONFIG}")
    
    try:
        # Step 1: Load clean data
        raw_data = load_clean_data()
        
        if not raw_data:
            logger.error("No data loaded. Exiting.")
            return
        
        # Step 2: Resample to 5-second intervals
        logger.info("Resampling data to 5-second intervals...")
        resampled_data = {}
        for key, df in raw_data.items():
            try:
                resampled_df = resample_to_5s(df)
                resampled_data[key] = resampled_df
                logger.info(f"Resampled {key}: {len(resampled_df)} records")
            except Exception as e:
                logger.error(f"Error resampling {key}: {e}")
                continue
        
        # Step 3: Apply percent-change transformation
        logger.info("Applying percent-change transformation...")
        transformed_data = {}
        for key, df in resampled_data.items():
            try:
                transformed_df = apply_percent_change_transformation(df)
                transformed_data[key] = transformed_df
                logger.info(f"Transformed {key}: {len(transformed_df)} records")
            except Exception as e:
                logger.error(f"Error transforming {key}: {e}")
                continue
        
        # Step 4: Create compound multivariate embedding structure
        combined_data, embedding_metadata = create_compound_multivariate_embedding_structure(transformed_data)
        
        # Step 5: Apply min-max scaling
        scaled_data, scaler = apply_min_max_scaling(combined_data)
        
        # Step 6: Create sequences
        contexts, targets = create_sequences(
            scaled_data, 
            CONFIG['context_length'], 
            CONFIG['target_length']
        )
        
        # Step 7: Split data
        data_splits = split_data(
            contexts, 
            targets, 
            CONFIG['train_ratio'], 
            CONFIG['val_ratio'], 
            CONFIG['test_ratio']
        )
        
        # Step 8: Save processed data
        save_processed_data(data_splits, scaler, embedding_metadata, CONFIG['output_path'])
        
        logger.info("Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 