#!/usr/bin/env python3
"""
Attention-based LOB Data Preparation (Version 4)
FIXES: Properly combines features from all exchange/pair combinations
Creates 240-feature vectors (12 combinations × 20 features each)
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration following the paper's methodology
CONFIG = {
    'data_path': 'data/full_lob_data/resampled_5s',
    'output_path': 'data/final_attention',
    'context_length': 120,  # 120 steps * 5s = 10 minutes
    'target_length': 24,    # 24 steps * 5s = 2 minutes
    'lob_levels': 5,
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'exchanges': ['binance_spot', 'binance_perp', 'bybit_spot'],
    'trading_pairs': ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT'],
    'overlap_ratio': 0.9,  # 90% overlap for maximum sequence generation
}

def parse_filename(filename: str) -> Tuple[str, str, str]:
    """Parse exchange, pair, and date from filename."""
    parts = filename.replace('.parquet', '').split('_')
    exchange = f"{parts[0]}_{parts[1]}"  # e.g., "binance_spot"
    pair = parts[2]  # e.g., "BTC-USDT"
    date = parts[3]  # e.g., "2025-06-01"
    return exchange, pair, date

def load_and_align_data() -> pd.DataFrame:
    """Load all files and align them by timestamp to create multi-exchange feature vectors."""
    logger.info("Loading and aligning data from all exchange/pair combinations...")
    
    data_path = CONFIG['data_path']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Get all files and organize by exchange/pair
    files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
    logger.info(f"Found {len(files)} parquet files")
    
    # Group files by exchange/pair
    exchange_pair_files = {}
    for file in files:
        try:
            exchange, pair, date = parse_filename(file)
            key = f"{exchange}_{pair}"
            if key not in exchange_pair_files:
                exchange_pair_files[key] = []
            exchange_pair_files[key].append(file)
        except Exception as e:
            logger.warning(f"Could not parse filename {file}: {e}")
            continue
    
    logger.info(f"Found data for {len(exchange_pair_files)} exchange/pair combinations:")
    for key, file_list in exchange_pair_files.items():
        logger.info(f"  {key}: {len(file_list)} files")
    
    # Load and combine all data
    all_dataframes = []
    
    for exchange_pair, file_list in exchange_pair_files.items():
        logger.info(f"Loading data for {exchange_pair}...")
        
        # Load all files for this exchange/pair
        exchange_data = []
        for file in file_list:
            try:
                file_path = os.path.join(data_path, file)
                df = pd.read_parquet(file_path)
                
                # Ensure timestamp is datetime and set as index
                if 'timestamp' not in df.columns:
                    if df.index.name != 'timestamp':
                        df.index.name = 'timestamp'
                    df = df.reset_index()
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                
                # Apply percent change transformation to price columns
                df = apply_percent_change_transformation(df)
                
                # Add exchange_pair prefix to column names to avoid conflicts
                df.columns = [f"{exchange_pair}_{col}" for col in df.columns]
                
                exchange_data.append(df)
                
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                continue
        
        if exchange_data:
            # Combine all files for this exchange/pair
            combined_exchange_data = pd.concat(exchange_data, axis=0)
            combined_exchange_data = combined_exchange_data.sort_index()
            
            # Remove duplicates (keep first occurrence)
            combined_exchange_data = combined_exchange_data[~combined_exchange_data.index.duplicated(keep='first')]
            
            all_dataframes.append(combined_exchange_data)
            logger.info(f"  Combined {exchange_pair}: {len(combined_exchange_data)} records")
    
    if not all_dataframes:
        raise ValueError("No valid data loaded")
    
    # Find common time range across all exchanges/pairs
    logger.info("Aligning timestamps across all exchange/pair combinations...")
    
    # Get the intersection of all timestamps
    common_timestamps = all_dataframes[0].index
    for df in all_dataframes[1:]:
        common_timestamps = common_timestamps.intersection(df.index)
    
    logger.info(f"Found {len(common_timestamps)} common timestamps")
    
    if len(common_timestamps) == 0:
        raise ValueError("No common timestamps found across exchange/pair combinations")
    
    # Align all dataframes to common timestamps
    aligned_dataframes = []
    for df in all_dataframes:
        aligned_df = df.loc[common_timestamps]
        aligned_dataframes.append(aligned_df)
    
    # Concatenate horizontally to create multi-exchange feature vectors
    final_combined_data = pd.concat(aligned_dataframes, axis=1)
    
    # Forward fill and fill remaining NaN values
    final_combined_data = final_combined_data.fillna(method='ffill')
    final_combined_data = final_combined_data.fillna(0)
    
    logger.info(f"Final combined dataset shape: {final_combined_data.shape}")
    logger.info(f"Features per exchange/pair: ~{final_combined_data.shape[1] / len(exchange_pair_files):.0f}")
    logger.info(f"Total features: {final_combined_data.shape[1]}")
    
    return final_combined_data

def apply_percent_change_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply percent-change transformation to price columns for stationarity."""
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
    
    return df_transformed

def create_sequences(df: pd.DataFrame, context_length: int, target_length: int, step_size: int) -> List[np.ndarray]:
    """Create sequences from the aligned multi-exchange data."""
    logger.info("Creating sequences from aligned multi-exchange data...")
    
    # Select only numeric columns for the model
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data = df[numeric_cols].values
    
    if len(data) < context_length + target_length:
        logger.warning(f"Data too short: {len(data)} < {context_length + target_length}")
        return []
    
    sequences = []
    sequence_length = context_length + target_length
    
    # Create overlapping sequences
    for i in range(0, len(data) - sequence_length + 1, step_size):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    
    logger.info(f"Created {len(sequences)} sequences of shape {sequences[0].shape if sequences else 'N/A'}")
    return sequences

def apply_min_max_scaling(sequences: List[np.ndarray]) -> Tuple[np.ndarray, MinMaxScaler]:
    """Apply MinMax scaling to the dataset."""
    logger.info("Applying MinMax scaling...")
    
    # Convert to numpy array
    data = np.array(sequences)
    original_shape = data.shape
    
    # Reshape for scaling (samples * timesteps, features)
    data_reshaped = data.reshape(-1, data.shape[-1])
    
    # Fit and transform
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_reshaped)
    
    # Reshape back to original
    scaled_data = scaled_data.reshape(original_shape)
    
    logger.info(f"Scaling applied to shape: {scaled_data.shape}")
    return scaled_data, scaler

def create_context_target_split(data: np.ndarray, context_length: int, target_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split sequences into context and target portions."""
    logger.info("Splitting sequences into context and target portions...")
    
    contexts = data[:, :context_length, :]  # First context_length timesteps
    targets = data[:, context_length:, :]   # Last target_length timesteps
    
    logger.info(f"Contexts shape: {contexts.shape}")
    logger.info(f"Targets shape: {targets.shape}")
    
    return contexts, targets

def split_data(contexts: np.ndarray, targets: np.ndarray, 
               train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple:
    """Split data into train/validation/test sets."""
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
        contexts=train_contexts,
        targets=train_targets
    )
    
    np.savez_compressed(
        os.path.join(output_path, 'validation.npz'),
        contexts=val_contexts,
        targets=val_targets
    )
    
    np.savez_compressed(
        os.path.join(output_path, 'test.npz'),
        contexts=test_contexts,
        targets=test_targets
    )
    
    # Save scaler
    import joblib
    joblib.dump(scaler, os.path.join(output_path, 'scaler.pkl'))
    
    # Save metadata
    with open(os.path.join(output_path, 'embedding_metadata.json'), 'w') as f:
        json.dump(embedding_metadata, f, indent=2)
    
    # Save configuration
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Create dataset statistics
    dataset_stats = {
        'total_sequences': len(train_contexts) + len(val_contexts) + len(test_contexts),
        'train_sequences': len(train_contexts),
        'val_sequences': len(val_contexts),
        'test_sequences': len(test_contexts),
        'num_features': train_contexts.shape[-1],
        'context_length': train_contexts.shape[1],
        'target_length': train_targets.shape[1],
        'created_at': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_path, 'dataset_stats.json'), 'w') as f:
        json.dump(dataset_stats, f, indent=2, default=str)
    
    logger.info("All data saved successfully!")
    return dataset_stats

def main():
    """Main data preparation pipeline."""
    logger.info("Starting attention-based LOB data preparation (Version 4)...")
    logger.info(f"Configuration: {CONFIG}")
    
    try:
        # Load and align all data by timestamp
        combined_data = load_and_align_data()
        
        if combined_data.empty:
            raise ValueError("No data loaded successfully")
        
        # Calculate sequence parameters
        context_length = CONFIG['context_length']
        target_length = CONFIG['target_length']
        step_size = int(context_length * (1 - CONFIG['overlap_ratio']))
        
        # Create sequences from aligned data
        sequences = create_sequences(combined_data, context_length, target_length, step_size)
        
        if not sequences:
            raise ValueError("No sequences were created")
        
        # Apply scaling
        scaled_data, scaler = apply_min_max_scaling(sequences)
        
        # Split into contexts and targets
        contexts, targets = create_context_target_split(scaled_data, context_length, target_length)
        
        # Split into train/val/test
        data_splits = split_data(contexts, targets, CONFIG['train_ratio'], CONFIG['val_ratio'], CONFIG['test_ratio'])
        
        # Create embedding metadata
        embedding_metadata = {
            'exchanges': CONFIG['exchanges'],
            'trading_pairs': CONFIG['trading_pairs'],
            'num_features': combined_data.shape[1],
            'features_per_exchange_pair': combined_data.shape[1] // (len(CONFIG['exchanges']) * len(CONFIG['trading_pairs'])),
            'context_length': CONFIG['context_length'],
            'target_length': CONFIG['target_length'],
            'total_sequences': len(sequences),
            'sequence_step_size': step_size,
            'overlap_ratio': CONFIG['overlap_ratio']
        }
        
        # Save all processed data
        dataset_stats = save_processed_data(data_splits, scaler, embedding_metadata, CONFIG['output_path'])
        
        logger.info("Data preparation completed successfully!")
        logger.info(f"Final dataset: {dataset_stats['total_sequences']} sequences with {dataset_stats['num_features']} features")
        logger.info(f"Feature breakdown: {len(CONFIG['exchanges'])} exchanges × {len(CONFIG['trading_pairs'])} pairs = {len(CONFIG['exchanges']) * len(CONFIG['trading_pairs'])} combinations")
        
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 