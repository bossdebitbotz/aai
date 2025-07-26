#!/usr/bin/env python3
"""
Attention-based LOB Data Preparation (Version 3)
Fixes major issue with session detection - treats each file as independent session
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler
import json

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

def load_and_process_files() -> List[np.ndarray]:
    """Load and process each file independently to maximize sequence generation."""
    logger.info("Loading and processing files independently...")
    
    data_path = CONFIG['data_path']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Get all resampled files
    files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
    logger.info(f"Found {len(files)} resampled parquet files")
    
    all_sequences = []
    total_sequences = 0
    
    context_length = CONFIG['context_length']
    target_length = CONFIG['target_length']
    sequence_length = context_length + target_length
    step_size = int(context_length * (1 - CONFIG['overlap_ratio']))
    
    logger.info(f"Sequence parameters: context={context_length}, target={target_length}, step_size={step_size}")
    
    # Process each file independently
    for file in files:
        try:
            file_path = os.path.join(data_path, file)
            df = pd.read_parquet(file_path)
            
            if len(df) < sequence_length:
                logger.debug(f"Skipping {file}: too short ({len(df)} < {sequence_length})")
                continue
                
            # Apply percent-change transformation
            df_transformed = apply_percent_change_transformation(df)
            
            # Create sequences from this file
            file_sequences = create_sequences_from_file(df_transformed, context_length, target_length, step_size)
            
            if len(file_sequences) > 0:
                all_sequences.extend(file_sequences)
                total_sequences += len(file_sequences)
                logger.debug(f"Created {len(file_sequences)} sequences from {file}")
            
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            continue
    
    logger.info(f"Total sequences created: {total_sequences}")
    return all_sequences

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

def create_sequences_from_file(df: pd.DataFrame, context_length: int, target_length: int, step_size: int) -> List[np.ndarray]:
    """Create sequences from a single file."""
    # Select only numeric columns for the model
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data = df[numeric_cols].values
    
    if len(data) < context_length + target_length:
        return []
    
    sequences = []
    sequence_length = context_length + target_length
    
    # Create overlapping sequences
    for i in range(0, len(data) - sequence_length + 1, step_size):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    
    return sequences

def create_compound_multivariate_embedding(all_sequences: List[np.ndarray]) -> Tuple[np.ndarray, Dict]:
    """Create compound multivariate embedding structure for the dataset."""
    logger.info("Creating compound multivariate embedding structure...")
    
    # Convert list of sequences to numpy array
    if not all_sequences:
        raise ValueError("No sequences to process")
    
    # Stack all sequences
    combined_sequences = np.array(all_sequences)
    logger.info(f"Combined sequences shape: {combined_sequences.shape}")
    
    # Create embedding metadata
    embedding_metadata = {
        'exchanges': CONFIG['exchanges'],
        'trading_pairs': CONFIG['trading_pairs'],
        'num_features': combined_sequences.shape[-1],
        'context_length': CONFIG['context_length'],
        'target_length': CONFIG['target_length'],
        'total_sequences': len(combined_sequences)
    }
    
    return combined_sequences, embedding_metadata

def apply_min_max_scaling(data: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """Apply min-max scaling to all features."""
    logger.info("Applying min-max scaling...")
    
    # Reshape data for scaling: (sequences, timesteps, features) -> (sequences*timesteps, features)
    original_shape = data.shape
    reshaped_data = data.reshape(-1, data.shape[-1])
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(reshaped_data)
    
    # Reshape back to original shape
    scaled_data = scaled_data.reshape(original_shape)
    
    logger.info(f"Applied min-max scaling to {data.shape[-1]} features")
    return scaled_data, scaler

def create_context_target_split(data: np.ndarray, context_length: int, target_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split sequences into context and target portions."""
    contexts = data[:, :context_length, :]
    targets = data[:, context_length:context_length + target_length, :]
    
    logger.info(f"Split into contexts: {contexts.shape}, targets: {targets.shape}")
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
        'context_length': CONFIG['context_length'],
        'target_length': CONFIG['target_length'],
        'num_features': embedding_metadata['num_features'],
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
        json.dump(dataset_stats, f, indent=2, default=str)
    
    logger.info("All data saved successfully!")
    return dataset_stats

def main():
    """Main data preparation pipeline."""
    logger.info("Starting attention-based LOB data preparation (Version 3)...")
    logger.info(f"Configuration: {CONFIG}")
    
    try:
        # Load and process all files independently
        all_sequences = load_and_process_files()
        
        if not all_sequences:
            raise ValueError("No sequences were created. Check your data files.")
        
        # Create compound multivariate embedding
        combined_data, embedding_metadata = create_compound_multivariate_embedding(all_sequences)
        
        # Apply scaling
        scaled_data, scaler = apply_min_max_scaling(combined_data)
        
        # Split into contexts and targets
        contexts, targets = create_context_target_split(
            scaled_data, CONFIG['context_length'], CONFIG['target_length']
        )
        
        # Split into train/val/test
        data_splits = split_data(
            contexts, targets,
            CONFIG['train_ratio'], CONFIG['val_ratio'], CONFIG['test_ratio']
        )
        
        # Save all processed data
        dataset_stats = save_processed_data(data_splits, scaler, embedding_metadata, CONFIG['output_path'])
        
        logger.info("Data preparation completed successfully!")
        logger.info(f"Final dataset: {dataset_stats['total_sequences']} sequences with {dataset_stats['num_features']} features")
        
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 