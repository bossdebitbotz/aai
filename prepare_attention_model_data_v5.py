#!/usr/bin/env python3
"""
Attention-based LOB Data Preparation (Version 5)
FIXES: Uses temporal alignment and missing value handling instead of intersection
RESULT: Should generate 100x more sequences by properly handling missing data
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime, timedelta

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
    'min_sequence_length': 144,  # 120 context + 24 target
    'step_size': 12,  # 90% overlap
    'missing_value': 0.0,  # Value to use for missing data
    'max_missing_ratio': 0.3,  # Max 30% missing values per sequence
}

def parse_filename(filename: str) -> Tuple[str, str, str]:
    """Parse filename to extract exchange, pair, and date."""
    parts = filename.replace('.parquet', '').split('_')
    if len(parts) >= 4:
        exchange = f"{parts[0]}_{parts[1]}" if len(parts) > 4 else parts[0]
        pair = parts[2] if len(parts) > 4 else parts[1]
        date = parts[3] if len(parts) > 4 else parts[2]
        return exchange, pair, date
    raise ValueError(f"Cannot parse filename: {filename}")

def load_and_create_temporal_grid() -> pd.DataFrame:
    """
    Load all files and create a temporal grid with missing value handling.
    Instead of intersection, we create a comprehensive time grid and fill missing values.
    """
    logger.info("Loading data with temporal grid approach...")
    
    data_path = CONFIG['data_path']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Get all files
    files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
    logger.info(f"Found {len(files)} parquet files")
    
    # Load all data and track time ranges
    all_data = {}
    all_timestamps = set()
    
    for file in files:
        try:
            exchange, pair, date = parse_filename(file)
            key = f"{exchange}_{pair}"
            
            file_path = os.path.join(data_path, file)
            df = pd.read_parquet(file_path)
            
            # Ensure timestamp handling
            if 'timestamp' not in df.columns:
                if df.index.name != 'timestamp':
                    df.index.name = 'timestamp'
                df = df.reset_index()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Apply percent change transformation
            df = apply_percent_change_transformation(df)
            
            # Add to collection
            if key not in all_data:
                all_data[key] = []
            
            all_data[key].append(df)
            all_timestamps.update(df.index)
            
            logger.info(f"Loaded {file}: {len(df)} records")
            
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
            continue
    
    logger.info(f"Loaded data for {len(all_data)} exchange/pair combinations")
    logger.info(f"Total unique timestamps: {len(all_timestamps):,}")
    
    # Create comprehensive time grid
    all_timestamps = sorted(all_timestamps)
    time_grid = pd.DatetimeIndex(all_timestamps)
    
    # Combine data for each exchange/pair
    combined_data = {}
    for key, dataframes in all_data.items():
        # Concatenate all sessions for this exchange/pair
        combined_df = pd.concat(dataframes, axis=0)
        combined_df = combined_df.sort_index()
        
        # Remove duplicates
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        # Reindex to full time grid with missing values
        combined_df = combined_df.reindex(time_grid, fill_value=CONFIG['missing_value'])
        
        # Rename columns with exchange/pair prefix
        combined_df.columns = [f"{key}_{col}" for col in combined_df.columns]
        
        combined_data[key] = combined_df
        logger.info(f"Combined {key}: {len(combined_df)} timestamps")
    
    # Concatenate all exchange/pair data horizontally
    final_data = pd.concat(list(combined_data.values()), axis=1)
    
    logger.info(f"Final temporal grid shape: {final_data.shape}")
    logger.info(f"Time range: {final_data.index.min()} to {final_data.index.max()}")
    
    return final_data

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

def create_sequences_with_missing_handling(df: pd.DataFrame) -> List[np.ndarray]:
    """
    Create sequences with intelligent missing value handling.
    Reject sequences with too many missing values but keep valid ones.
    """
    logger.info("Creating sequences with missing value handling...")
    
    context_length = CONFIG['context_length']
    target_length = CONFIG['target_length']
    sequence_length = context_length + target_length
    step_size = CONFIG['step_size']
    max_missing_ratio = CONFIG['max_missing_ratio']
    
    # Ensure all columns are numeric
    logger.info(f"DataFrame shape before cleaning: {df.shape}")
    logger.info(f"DataFrame dtypes: {df.dtypes.value_counts()}")
    
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]
    
    logger.info(f"After selecting numeric columns: {df_numeric.shape}")
    logger.info(f"Numeric dtypes: {df_numeric.dtypes.value_counts()}")
    
    data_array = df_numeric.values
    timestamps = df.index
    
    sequences = []
    valid_sequences = 0
    rejected_sequences = 0
    
    # Generate sequences with overlap
    for start_idx in range(0, len(data_array) - sequence_length + 1, step_size):
        end_idx = start_idx + sequence_length
        
        sequence = data_array[start_idx:end_idx]
        
        # Check missing value ratio
        missing_ratio = (sequence == CONFIG['missing_value']).mean()
        
        if missing_ratio <= max_missing_ratio:
            sequences.append(sequence)
            valid_sequences += 1
        else:
            rejected_sequences += 1
    
    logger.info(f"Generated sequences:")
    logger.info(f"  Valid: {valid_sequences:,}")
    logger.info(f"  Rejected (too many missing): {rejected_sequences:,}")
    logger.info(f"  Total attempted: {valid_sequences + rejected_sequences:,}")
    logger.info(f"  Success rate: {valid_sequences/(valid_sequences + rejected_sequences)*100:.1f}%")
    
    if len(sequences) == 0:
        raise ValueError("No valid sequences generated")
    
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

def create_embedding_metadata(df: pd.DataFrame) -> Dict:
    """Create metadata for compound multivariate embedding."""
    logger.info("Creating embedding metadata...")
    
    columns = df.columns.tolist()
    exchanges = CONFIG['exchanges']
    trading_pairs = CONFIG['trading_pairs']
    levels = list(range(1, CONFIG['lob_levels'] + 1))
    order_types = ['bid', 'ask']
    features = ['price', 'volume']
    
    # Create column mapping
    column_mapping = {}
    for i, col in enumerate(columns):
        # Parse column name: exchange_pair_feature_info
        parts = col.split('_')
        if len(parts) >= 4:
            exchange = f"{parts[0]}_{parts[1]}"
            pair = parts[2]
            feature_info = "_".join(parts[3:])
            
            # Extract level and type from feature_info
            if 'bid' in feature_info:
                order_type = 'bid'
                level_str = feature_info.replace('bid_', '').replace('_price', '').replace('_volume', '')
            elif 'ask' in feature_info:
                order_type = 'ask' 
                level_str = feature_info.replace('ask_', '').replace('_price', '').replace('_volume', '')
            else:
                continue
                
            try:
                level = int(level_str.split('_')[0])
            except:
                level = 1
                
            feature_type = 'price' if 'price' in feature_info else 'volume'
            
            column_mapping[col] = {
                'index': i,
                'exchange': exchange,
                'trading_pair': pair,
                'level': level,
                'order_type': order_type,
                'feature_type': feature_type
            }
    
    metadata = {
        'columns': columns,
        'column_mapping': column_mapping,
        'unique_attributes': {
            'exchanges': exchanges,
            'trading_pairs': trading_pairs,
            'levels': levels,
            'order_types': order_types,
            'features': features
        },
        'num_features': len(columns),
        'context_length': CONFIG['context_length'],
        'target_length': CONFIG['target_length']
    }
    
    logger.info(f"Created metadata for {len(columns)} features")
    return metadata

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
    
    # Save dataset statistics
    stats = {
        'total_sequences': len(train_contexts) + len(val_contexts) + len(test_contexts),
        'train_sequences': len(train_contexts),
        'validation_sequences': len(val_contexts),
        'test_sequences': len(test_contexts),
        'feature_count': train_contexts.shape[-1],
        'context_length': train_contexts.shape[1],
        'target_length': train_targets.shape[1],
        'data_shape': {
            'train_contexts': train_contexts.shape,
            'train_targets': train_targets.shape,
            'val_contexts': val_contexts.shape,
            'val_targets': val_targets.shape,
            'test_contexts': test_contexts.shape,
            'test_targets': test_targets.shape
        },
        'missing_value_handling': {
            'missing_value': CONFIG['missing_value'],
            'max_missing_ratio': CONFIG['max_missing_ratio'],
            'step_size': CONFIG['step_size']
        }
    }
    
    with open(os.path.join(output_path, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("✅ All data saved successfully!")
    logger.info(f"📊 Dataset Summary:")
    logger.info(f"   Total sequences: {stats['total_sequences']:,}")
    logger.info(f"   Features: {stats['feature_count']}")
    logger.info(f"   Context length: {stats['context_length']}")
    logger.info(f"   Target length: {stats['target_length']}")

def main():
    """Main execution function."""
    logger.info("🚀 Starting LOB attention model data preparation (V5 - Fixed)")
    logger.info("=" * 60)
    
    try:
        # 1. Load and create temporal grid with missing value handling
        df = load_and_create_temporal_grid()
        
        # 2. Create sequences with intelligent missing value handling
        sequences = create_sequences_with_missing_handling(df)
        
        # 3. Apply scaling
        scaled_sequences, scaler = apply_min_max_scaling(sequences)
        
        # 4. Split into context and target
        contexts, targets = create_context_target_split(
            scaled_sequences, CONFIG['context_length'], CONFIG['target_length']
        )
        
        # 5. Split into train/val/test
        data_splits = split_data(
            contexts, targets, 
            CONFIG['train_ratio'], CONFIG['val_ratio'], CONFIG['test_ratio']
        )
        
        # 6. Create embedding metadata (need to pass the original df before sequence generation)
        # Recreate the clean dataframe for metadata
        df_clean = load_and_create_temporal_grid()
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean_numeric = df_clean[numeric_columns]
        embedding_metadata = create_embedding_metadata(df_clean_numeric)
        
        # 7. Save everything
        save_processed_data(data_splits, scaler, embedding_metadata, CONFIG['output_path'])
        
        logger.info("🎉 Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in data preparation: {e}")
        raise

if __name__ == "__main__":
    main() 