#!/usr/bin/env python3
"""
Binance Perpetual-Only LOB Data Preparation

This script extracts only the Binance perpetual features from the full 240-feature dataset
to create a specialized 80-feature dataset for training a Binance perp-focused model.

Key Features:
1. Extracts 80 Binance perp features (4 pairs × 20 features each)
2. Maintains same temporal structure (120 context, 24 target)
3. Creates specialized embedding metadata for perp-only features
4. Optimized for perpetual market characteristics
"""

import os
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SOURCE_DATA_DIR = "data/final_attention"
OUTPUT_DATA_DIR = "data/final_binance_perp"
CONTEXT_LENGTH = 120  # 120 steps * 5s = 10 minutes
TARGET_LENGTH = 24    # 24 steps * 5s = 2 minutes

# Binance perp specific configuration
BINANCE_PERP_PAIRS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT']
LOB_LEVELS = 5

def load_source_metadata() -> Dict:
    """Load the original 240-feature metadata."""
    logger.info("Loading source embedding metadata...")
    
    metadata_path = os.path.join(SOURCE_DATA_DIR, 'embedding_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded metadata for {metadata['num_features']} features")
    return metadata

def extract_binance_perp_indices(source_metadata: Dict) -> List[int]:
    """Extract the column indices for Binance perpetual features only."""
    logger.info("Extracting Binance perpetual feature indices...")
    
    binance_perp_indices = []
    column_mapping = source_metadata['column_mapping']
    
    for col_name, col_info in column_mapping.items():
        if col_info['exchange'] == 'binance_perp':
            binance_perp_indices.append(col_info['column_idx'])
    
    binance_perp_indices.sort()  # Ensure proper ordering
    
    logger.info(f"Found {len(binance_perp_indices)} Binance perpetual features")
    logger.info(f"Index range: {min(binance_perp_indices)} to {max(binance_perp_indices)}")
    
    # Verify we have the expected number
    expected_features = len(BINANCE_PERP_PAIRS) * LOB_LEVELS * 2 * 2  # pairs × levels × types × features
    if len(binance_perp_indices) != expected_features:
        raise ValueError(f"Expected {expected_features} features, found {len(binance_perp_indices)}")
    
    return binance_perp_indices

def create_binance_perp_metadata(source_metadata: Dict, perp_indices: List[int]) -> Dict:
    """Create new metadata structure for Binance perp-only features."""
    logger.info("Creating Binance perpetual metadata structure...")
    
    # Extract original columns and mappings for perp features
    source_columns = source_metadata['columns']
    source_mapping = source_metadata['column_mapping']
    
    perp_metadata = {
        'exchanges': [],
        'trading_pairs': [],
        'order_types': [],
        'features': [],
        'levels': [],
        'column_mapping': {},
        'unique_attributes': {
            'exchanges': ['binance_perp'],  # Only one exchange
            'trading_pairs': sorted(BINANCE_PERP_PAIRS),
            'order_types': ['bid', 'ask'],
            'features': ['price', 'volume'],
            'levels': list(range(1, LOB_LEVELS + 1))
        },
        'columns': [],
        'num_features': len(perp_indices)
    }
    
    # Create new column mapping with renumbered indices
    for new_idx, orig_idx in enumerate(perp_indices):
        orig_col_name = source_columns[orig_idx]
        orig_col_info = source_mapping[orig_col_name]
        
        # Add to lists
        perp_metadata['exchanges'].append(orig_col_info['exchange'])
        perp_metadata['trading_pairs'].append(orig_col_info['trading_pair'])
        perp_metadata['order_types'].append(orig_col_info['order_type'])
        perp_metadata['features'].append(orig_col_info['feature_type'])
        perp_metadata['levels'].append(orig_col_info['level'])
        perp_metadata['columns'].append(orig_col_name)
        
        # Create new mapping with updated column index
        perp_metadata['column_mapping'][orig_col_name] = {
            'exchange': orig_col_info['exchange'],
            'trading_pair': orig_col_info['trading_pair'],
            'order_type': orig_col_info['order_type'],
            'feature_type': orig_col_info['feature_type'],
            'level': orig_col_info['level'],
            'column_idx': new_idx  # New index in the 80-feature space
        }
    
    logger.info(f"Created metadata for {perp_metadata['num_features']} Binance perp features")
    logger.info(f"Pairs: {perp_metadata['unique_attributes']['trading_pairs']}")
    
    return perp_metadata

def extract_perp_data(source_file: str, output_file: str, perp_indices: List[int]) -> Dict:
    """Extract Binance perp features from source data file."""
    logger.info(f"Extracting Binance perp data from {source_file}...")
    
    # Load source data
    with np.load(source_file) as data:
        source_x = data['x']  # Shape: (sequences, context_len, 240_features)
        source_y = data['y']  # Shape: (sequences, target_len, 240_features)
    
    logger.info(f"Source data shape - X: {source_x.shape}, Y: {source_y.shape}")
    
    # Extract only Binance perp features
    perp_x = source_x[:, :, perp_indices]  # Shape: (sequences, context_len, 80_features)
    perp_y = source_y[:, :, perp_indices]  # Shape: (sequences, target_len, 80_features)
    
    logger.info(f"Extracted perp data shape - X: {perp_x.shape}, Y: {perp_y.shape}")
    
    # Save extracted data
    np.savez_compressed(output_file, x=perp_x, y=perp_y)
    
    stats = {
        'num_sequences': perp_x.shape[0],
        'context_length': perp_x.shape[1],
        'target_length': perp_y.shape[1],
        'num_features': perp_x.shape[2],
        'file_size_mb': os.path.getsize(output_file) / (1024 * 1024)
    }
    
    logger.info(f"Saved {stats['num_sequences']} sequences to {output_file}")
    logger.info(f"File size: {stats['file_size_mb']:.1f} MB")
    
    return stats

def create_dataset_summary(perp_metadata: Dict, stats: Dict) -> Dict:
    """Create a summary of the Binance perp dataset."""
    summary = {
        'dataset_type': 'binance_perp_only',
        'description': 'Specialized dataset containing only Binance perpetual features',
        'created_from': 'data/final_attention (240-feature dataset)',
        'num_features': perp_metadata['num_features'],
        'feature_breakdown': {
            'exchanges': 1,
            'trading_pairs': len(perp_metadata['unique_attributes']['trading_pairs']),
            'levels': len(perp_metadata['unique_attributes']['levels']),
            'types': len(perp_metadata['unique_attributes']['order_types']),
            'features': len(perp_metadata['unique_attributes']['features'])
        },
        'temporal_structure': {
            'context_length': CONTEXT_LENGTH,
            'target_length': TARGET_LENGTH,
            'frequency': '5_seconds'
        },
        'data_statistics': stats,
        'trading_pairs': perp_metadata['unique_attributes']['trading_pairs'],
        'optimization_target': 'binance_perpetual_execution'
    }
    
    return summary

def main():
    """Main execution function."""
    logger.info("Starting Binance Perpetual data extraction...")
    
    # Create output directory
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    # Load source metadata and extract perp indices
    source_metadata = load_source_metadata()
    perp_indices = extract_binance_perp_indices(source_metadata)
    
    # Create new metadata for perp-only features
    perp_metadata = create_binance_perp_metadata(source_metadata, perp_indices)
    
    # Extract data for each split
    data_files = ['train.npz', 'validation.npz', 'test.npz']
    all_stats = {}
    
    for data_file in data_files:
        source_path = os.path.join(SOURCE_DATA_DIR, data_file)
        output_path = os.path.join(OUTPUT_DATA_DIR, data_file)
        
        if not os.path.exists(source_path):
            logger.warning(f"Source file not found: {source_path}")
            continue
        
        stats = extract_perp_data(source_path, output_path, perp_indices)
        all_stats[data_file.replace('.npz', '')] = stats
    
    # Save metadata
    metadata_path = os.path.join(OUTPUT_DATA_DIR, 'embedding_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(perp_metadata, f, indent=2)
    logger.info(f"Saved Binance perp metadata to {metadata_path}")
    
    # Create and save dataset summary
    summary = create_dataset_summary(perp_metadata, all_stats)
    summary_path = os.path.join(OUTPUT_DATA_DIR, 'dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved dataset summary to {summary_path}")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("BINANCE PERPETUAL DATA EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {OUTPUT_DATA_DIR}")
    logger.info(f"Features: {perp_metadata['num_features']} (reduced from 240)")
    logger.info(f"Trading pairs: {', '.join(perp_metadata['unique_attributes']['trading_pairs'])}")
    
    for split, stats in all_stats.items():
        logger.info(f"{split.capitalize()}: {stats['num_sequences']} sequences, {stats['file_size_mb']:.1f} MB")
    
    logger.info("\nNext steps:")
    logger.info("1. Train specialized Binance perp model")
    logger.info("2. Compare performance to general 240-feature model")
    logger.info("3. Optimize for perpetual market characteristics")

if __name__ == "__main__":
    main() 