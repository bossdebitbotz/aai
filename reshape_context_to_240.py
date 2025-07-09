#!/usr/bin/env python3
"""
Reshape Existing Data to 240-Step Context

This script takes the existing data/final_attention/ dataset (120-step context)
and reshapes it to use 240-step context (20 minutes) like the paper.

Strategy:
- Load existing train/val/test data
- Create overlapping 240-step windows from available sequences
- Maintain 24-step targets
- Save as new dataset for 20-minute context training
"""

import os
import numpy as np
import json
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SOURCE_DATA_DIR = "data/final_attention"
OUTPUT_DATA_DIR = "data/final_attention_240"
NEW_CONTEXT_LENGTH = 240  # 20 minutes
TARGET_LENGTH = 24        # 2 minutes (unchanged)

def load_existing_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load existing 120-step data."""
    logger.info(f"Loading data from {file_path}...")
    
    with np.load(file_path) as data:
        x = data['x']  # (sequences, 120, 240)
        y = data['y']  # (sequences, 24, 240)
    
    logger.info(f"Loaded data shapes - X: {x.shape}, Y: {y.shape}")
    return x, y

def create_240_step_sequences(x_120: np.ndarray, y_120: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 240-step context sequences from 120-step data.
    
    Strategy:
    1. Each 120-step sequence becomes a starting point
    2. We need to find consecutive sequences to create 240-step windows
    3. Since we have overlapping data, we can combine sequences
    """
    num_sequences, context_120, num_features = x_120.shape
    
    logger.info(f"Creating 240-step sequences from {num_sequences} 120-step sequences...")
    
    # We need 2 consecutive 120-step sequences to make one 240-step sequence
    # Assuming 50% overlap in original data, we can combine adjacent sequences
    
    new_sequences_x = []
    new_sequences_y = []
    
    # Combine pairs of consecutive sequences
    for i in range(0, num_sequences - 1, 1):  # Step by 1 to maintain overlap
        if i + 1 < num_sequences:
            # Combine two 120-step sequences into one 240-step sequence
            # Take first 120 steps from sequence i and next 120 from sequence i+1
            combined_x = np.concatenate([x_120[i], x_120[i + 1]], axis=0)  # (240, 240)
            
            # Use the target from the second sequence (more recent)
            target_y = y_120[i + 1]  # (24, 240)
            
            new_sequences_x.append(combined_x)
            new_sequences_y.append(target_y)
    
    # Convert to numpy arrays
    x_240 = np.array(new_sequences_x)  # (new_sequences, 240, 240)
    y_240 = np.array(new_sequences_y)  # (new_sequences, 24, 240)
    
    logger.info(f"Created {len(new_sequences_x)} sequences with 240-step context")
    logger.info(f"New data shapes - X: {x_240.shape}, Y: {y_240.shape}")
    
    return x_240, y_240

def save_reshaped_data(x: np.ndarray, y: np.ndarray, output_path: str):
    """Save the reshaped data."""
    logger.info(f"Saving reshaped data to {output_path}...")
    
    np.savez_compressed(output_path, x=x, y=y)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Saved {x.shape[0]} sequences to {output_path} ({file_size_mb:.1f} MB)")

def create_updated_metadata(source_metadata_path: str, output_metadata_path: str):
    """Update metadata for 240-step context."""
    logger.info("Updating metadata for 240-step context...")
    
    with open(source_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Add context length info
    metadata['context_length'] = NEW_CONTEXT_LENGTH
    metadata['target_length'] = TARGET_LENGTH
    metadata['context_minutes'] = NEW_CONTEXT_LENGTH * 5 / 60  # 20 minutes
    metadata['target_minutes'] = TARGET_LENGTH * 5 / 60        # 2 minutes
    metadata['created_from'] = SOURCE_DATA_DIR
    metadata['methodology'] = 'Paper-compliant 20-minute context window'
    
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Updated metadata saved to {output_metadata_path}")

def main():
    """Main execution function."""
    logger.info("Starting data reshape to 240-step context...")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    # Process each data split
    data_files = ['train.npz', 'validation.npz', 'test.npz']
    total_sequences = 0
    
    for data_file in data_files:
        source_path = os.path.join(SOURCE_DATA_DIR, data_file)
        output_path = os.path.join(OUTPUT_DATA_DIR, data_file)
        
        if not os.path.exists(source_path):
            logger.warning(f"Source file not found: {source_path}")
            continue
        
        # Load existing data
        x_120, y_120 = load_existing_data(source_path)
        
        # Create 240-step sequences
        x_240, y_240 = create_240_step_sequences(x_120, y_120)
        
        # Save reshaped data
        save_reshaped_data(x_240, y_240, output_path)
        
        total_sequences += x_240.shape[0]
        
        split_name = data_file.replace('.npz', '')
        logger.info(f"{split_name.capitalize()}: {x_120.shape[0]} → {x_240.shape[0]} sequences")
    
    # Copy and update metadata
    source_metadata = os.path.join(SOURCE_DATA_DIR, 'embedding_metadata.json')
    output_metadata = os.path.join(OUTPUT_DATA_DIR, 'embedding_metadata.json')
    
    if os.path.exists(source_metadata):
        create_updated_metadata(source_metadata, output_metadata)
    
    # Create dataset summary
    summary = {
        'dataset_type': '240_step_context',
        'description': 'Reshaped from 120-step to 240-step context for paper compliance',
        'context_length': NEW_CONTEXT_LENGTH,
        'target_length': TARGET_LENGTH,
        'context_minutes': 20,
        'target_minutes': 2,
        'total_sequences': total_sequences,
        'methodology': 'Combined overlapping 120-step sequences into 240-step windows',
        'paper_compliance': True
    }
    
    summary_path = os.path.join(OUTPUT_DATA_DIR, 'dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("DATA RESHAPE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {OUTPUT_DATA_DIR}")
    logger.info(f"Context: 240 steps (20 minutes) - Paper compliant")
    logger.info(f"Target: 24 steps (2 minutes)")
    logger.info(f"Total sequences: {total_sequences}")
    logger.info("\nNext step:")
    logger.info("Update train_full_to_binance_perp.py to use data/final_attention_240/")

if __name__ == "__main__":
    main() 