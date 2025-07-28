#!/usr/bin/env python3
"""
Debug script to check actual data shapes and determine correct target_steps
"""

import numpy as np
import json
import os

def debug_data_shapes():
    """Check the actual shapes in the data files."""
    print("🔍 Debugging Data Shapes...")
    
    data_dir = "data/final_attention"
    
    # Load metadata
    with open(os.path.join(data_dir, 'embedding_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Metadata:")
    print(f"  - Total features: {metadata['num_features']}")
    print(f"  - Total columns: {len(metadata['columns'])}")
    
    # Check all data files
    for split in ['train', 'validation', 'test']:
        file_path = os.path.join(data_dir, f'{split}.npz')
        if os.path.exists(file_path):
            with np.load(file_path) as data:
                print(f"\n📁 {split.upper()} data:")
                print(f"  - Keys: {list(data.keys())}")
                if 'contexts' in data:
                    contexts = data['contexts']
                    print(f"  - Contexts shape: {contexts.shape}")
                if 'targets' in data:
                    targets = data['targets']
                    print(f"  - Targets shape: {targets.shape}")
                    print(f"  - Target time steps: {targets.shape[1]}")
                    print(f"  - Target features: {targets.shape[2]}")
    
    # Find WLD binance_perp indices
    wld_indices = []
    for i, col_name in enumerate(metadata['columns']):
        col_info = metadata['column_mapping'][col_name]
        if (col_info['exchange'] == 'binance_perp' and 
            col_info['trading_pair'] == 'WLD-USDT'):
            wld_indices.append(i)
    
    print(f"\n🎯 WLD Binance Perp Analysis:")
    print(f"  - WLD feature count: {len(wld_indices)}")
    print(f"  - WLD indices: {wld_indices}")
    
    # Check a sample to see actual target steps
    if os.path.exists(os.path.join(data_dir, 'train.npz')):
        with np.load(os.path.join(data_dir, 'train.npz')) as data:
            targets = data['targets']
            sample_target = targets[0, :, wld_indices[0]]  # First sequence, first WLD feature
            
            print(f"\n📈 Sample Target Analysis (first WLD feature):")
            print(f"  - Sample values: {sample_target[:10]}")
            print(f"  - Non-zero steps: {np.count_nonzero(sample_target)}")
            print(f"  - Target length used in training: {targets.shape[1]}")
            
            # Check what target_steps was likely used
            print(f"\n💡 Recommendation:")
            print(f"  - Use target_steps = {targets.shape[1]} (full target length)")

if __name__ == "__main__":
    debug_data_shapes() 