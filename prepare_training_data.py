#!/usr/bin/env python3

import os
import numpy as np
import pickle
import json

def prepare_training_data():
    """Convert attention model data to training format"""
    
    # Load the existing processed data
    data_dir = os.path.join(os.getcwd(), 'data', 'final_attention')
    
    print("Loading processed attention model data...")
    
    # Load train data
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    val_data = np.load(os.path.join(data_dir, 'validation.npz'))
    
    print(f"Train data keys: {list(train_data.keys())}")
    print(f"Validation data keys: {list(val_data.keys())}")
    
    # Load config to understand the data structure
    with open(os.path.join(data_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    print(f"Config: {config}")
    
    # Get sequences and targets using correct keys
    train_x = train_data['x']  # Context sequences [N, context_length, features]
    train_y = train_data['y']  # Target sequences [N, target_length, features]
    
    val_x = val_data['x']
    val_y = val_data['y']
    
    print(f"Train X shape: {train_x.shape}")
    print(f"Train Y shape: {train_y.shape}")
    print(f"Val X shape: {val_x.shape}")
    print(f"Val Y shape: {val_y.shape}")
    
    # Combine sequences and targets to create full sequences
    # The training script expects [N, context_length + 1, features] 
    # where the last timestep is the target
    
    context_length = config['context_length']  # Should be 120
    target_length = 1  # We only predict 1 step ahead for the transformer
    
    print(f"Context length: {context_length}")
    print(f"Original target length: {config['target_length']}")
    print(f"Using target length: {target_length}")
    
    # Create combined sequences: [context] + [first_target_step]
    train_combined = np.concatenate([
        train_x,                    # [N, 120, features] - context
        train_y[:, :1, :]          # [N, 1, features] - just first target step
    ], axis=1)  # [N, 121, features]
    
    val_combined = np.concatenate([
        val_x,                     # [N, 120, features] - context  
        val_y[:, :1, :]           # [N, 1, features] - just first target step
    ], axis=1)
    
    # Combine train and validation for the final dataset
    all_sequences = np.concatenate([train_combined, val_combined], axis=0)
    
    print(f"Combined sequences shape: {all_sequences.shape}")
    print(f"Total sequences: {len(all_sequences)}")
    
    # Create the data structure expected by training script
    training_data = {
        'sequences': all_sequences,
        'context_length': context_length,
        'target_length': target_length,
        'n_features': all_sequences.shape[-1],
        'n_sequences': len(all_sequences)
    }
    
    # Save the training data
    output_path = os.path.join(os.getcwd(), 'data', 'lob_sequences_btc_1min.pkl')
    
    print(f"Saving training data to: {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    print("Training data saved successfully!")
    print(f"Data summary:")
    print(f"  - Sequences: {training_data['n_sequences']}")
    print(f"  - Features: {training_data['n_features']}")
    print(f"  - Context length: {training_data['context_length']}")
    print(f"  - Target length: {training_data['target_length']}")
    print(f"  - Total sequence length: {training_data['context_length'] + training_data['target_length']}")

if __name__ == "__main__":
    prepare_training_data() 