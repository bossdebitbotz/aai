#!/usr/bin/env python3

import os
import numpy as np
import pickle
import json

def prepare_challenging_lob_data():
    """Convert existing attention model data to a more challenging prediction task"""
    
    # Load the existing processed data
    data_dir = os.path.join(os.getcwd(), 'data', 'final_attention')
    
    print("Loading existing processed attention model data...")
    
    # Load train data
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    val_data = np.load(os.path.join(data_dir, 'validation.npz'))
    
    train_x = train_data['x']  # Context sequences [N, 120, 240]
    train_y = train_data['y']  # Target sequences [N, 24, 240]
    
    val_x = val_data['x']
    val_y = val_data['y']
    
    print(f"Train X shape: {train_x.shape}")
    print(f"Train Y shape: {train_y.shape}")
    print(f"Val X shape: {val_x.shape}")
    print(f"Val Y shape: {val_y.shape}")
    
    # Strategy 1: Predict further into the future (10 steps ahead instead of 1)
    print("Creating challenging prediction task: 10 steps ahead...")
    
    # Use only every 2nd timestep to reduce autocorrelation
    print("Subsampling to reduce autocorrelation...")
    train_x_sub = train_x[:, ::2, :]  # [N, 60, 240] - every 2nd timestep
    train_y_sub = train_y[:, ::2, :]  # [N, 12, 240] - every 2nd timestep
    
    val_x_sub = val_x[:, ::2, :]
    val_y_sub = val_y[:, ::2, :]
    
    print(f"After subsampling - Train X: {train_x_sub.shape}, Train Y: {train_y_sub.shape}")
    
    # Create sequences where we predict 10 steps ahead
    target_steps_ahead = 10
    context_length = 60  # Use 60 timesteps as context
    
    train_sequences = []
    val_sequences = []
    
    # Process training data
    for i in range(len(train_x_sub)):
        context = train_x_sub[i]  # [60, 240]
        targets = train_y_sub[i]  # [12, 240]
        
        if len(targets) >= target_steps_ahead:
            # Predict the 10th step ahead
            target = targets[target_steps_ahead-1:target_steps_ahead]  # [1, 240]
            
            # Combine context + target for training format
            sequence = np.concatenate([context, target], axis=0)  # [61, 240]
            train_sequences.append(sequence)
    
    # Process validation data
    for i in range(len(val_x_sub)):
        context = val_x_sub[i]  # [60, 240]
        targets = val_y_sub[i]  # [12, 240]
        
        if len(targets) >= target_steps_ahead:
            target = targets[target_steps_ahead-1:target_steps_ahead]  # [1, 240]
            sequence = np.concatenate([context, target], axis=0)  # [61, 240]
            val_sequences.append(sequence)
    
    train_sequences = np.array(train_sequences)
    val_sequences = np.array(val_sequences)
    
    print(f"Created {len(train_sequences)} training sequences")
    print(f"Created {len(val_sequences)} validation sequences")
    print(f"Sequence shape: {train_sequences.shape}")
    
    # Combine all sequences
    all_sequences = np.concatenate([train_sequences, val_sequences], axis=0)
    
    # Strategy 2: Create price direction prediction task
    print("Creating price direction prediction task...")
    
    # Extract first few features (bid/ask prices) for direction prediction
    price_features = all_sequences[:, :, :20]  # First 20 features (likely price-related)
    
    # Calculate price movement direction (up/down) for each price feature
    context_prices = price_features[:, :-1, :]  # [N, 60, 20]
    target_prices = price_features[:, -1:, :]   # [N, 1, 20]
    
    # Create direction targets: 1 if price goes up, 0 if down
    last_context_prices = context_prices[:, -1:, :]  # [N, 1, 20]
    price_changes = target_prices - last_context_prices  # [N, 1, 20]
    direction_targets = (price_changes > 0).astype(np.float32)  # [N, 1, 20]
    
    # Combine with original features for multi-task learning
    print("Creating multi-task targets (price level + direction)...")
    
    # Original target (price levels)
    price_level_targets = all_sequences[:, -1:, :]  # [N, 1, 240]
    
    # Direction targets (expanded to match dimensions)
    direction_expanded = np.zeros_like(price_level_targets)
    direction_expanded[:, :, :20] = direction_targets  # Only first 20 features get direction
    
    # Combine: 50% weight on price levels, 50% weight on direction
    combined_targets = 0.5 * price_level_targets + 0.5 * direction_expanded
    
    # Create final training data
    print("Creating final challenging dataset...")
    
    # Use the original context length but with the challenging multi-task target
    contexts = all_sequences[:, :-1, :]  # [N, 60, 240]
    targets = combined_targets            # [N, 1, 240]
    
    # Recombine for the expected format
    final_sequences = np.concatenate([contexts, targets], axis=1)  # [N, 61, 240]
    
    print(f"Final sequences shape: {final_sequences.shape}")
    
    # Check the new autocorrelation
    print("Analyzing new data characteristics...")
    
    # Calculate variance
    sequence_variance = np.var(final_sequences, axis=1).mean()
    print(f"Average variance within sequences: {sequence_variance:.6f}")
    
    # Calculate autocorrelation between consecutive timesteps
    autocorr_sum = 0
    for i in range(final_sequences.shape[1] - 1):
        corr = np.corrcoef(final_sequences[:, i, :].flatten(), 
                          final_sequences[:, i+1, :].flatten())[0, 1]
        autocorr_sum += corr
    
    avg_autocorr = autocorr_sum / (final_sequences.shape[1] - 1)
    print(f"Average temporal autocorrelation: {avg_autocorr:.6f}")
    
    # Create the data structure expected by training script
    training_data = {
        'sequences': final_sequences,
        'context_length': contexts.shape[1],  # 60
        'target_length': 1,
        'n_features': final_sequences.shape[-1],
        'n_sequences': len(final_sequences),
        'prediction_type': 'multi_task_10_steps_ahead',
        'autocorrelation': avg_autocorr,
        'variance': sequence_variance
    }
    
    # Save the challenging training data
    output_path = os.path.join(os.getcwd(), 'data', 'lob_sequences_btc_1min.pkl')
    
    print(f"Saving challenging training data to: {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    print("Challenging training data saved successfully!")
    print(f"Data summary:")
    print(f"  - Sequences: {training_data['n_sequences']}")
    print(f"  - Features: {training_data['n_features']}")
    print(f"  - Context length: {training_data['context_length']}")
    print(f"  - Target length: {training_data['target_length']}")
    print(f"  - Prediction task: {training_data['prediction_type']}")
    print(f"  - Autocorrelation: {training_data['autocorrelation']:.6f}")
    print(f"  - Variance: {training_data['variance']:.6f}")
    
    if avg_autocorr < 0.9:
        print("✅ Good! Autocorrelation is now below 0.9")
    else:
        print("⚠️  Autocorrelation still high, prediction may still be too easy")

if __name__ == "__main__":
    prepare_challenging_lob_data() 