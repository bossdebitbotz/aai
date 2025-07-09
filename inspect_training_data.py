#!/usr/bin/env python3

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

def inspect_training_data():
    """Inspect the training data to understand why model converges so fast"""
    
    # Load the training data
    data_path = os.path.join(os.getcwd(), 'data', 'lob_sequences_btc_1min.pkl')
    
    print("Loading training data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']  # [N, total_len, features]
    print(f"Data shape: {sequences.shape}")
    print(f"Data type: {sequences.dtype}")
    
    # Check basic statistics
    print(f"\nData statistics:")
    print(f"Min: {sequences.min():.6f}")
    print(f"Max: {sequences.max():.6f}")
    print(f"Mean: {sequences.mean():.6f}")
    print(f"Std: {sequences.std():.6f}")
    
    # Check for NaN or infinite values
    print(f"\nData quality:")
    print(f"NaN count: {np.isnan(sequences).sum()}")
    print(f"Infinite count: {np.isinf(sequences).sum()}")
    print(f"Zero count: {(sequences == 0).sum()}")
    
    # Look at first few sequences
    print(f"\nFirst sequence context (first 5 timesteps, first 10 features):")
    print(sequences[0, :5, :10])
    
    print(f"\nFirst sequence target (last timestep, first 10 features):")
    print(sequences[0, -1, :10])
    
    # Check if sequences are too similar (potential data leakage)
    print(f"\nChecking sequence diversity...")
    
    # Compare first 3 sequences
    for i in range(min(3, len(sequences))):
        for j in range(i+1, min(3, len(sequences))):
            diff = np.abs(sequences[i] - sequences[j]).mean()
            print(f"Average difference between sequence {i} and {j}: {diff:.6f}")
    
    # Check if context and target are too similar (major red flag)
    context = sequences[:, :-1, :]  # [N, 120, features]
    target = sequences[:, -1:, :]   # [N, 1, features]
    
    # Compare last context step with target
    last_context = sequences[:, -2:-1, :]  # [N, 1, features] - second to last
    target_step = sequences[:, -1:, :]     # [N, 1, features] - last
    
    context_target_diff = np.abs(last_context - target_step).mean()
    print(f"\nAverage difference between last context and target: {context_target_diff:.6f}")
    
    # This should NOT be tiny - if it is, we have data leakage
    if context_target_diff < 0.001:
        print("🚨 WARNING: Context and target are extremely similar! Possible data leakage.")
        print("The model might just be learning to copy the last input.")
    
    # Check variance within sequences
    sequence_variance = np.var(sequences, axis=1).mean()  # Variance across time for each sequence
    print(f"Average variance within sequences: {sequence_variance:.6f}")
    
    if sequence_variance < 0.001:
        print("🚨 WARNING: Very low variance within sequences! Data might be too static.")
    
    # Check feature-wise statistics
    print(f"\nFeature-wise statistics (first 10 features):")
    for i in range(min(10, sequences.shape[-1])):
        feature_data = sequences[:, :, i]
        print(f"Feature {i}: min={feature_data.min():.6f}, max={feature_data.max():.6f}, "
              f"mean={feature_data.mean():.6f}, std={feature_data.std():.6f}")
    
    # Check if all features have same scale (another potential issue)
    feature_stds = np.std(sequences, axis=(0, 1))  # Std for each feature across all data
    print(f"\nFeature standard deviations range: {feature_stds.min():.6f} to {feature_stds.max():.6f}")
    
    # Count features with very low variance
    low_variance_features = (feature_stds < 0.001).sum()
    print(f"Features with std < 0.001: {low_variance_features} out of {len(feature_stds)}")
    
    if low_variance_features > 0.5 * len(feature_stds):
        print("🚨 WARNING: Many features have very low variance!")
    
    # Check temporal patterns
    print(f"\nTemporal analysis:")
    
    # Look at autocorrelation - how similar is each timestep to the next?
    autocorr_sum = 0
    for i in range(sequences.shape[1] - 1):
        corr = np.corrcoef(sequences[:, i, :].flatten(), sequences[:, i+1, :].flatten())[0, 1]
        autocorr_sum += corr
    
    avg_autocorr = autocorr_sum / (sequences.shape[1] - 1)
    print(f"Average temporal autocorrelation: {avg_autocorr:.6f}")
    
    if avg_autocorr > 0.99:
        print("🚨 WARNING: Extremely high temporal autocorrelation! Data changes very little over time.")
    
    # Sample some actual values to see if they make sense
    print(f"\nSample values from different parts of first sequence:")
    print(f"Timestep 0, features 0-5: {sequences[0, 0, :5]}")
    print(f"Timestep 60, features 0-5: {sequences[0, 60, :5]}")
    print(f"Timestep 119, features 0-5: {sequences[0, 119, :5]}")
    print(f"Timestep 120 (target), features 0-5: {sequences[0, 120, :5]}")

if __name__ == "__main__":
    inspect_training_data() 