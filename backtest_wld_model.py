#!/usr/bin/env python3
"""
WLD-Specific Attention Model Backtest

This script backtests models that predict only WLD-USDT features (20 features)
instead of all Binance perpetual features (80 features).

Handles:
- WLD-only predictions (20 features)
- Variable prediction lengths (24, 36, 60 steps)  
- Proper target feature extraction
- WLD-specific trading simulation
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import warnings
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def load_model_and_data(model_path):
    """Load the WLD model and extract target configuration."""
    logger.info("Loading WLD model and test data...")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    target_feature_indices = checkpoint['target_feature_indices']
    target_pair = checkpoint.get('target_pair', 'WLD-USDT')
    target_steps = checkpoint.get('target_steps', 24)
    
    logger.info(f"Model target pair: {target_pair}")
    logger.info(f"Model target steps: {target_steps}")
    logger.info(f"Model target features: {len(target_feature_indices)}")
    
    # Load test data
    test_data_path = 'data/final_attention_240/test.npz'
    if not os.path.exists(test_data_path):
        test_data_path = 'data/final_attention/test.npz'
    
    with np.load(test_data_path) as data:
        test_x = data['x']  # (samples, context_len, all_features)
        test_y = data['y']  # (samples, target_len, all_features)
    
    # Extract only WLD target features from test data
    # Ensure we have enough target steps in the test data
    if test_y.shape[1] < target_steps:
        logger.warning(f"Test data only has {test_y.shape[1]} target steps, but model expects {target_steps}")
        logger.info(f"Using available {test_y.shape[1]} steps instead")
        actual_target_steps = test_y.shape[1]
    else:
        actual_target_steps = target_steps
    
    test_y_wld = test_y[:, :actual_target_steps, target_feature_indices]
    
    logger.info(f"Test data: {test_x.shape[0]} sequences")
    logger.info(f"Context: {test_x.shape[1]} steps, {test_x.shape[2]} features")
    logger.info(f"Target: {test_y_wld.shape[1]} steps, {test_y_wld.shape[2]} WLD features")
    
    return test_x, test_y_wld, target_feature_indices, actual_target_steps, checkpoint

def create_model(checkpoint, target_feature_indices, target_steps, force_cpu=False):
    """Recreate the model architecture."""
    # Import the model architecture (assuming it's in the same directory)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Load embedding metadata
    try:
        with open('data/final_attention_240/embedding_metadata.json', 'r') as f:
            embedding_metadata = json.load(f)
    except FileNotFoundError:
        with open('data/final_attention/embedding_metadata.json', 'r') as f:
            embedding_metadata = json.load(f)
    
    # Create model (using the same architecture from training script)
    from train_full_to_binance_perp_dataparallel import FullToBinancePerpForecaster
    
    model = FullToBinancePerpForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=128,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=3,
        dropout=0.1,
        target_len=target_steps
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to device
    if torch.cuda.is_available() and not force_cpu:
        model = model.cuda()
        logger.info(f"Using device: cuda")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        logger.info("Model loaded on GPU successfully")
    else:
        logger.info("Model loaded on CPU (memory-safe mode)")
    
    return model

def run_inference(model, test_x, target_steps, batch_size=1):  # Reduced from 8 to 1
    """Run model inference on test data."""
    logger.info("Running inference on test data...")
    logger.info(f"Using batch size: {batch_size} (reduced for memory efficiency)")
    
    # Check if model is on GPU
    model_on_gpu = next(model.parameters()).is_cuda
    device_name = "GPU" if model_on_gpu else "CPU"
    logger.info(f"Model is on: {device_name}")
    
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            try:
                batch_x = torch.FloatTensor(test_x[i:i+batch_size])
                
                # Create decoder input (zeros for inference)
                batch_size_actual = batch_x.shape[0]
                decoder_input = torch.zeros(batch_size_actual, target_steps, len(model.target_feature_indices))
                
                # Move to same device as model
                if model_on_gpu:
                    batch_x = batch_x.cuda()
                    decoder_input = decoder_input.cuda()
                
                # Forward pass
                batch_pred = model(batch_x, decoder_input)
                predictions.append(batch_pred.cpu().numpy())
                
                if (i // batch_size) % 10 == 0:
                    logger.info(f"Processed {i + batch_size_actual}/{len(test_x)} sequences")
                    
            except RuntimeError as e:
                if "memory" in str(e).lower():
                    logger.error(f"Memory error at batch {i}: {e}")
                    logger.info("Try using --cpu flag for CPU inference")
                    raise
                else:
                    logger.error(f"Runtime error at batch {i}: {e}")
                    raise
    
    predictions = np.concatenate(predictions, axis=0)
    logger.info(f"Inference completed: {len(predictions)} predictions")
    
    return predictions

def calculate_metrics(predictions, targets):
    """Calculate accuracy metrics."""
    logger.info("Calculating accuracy metrics...")
    
    # Flatten for overall metrics
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)
    
    mse = mean_squared_error(target_flat, pred_flat)
    mae = mean_absolute_error(target_flat, pred_flat)
    r2 = r2_score(target_flat, pred_flat)
    
    # Directional accuracy (for price changes)
    # Use the first feature (likely price) for directional accuracy
    pred_diff = np.diff(predictions[:, :, 0], axis=1)
    target_diff = np.diff(targets[:, :, 0], axis=1)
    
    directional_accuracy = np.mean(np.sign(pred_diff) == np.sign(target_diff))
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': directional_accuracy
    }

def simulate_wld_trading(predictions, targets, threshold=0.001):
    """Simulate trading performance for WLD predictions."""
    logger.info("Simulating WLD trading performance...")
    
    # Use first feature (likely price) for trading simulation
    pred_prices = predictions[:, -1, 0]  # Last predicted price
    target_prices = targets[:, -1, 0]    # Actual final price
    
    # Trading signals based on predicted direction
    signals = np.sign(pred_prices)  # 1 for buy, -1 for sell
    returns = signals * target_prices  # Simple return simulation
    
    # Calculate trading metrics
    hit_rate = np.mean(returns > 0)
    total_return = np.sum(returns)
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
    
    return {
        'hit_rate': hit_rate,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(returns)
    }

def create_visualizations(predictions, targets, save_dir):
    """Create visualization plots."""
    logger.info("Creating visualizations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Prediction vs Actual (first few samples)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i in range(4):
        row, col = i // 2, i % 2
        
        # Plot first feature (price) over time
        time_steps = range(predictions.shape[1])
        axes[row, col].plot(time_steps, targets[i, :, 0], 'b-', label='Actual', alpha=0.7)
        axes[row, col].plot(time_steps, predictions[i, :, 0], 'r--', label='Predicted', alpha=0.7)
        axes[row, col].set_title(f'WLD Price Prediction - Sample {i+1}')
        axes[row, col].set_xlabel('Time Steps')
        axes[row, col].set_ylabel('Price (normalized)')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'wld_predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Error distribution
    errors = (predictions - targets).reshape(-1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
    plt.title('WLD Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'wld_error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_results(metrics, trading_metrics, save_dir):
    """Save results to files."""
    logger.info("Saving results...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(save_dir, 'wld_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(save_dir, 'wld_trading_metrics.json'), 'w') as f:
        json.dump(trading_metrics, f, indent=2)
    
    # Save summary report
    with open(os.path.join(save_dir, 'wld_backtest_report.txt'), 'w') as f:
        f.write("WLD Model Backtest Results\n")
        f.write("==========================\n\n")
        f.write(f"Prediction Accuracy:\n")
        f.write(f"  MSE: {metrics['mse']:.6f}\n")
        f.write(f"  MAE: {metrics['mae']:.6f}\n")
        f.write(f"  R²: {metrics['r2']:.6f}\n")
        f.write(f"  Directional Accuracy: {metrics['directional_accuracy']:.3f}\n\n")
        f.write(f"Trading Performance:\n")
        f.write(f"  Hit Rate: {trading_metrics['hit_rate']:.3f}\n")
        f.write(f"  Total Return: {trading_metrics['total_return']:.6f}\n")
        f.write(f"  Sharpe Ratio: {trading_metrics['sharpe_ratio']:.2f}\n")
        f.write(f"  Number of Trades: {trading_metrics['num_trades']}\n")

def main():
    parser = argparse.ArgumentParser(description="Backtest WLD-specific attention model")
    parser.add_argument("--model_path", required=True, help="Path to the trained WLD model")
    parser.add_argument("--output_dir", default="wld_backtest_results", help="Output directory for results")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference (slower but memory-safe)")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting WLD Model Backtest")
    if args.cpu:
        logger.info("⚠️  Using CPU inference (memory-safe mode)")
    
    # Load model and data
    test_x, test_y_wld, target_feature_indices, actual_target_steps, checkpoint = load_model_and_data(args.model_path)
    
    # Get the model's expected target steps from checkpoint
    model_target_steps = checkpoint.get('target_steps', 24)
    
    # Create model with the original target steps it was trained with
    try:
        model = create_model(checkpoint, target_feature_indices, model_target_steps, force_cpu=args.cpu)
    except RuntimeError as e:
        if "memory" in str(e).lower():
            logger.error("GPU memory insufficient, falling back to CPU")
            model = create_model(checkpoint, target_feature_indices, model_target_steps, force_cpu=True)
        else:
            raise
    
    # Run inference with model's expected target steps
    try:
        predictions = run_inference(model, test_x, model_target_steps)
    except RuntimeError as e:
        if "memory" in str(e).lower():
            logger.error("GPU inference failed, retrying on CPU")
            model = create_model(checkpoint, target_feature_indices, model_target_steps, force_cpu=True)
            predictions = run_inference(model, test_x, model_target_steps)
        else:
            raise
    
    # If model produces more steps than we have in test data, truncate predictions
    if predictions.shape[1] > actual_target_steps:
        logger.info(f"Truncating predictions from {predictions.shape[1]} to {actual_target_steps} steps to match test data")
        predictions = predictions[:, :actual_target_steps, :]
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, test_y_wld)
    
    # Simulate trading
    trading_metrics = simulate_wld_trading(predictions, test_y_wld)
    
    # Create visualizations
    create_visualizations(predictions, test_y_wld, args.output_dir)
    
    # Save results
    save_results(metrics, trading_metrics, args.output_dir)
    
    # Print results
    logger.info("=" * 60)
    logger.info("WLD BACKTEST RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Overall MSE: {metrics['mse']:.6f}")
    logger.info(f"Overall MAE: {metrics['mae']:.6f}")
    logger.info(f"Overall R²: {metrics['r2']:.6f}")
    logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.3f}")
    logger.info(f"Trading Hit Rate: {trading_metrics['hit_rate']:.3f}")
    logger.info(f"Trading Sharpe Ratio: {trading_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Results saved to: {args.output_dir}/")
    logger.info("🎉 WLD Backtest completed successfully!")

if __name__ == "__main__":
    main() 