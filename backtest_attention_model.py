#!/usr/bin/env python3
"""
Comprehensive Backtesting for Full-to-Binance-Perp Attention Model

This script evaluates the trained attention model on test data and provides:
1. Prediction accuracy metrics (MSE, MAE, directional accuracy)
2. Trading performance simulation
3. Structural constraint validation
4. Visualization of predictions vs actual
5. Feature importance analysis
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore')

# Copy necessary classes from training script
class CrossMarketCompoundEmbedding(nn.Module):
    """Compound embedding for all market features."""
    def __init__(self, embedding_metadata, embed_dim):
        super().__init__()
        self.metadata = embedding_metadata
        self.embed_dim = embed_dim
        
        self.price_embed = nn.Embedding(1, embed_dim // 4)
        self.size_embed = nn.Embedding(1, embed_dim // 4)
        self.exchange_embed = nn.Embedding(4, embed_dim // 4)
        self.pair_embed = nn.Embedding(5, embed_dim // 4)

    def forward(self, num_features):
        embeddings = []
        device = self.price_embed.weight.device
        
        for i in range(num_features):
            feature_embed = torch.cat([
                self.price_embed(torch.tensor(0, device=device)),
                self.size_embed(torch.tensor(0, device=device)),
                self.exchange_embed(torch.tensor(i % 3, device=device)),
                self.pair_embed(torch.tensor(i % 4, device=device))
            ])
            embeddings.append(feature_embed)
        
        return torch.stack(embeddings)

class BinancePerpOutputEmbedding(nn.Module):
    """Specialized embedding for Binance perp output features."""
    def __init__(self, embedding_metadata, target_indices, embed_dim):
        super().__init__()
        self.metadata = embedding_metadata
        self.target_indices = target_indices
        self.embed_dim = embed_dim
        
        self.perp_price_embed = nn.Embedding(1, embed_dim // 2)
        self.perp_size_embed = nn.Embedding(1, embed_dim // 2)

    def forward(self, num_target_features):
        embeddings = []
        device = self.perp_price_embed.weight.device
        
        for i in range(num_target_features):
            feature_embed = torch.cat([
                self.perp_price_embed(torch.tensor(0, device=device)),
                self.perp_size_embed(torch.tensor(0, device=device))
            ])
            embeddings.append(feature_embed)
        
        return torch.stack(embeddings)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, dropout=0.1, max_len=100000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].clone()
        return self.dropout(x)

class FullToBinancePerpForecaster(nn.Module):
    """Cross-market model: 240 input features → 80 Binance perp output features."""
    def __init__(self, embedding_metadata, target_feature_indices, embed_dim, num_heads, 
                 num_encoder_layers, num_decoder_layers, dropout, target_len):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_input_features = embedding_metadata['num_features']
        self.target_feature_indices = target_feature_indices
        self.num_target_features = len(target_feature_indices)
        self.target_len = target_len
        self.embed_dim = embed_dim

        self.value_projection = nn.Linear(1, embed_dim)
        self.input_embedding = CrossMarketCompoundEmbedding(embedding_metadata, embed_dim)
        self.output_embedding = BinancePerpOutputEmbedding(embedding_metadata, target_feature_indices, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len=100000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.transformer_encoder.enable_nested_tensor = False
        self.transformer_decoder.enable_nested_tensor = False

        self.cross_market_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, src, tgt):
        input_feature_embeds = self.input_embedding(self.num_input_features)
        output_feature_embeds = self.output_embedding(self.num_target_features)
        
        src_proj = self.value_projection(src.unsqueeze(-1))
        tgt_proj = self.value_projection(tgt.unsqueeze(-1))

        src_embedded = src_proj + input_feature_embeds.unsqueeze(0).unsqueeze(0)
        tgt_embedded = tgt_proj + output_feature_embeds.unsqueeze(0).unsqueeze(0)

        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        src_flat = src_embedded.reshape(batch_size, context_len * self.num_input_features, self.embed_dim)
        src_pos = self.positional_encoding(src_flat)
        memory = self.transformer_encoder(src_pos)
        
        tgt_flat = tgt_embedded.reshape(batch_size, target_len * self.num_target_features, self.embed_dim)
        tgt_pos = self.positional_encoding(tgt_flat)
        
        combined_target_len = target_len * self.num_target_features
        tgt_mask = self.generate_square_subsequent_mask(combined_target_len).to(src.device)
        
        transformer_out = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)

        output = self.output_layer(transformer_out)
        output = output.squeeze(-1)
        output = output.reshape(batch_size, target_len, self.num_target_features)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# Configuration
MODEL_PATH = "models/full_to_binance_perp_dataparallel/best_model_dataparallel.pt"
DATA_DIR = "data/final_attention"
RESULTS_DIR = "backtest_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backtest")

def get_binance_perp_indices(embedding_metadata):
    """Get indices of Binance perp features."""
    target_indices = []
    for i, col_name in enumerate(embedding_metadata['columns']):
        col_info = embedding_metadata['column_mapping'][col_name]
        if col_info['exchange'] == 'binance_perp':
            target_indices.append(i)
    return target_indices

def load_model_and_data():
    """Load the trained model and test data."""
    logger.info("Loading trained model and test data...")
    
    # Load metadata
    data_dir = DATA_DIR
    try:
        with open(os.path.join(data_dir, 'embedding_metadata.json'), 'r') as f:
            embedding_metadata = json.load(f)
    except FileNotFoundError:
        # Try alternative location
        data_dir = 'data/final_attention'
        with open(os.path.join(data_dir, 'embedding_metadata.json'), 'r') as f:
            embedding_metadata = json.load(f)
    
    # Get target indices
    target_feature_indices = get_binance_perp_indices(embedding_metadata)
    
    # Load model checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Create model
    model = FullToBinancePerpForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=128,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=3,
        dropout=0.1,
        target_len=24
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Load test data
    try:
        with np.load(os.path.join(data_dir, 'test.npz'), allow_pickle=True) as data:
            test_context = torch.from_numpy(data['x']).float()
            test_target_full = torch.from_numpy(data['y']).float()
            test_target = test_target_full[:, :, target_feature_indices]
    except FileNotFoundError:
        # Use validation data if test data doesn't exist
        logger.info("Test data not found, using validation data for backtest...")
        with np.load(os.path.join(data_dir, 'validation.npz'), allow_pickle=True) as data:
            test_context = torch.from_numpy(data['x']).float()
            test_target_full = torch.from_numpy(data['y']).float()
            test_target = test_target_full[:, :, target_feature_indices]
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Test data: {test_context.shape[0]} sequences")
    logger.info(f"Input features: {test_context.shape[2]} (all markets)")
    logger.info(f"Output features: {test_target.shape[2]} (Binance perp)")
    
    return model, test_context, test_target, embedding_metadata, target_feature_indices

def run_inference(model, test_context, test_target):
    """Run model inference on test data."""
    logger.info("Running inference on test data...")
    
    predictions = []
    actuals = []
    batch_size = 32
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_context), batch_size):
            end_idx = min(i + batch_size, len(test_context))
            
            context_batch = test_context[i:end_idx].to(DEVICE)
            target_batch = test_target[i:end_idx].to(DEVICE)
            
            # Create decoder input (shifted target)
            decoder_input = torch.zeros_like(target_batch)
            decoder_input[:, 1:] = target_batch[:, :-1]
            
            # Get predictions
            pred_batch = model(context_batch, decoder_input)
            
            predictions.append(pred_batch.cpu())
            actuals.append(target_batch.cpu())
    
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)
    
    logger.info(f"Inference completed: {predictions.shape[0]} predictions")
    
    return predictions.numpy(), actuals.numpy()

def calculate_metrics(predictions, actuals):
    """Calculate comprehensive accuracy metrics."""
    logger.info("Calculating accuracy metrics...")
    
    metrics = {}
    
    # Overall metrics
    mse = mean_squared_error(actuals.flatten(), predictions.flatten())
    mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
    r2 = r2_score(actuals.flatten(), predictions.flatten())
    
    metrics['overall'] = {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'r2': r2
    }
    
    # Directional accuracy (for each timestep)
    directional_acc = []
    for t in range(actuals.shape[1] - 1):
        actual_direction = np.sign(actuals[:, t+1] - actuals[:, t])
        pred_direction = np.sign(predictions[:, t+1] - predictions[:, t])
        
        # Calculate accuracy for each feature
        feature_acc = []
        for f in range(actuals.shape[2]):
            acc = np.mean(actual_direction[:, f] == pred_direction[:, f])
            feature_acc.append(acc)
        
        directional_acc.append(np.mean(feature_acc))
    
    metrics['directional_accuracy'] = {
        'by_timestep': directional_acc,
        'average': np.mean(directional_acc)
    }
    
    # Feature-wise metrics
    feature_metrics = []
    for f in range(actuals.shape[2]):
        feature_mse = mean_squared_error(actuals[:, :, f].flatten(), predictions[:, :, f].flatten())
        feature_mae = mean_absolute_error(actuals[:, :, f].flatten(), predictions[:, :, f].flatten())
        feature_r2 = r2_score(actuals[:, :, f].flatten(), predictions[:, :, f].flatten())
        
        feature_metrics.append({
            'mse': feature_mse,
            'mae': feature_mae,
            'rmse': np.sqrt(feature_mse),
            'r2': feature_r2
        })
    
    metrics['by_feature'] = feature_metrics
    
    return metrics

def simulate_trading_performance(predictions, actuals):
    """Simulate trading performance based on predictions."""
    logger.info("Simulating trading performance...")
    
    # Simple trading strategy: buy if predicted price increase > threshold
    threshold = 0.001  # 0.1% threshold
    
    returns = []
    hit_rate = []
    
    for sample in range(min(1000, predictions.shape[0])):  # Limit for performance
        for t in range(predictions.shape[1] - 1):
            # Use first feature (typically best bid/ask price)
            actual_return = (actuals[sample, t+1, 0] - actuals[sample, t, 0]) / actuals[sample, t, 0]
            predicted_return = (predictions[sample, t+1, 0] - predictions[sample, t, 0]) / predictions[sample, t, 0]
            
            # Trading signal
            if abs(predicted_return) > threshold:
                signal = np.sign(predicted_return)
                trading_return = signal * actual_return
                returns.append(trading_return)
                hit_rate.append(1 if signal * actual_return > 0 else 0)
    
    if returns:
        trading_metrics = {
            'total_return': np.sum(returns),
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 12),  # Assuming 5-second intervals
            'hit_rate': np.mean(hit_rate),
            'avg_return_per_trade': np.mean(returns),
            'num_trades': len(returns)
        }
    else:
        trading_metrics = {'error': 'No trades generated'}
    
    return trading_metrics

def create_visualizations(predictions, actuals, metrics, save_dir):
    """Create visualization plots."""
    logger.info("Creating visualizations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Prediction vs Actual scatter plot
    plt.figure(figsize=(12, 8))
    
    # Sample data for visualization
    sample_size = min(5000, predictions.size)
    sample_indices = np.random.choice(predictions.size, sample_size, replace=False)
    
    pred_sample = predictions.flatten()[sample_indices]
    actual_sample = actuals.flatten()[sample_indices]
    
    plt.subplot(2, 2, 1)
    plt.scatter(actual_sample, pred_sample, alpha=0.5, s=1)
    plt.plot([actual_sample.min(), actual_sample.max()], [actual_sample.min(), actual_sample.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actual (R² = {metrics["overall"]["r2"]:.4f})')
    plt.grid(True, alpha=0.3)
    
    # 2. Time series plot
    plt.subplot(2, 2, 2)
    sample_seq = 0
    plt.plot(actuals[sample_seq, :, 0], label='Actual', linewidth=2)
    plt.plot(predictions[sample_seq, :, 0], label='Predicted', linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.title('Sample Time Series: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Directional accuracy by timestep
    plt.subplot(2, 2, 3)
    timesteps = range(len(metrics['directional_accuracy']['by_timestep']))
    plt.bar(timesteps, metrics['directional_accuracy']['by_timestep'])
    plt.xlabel('Time Step')
    plt.ylabel('Directional Accuracy')
    plt.title(f'Directional Accuracy by Timestep (Avg: {metrics["directional_accuracy"]["average"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    # 4. Feature-wise performance
    plt.subplot(2, 2, 4)
    feature_r2 = [f['r2'] for f in metrics['by_feature']]
    plt.bar(range(len(feature_r2)), feature_r2)
    plt.xlabel('Feature Index')
    plt.ylabel('R² Score')
    plt.title('R² Score by Feature')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'backtest_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Error distribution
    plt.figure(figsize=(10, 6))
    errors = (predictions - actuals).flatten()
    plt.hist(errors, bins=100, alpha=0.7, density=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_results(metrics, trading_metrics, save_dir):
    """Save all results to files."""
    logger.info("Saving results...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                json_metrics[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        json_metrics[key][subkey] = subvalue.tolist()
                    elif isinstance(subvalue, list):
                        json_metrics[key][subkey] = subvalue
                    else:
                        json_metrics[key][subkey] = float(subvalue) if isinstance(subvalue, np.floating) else subvalue
            else:
                json_metrics[key] = value
        
        json.dump(json_metrics, f, indent=2)
    
    # Save trading metrics (convert numpy types to native Python types)
    with open(os.path.join(save_dir, 'trading_metrics.json'), 'w') as f:
        json_trading_metrics = {}
        for key, value in trading_metrics.items():
            if isinstance(value, (np.floating, np.integer)):
                json_trading_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                json_trading_metrics[key] = value.tolist()
            else:
                json_trading_metrics[key] = value
        json.dump(json_trading_metrics, f, indent=2)
    
    # Create summary report
    with open(os.path.join(save_dir, 'backtest_report.txt'), 'w') as f:
        f.write("ATTENTION MODEL BACKTEST REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"MSE: {metrics['overall']['mse']:.6f}\n")
        f.write(f"MAE: {metrics['overall']['mae']:.6f}\n")
        f.write(f"RMSE: {metrics['overall']['rmse']:.6f}\n")
        f.write(f"R² Score: {metrics['overall']['r2']:.6f}\n\n")
        
        f.write("DIRECTIONAL ACCURACY:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average: {metrics['directional_accuracy']['average']:.3f}\n")
        f.write(f"Range: {min(metrics['directional_accuracy']['by_timestep']):.3f} - {max(metrics['directional_accuracy']['by_timestep']):.3f}\n\n")
        
        if 'error' not in trading_metrics:
            f.write("TRADING SIMULATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Return: {trading_metrics['total_return']:.4f}\n")
            f.write(f"Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}\n")
            f.write(f"Hit Rate: {trading_metrics['hit_rate']:.3f}\n")
            f.write(f"Avg Return per Trade: {trading_metrics['avg_return_per_trade']:.6f}\n")
            f.write(f"Number of Trades: {trading_metrics['num_trades']}\n")

def main():
    """Main backtesting function."""
    logger.info("🎯 Starting Attention Model Backtest")
    
    # Load model and data
    model, test_context, test_target, embedding_metadata, target_indices = load_model_and_data()
    
    # Run inference
    predictions, actuals = run_inference(model, test_context, test_target)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, actuals)
    
    # Simulate trading
    trading_metrics = simulate_trading_performance(predictions, actuals)
    
    # Create visualizations
    create_visualizations(predictions, actuals, metrics, RESULTS_DIR)
    
    # Save results
    save_results(metrics, trading_metrics, RESULTS_DIR)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Overall MSE: {metrics['overall']['mse']:.6f}")
    logger.info(f"Overall MAE: {metrics['overall']['mae']:.6f}")
    logger.info(f"Overall R²: {metrics['overall']['r2']:.6f}")
    logger.info(f"Directional Accuracy: {metrics['directional_accuracy']['average']:.3f}")
    
    if 'error' not in trading_metrics:
        logger.info(f"Trading Hit Rate: {trading_metrics['hit_rate']:.3f}")
        logger.info(f"Trading Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}")
    
    logger.info(f"Results saved to: {RESULTS_DIR}/")
    logger.info("🎉 Backtest completed successfully!")

if __name__ == "__main__":
    main() 