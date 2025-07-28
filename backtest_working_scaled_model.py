#!/usr/bin/env python3
"""
Comprehensive Backtesting Script for Working Scaled Model

Tests the trained WLD 1-minute model on out-of-sample test data
and provides trading-relevant performance metrics.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
FINAL_DATA_DIR = "data/final_attention"
MODEL_PATH = "models/working_scaled_multigpu/best_model_scaled.pt"
RESULTS_DIR = "backtest_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BacktestDataset(Dataset):
    """Dataset for backtesting."""
    def __init__(self, file_path, target_feature_indices, target_steps):
        with np.load(file_path, allow_pickle=True) as data:
            self.context = torch.from_numpy(data['contexts']).float()
            self.target_full = torch.from_numpy(data['targets']).float()
            self.target = self.target_full[:, :target_steps, target_feature_indices]
            self.len = self.context.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]

class WorkingScaledDataset(Dataset):
    """Working scaled dataset for model recreation."""
    def __init__(self, file_path, target_feature_indices, target_steps):
        with np.load(file_path, allow_pickle=True) as data:
            self.context = torch.from_numpy(data['contexts']).float()
            self.target_full = torch.from_numpy(data['targets']).float()
            self.target = self.target_full[:, :target_steps, target_feature_indices]
            self.len = self.context.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]

class ScaledCrossMarketEmbedding(nn.Module):
    """Cross-market embedding from original model."""
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

class ScaledBinancePerpEmbedding(nn.Module):
    """Binance perp embedding from original model."""
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

class ScaledPositionalEncoding(nn.Module):
    """Positional encoding from original model."""
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

class ScaledMultiGPUForecaster(nn.Module):
    """Recreation of the working scaled model architecture."""
    def __init__(self, embedding_metadata, target_feature_indices, embed_dim, num_heads, 
                 num_encoder_layers, num_decoder_layers, dropout, target_len, num_target_features):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_input_features = embedding_metadata['num_features']
        self.target_feature_indices = target_feature_indices
        self.num_target_features = num_target_features
        self.target_len = target_len
        self.embed_dim = embed_dim

        self.value_projection = nn.Linear(1, embed_dim)
        self.input_embedding = ScaledCrossMarketEmbedding(embedding_metadata, embed_dim)
        self.output_embedding = ScaledBinancePerpEmbedding(embedding_metadata, target_feature_indices, embed_dim)
        self.positional_encoding = ScaledPositionalEncoding(embed_dim, dropout, max_len=100000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout, 
            activation='gelu', 
            batch_first=True, 
            norm_first=True
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout, 
            activation='gelu', 
            batch_first=True, 
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.transformer_encoder.enable_nested_tensor = False
        self.transformer_decoder.enable_nested_tensor = False

        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1)
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

def get_target_indices(embedding_metadata):
    """Get WLD binance_perp feature indices."""
    target_indices = []
    
    for i, col_name in enumerate(embedding_metadata['columns']):
        col_info = embedding_metadata['column_mapping'][col_name]
        if (col_info['exchange'] == 'binance_perp' and 
            col_info['trading_pair'] == 'WLD-USDT'):
            target_indices.append(i)
    
    return target_indices

def load_model_and_data():
    """Load the trained model and test data."""
    print("📚 Loading Model and Data...")
    
    # Load metadata
    with open(os.path.join(FINAL_DATA_DIR, 'embedding_metadata.json'), 'r') as f:
        embedding_metadata = json.load(f)
    
    target_feature_indices = get_target_indices(embedding_metadata)
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    print(f"  Model trained for {checkpoint['epoch']} epochs")
    print(f"  Best validation loss: {checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A')):.6f}")
    
    # Extract model configuration with fallbacks
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        print("  ✅ Found saved model config")
    else:
        # Use defaults based on train_working_scaled.py architecture
        model_config = {
            'embed_dim': 256,
            'num_heads': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 4
        }
        print("  ⚠️  No model config found, using defaults")
    
    # Determine target_steps from data or use default
    target_steps = model_config.get('target_steps', 24)  # Default from train_working_scaled.py
    
    print(f"  📊 Model config: embed_dim={model_config.get('embed_dim', 256)}, "
          f"heads={model_config.get('num_heads', 8)}, "
          f"target_steps={target_steps}")
    
    # Recreate model
    model = ScaledMultiGPUForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=model_config.get('embed_dim', 256),
        num_heads=model_config.get('num_heads', 8),
        num_encoder_layers=model_config.get('num_encoder_layers', 6),
        num_decoder_layers=model_config.get('num_decoder_layers', 4),
        dropout=0.1,  # Not saved in config
        target_len=target_steps,
        num_target_features=len(target_feature_indices)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Load test dataset
    test_dataset = BacktestDataset(
        os.path.join(FINAL_DATA_DIR, 'test.npz'),
        target_feature_indices,
        target_steps
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"  ✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  ✅ Test dataset: {len(test_dataset)} sequences")
    print()
    
    return model, test_loader, target_feature_indices, embedding_metadata

def run_backtest(model, test_loader):
    """Run comprehensive backtest."""
    print("🎯 Running Backtest...")
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (context, target) in enumerate(test_loader):
            context, target = context.to(DEVICE), target.to(DEVICE)
            
            # Create decoder input (shifted target)
            decoder_input = torch.zeros_like(target)
            decoder_input[:, 1:] = target[:, :-1]
            
            # Get predictions
            predictions = model(context, decoder_input)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Concatenate all results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    print(f"  ✅ Backtest complete: {predictions.shape[0]} sequences")
    print(f"  📊 Prediction shape: {predictions.shape}")
    print()
    
    return predictions, targets

def calculate_metrics(predictions, targets):
    """Calculate comprehensive performance metrics."""
    print("📊 Calculating Performance Metrics...")
    
    # Flatten for overall metrics
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)
    
    # Basic regression metrics
    mse = mean_squared_error(target_flat, pred_flat)
    mae = mean_absolute_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(target_flat, pred_flat)
    correlation, _ = pearsonr(pred_flat, target_flat)
    
    # Trading-specific metrics
    # Directional accuracy (for each time step)
    directional_accuracy = []
    for t in range(predictions.shape[1]):
        if t == 0:
            continue  # Can't compute direction for first step
        
        pred_direction = np.sign(predictions[:, t] - predictions[:, t-1])
        target_direction = np.sign(targets[:, t] - targets[:, t-1])
        
        # Accuracy per feature
        feature_accuracies = []
        for f in range(predictions.shape[2]):
            accuracy = np.mean(pred_direction[:, f] == target_direction[:, f])
            feature_accuracies.append(accuracy)
        
        directional_accuracy.append(np.mean(feature_accuracies))
    
    avg_directional_accuracy = np.mean(directional_accuracy)
    
    # Price movement magnitude accuracy
    pred_returns = np.diff(predictions, axis=1)
    target_returns = np.diff(targets, axis=1)
    
    return_correlation = pearsonr(pred_returns.flatten(), target_returns.flatten())[0]
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation,
        'directional_accuracy': avg_directional_accuracy,
        'return_correlation': return_correlation,
        'total_predictions': len(pred_flat)
    }
    
    # Print results
    print("  📈 Regression Metrics:")
    print(f"    MSE: {mse:.8f}")
    print(f"    MAE: {mae:.8f}")
    print(f"    RMSE: {rmse:.8f}")
    print(f"    R²: {r2:.6f}")
    print(f"    Correlation: {correlation:.6f}")
    print()
    print("  🎯 Trading Metrics:")
    print(f"    Directional Accuracy: {avg_directional_accuracy:.4f} ({avg_directional_accuracy*100:.2f}%)")
    print(f"    Return Correlation: {return_correlation:.6f}")
    print()
    
    return metrics

def create_visualizations(predictions, targets, save_dir):
    """Create comprehensive visualizations."""
    print("📊 Creating Visualizations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Prediction vs Target scatter plot (sample)
    plt.figure(figsize=(12, 8))
    
    # Take first 1000 samples and first feature for visualization
    sample_size = min(1000, predictions.shape[0])
    sample_pred = predictions[:sample_size, :, 0].flatten()
    sample_target = targets[:sample_size, :, 0].flatten()
    
    plt.subplot(2, 2, 1)
    plt.scatter(sample_target, sample_pred, alpha=0.5, s=1)
    plt.plot([sample_target.min(), sample_target.max()], 
             [sample_target.min(), sample_target.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual (Feature 0)')
    plt.grid(True, alpha=0.3)
    
    # 2. Time series comparison (first sequence)
    plt.subplot(2, 2, 2)
    time_steps = range(predictions.shape[1])
    plt.plot(time_steps, targets[0, :, 0], 'b-', label='Actual', linewidth=2)
    plt.plot(time_steps, predictions[0, :, 0], 'r--', label='Predicted', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Time Series Prediction (First Sequence, Feature 0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals distribution
    plt.subplot(2, 2, 3)
    residuals = (sample_pred - sample_target)
    plt.hist(residuals, bins=50, alpha=0.7, density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residuals Distribution')
    plt.grid(True, alpha=0.3)
    
    # 4. Feature-wise performance
    plt.subplot(2, 2, 4)
    feature_mse = []
    for f in range(min(10, predictions.shape[2])):  # First 10 features
        feature_pred = predictions[:, :, f].flatten()
        feature_target = targets[:, :, f].flatten()
        feature_mse.append(mean_squared_error(feature_target, feature_pred))
    
    plt.bar(range(len(feature_mse)), feature_mse)
    plt.xlabel('Feature Index')
    plt.ylabel('MSE')
    plt.title('Per-Feature MSE')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'backtest_overview.png'), dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved overview plot: {save_dir}/backtest_overview.png")
    
    # 5. Detailed time series (multiple sequences)
    plt.figure(figsize=(15, 10))
    
    for i in range(min(5, predictions.shape[0])):
        plt.subplot(3, 2, i+1)
        time_steps = range(predictions.shape[1])
        plt.plot(time_steps, targets[i, :, 0], 'b-', label='Actual', alpha=0.8)
        plt.plot(time_steps, predictions[i, :, 0], 'r--', label='Predicted', alpha=0.8)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'Sequence {i+1} (Feature 0)')
        if i == 0:
            plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detailed_sequences.png'), dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved detailed sequences: {save_dir}/detailed_sequences.png")
    
    plt.close('all')  # Close all figures to save memory

def save_results(metrics, predictions, targets, save_dir):
    """Save backtest results."""
    print("💾 Saving Results...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions and targets (compressed)
    np.savez_compressed(
        os.path.join(save_dir, 'predictions.npz'),
        predictions=predictions,
        targets=targets
    )
    
    # Save summary report
    with open(os.path.join(save_dir, 'backtest_report.txt'), 'w') as f:
        f.write("WLD 1-Minute Binance Perp Model Backtest Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write("Model Performance:\n")
        f.write(f"  - MSE: {metrics['mse']:.8f}\n")
        f.write(f"  - MAE: {metrics['mae']:.8f}\n")
        f.write(f"  - R²: {metrics['r2']:.6f}\n")
        f.write(f"  - Correlation: {metrics['correlation']:.6f}\n")
        f.write(f"  - Directional Accuracy: {metrics['directional_accuracy']*100:.2f}%\n")
        f.write(f"  - Return Correlation: {metrics['return_correlation']:.6f}\n\n")
        
        f.write(f"Data Summary:\n")
        f.write(f"  - Total predictions: {metrics['total_predictions']:,}\n")
        f.write(f"  - Prediction shape: {predictions.shape}\n")
        f.write(f"  - Target shape: {targets.shape}\n")
    
    print(f"  ✅ Results saved to: {save_dir}")

def main():
    """Main backtesting function."""
    print("=" * 70)
    print("🚀 WLD 1-MINUTE MODEL BACKTEST")
    print("=" * 70)
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_DIR, f"backtest_{timestamp}")
    
    try:
        # Load model and data
        model, test_loader, target_indices, metadata = load_model_and_data()
        
        # Run backtest
        predictions, targets = run_backtest(model, test_loader)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, targets)
        
        # Create visualizations
        create_visualizations(predictions, targets, results_dir)
        
        # Save results
        save_results(metrics, predictions, targets, results_dir)
        
        print("=" * 70)
        print("✅ BACKTEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print("🎯 Key Results:")
        print(f"  📊 R² Score: {metrics['r2']:.4f}")
        print(f"  🎯 Directional Accuracy: {metrics['directional_accuracy']*100:.2f}%")
        print(f"  📈 Correlation: {metrics['correlation']:.4f}")
        print()
        print(f"📁 Results saved to: {results_dir}")
        print("🖼️  Check the generated plots for visual analysis!")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 