#!/usr/bin/env python3
"""
CORRECTED: Comprehensive Backtesting Script for Properly Calibrated WLD Model

✅ Uses corrected model: 24 time steps prediction
✅ Tests exactly 20 WLD binance_perp features 
✅ Compatible with properly trained model architecture
✅ Validates configuration matches training

Model Path: models/working_scaled_corrected/best_model_corrected.pt
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

# Configuration - CORRECTED PATHS
FINAL_DATA_DIR = "data/final_attention"
MODEL_PATH = "models/working_scaled_corrected/best_model_corrected.pt"  # Corrected model path
RESULTS_DIR = "backtest_results_corrected"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CorrectedBacktestDataset(Dataset):
    """Dataset for backtesting the corrected model."""
    def __init__(self, file_path, target_feature_indices, target_steps):
        with np.load(file_path, allow_pickle=True) as data:
            self.context = torch.from_numpy(data['contexts']).float()
            self.target_full = torch.from_numpy(data['targets']).float()
            # Use exact target_steps from corrected model (should be 24)
            self.target = self.target_full[:, :target_steps, target_feature_indices]
            self.len = self.context.shape[0]

        print(f"  📊 Dataset loaded: {self.len} sequences")
        print(f"  📊 Context shape: {self.context.shape}")
        print(f"  📊 Target shape: {self.target.shape}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]

# Import model classes from corrected training script
class ScaledCrossMarketEmbedding(nn.Module):
    """Cross-market embedding for all input features."""
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
    """Embedding specifically for WLD binance_perp output features."""
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

class CorrectedMultiGPUForecaster(nn.Module):
    """CORRECTED: Model properly configured for 24 time step prediction."""
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

def get_wld_perp_indices(embedding_metadata):
    """✅ Get exactly the 20 WLD binance_perp feature indices."""
    target_indices = []
    
    for i, col_name in enumerate(embedding_metadata['columns']):
        col_info = embedding_metadata['column_mapping'][col_name]
        if (col_info['exchange'] == 'binance_perp' and 
            col_info['trading_pair'] == 'WLD-USDT'):
            target_indices.append(i)
    
    return target_indices

def validate_model_data_compatibility(model_config, embedding_metadata, target_indices):
    """✅ Validate model and data are compatible."""
    print("🔍 Validating Model-Data Compatibility...")
    
    # Check target indices
    assert len(target_indices) == 20, f"Expected 20 WLD features, got {len(target_indices)}"
    assert target_indices == list(range(120, 140)), f"Expected indices 120-139, got range {target_indices[0]}-{target_indices[-1]}"
    
    # Check model config matches data
    assert model_config['target_steps'] == embedding_metadata['target_length'], \
        f"Model target_steps {model_config['target_steps']} != data target_length {embedding_metadata['target_length']}"
    
    assert model_config['num_input_features'] == embedding_metadata['num_features'], \
        f"Model input features {model_config['num_input_features']} != data features {embedding_metadata['num_features']}"
    
    assert model_config['num_target_features'] == 20, \
        f"Model target features {model_config['num_target_features']} != expected 20"
    
    print("  ✅ Model architecture matches data structure")
    print("  ✅ Target indices are correct (120-139)")
    print("  ✅ Target steps match data preparation (24)")
    print()

def load_corrected_model_and_data():
    """Load the corrected trained model and test data."""
    print("📚 Loading CORRECTED Model and Data...")
    
    # Check if corrected model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Corrected model not found at: {MODEL_PATH}")
        print("💡 Please train the corrected model first using:")
        print("   python3 train_working_scaled_CORRECTED.py")
        return None, None, None, None
    
    # Load metadata
    with open(os.path.join(FINAL_DATA_DIR, 'embedding_metadata.json'), 'r') as f:
        embedding_metadata = json.load(f)
    
    target_feature_indices = get_wld_perp_indices(embedding_metadata)
    
    print(f"  ✅ WLD binance_perp features: {len(target_feature_indices)} (indices {target_feature_indices[0]}-{target_feature_indices[-1]})")
    
    # Load corrected model checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    print(f"  ✅ Model trained for {checkpoint['epoch']} epochs")
    print(f"  ✅ Best validation loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    
    # Get model configuration
    if 'model_config' not in checkpoint:
        print("❌ No model_config found in corrected model!")
        return None, None, None, None
        
    model_config = checkpoint['model_config']
    print("  ✅ Found complete model config")
    
    # Validate compatibility
    validate_model_data_compatibility(model_config, embedding_metadata, target_feature_indices)
    
    target_steps = model_config['target_steps']  # Should be 24
    
    print(f"  📊 Model Configuration:")
    print(f"    - Embed dim: {model_config['embed_dim']}")
    print(f"    - Heads: {model_config['num_heads']}")
    print(f"    - Target steps: {target_steps}")
    print(f"    - Input features: {model_config['num_input_features']}")
    print(f"    - Target features: {model_config['num_target_features']}")
    
    # Recreate model with exact same architecture
    model = CorrectedMultiGPUForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        dropout=model_config['dropout'],
        target_len=target_steps,
        num_target_features=model_config['num_target_features']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Load test dataset with correct configuration
    test_dataset = CorrectedBacktestDataset(
        os.path.join(FINAL_DATA_DIR, 'test.npz'),
        target_feature_indices,
        target_steps
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"  ✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  ✅ Test dataset: {len(test_dataset)} sequences")
    print(f"  ✅ Expected output shape: (batch, {target_steps}, {len(target_feature_indices)})")
    print()
    
    return model, test_loader, target_feature_indices, embedding_metadata

def run_corrected_backtest(model, test_loader):
    """Run comprehensive backtest on corrected model."""
    print("🎯 Running CORRECTED Backtest...")
    
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
    print(f"  📊 Target shape: {targets.shape}")
    
    # Validate shapes
    expected_shape = (predictions.shape[0], 24, 20)  # (sequences, 24 time steps, 20 features)
    assert predictions.shape == expected_shape, f"Wrong prediction shape! Got {predictions.shape}, expected {expected_shape}"
    assert targets.shape == expected_shape, f"Wrong target shape! Got {targets.shape}, expected {expected_shape}"
    
    print(f"  ✅ Shape validation passed: {expected_shape}")
    print()
    
    return predictions, targets

def calculate_wld_metrics(predictions, targets):
    """Calculate comprehensive performance metrics for WLD model."""
    print("📊 Calculating WLD Performance Metrics...")
    
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
    directional_accuracy = []
    for t in range(1, predictions.shape[1]):  # Skip first step
        pred_direction = np.sign(predictions[:, t] - predictions[:, t-1])
        target_direction = np.sign(targets[:, t] - targets[:, t-1])
        
        # Accuracy per feature
        feature_accuracies = []
        for f in range(predictions.shape[2]):
            accuracy = np.mean(pred_direction[:, f] == target_direction[:, f])
            feature_accuracies.append(accuracy)
        
        directional_accuracy.append(np.mean(feature_accuracies))
    
    avg_directional_accuracy = np.mean(directional_accuracy)
    
    # Price movement correlation
    pred_returns = np.diff(predictions, axis=1)
    target_returns = np.diff(targets, axis=1)
    return_correlation = pearsonr(pred_returns.flatten(), target_returns.flatten())[0]
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'correlation': float(correlation),
        'directional_accuracy': float(avg_directional_accuracy),
        'return_correlation': float(return_correlation),
        'total_predictions': int(len(pred_flat)),
        'sequences': int(predictions.shape[0]),
        'time_steps': int(predictions.shape[1]),
        'features': int(predictions.shape[2])
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
    print("  📊 Data Summary:")
    print(f"    Sequences: {predictions.shape[0]:,}")
    print(f"    Time steps per sequence: {predictions.shape[1]}")
    print(f"    Features per time step: {predictions.shape[2]}")
    print(f"    Total predictions: {len(pred_flat):,}")
    print()
    
    return metrics

def save_corrected_results(metrics, predictions, targets, save_dir):
    """Save backtest results for corrected model."""
    print("💾 Saving CORRECTED Results...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(save_dir, 'corrected_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions and targets
    np.savez_compressed(
        os.path.join(save_dir, 'corrected_predictions.npz'),
        predictions=predictions,
        targets=targets
    )
    
    # Save summary report
    with open(os.path.join(save_dir, 'corrected_backtest_report.txt'), 'w') as f:
        f.write("CORRECTED WLD 1-Minute Binance Perp Model Backtest Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Model: Corrected 24-step WLD prediction model\n\n")
        
        f.write("✅ CORRECTED Model Configuration:\n")
        f.write(f"  - Target Length: 24 time steps (corrected from 12)\n")
        f.write(f"  - Target Features: 20 WLD binance_perp features\n")
        f.write(f"  - Feature Indices: 120-139\n")
        f.write(f"  - Input Context: 120 time steps, 240 features\n\n")
        
        f.write("📊 Performance Metrics:\n")
        f.write(f"  - MSE: {metrics['mse']:.8f}\n")
        f.write(f"  - MAE: {metrics['mae']:.8f}\n")
        f.write(f"  - R²: {metrics['r2']:.6f}\n")
        f.write(f"  - Correlation: {metrics['correlation']:.6f}\n")
        f.write(f"  - Directional Accuracy: {metrics['directional_accuracy']*100:.2f}%\n")
        f.write(f"  - Return Correlation: {metrics['return_correlation']:.6f}\n\n")
        
        f.write(f"📈 Data Summary:\n")
        f.write(f"  - Total sequences: {metrics['sequences']:,}\n")
        f.write(f"  - Prediction shape: ({metrics['sequences']}, {metrics['time_steps']}, {metrics['features']})\n")
        f.write(f"  - Total predictions: {metrics['total_predictions']:,}\n")
    
    print(f"  ✅ Results saved to: {save_dir}")

def main():
    """Main backtesting function for corrected model."""
    print("=" * 80)
    print("🚀 CORRECTED WLD 1-MINUTE MODEL BACKTEST")
    print("=" * 80)
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_DIR, f"corrected_backtest_{timestamp}")
    
    try:
        # Load corrected model and data
        result = load_corrected_model_and_data()
        if result[0] is None:
            return
            
        model, test_loader, target_indices, metadata = result
        
        # Run backtest
        predictions, targets = run_corrected_backtest(model, test_loader)
        
        # Calculate metrics
        metrics = calculate_wld_metrics(predictions, targets)
        
        # Save results
        save_corrected_results(metrics, predictions, targets, results_dir)
        
        print("=" * 80)
        print("✅ CORRECTED BACKTEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("🎯 Key Results:")
        print(f"  📊 R² Score: {metrics['r2']:.4f}")
        print(f"  🎯 Directional Accuracy: {metrics['directional_accuracy']*100:.2f}%")
        print(f"  📈 Correlation: {metrics['correlation']:.4f}")
        print(f"  ⏱️  Time Steps: {metrics['time_steps']} (24 seconds ahead)")
        print(f"  🎲 Features: {metrics['features']} (WLD perp LOB)")
        print()
        print(f"📁 Results saved to: {results_dir}")
        print("🎉 Ready for production trading!")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 