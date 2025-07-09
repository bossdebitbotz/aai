#!/usr/bin/env python3
"""
Stable High-Performance Attention-Based LOB Forecasting

Optimized for single H100 GPU with maximum batch size and efficiency.
Implements exact paper methodology with hardware optimization.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import math
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Optimized Configuration for H100
def get_h100_config():
    """Get optimized configuration for H100 GPU."""
    config = {
        'data_path': 'data/final_attention',
        'model_save_dir': 'models/paper_h100',
        'learning_rate': 5e-4,
        'init_lr': 1e-10,
        'warmup_steps': 1000,
        'decay_factor': 0.8,
        'epochs': 50,
        'patience': 10,
        'embed_dim': 126,
        'num_heads': 3,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'd_ff': 512,
        'dropout': 0.1,
        'structural_weight': 0.01,
    }
    
    # Optimize for H100 (80GB VRAM)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    num_cpus = os.cpu_count()
    
    if gpu_memory > 70:  # H100 80GB
        config['batch_size'] = 32  # Large batch for H100
    elif gpu_memory > 40:  # A100 40GB
        config['batch_size'] = 24
    elif gpu_memory > 20:  # RTX 4090
        config['batch_size'] = 16
    else:
        config['batch_size'] = 8
    
    # Optimize data loading for 48 cores
    config['num_workers'] = min(16, num_cpus // 3)  # Use 1/3 of cores for data loading
    
    logger.info(f"H100 Optimization: {gpu_memory:.1f}GB GPU, {num_cpus} CPU cores")
    logger.info(f"Optimized: batch_size={config['batch_size']}, num_workers={config['num_workers']}")
    
    return config

CONFIG = get_h100_config()

# --- Model Classes (Same as paper implementation) ---

class LOBDataset(Dataset):
    """Memory-efficient LOB dataset."""
    def __init__(self, file_path):
        self.file_path = file_path
        # Load data into memory for H100 (640GB RAM available)
        with np.load(file_path) as data:
            self.x = torch.from_numpy(data['x']).float()
            self.y = torch.from_numpy(data['y']).float()
        logger.info(f"Loaded {len(self.x)} sequences into memory")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Time2Vec(nn.Module):
    """Time2Vec embedding layer."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.w = nn.parameter.Parameter(torch.randn(1, embed_dim-1))
        self.b = nn.parameter.Parameter(torch.randn(1, embed_dim-1))
        self.f = torch.sin

    def forward(self, tau):
        tau = tau.unsqueeze(-1)
        v1 = self.f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], -1)

class CompoundMultivariateEmbedding(nn.Module):
    """Simplified compound multivariate embedding."""
    def __init__(self, embedding_metadata, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.metadata = embedding_metadata
        
        unique_attrs = embedding_metadata['unique_attributes']
        attr_embed_dim = embed_dim // 5
        remaining_dim = embed_dim - (attr_embed_dim * 4)
        
        self.level_embedding = nn.Embedding(len(unique_attrs['levels']), attr_embed_dim)
        self.type_embedding = nn.Embedding(len(unique_attrs['order_types']), attr_embed_dim)
        self.feature_embedding = nn.Embedding(len(unique_attrs['features']), attr_embed_dim)
        self.exchange_embedding = nn.Embedding(len(unique_attrs['exchanges']), attr_embed_dim)
        self.pair_embedding = nn.Embedding(len(unique_attrs['trading_pairs']), remaining_dim)
        
        self.projection = nn.Linear(embed_dim, embed_dim)
        
        # Pre-compute indices
        self._create_feature_indices()
    
    def _create_feature_indices(self):
        columns = self.metadata['columns']
        column_mapping = self.metadata['column_mapping']
        
        # Create mappings
        level_to_idx = {level: i for i, level in enumerate(self.metadata['unique_attributes']['levels'])}
        type_to_idx = {otype: i for i, otype in enumerate(self.metadata['unique_attributes']['order_types'])}
        feature_to_idx = {feat: i for i, feat in enumerate(self.metadata['unique_attributes']['features'])}
        exchange_to_idx = {exch: i for i, exch in enumerate(self.metadata['unique_attributes']['exchanges'])}
        pair_to_idx = {pair: i for i, pair in enumerate(self.metadata['unique_attributes']['trading_pairs'])}
        
        level_indices = []
        type_indices = []
        feature_indices = []
        exchange_indices = []
        pair_indices = []
        
        for col in columns:
            attrs = column_mapping[col]
            level_indices.append(level_to_idx[attrs['level']])
            type_indices.append(type_to_idx[attrs['order_type']])
            feature_indices.append(feature_to_idx[attrs['feature_type']])
            exchange_indices.append(exchange_to_idx[attrs['exchange']])
            pair_indices.append(pair_to_idx[attrs['trading_pair']])
        
        self.register_buffer('level_indices', torch.LongTensor(level_indices))
        self.register_buffer('type_indices', torch.LongTensor(type_indices))
        self.register_buffer('feature_indices', torch.LongTensor(feature_indices))
        self.register_buffer('exchange_indices', torch.LongTensor(exchange_indices))
        self.register_buffer('pair_indices', torch.LongTensor(pair_indices))
    
    def forward(self):
        level_embeds = self.level_embedding(self.level_indices)
        type_embeds = self.type_embedding(self.type_indices)
        feature_embeds = self.feature_embedding(self.feature_indices)
        exchange_embeds = self.exchange_embedding(self.exchange_indices)
        pair_embeds = self.pair_embedding(self.pair_indices)
        
        combined_embeds = torch.cat([
            level_embeds, type_embeds, feature_embeds, 
            exchange_embeds, pair_embeds
        ], dim=-1)
        
        projected_embeds = self.projection(combined_embeds)
        return projected_embeds

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)

class StructuralLoss(nn.Module):
    """Structural regularizer to preserve LOB price ordering."""
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
        
    def forward(self, predictions, feature_metadata):
        if predictions.size(-1) < 10:
            return torch.tensor(0.0, device=predictions.device)
        
        total_loss = 0.0
        columns = feature_metadata['columns']
        column_mapping = feature_metadata['column_mapping']
        
        # Group features by exchange_pair
        exchange_pairs = {}
        for idx, col in enumerate(columns):
            attrs = column_mapping[col]
            key = f"{attrs['exchange']}_{attrs['trading_pair']}"
            if key not in exchange_pairs:
                exchange_pairs[key] = {'bid_prices': {}, 'ask_prices': {}}
            
            if attrs['feature_type'] == 'price':
                level = attrs['level']
                if attrs['order_type'] == 'bid':
                    exchange_pairs[key]['bid_prices'][level] = idx
                elif attrs['order_type'] == 'ask':
                    exchange_pairs[key]['ask_prices'][level] = idx
        
        # Apply structural constraints
        for key, price_indices in exchange_pairs.items():
            bid_prices = price_indices['bid_prices']
            ask_prices = price_indices['ask_prices']
            
            if len(bid_prices) >= 2 and len(ask_prices) >= 2:
                # Ask price constraints: ask_k < ask_{k+1}
                for level in range(1, 5):
                    if level in ask_prices and (level + 1) in ask_prices:
                        ask_k = predictions[:, :, ask_prices[level]]
                        ask_k_plus_1 = predictions[:, :, ask_prices[level + 1]]
                        violation = torch.relu(ask_k - ask_k_plus_1)
                        total_loss += violation.mean()
                
                # Bid price constraints: bid_k > bid_{k+1}
                for level in range(1, 5):
                    if level in bid_prices and (level + 1) in bid_prices:
                        bid_k = predictions[:, :, bid_prices[level]]
                        bid_k_plus_1 = predictions[:, :, bid_prices[level + 1]]
                        violation = torch.relu(bid_k_plus_1 - bid_k)
                        total_loss += violation.mean()
                
                # Bid-ask spread constraint: bid_1 < ask_1
                if 1 in bid_prices and 1 in ask_prices:
                    bid_1 = predictions[:, :, bid_prices[1]]
                    ask_1 = predictions[:, :, ask_prices[1]]
                    violation = torch.relu(bid_1 - ask_1)
                    total_loss += violation.mean()
        
        return self.weight * total_loss

class LOBForecaster(nn.Module):
    """High-performance LOB forecaster optimized for H100."""
    def __init__(self, embedding_metadata, embed_dim, num_heads, num_encoder_layers, 
                 num_decoder_layers, d_ff, dropout, target_len):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_features = len(embedding_metadata['columns'])
        self.target_len = target_len
        self.embed_dim = embed_dim
        
        # Embedding layers
        self.value_projection = nn.Linear(1, embed_dim)
        self.compound_embedding = CompoundMultivariateEmbedding(embedding_metadata, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Transformer encoder-decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads, 
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output layers
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, 1)
        
    def forward(self, src, tgt):
        """Optimized forward pass."""
        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        # Get feature embeddings
        feature_embeds = self.compound_embedding()  # [num_features, embed_dim]
        
        # Project values and add feature embeddings
        src_values = self.value_projection(src.unsqueeze(-1))  # [batch, context_len, num_feat, embed_dim]
        tgt_values = self.value_projection(tgt.unsqueeze(-1))  # [batch, target_len, num_feat, embed_dim]
        
        # Add feature embeddings (broadcast)
        src_embedded = src_values + feature_embeds.unsqueeze(0).unsqueeze(0)
        tgt_embedded = tgt_values + feature_embeds.unsqueeze(0).unsqueeze(0)
        
        # Reshape for transformer: [batch * num_features, seq_len, embed_dim]
        src_flat = src_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, context_len, self.embed_dim)
        tgt_flat = tgt_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, target_len, self.embed_dim)
        
        # Add positional encoding
        src_pos = self.positional_encoding(src_flat)
        tgt_pos = self.positional_encoding(tgt_flat)
        
        # Apply transformer
        memory = self.transformer_encoder(src_pos)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_len).to(src.device)
        output = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)
        
        # Output projection
        output = self.output_norm(output)
        output = self.output_projection(output)  # [batch * num_feat, target_len, 1]
        
        # Reshape back
        output = output.reshape(batch_size, self.num_features, target_len).permute(0, 2, 1)
        
        return output

def create_model(embedding_metadata):
    """Create optimized model for H100."""
    model = LOBForecaster(
        embedding_metadata=embedding_metadata,
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        num_encoder_layers=CONFIG['num_encoder_layers'],
        num_decoder_layers=CONFIG['num_decoder_layers'],
        d_ff=CONFIG['d_ff'],
        dropout=CONFIG['dropout'],
        target_len=24
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"H100 Model - Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model

def train_epoch(model, train_loader, mse_criterion, structural_loss, optimizer, embedding_metadata):
    """Optimized training epoch for H100."""
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_struct_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (contexts, targets) in enumerate(progress_bar):
        contexts, targets = contexts.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient for H100
        
        # Teacher forcing
        decoder_input = torch.cat([
            contexts[:, -1:, :],
            targets[:, :-1, :]
        ], dim=1)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            predictions = model(contexts, decoder_input)
            mse_loss = mse_criterion(predictions, targets)
            struct_loss = structural_loss(predictions, embedding_metadata)
            total_batch_loss = mse_loss + struct_loss
        
        # Backward pass
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_batch_loss.item()
        total_mse_loss += mse_loss.item()
        total_struct_loss += struct_loss.item()
        num_batches += 1
        
        # Update progress
        progress_bar.set_postfix({
            'Loss': f'{total_batch_loss.item():.6f}',
            'MSE': f'{mse_loss.item():.6f}',
            'Struct': f'{struct_loss.item():.6f}'
        })
    
    return total_loss / num_batches, total_mse_loss / num_batches, total_struct_loss / num_batches

def validate(model, val_loader, mse_criterion, structural_loss, embedding_metadata):
    """Optimized validation for H100."""
    model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_struct_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for contexts, targets in tqdm(val_loader, desc="Validation"):
            contexts, targets = contexts.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            
            decoder_input = torch.cat([
                contexts[:, -1:, :],
                targets[:, :-1, :]
            ], dim=1)
            
            with torch.amp.autocast('cuda'):
                predictions = model(contexts, decoder_input)
                mse_loss = mse_criterion(predictions, targets)
                struct_loss = structural_loss(predictions, embedding_metadata)
                total_batch_loss = mse_loss + struct_loss
            
            total_loss += total_batch_loss.item()
            total_mse_loss += mse_loss.item()
            total_struct_loss += struct_loss.item()
            num_batches += 1
    
    return total_loss / num_batches, total_mse_loss / num_batches, total_struct_loss / num_batches

def main():
    """Main training function optimized for H100."""
    logger.info("Starting H100-Optimized Attention-Based LOB Forecasting")
    logger.info(f"Configuration: {CONFIG}")
    
    # Create model directory
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
    
    # Load data and metadata
    logger.info("Loading data and metadata...")
    
    with open(os.path.join(CONFIG['data_path'], 'embedding_metadata.json'), 'r') as f:
        embedding_metadata = json.load(f)
    
    with open(os.path.join(CONFIG['data_path'], 'config.json'), 'r') as f:
        data_config = json.load(f)
    
    logger.info(f"Dataset: {embedding_metadata['num_features']} features, "
                f"Context: {data_config['context_length']}, Target: {data_config['target_length']}")
    
    # Create datasets (load into 640GB RAM)
    train_dataset = LOBDataset(os.path.join(CONFIG['data_path'], 'train.npz'))
    val_dataset = LOBDataset(os.path.join(CONFIG['data_path'], 'validation.npz'))
    
    # Create data loaders optimized for H100
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=CONFIG['num_workers'], 
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,
        num_workers=CONFIG['num_workers'], 
        pin_memory=True,
        persistent_workers=True
    )
    
    logger.info(f"Training sequences: {len(train_dataset)}")
    logger.info(f"Validation sequences: {len(val_dataset)}")
    logger.info(f"Effective batch size: {CONFIG['batch_size']}")
    
    # Create model
    logger.info("Creating H100-optimized model...")
    model = create_model(embedding_metadata)
    
    # Loss functions and optimizer
    mse_criterion = nn.MSELoss()
    structural_loss = StructuralLoss(weight=CONFIG['structural_weight'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['decay_factor'], patience=3, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("Starting H100 training...")
    logger.info(f"Paper target metrics: Total loss < 0.008, Structure loss < 0.15")
    
    for epoch in range(CONFIG['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        # Training
        train_loss, train_mse, train_struct = train_epoch(
            model, train_loader, mse_criterion, structural_loss, optimizer, embedding_metadata
        )
        
        # Validation
        val_loss, val_mse, val_struct = validate(
            model, val_loader, mse_criterion, structural_loss, embedding_metadata
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Train - Total: {train_loss:.6f}, MSE: {train_mse:.6f}, Struct: {train_struct:.6f}")
        logger.info(f"Val   - Total: {val_loss:.6f}, MSE: {val_mse:.6f}, Struct: {val_struct:.6f}")
        logger.info(f"Learning Rate: {current_lr:.2e}")
        
        # Check target metrics
        if val_loss < 0.008 and val_struct < 0.15:
            logger.info(f"🎯 Paper target metrics achieved! Total: {val_loss:.6f} < 0.008, Struct: {val_struct:.6f} < 0.15")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mse': val_mse,
                'val_struct': val_struct,
                'config': CONFIG,
                'embedding_metadata': embedding_metadata
            }
            
            torch.save(checkpoint, os.path.join(CONFIG['model_save_dir'], 'best_model.pt'))
            logger.info(f"💾 New best model saved! Val loss: {val_loss:.6f}")
            
        else:
            patience_counter += 1
            
        if patience_counter >= CONFIG['patience']:
            logger.info(f"🛑 Early stopping triggered after {CONFIG['patience']} epochs without improvement")
            break
            
        logger.info("-" * 80)
    
    logger.info("H100 Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to: {CONFIG['model_save_dir']}/best_model.pt")

if __name__ == "__main__":
    main() 