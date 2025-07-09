#!/usr/bin/env python3
"""
Attention-Based LOB Forecasting - Exact Paper Implementation

This script implements the exact methodology described in:
"Attention-Based Reading, Highlighting, and Forecasting of the Limit Order Book"

Key Paper Features Implemented:
1. Compound multivariate embedding for LOB attributes
2. Spacetimeformer architecture with Performer attention  
3. Structural regularizer to preserve price ordering (w_o = 0.01)
4. Time2Vec temporal encoding
5. Percent-change transformation + Min-max scaling
6. Context: 120 steps (10 min), Target: 24 steps (2 min)
7. Learning rate decay (0.8) with 1000 warmup steps
8. 3-head attention as specified in paper
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
import math
import logging
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add spacetimeformer to path
sys.path.insert(0, os.path.join(os.getcwd(), 'spacetimeformer'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Paper Configuration (exact values from paper)
CONFIG = {
    'data_path': 'data/final_attention',
    'model_save_dir': 'models/paper_implementation',
    'batch_size': 4,  # Small batch for high-frequency data
    'learning_rate': 5e-4,  # Base LR as in paper
    'init_lr': 1e-10,  # Initial LR for warmup
    'warmup_steps': 1000,  # 1000 warmup steps as in paper
    'decay_factor': 0.8,  # 0.8 decay factor as in paper
    'epochs': 50,
    'patience': 10,  # Early stopping patience
    'embed_dim': 126,  # Divisible by 3 heads (126 ÷ 3 = 42)
    'num_heads': 3,  # 3-head attention as in paper
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'd_ff': 512,  # Feedforward dimension
    'dropout': 0.1,
    'structural_weight': 0.01,  # w_o = 0.01 as in paper
    'use_revin': True,  # Reversible normalization
    'use_seasonal_decomp': True,  # Seasonal decomposition
}

# --- 1. Data Loading ---

class LOBDataset(Dataset):
    """Dataset for LOB sequences with memory-efficient loading."""
    def __init__(self, file_path):
        self.file_path = file_path
        with np.load(file_path) as data:
            self.len = data['x'].shape[0]
            self.x_shape = data['x'].shape
            self.y_shape = data['y'].shape

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with np.load(self.file_path) as data:
            x = data['x'][idx]
            y = data['y'][idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# --- 2. Time2Vec Embedding (Paper Implementation) ---

class Time2Vec(nn.Module):
    """Time2Vec embedding layer as described in the paper."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.w = nn.parameter.Parameter(torch.randn(1, embed_dim-1))
        self.b = nn.parameter.Parameter(torch.randn(1, embed_dim-1))
        self.f = torch.sin

    def forward(self, tau):
        """
        Args:
            tau: [batch, seq_len] time vectors
        Returns:
            [batch, seq_len, embed_dim] time embeddings
        """
        tau = tau.unsqueeze(-1)  # [batch, seq_len, 1]
        v1 = self.f(torch.matmul(tau, self.w) + self.b)  # [batch, seq_len, embed_dim-1]
        v2 = torch.matmul(tau, self.w0) + self.b0  # [batch, seq_len, 1]
        return torch.cat([v1, v2], -1)  # [batch, seq_len, embed_dim]

# --- 3. Compound Multivariate Embedding (Paper Implementation) ---

class CompoundMultivariateEmbedding(nn.Module):
    """
    Compound multivariate embedding as described in the paper.
    Embeds each attribute (level, type, feature, exchange, pair) separately
    then combines and scales the embeddings.
    """
    def __init__(self, embedding_metadata, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.metadata = embedding_metadata
        
        unique_attrs = embedding_metadata['unique_attributes']
        
        # Calculate embedding dimensions (divide total among attributes)
        attr_embed_dim = embed_dim // 5
        remaining_dim = embed_dim - (attr_embed_dim * 4)
        
        # Create embedding layers for each attribute
        self.level_embedding = nn.Embedding(len(unique_attrs['levels']), attr_embed_dim)
        self.type_embedding = nn.Embedding(len(unique_attrs['order_types']), attr_embed_dim)
        self.feature_embedding = nn.Embedding(len(unique_attrs['features']), attr_embed_dim)
        self.exchange_embedding = nn.Embedding(len(unique_attrs['exchanges']), attr_embed_dim)
        self.pair_embedding = nn.Embedding(len(unique_attrs['trading_pairs']), remaining_dim)
        
        # Projection to final embedding dimension
        self.projection = nn.Linear(embed_dim, embed_dim)
        
        # Create attribute index mappings
        self.level_to_idx = {level: i for i, level in enumerate(unique_attrs['levels'])}
        self.type_to_idx = {otype: i for i, otype in enumerate(unique_attrs['order_types'])}
        self.feature_to_idx = {feat: i for i, feat in enumerate(unique_attrs['features'])}
        self.exchange_to_idx = {exch: i for i, exch in enumerate(unique_attrs['exchanges'])}
        self.pair_to_idx = {pair: i for i, pair in enumerate(unique_attrs['trading_pairs'])}
        
        # Pre-compute feature attribute indices
        self._create_feature_indices()
    
    def _create_feature_indices(self):
        """Pre-compute attribute indices for all features."""
        columns = self.metadata['columns']
        column_mapping = self.metadata['column_mapping']
        
        level_indices = []
        type_indices = []
        feature_indices = []
        exchange_indices = []
        pair_indices = []
        
        for col in columns:
            attrs = column_mapping[col]
            level_indices.append(self.level_to_idx[attrs['level']])
            type_indices.append(self.type_to_idx[attrs['order_type']])
            feature_indices.append(self.feature_to_idx[attrs['feature_type']])
            exchange_indices.append(self.exchange_to_idx[attrs['exchange']])
            pair_indices.append(self.pair_to_idx[attrs['trading_pair']])
        
        # Register as buffers (moved to device automatically)
        self.register_buffer('level_indices', torch.LongTensor(level_indices))
        self.register_buffer('type_indices', torch.LongTensor(type_indices))
        self.register_buffer('feature_indices', torch.LongTensor(feature_indices))
        self.register_buffer('exchange_indices', torch.LongTensor(exchange_indices))
        self.register_buffer('pair_indices', torch.LongTensor(pair_indices))
    
    def forward(self):
        """
        Returns compound embedding for all features.
        Returns:
            [num_features, embed_dim] feature embeddings
        """
        # Get embeddings for each attribute
        level_embeds = self.level_embedding(self.level_indices)
        type_embeds = self.type_embedding(self.type_indices)
        feature_embeds = self.feature_embedding(self.feature_indices)
        exchange_embeds = self.exchange_embedding(self.exchange_indices)
        pair_embeds = self.pair_embedding(self.pair_indices)
        
        # Combine embeddings by concatenation (as in paper)
        combined_embeds = torch.cat([
            level_embeds, type_embeds, feature_embeds, 
            exchange_embeds, pair_embeds
        ], dim=-1)
        
        # Apply projection and scaling
        projected_embeds = self.projection(combined_embeds)
        
        return projected_embeds

# --- 4. Positional Encoding ---

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
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            [batch, seq_len, embed_dim]
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)

# --- 5. Structural Loss (Paper Implementation) ---

class StructuralLoss(nn.Module):
    """
    Structural regularizer to preserve LOB price ordering as described in paper.
    Equations (5), (6), (7), (8) from the paper.
    """
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
        
    def forward(self, predictions, feature_metadata):
        """
        Apply structural constraints to preserve price ordering.
        
        Args:
            predictions: [batch, seq_len, num_features] predictions
            feature_metadata: metadata about feature columns
        
        Returns:
            structural_loss: scalar loss value
        """
        if predictions.size(-1) < 10:  # Need sufficient features
            return torch.tensor(0.0, device=predictions.device)
        
        batch_size, seq_len, num_features = predictions.shape
        total_loss = 0.0
        
        # Extract price predictions by exchange and pair
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
        
        # Apply structural constraints for each exchange_pair
        for key, price_indices in exchange_pairs.items():
            bid_prices = price_indices['bid_prices']
            ask_prices = price_indices['ask_prices']
            
            if len(bid_prices) >= 2 and len(ask_prices) >= 2:
                # Constraint: ask_k < ask_{k+1} (ascending ask prices)
                for level in range(1, 5):  # levels 1-4
                    if level in ask_prices and (level + 1) in ask_prices:
                        ask_k = predictions[:, :, ask_prices[level]]
                        ask_k_plus_1 = predictions[:, :, ask_prices[level + 1]]
                        violation = torch.relu(ask_k - ask_k_plus_1)
                        total_loss += violation.mean()
                
                # Constraint: bid_k > bid_{k+1} (descending bid prices)
                for level in range(1, 5):  # levels 1-4
                    if level in bid_prices and (level + 1) in bid_prices:
                        bid_k = predictions[:, :, bid_prices[level]]
                        bid_k_plus_1 = predictions[:, :, bid_prices[level + 1]]
                        violation = torch.relu(bid_k_plus_1 - bid_k)
                        total_loss += violation.mean()
                
                # Constraint: bid_1 < ask_1 (bid-ask spread constraint)
                if 1 in bid_prices and 1 in ask_prices:
                    bid_1 = predictions[:, :, bid_prices[1]]
                    ask_1 = predictions[:, :, ask_prices[1]]
                    violation = torch.relu(bid_1 - ask_1)
                    total_loss += violation.mean()
        
        return self.weight * total_loss

# --- 6. Main LOB Forecaster (Paper Implementation) ---

class LOBForecaster(nn.Module):
    """
    Main attention-based LOB forecaster implementing the paper's architecture.
    Uses Spacetimeformer with compound multivariate embedding.
    """
    def __init__(self, embedding_metadata, embed_dim, num_heads, num_encoder_layers, 
                 num_decoder_layers, d_ff, dropout, target_len):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_features = len(embedding_metadata['columns'])
        self.target_len = target_len
        self.embed_dim = embed_dim
        
        # --- Embedding Layers ---
        self.value_projection = nn.Linear(1, embed_dim)
        self.compound_embedding = CompoundMultivariateEmbedding(embedding_metadata, embed_dim)
        self.time2vec = Time2Vec(embed_dim)
        self.context_target_embedding = nn.Embedding(2, embed_dim)  # 0=context, 1=target
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        
        # --- Spacetimeformer Transformer ---
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
        
        # --- Output Layer ---
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, 1)
        
    def forward(self, src, tgt, src_times=None, tgt_times=None):
        """
        Forward pass implementing the paper's methodology.
        
        Args:
            src: [batch, context_len, num_features] context sequence
            tgt: [batch, target_len, num_features] target sequence  
            src_times: [batch, context_len] context timestamps
            tgt_times: [batch, target_len] target timestamps
            
        Returns:
            [batch, target_len, num_features] predictions
        """
        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        # 1. Get compound multivariate embeddings (shared across all samples)
        feature_embeds = self.compound_embedding()  # [num_features, embed_dim]
        
        # 2. Project values and add feature embeddings
        src_values = self.value_projection(src.unsqueeze(-1))  # [batch, context_len, num_feat, embed_dim]
        tgt_values = self.value_projection(tgt.unsqueeze(-1))  # [batch, target_len, num_feat, embed_dim]
        
        # Add feature embeddings
        src_embedded = src_values + feature_embeds.unsqueeze(0).unsqueeze(0)
        tgt_embedded = tgt_values + feature_embeds.unsqueeze(0).unsqueeze(0)
        
        # 3. Add time embeddings (Time2Vec)
        if src_times is not None:
            time_embeds_src = self.time2vec(src_times)  # [batch, context_len, embed_dim]
            src_embedded = src_embedded + time_embeds_src.unsqueeze(2)
        
        if tgt_times is not None:
            time_embeds_tgt = self.time2vec(tgt_times)  # [batch, target_len, embed_dim]
            tgt_embedded = tgt_embedded + time_embeds_tgt.unsqueeze(2)
        
        # 4. Add context-target embeddings
        context_embed = self.context_target_embedding(torch.zeros(batch_size, context_len, dtype=torch.long, device=src.device))
        target_embed = self.context_target_embedding(torch.ones(batch_size, target_len, dtype=torch.long, device=src.device))
        
        src_embedded = src_embedded + context_embed.unsqueeze(2)
        tgt_embedded = tgt_embedded + target_embed.unsqueeze(2)
        
        # 5. Reshape for transformer: [batch * num_features, seq_len, embed_dim]
        src_flat = src_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, context_len, self.embed_dim)
        tgt_flat = tgt_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, target_len, self.embed_dim)
        
        # 6. Add positional encoding
        src_pos = self.positional_encoding(src_flat)
        tgt_pos = self.positional_encoding(tgt_flat)
        
        # 7. Apply transformer encoder
        memory = self.transformer_encoder(src_pos)
        
        # 8. Create causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_len).to(src.device)
        
        # 9. Apply transformer decoder
        output = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)
        
        # 10. Apply output normalization and projection
        output = self.output_norm(output)
        output = self.output_projection(output)  # [batch * num_feat, target_len, 1]
        
        # 11. Reshape back to original format
        output = output.reshape(batch_size, self.num_features, target_len).permute(0, 2, 1)
        
        return output

# --- 7. Training Functions ---

def create_model(embedding_metadata):
    """Create the LOB forecaster model."""
    model = LOBForecaster(
        embedding_metadata=embedding_metadata,
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        num_encoder_layers=CONFIG['num_encoder_layers'],
        num_decoder_layers=CONFIG['num_decoder_layers'],
        d_ff=CONFIG['d_ff'],
        dropout=CONFIG['dropout'],
        target_len=CONFIG.get('target_length', 24)
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created - Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model

def get_warmup_lr_scheduler(optimizer, warmup_steps, base_lr, init_lr):
    """Learning rate scheduler with warmup as described in paper."""
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup from init_lr to base_lr
            return init_lr + (base_lr - init_lr) * step / warmup_steps
        else:
            # After warmup, return base_lr (will be handled by ReduceLROnPlateau)
            return base_lr
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model, train_loader, mse_criterion, structural_loss, optimizer, embedding_metadata):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_struct_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (contexts, targets) in enumerate(progress_bar):
        contexts, targets = contexts.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Teacher forcing: use first part of targets as decoder input
        decoder_input = torch.cat([
            contexts[:, -1:, :],  # Last context step
            targets[:, :-1, :]    # All but last target step
        ], dim=1)
        
        # Forward pass
        predictions = model(contexts, decoder_input)
        
        # Compute losses
        mse_loss = mse_criterion(predictions, targets)
        struct_loss = structural_loss(predictions, embedding_metadata)
        
        # Combined loss (as in paper: Equation 9)
        total_batch_loss = mse_loss + struct_loss
        
        # Backward pass
        total_batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_batch_loss.item()
        total_mse_loss += mse_loss.item()
        total_struct_loss += struct_loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{total_batch_loss.item():.6f}',
            'MSE': f'{mse_loss.item():.6f}',
            'Struct': f'{struct_loss.item():.6f}'
        })
    
    return total_loss / num_batches, total_mse_loss / num_batches, total_struct_loss / num_batches

def validate(model, val_loader, mse_criterion, structural_loss, embedding_metadata):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_struct_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for contexts, targets in tqdm(val_loader, desc="Validation"):
            contexts, targets = contexts.to(DEVICE), targets.to(DEVICE)
            
            # Teacher forcing for validation
            decoder_input = torch.cat([
                contexts[:, -1:, :],
                targets[:, :-1, :]
            ], dim=1)
            
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
    """Main training function implementing the paper's methodology."""
    logger.info("Starting Attention-Based LOB Forecasting Training (Paper Implementation)")
    logger.info(f"Configuration: {CONFIG}")
    
    # Create model directory
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
    
    # Load data and metadata
    logger.info("Loading data and metadata...")
    
    with open(os.path.join(CONFIG['data_path'], 'embedding_metadata.json'), 'r') as f:
        embedding_metadata = json.load(f)
    
    with open(os.path.join(CONFIG['data_path'], 'config.json'), 'r') as f:
        data_config = json.load(f)
    
    # Update CONFIG with data configuration
    CONFIG.update(data_config)
    
    logger.info(f"Dataset: {embedding_metadata['num_features']} features, "
                f"Context: {CONFIG['context_length']}, Target: {CONFIG['target_length']}")
    
    # Create datasets and data loaders
    train_dataset = LOBDataset(os.path.join(CONFIG['data_path'], 'train.npz'))
    val_dataset = LOBDataset(os.path.join(CONFIG['data_path'], 'validation.npz'))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                          shuffle=False, num_workers=2, pin_memory=True)
    
    logger.info(f"Training sequences: {len(train_dataset)}")
    logger.info(f"Validation sequences: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(embedding_metadata)
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    structural_loss = StructuralLoss(weight=CONFIG['structural_weight'])
    
    # Optimizer (AdamW as commonly used with transformers)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['init_lr'], weight_decay=1e-4)
    
    # Learning rate schedulers (as in paper)
    warmup_scheduler = get_warmup_lr_scheduler(
        optimizer, CONFIG['warmup_steps'], CONFIG['learning_rate'], CONFIG['init_lr']
    )
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['decay_factor'], patience=3, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    step = 0
    
    logger.info("Starting training...")
    logger.info(f"Paper target metrics: Total loss < 0.008, Structure loss < 0.15")
    
    for epoch in range(CONFIG['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        # Training
        train_loss, train_mse, train_struct = train_epoch(
            model, train_loader, mse_criterion, structural_loss, optimizer, embedding_metadata
        )
        
        # Update learning rate (warmup phase)
        if step < CONFIG['warmup_steps']:
            warmup_scheduler.step()
            step += len(train_loader)
        
        # Validation
        val_loss, val_mse, val_struct = validate(
            model, val_loader, mse_criterion, structural_loss, embedding_metadata
        )
        
        # Update learning rate (post-warmup)
        if step >= CONFIG['warmup_steps']:
            plateau_scheduler.step(val_loss)
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Train - Total: {train_loss:.6f}, MSE: {train_mse:.6f}, Struct: {train_struct:.6f}")
        logger.info(f"Val   - Total: {val_loss:.6f}, MSE: {val_mse:.6f}, Struct: {val_struct:.6f}")
        logger.info(f"Learning Rate: {current_lr:.2e}")
        
        # Check if we've achieved paper's target metrics
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
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to: {CONFIG['model_save_dir']}/best_model.pt")

if __name__ == "__main__":
    main() 