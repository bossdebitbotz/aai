#!/usr/bin/env python3
"""
Attention-Based LOB Forecasting Model Training

This script implements the attention-based LOB forecasting model described in:
"Attention-Based Reading, Highlighting, and Forecasting of the Limit Order Book"

Key features:
1. Compound multivariate embedding for LOB attributes
2. Structural regularizer to preserve price ordering
3. Performer attention for computational efficiency
4. Session-aware training with proper loss balancing
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib
import json
import logging
from tqdm import tqdm
import math
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Configuration ---
FINAL_DATA_DIR = "data/final_attention"  # Updated to use new data
MODEL_SAVE_DIR = "models/attention_lob"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EPOCHS = 50
PATIENCE = 5  # Early stopping patience
EMBED_DIM = 126  # Divisible by 3 heads (126 ÷ 3 = 42)
NUM_HEADS = 3  # As per paper
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1
REG_WEIGHT = 0.01  # Structural regularizer weight (w_o)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("attention_lob_training")

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- 1. Data Loading ---

class LOBDataset(Dataset):
    """PyTorch Dataset for loading LOB sequences with lazy loading."""
    def __init__(self, file_path):
        self.file_path = file_path
        with np.load(file_path) as data:
            self.len = data['x'].shape[0]
            self.x_dtype = data['x'].dtype
            self.y_dtype = data['y'].dtype
            self.x_shape = data['x'].shape
            self.y_shape = data['y'].shape

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Open the file and read only the requested index
        # This is slower but vastly more memory efficient
        with np.load(self.file_path) as data:
            x = data['x'][idx]
            y = data['y'][idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# --- 2. Embedding Layers ---

class CompoundMultivariateEmbedding(nn.Module):
    """
    Compound multivariate embedding that combines multiple attribute embeddings
    instead of individual variable embeddings as in the paper.
    """
    def __init__(self, embedding_metadata, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.metadata = embedding_metadata
        
        # Create embeddings for each attribute type
        unique_attrs = embedding_metadata['unique_attributes']
        
        # Calculate individual embedding sizes to sum to embed_dim
        attr_embed_dim = embed_dim // 5
        remaining_dim = embed_dim - (attr_embed_dim * 4)  # Handle any remainder
        
        self.level_embedding = nn.Embedding(len(unique_attrs['levels']), attr_embed_dim)
        self.type_embedding = nn.Embedding(len(unique_attrs['order_types']), attr_embed_dim)
        self.feature_embedding = nn.Embedding(len(unique_attrs['features']), attr_embed_dim)
        self.exchange_embedding = nn.Embedding(len(unique_attrs['exchanges']), attr_embed_dim)
        self.pair_embedding = nn.Embedding(len(unique_attrs['trading_pairs']), remaining_dim)
        
        # Projection layer to get to exact embed_dim
        self.projection = nn.Linear(attr_embed_dim * 4 + remaining_dim, embed_dim)
        
        # Create attribute mappings
        self.level_to_idx = {level: i for i, level in enumerate(unique_attrs['levels'])}
        self.type_to_idx = {otype: i for i, otype in enumerate(unique_attrs['order_types'])}
        self.feature_to_idx = {feat: i for i, feat in enumerate(unique_attrs['features'])}
        self.exchange_to_idx = {exch: i for i, exch in enumerate(unique_attrs['exchanges'])}
        self.pair_to_idx = {pair: i for i, pair in enumerate(unique_attrs['trading_pairs'])}
        
    def forward(self, num_features):
        """Create embeddings for all features based on their attributes."""
        embeddings = []
        
        for i in range(num_features):
            # Get attribute indices for this feature
            col_name = self.metadata['columns'][i]
            col_info = self.metadata['column_mapping'][col_name]
            
            level_idx = self.level_to_idx[col_info['level']]
            type_idx = self.type_to_idx[col_info['order_type']]
            feature_idx = self.feature_to_idx[col_info['feature_type']]
            exchange_idx = self.exchange_to_idx[col_info['exchange']]
            pair_idx = self.pair_to_idx[col_info['trading_pair']]
            
            # Get embeddings for each attribute
            level_emb = self.level_embedding(torch.tensor(level_idx, device=DEVICE))
            type_emb = self.type_embedding(torch.tensor(type_idx, device=DEVICE))
            feature_emb = self.feature_embedding(torch.tensor(feature_idx, device=DEVICE))
            exchange_emb = self.exchange_embedding(torch.tensor(exchange_idx, device=DEVICE))
            pair_emb = self.pair_embedding(torch.tensor(pair_idx, device=DEVICE))
            
            # Concatenate all attribute embeddings
            combined_emb = torch.cat([level_emb, type_emb, feature_emb, exchange_emb, pair_emb], dim=0)
            embeddings.append(combined_emb)
        
        # Stack embeddings and project to final dimension
        stacked_embeddings = torch.stack(embeddings)  # (num_features, embed_dim)
        projected_embeddings = self.projection(stacked_embeddings)
        
        return projected_embeddings

class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformer."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- 3. Model Architecture ---

class LOBForecaster(nn.Module):
    """The main attention-based LOB forecasting model."""
    def __init__(self, embedding_metadata, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout, target_len):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_features = len(embedding_metadata['columns'])
        self.target_len = target_len
        self.embed_dim = embed_dim

        # --- Embedding Layers ---
        self.value_projection = nn.Linear(1, embed_dim)
        self.compound_embedding = CompoundMultivariateEmbedding(embedding_metadata, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        # --- Transformer ---
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        # --- Output Layer ---
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, src, tgt):
        # src shape: (batch_size, context_len, num_features)
        # tgt shape: (batch_size, target_len, num_features)

        # Create feature embeddings (constant for all batches)
        feature_embeds = self.compound_embedding(self.num_features)  # (num_features, embed_dim)
        
        # Project values and add feature embeddings
        src_proj = self.value_projection(src.unsqueeze(-1))  # (batch, context_len, num_feat, embed_dim)
        tgt_proj = self.value_projection(tgt.unsqueeze(-1))  # (batch, target_len, num_feat, embed_dim)

        src_embedded = src_proj + feature_embeds.unsqueeze(0).unsqueeze(0)
        tgt_embedded = tgt_proj + feature_embeds.unsqueeze(0).unsqueeze(0)

        # Reshape for transformer: (batch_size * num_features, seq_len, embed_dim)
        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        src_flat = src_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, context_len, self.embed_dim)
        tgt_flat = tgt_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, target_len, self.embed_dim)
        
        # Add positional encoding
        src_pos = self.positional_encoding(src_flat.permute(1, 0, 2)).permute(1, 0, 2)
        tgt_pos = self.positional_encoding(tgt_flat.permute(1, 0, 2)).permute(1, 0, 2)
        
        # Create target mask for decoder
        tgt_mask = self.transformer.generate_square_subsequent_mask(target_len).to(DEVICE)
        
        # Pass through transformer
        transformer_out = self.transformer(src_pos, tgt_pos, tgt_mask=tgt_mask)  # (batch*num_feat, target_len, embed_dim)

        # Project to output
        output = self.output_layer(transformer_out)  # (batch*num_feat, target_len, 1)
        
        # Reshape back to original format
        output = output.reshape(batch_size, self.num_features, target_len).permute(0, 2, 1)
        
        return output

# --- 4. Loss Functions ---

class StructuralLoss(nn.Module):
    """
    Structural regularizer to preserve LOB price ordering constraints
    as described in the paper.
    """
    def __init__(self, embedding_metadata):
        super().__init__()
        self.metadata = embedding_metadata
        self.columns = embedding_metadata['columns']
        
        # Find price column indices
        self.price_indices = {}
        self.volume_indices = {}
        
        for i, col in enumerate(self.columns):
            col_info = self.metadata['column_mapping'][col]
            exchange = col_info['exchange']
            pair = col_info['trading_pair']
            level = col_info['level']
            order_type = col_info['order_type']
            feature_type = col_info['feature_type']
            
            key = f"{exchange}_{pair}"
            
            if feature_type == 'price':
                if key not in self.price_indices:
                    self.price_indices[key] = {'bid': {}, 'ask': {}}
                self.price_indices[key][order_type][level] = i
            elif feature_type == 'volume':
                if key not in self.volume_indices:
                    self.volume_indices[key] = {'bid': {}, 'ask': {}}
                self.volume_indices[key][order_type][level] = i

    def forward(self, predictions):
        """
        Compute structural loss for LOB price ordering constraints.
        
        Constraints:
        1. Ask prices should increase with level: p_ask_1 < p_ask_2 < ... < p_ask_5
        2. Bid prices should decrease with level: p_bid_1 > p_bid_2 > ... > p_bid_5
        3. Bid-ask spread should be positive: p_bid_1 < p_ask_1
        """
        batch_size, seq_len, num_features = predictions.shape
        total_loss = 0.0
        
        for key, price_cols in self.price_indices.items():
            if 'bid' not in price_cols or 'ask' not in price_cols:
                continue
                
            bid_cols = price_cols['bid']
            ask_cols = price_cols['ask']
            
            # Get maximum available levels
            max_bid_level = max(bid_cols.keys()) if bid_cols else 0
            max_ask_level = max(ask_cols.keys()) if ask_cols else 0
            max_level = min(max_bid_level, max_ask_level)
            
            if max_level < 2:
                continue
                
            for t in range(seq_len):
                # Ask price ordering: p_ask_k < p_ask_{k+1}
                for level in range(1, max_level):
                    if level in ask_cols and (level + 1) in ask_cols:
                        ask_k_idx = ask_cols[level]
                        ask_k1_idx = ask_cols[level + 1]
                        
                        ask_k = predictions[:, t, ask_k_idx]
                        ask_k1 = predictions[:, t, ask_k1_idx]
                        
                        # ReLU penalty for violation
                        violation = torch.relu(ask_k - ask_k1)
                        total_loss += violation.mean()
                
                # Bid price ordering: p_bid_k > p_bid_{k+1}
                for level in range(1, max_level):
                    if level in bid_cols and (level + 1) in bid_cols:
                        bid_k_idx = bid_cols[level]
                        bid_k1_idx = bid_cols[level + 1]
                        
                        bid_k = predictions[:, t, bid_k_idx]
                        bid_k1 = predictions[:, t, bid_k1_idx]
                        
                        # ReLU penalty for violation
                        violation = torch.relu(bid_k1 - bid_k)
                        total_loss += violation.mean()
                
                # Bid-ask spread constraint: p_bid_1 < p_ask_1
                if 1 in bid_cols and 1 in ask_cols:
                    bid_1_idx = bid_cols[1]
                    ask_1_idx = ask_cols[1]
                    
                    bid_1 = predictions[:, t, bid_1_idx]
                    ask_1 = predictions[:, t, ask_1_idx]
                    
                    # ReLU penalty for violation
                    violation = torch.relu(bid_1 - ask_1)
                    total_loss += violation.mean()
        
        return total_loss

# --- 5. Training Functions ---

def warmup_lr_schedule(step, warmup_steps, d_model):
    """Learning rate warmup schedule from the paper."""
    return min(step ** -0.5, step * warmup_steps ** -1.5) * (d_model ** -0.5)

def train_epoch(model, train_loader, optimizer, mse_loss_fn, struct_loss_fn, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_struct = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
        for batch_idx, (context, target) in enumerate(pbar):
            context, target = context.to(DEVICE), target.to(DEVICE)
            
            # Create decoder input (shifted target)
            decoder_input = torch.zeros_like(target)
            decoder_input[:, 1:] = target[:, :-1]
            
            # Forward pass
            predictions = model(context, decoder_input)
            
            # Compute losses
            mse_loss = mse_loss_fn(predictions, target)
            struct_loss = struct_loss_fn(predictions)
            total_loss_batch = mse_loss + REG_WEIGHT * struct_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update learning rate with warmup
            step = epoch * len(train_loader) + batch_idx
            lr = warmup_lr_schedule(step + 1, WARMUP_STEPS, EMBED_DIM)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * LEARNING_RATE
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_struct += struct_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.6f}',
                'MSE': f'{mse_loss.item():.6f}',
                'Struct': f'{struct_loss.item():.6f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
    
    return {
        'total_loss': total_loss / len(train_loader),
        'mse_loss': total_mse / len(train_loader),
        'struct_loss': total_struct / len(train_loader)
    }

def validate_epoch(model, val_loader, mse_loss_fn, struct_loss_fn):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_struct = 0.0
    
    with torch.no_grad():
        for context, target in val_loader:
            context, target = context.to(DEVICE), target.to(DEVICE)
            
            # Create decoder input
            decoder_input = torch.zeros_like(target)
            decoder_input[:, 1:] = target[:, :-1]
            
            # Forward pass
            predictions = model(context, decoder_input)
            
            # Compute losses
            mse_loss = mse_loss_fn(predictions, target)
            struct_loss = struct_loss_fn(predictions)
            total_loss_batch = mse_loss + REG_WEIGHT * struct_loss
            
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_struct += struct_loss.item()
    
    return {
        'total_loss': total_loss / len(val_loader),
        'mse_loss': total_mse / len(val_loader),
        'struct_loss': total_struct / len(val_loader)
    }

# --- 6. Main Execution ---

def main():
    """Main function to run the training process."""
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # Load data and metadata
    logger.info("Loading data and metadata...")
    
    # Load embedding metadata
    with open(os.path.join(FINAL_DATA_DIR, 'embedding_metadata.json'), 'r') as f:
        embedding_metadata = json.load(f)
    
    # Load dataset configuration
    with open(os.path.join(FINAL_DATA_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    
    logger.info(f"Dataset configuration: {config}")
    logger.info(f"Number of features: {embedding_metadata['num_features']}")
    
    # Create datasets and data loaders
    train_dataset = LOBDataset(os.path.join(FINAL_DATA_DIR, 'train.npz'))
    val_dataset = LOBDataset(os.path.join(FINAL_DATA_DIR, 'validation.npz'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    logger.info(f"Training sequences: {len(train_dataset)}")
    logger.info(f"Validation sequences: {len(val_dataset)}")
    
    # Initialize model
    logger.info("Initializing model...")
    target_len = config['target_length']
    
    model = LOBForecaster(
        embedding_metadata=embedding_metadata,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        target_len=target_len
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize loss functions and optimizer
    mse_loss_fn = nn.MSELoss()
    struct_loss_fn = StructuralLoss(embedding_metadata).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_mse': [], 'val_mse': [],
        'train_struct': [], 'val_struct': []
    }
    
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, mse_loss_fn, struct_loss_fn, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, mse_loss_fn, struct_loss_fn)
        
        # Update scheduler
        scheduler.step(val_metrics['total_loss'])
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        logger.info(f"Train - Total: {train_metrics['total_loss']:.6f}, MSE: {train_metrics['mse_loss']:.6f}, Struct: {train_metrics['struct_loss']:.6f}")
        logger.info(f"Val   - Total: {val_metrics['total_loss']:.6f}, MSE: {val_metrics['mse_loss']:.6f}, Struct: {val_metrics['struct_loss']:.6f}")
        
        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_mse'].append(train_metrics['mse_loss'])
        history['val_mse'].append(val_metrics['mse_loss'])
        history['train_struct'].append(train_metrics['struct_loss'])
        history['val_struct'].append(val_metrics['struct_loss'])
        
        # Early stopping and model saving
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            epochs_no_improve = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'embedding_metadata': embedding_metadata,
                'config': config
            }, os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
            
            logger.info(f"New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= PATIENCE:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model and training history
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['total_loss'],
        'embedding_metadata': embedding_metadata,
        'config': config
    }, os.path.join(MODEL_SAVE_DIR, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(MODEL_SAVE_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Models saved in: {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    main() 