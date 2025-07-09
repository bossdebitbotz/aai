#!/usr/bin/env python3
"""
Full Market Context to Binance Perpetual Prediction Model

This model uses ALL available market data (240 features) as input context
to predict ONLY Binance perpetual futures (80 features) as output.

Strategy:
- Input: All 240 features (3 exchanges × 4 pairs × 20 features)  
- Context: 20 minutes (240 steps × 5 seconds) - matches paper
- Output: Only 80 Binance perp features (4 pairs × 20 features)
- Target: 2 minutes (24 steps × 5 seconds)
- Leverages cross-market arbitrage and leading indicators
- H100 VPS optimized for maximum performance
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import logging
from tqdm import tqdm
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# --- H100 VPS Configuration ---
FINAL_DATA_DIR = "data/final_attention"  # Full 240-feature dataset
MODEL_SAVE_DIR = "models/full_to_binance_perp"
BATCH_SIZE = 32  # H100 optimized
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EPOCHS = 100
PATIENCE = 10
EMBED_DIM = 240  # Match input features for efficiency
NUM_HEADS = 8  # H100 optimized
NUM_ENCODER_LAYERS = 6  # Deeper for cross-market learning
NUM_DECODER_LAYERS = 4
DROPOUT = 0.1
REG_WEIGHT = 0.01  # Structural regularizer weight
NUM_WORKERS = 16  # H100 VPS optimized

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("full_to_binance_perp_training")

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- 1. Data Loading ---

class FullToBinancePerpDataset(Dataset):
    """Dataset that uses all 240 features as input, only Binance perp as output."""
    def __init__(self, file_path, target_feature_indices):
        self.file_path = file_path
        self.target_feature_indices = target_feature_indices
        
        with np.load(file_path) as data:
            self.len = data['x'].shape[0]
            self.x_shape = data['x'].shape  # (sequences, 240, 240) - 20min context
            self.y_shape = data['y'].shape  # (sequences, 24, 240)
            
        # New target shape: only Binance perp features
        self.target_y_shape = (self.len, self.y_shape[1], len(self.target_feature_indices))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with np.load(self.file_path) as data:
            x = data['x'][idx]  # All 240 features as input
            y_full = data['y'][idx]  # Full 240 features
            
            # Extract only Binance perp features for target
            y_target = y_full[:, self.target_feature_indices]
            
        return torch.from_numpy(x).float(), torch.from_numpy(y_target).float()

# --- 2. Embedding Layers ---

class CrossMarketCompoundEmbedding(nn.Module):
    """Full compound embedding for all 240 input features."""
    def __init__(self, embedding_metadata, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.metadata = embedding_metadata
        
        # Create embeddings for each attribute type
        unique_attrs = embedding_metadata['unique_attributes']
        
        # Calculate individual embedding sizes
        attr_embed_dim = embed_dim // 5
        remaining_dim = embed_dim - (attr_embed_dim * 4)
        
        self.level_embedding = nn.Embedding(len(unique_attrs['levels']), attr_embed_dim)
        self.type_embedding = nn.Embedding(len(unique_attrs['order_types']), attr_embed_dim)
        self.feature_embedding = nn.Embedding(len(unique_attrs['features']), attr_embed_dim)
        self.exchange_embedding = nn.Embedding(len(unique_attrs['exchanges']), attr_embed_dim)
        self.pair_embedding = nn.Embedding(len(unique_attrs['trading_pairs']), remaining_dim)
        
        # Projection layer
        self.projection = nn.Linear(attr_embed_dim * 4 + remaining_dim, embed_dim)
        
        # Create attribute mappings
        self.level_to_idx = {level: i for i, level in enumerate(unique_attrs['levels'])}
        self.type_to_idx = {otype: i for i, otype in enumerate(unique_attrs['order_types'])}
        self.feature_to_idx = {feat: i for i, feat in enumerate(unique_attrs['features'])}
        self.exchange_to_idx = {exch: i for i, exch in enumerate(unique_attrs['exchanges'])}
        self.pair_to_idx = {pair: i for i, pair in enumerate(unique_attrs['trading_pairs'])}
        
    def forward(self, num_features):
        """Create embeddings for all input features."""
        embeddings = []
        
        for i in range(num_features):
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
            
            # Concatenate and project
            combined_emb = torch.cat([level_emb, type_emb, feature_emb, exchange_emb, pair_emb], dim=0)
            embeddings.append(combined_emb)
        
        stacked_embeddings = torch.stack(embeddings)
        projected_embeddings = self.projection(stacked_embeddings)
        
        return projected_embeddings

class BinancePerpOutputEmbedding(nn.Module):
    """Specialized embedding for Binance perp output features only."""
    def __init__(self, embedding_metadata, target_indices, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.metadata = embedding_metadata
        self.target_indices = target_indices
        
        # Simplified for Binance perp only (no exchange dimension needed)
        attr_embed_dim = embed_dim // 4
        
        self.type_embedding = nn.Embedding(2, attr_embed_dim)  # bid, ask
        self.feature_embedding = nn.Embedding(2, attr_embed_dim)  # price, volume
        self.level_embedding = nn.Embedding(5, attr_embed_dim)  # 1-5
        self.pair_embedding = nn.Embedding(4, attr_embed_dim)  # 4 pairs
        
        # Projection layer
        self.projection = nn.Linear(attr_embed_dim * 4, embed_dim)
        
        # Create mappings for Binance perp features
        self.type_to_idx = {'bid': 0, 'ask': 1}
        self.feature_to_idx = {'price': 0, 'volume': 1}
        self.level_to_idx = {i: i-1 for i in range(1, 6)}
        self.pair_to_idx = {'BTC-USDT': 0, 'ETH-USDT': 1, 'SOL-USDT': 2, 'WLD-USDT': 3}
        
    def forward(self, num_target_features):
        """Create embeddings for Binance perp output features."""
        embeddings = []
        
        for i in range(num_target_features):
            # Get original column info for target feature
            orig_idx = self.target_indices[i]
            col_name = self.metadata['columns'][orig_idx]
            col_info = self.metadata['column_mapping'][col_name]
            
            type_idx = self.type_to_idx[col_info['order_type']]
            feature_idx = self.feature_to_idx[col_info['feature_type']]
            level_idx = self.level_to_idx[col_info['level']]
            pair_idx = self.pair_to_idx[col_info['trading_pair']]
            
            # Get embeddings
            type_emb = self.type_embedding(torch.tensor(type_idx, device=DEVICE))
            feature_emb = self.feature_embedding(torch.tensor(feature_idx, device=DEVICE))
            level_emb = self.level_embedding(torch.tensor(level_idx, device=DEVICE))
            pair_emb = self.pair_embedding(torch.tensor(pair_idx, device=DEVICE))
            
            # Concatenate and project
            combined_emb = torch.cat([type_emb, feature_emb, level_emb, pair_emb], dim=0)
            embeddings.append(combined_emb)
        
        stacked_embeddings = torch.stack(embeddings)
        projected_embeddings = self.projection(stacked_embeddings)
        
        return projected_embeddings

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
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

class FullToBinancePerpForecaster(nn.Module):
    """Cross-market model: 240 input features → 80 Binance perp output features."""
    def __init__(self, embedding_metadata, target_feature_indices, embed_dim, num_heads, 
                 num_encoder_layers, num_decoder_layers, dropout, target_len):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_input_features = embedding_metadata['num_features']  # 240
        self.target_feature_indices = target_feature_indices
        self.num_target_features = len(target_feature_indices)  # 80
        self.target_len = target_len
        self.embed_dim = embed_dim

        # --- Input Embedding (All 240 features) ---
        self.value_projection = nn.Linear(1, embed_dim)
        self.input_embedding = CrossMarketCompoundEmbedding(embedding_metadata, embed_dim)
        
        # --- Output Embedding (80 Binance perp features) ---
        self.output_embedding = BinancePerpOutputEmbedding(embedding_metadata, target_feature_indices, embed_dim)
        
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        # --- Cross-Market Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training
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

        # --- Cross-Market Attention Layer ---
        self.cross_market_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # --- Output Projection ---
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, src, tgt):
        # src shape: (batch_size, 240, 240) - 20min context, ALL market data
        # tgt shape: (batch_size, 24, 80) - 2min target, Binance perp only

        # Create embeddings
        input_feature_embeds = self.input_embedding(self.num_input_features)  # (240, embed_dim)
        output_feature_embeds = self.output_embedding(self.num_target_features)  # (80, embed_dim)
        
        # Project input values and add feature embeddings
        src_proj = self.value_projection(src.unsqueeze(-1))  # (batch, context_len, 240, embed_dim)
        tgt_proj = self.value_projection(tgt.unsqueeze(-1))  # (batch, target_len, 80, embed_dim)

        src_embedded = src_proj + input_feature_embeds.unsqueeze(0).unsqueeze(0)
        tgt_embedded = tgt_proj + output_feature_embeds.unsqueeze(0).unsqueeze(0)

        # Reshape for transformer
        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        # Encoder processes ALL market data
        src_flat = src_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_input_features, context_len, self.embed_dim)
        src_pos = self.positional_encoding(src_flat.permute(1, 0, 2)).permute(1, 0, 2)
        memory = self.transformer_encoder(src_pos)
        
        # Decoder focuses on Binance perp
        tgt_flat = tgt_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_target_features, target_len, self.embed_dim)
        tgt_pos = self.positional_encoding(tgt_flat.permute(1, 0, 2)).permute(1, 0, 2)
        
        # Create target mask
        tgt_mask = self.generate_square_subsequent_mask(target_len).to(DEVICE)
        
        # Cross-market decoding
        transformer_out = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)

        # Project to output
        output = self.output_layer(transformer_out)  # (batch*80, target_len, 1)
        
        # Reshape back to target format
        output = output.reshape(batch_size, self.num_target_features, target_len).permute(0, 2, 1)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# --- 4. Loss Functions ---

class BinancePerpStructuralLoss(nn.Module):
    """Structural regularizer for Binance perp predictions."""
    def __init__(self, embedding_metadata, target_feature_indices):
        super().__init__()
        self.metadata = embedding_metadata
        self.target_indices = target_feature_indices
        
        # Map target indices to price columns by trading pair
        self.price_indices_by_pair = {}
        
        for local_idx, global_idx in enumerate(target_feature_indices):
            col_name = self.metadata['columns'][global_idx]
            col_info = self.metadata['column_mapping'][col_name]
            
            if col_info['feature_type'] == 'price':
                pair = col_info['trading_pair']
                if pair not in self.price_indices_by_pair:
                    self.price_indices_by_pair[pair] = {'bid': {}, 'ask': {}}
                
                order_type = col_info['order_type']
                level = col_info['level']
                self.price_indices_by_pair[pair][order_type][level] = local_idx

    def forward(self, predictions):
        """Compute structural loss for Binance perp predictions."""
        batch_size, seq_len, num_target_features = predictions.shape
        total_loss = 0.0
        
        for pair, price_indices in self.price_indices_by_pair.items():
            bid_cols = price_indices['bid']
            ask_cols = price_indices['ask']
            
            max_level = min(max(bid_cols.keys()) if bid_cols else 0, 
                           max(ask_cols.keys()) if ask_cols else 0)
            
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
                        
                        violation = torch.relu(ask_k - ask_k1)
                        total_loss += violation.mean()
                
                # Bid price ordering: p_bid_k > p_bid_{k+1}
                for level in range(1, max_level):
                    if level in bid_cols and (level + 1) in bid_cols:
                        bid_k_idx = bid_cols[level]
                        bid_k1_idx = bid_cols[level + 1]
                        
                        bid_k = predictions[:, t, bid_k_idx]
                        bid_k1 = predictions[:, t, bid_k1_idx]
                        
                        violation = torch.relu(bid_k1 - bid_k)
                        total_loss += violation.mean()
                
                # Bid-ask spread constraint: p_bid_1 < p_ask_1
                if 1 in bid_cols and 1 in ask_cols:
                    bid_1_idx = bid_cols[1]
                    ask_1_idx = ask_cols[1]
                    
                    bid_1 = predictions[:, t, bid_1_idx]
                    ask_1 = predictions[:, t, ask_1_idx]
                    
                    violation = torch.relu(bid_1 - ask_1)
                    total_loss += violation.mean()
        
        return total_loss

# --- 5. Helper Functions ---

def get_binance_perp_indices(embedding_metadata):
    """Get indices of Binance perp features from the full 240-feature set."""
    target_indices = []
    
    for i, col_name in enumerate(embedding_metadata['columns']):
        col_info = embedding_metadata['column_mapping'][col_name]
        if col_info['exchange'] == 'binance_perp':
            target_indices.append(i)
    
    return target_indices

def warmup_lr_schedule(step, warmup_steps, d_model):
    """Learning rate warmup schedule."""
    return min(step ** -0.5, step * warmup_steps ** -1.5) * (d_model ** -0.5)

# --- 6. Training Functions ---

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

# --- 7. Main Execution ---

def main():
    """Main training function."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"{MODEL_SAVE_DIR}_{timestamp}"
    os.makedirs(model_save_path, exist_ok=True)

    logger.info("Starting Full Market → Binance Perp Model Training...")
    logger.info("=" * 60)
    
    # Load embedding metadata
    with open(os.path.join(FINAL_DATA_DIR, 'embedding_metadata.json'), 'r') as f:
        embedding_metadata = json.load(f)
    
    # Get Binance perp target indices
    target_feature_indices = get_binance_perp_indices(embedding_metadata)
    
    logger.info(f"Input features: {embedding_metadata['num_features']} (all markets)")
    logger.info(f"Output features: {len(target_feature_indices)} (Binance perp only)")
    logger.info(f"Cross-market strategy: Leverage all data to predict Binance perp")
    
    # Create datasets
    train_dataset = FullToBinancePerpDataset(
        os.path.join(FINAL_DATA_DIR, 'train.npz'), 
        target_feature_indices
    )
    val_dataset = FullToBinancePerpDataset(
        os.path.join(FINAL_DATA_DIR, 'validation.npz'), 
        target_feature_indices
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    logger.info(f"Training sequences: {len(train_dataset)}")
    logger.info(f"Validation sequences: {len(val_dataset)}")
    
    # Initialize model
    logger.info("Initializing cross-market model...")
    model = FullToBinancePerpForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        target_len=24
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Initialize loss functions and optimizer
    mse_loss_fn = nn.MSELoss()
    struct_loss_fn = BinancePerpStructuralLoss(embedding_metadata, target_feature_indices).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_mse': [], 'val_mse': [],
        'train_struct': [], 'val_struct': []
    }

    logger.info(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, mse_loss_fn, struct_loss_fn, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, mse_loss_fn, struct_loss_fn)
        
        # Update scheduler
        scheduler.step(val_metrics['total_loss'])
        
        # Log metrics
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"  Train - Loss: {train_metrics['total_loss']:.6f}, MSE: {train_metrics['mse_loss']:.6f}, Struct: {train_metrics['struct_loss']:.6f}")
        logger.info(f"  Val   - Loss: {val_metrics['total_loss']:.6f}, MSE: {val_metrics['mse_loss']:.6f}, Struct: {val_metrics['struct_loss']:.6f}")
        
        # Save history
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
                'val_mse': val_metrics['mse_loss'],
                'val_struct': val_metrics['struct_loss'],
                'embedding_metadata': embedding_metadata,
                'target_feature_indices': target_feature_indices,
                'config': {
                    'embed_dim': EMBED_DIM,
                    'num_heads': NUM_HEADS,
                    'num_encoder_layers': NUM_ENCODER_LAYERS,
                    'num_decoder_layers': NUM_DECODER_LAYERS,
                    'dropout': DROPOUT,
                    'reg_weight': REG_WEIGHT,
                    'strategy': 'full_market_to_binance_perp'
                }
            }, os.path.join(model_save_path, 'best_model.pth'))
            
            logger.info(f"  → New best model saved! Val loss: {val_metrics['total_loss']:.6f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= PATIENCE:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Save training history
    history_path = os.path.join(model_save_path, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("CROSS-MARKET BINANCE PERP MODEL TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Strategy: 240 input features → 80 Binance perp predictions")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"Training took {epoch} epochs")

if __name__ == "__main__":
    main() 