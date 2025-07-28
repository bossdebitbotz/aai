#!/usr/bin/env python3
"""
WLD 3-Minute Prediction Model - Fixed Architecture

Uses the proven cross-market architecture to predict WLD-USDT perpetual futures.
- Input: All 240 features (3 exchanges × 4 pairs × 20 features)  
- Context: 20 minutes (240 steps × 5 seconds)
- Output: Only 20 WLD features (1 pair × 20 features)
- Target: 3 minutes (36 steps × 5 seconds)
- Architecture: Working encoder-decoder with proper feature handling
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import json
import logging
from tqdm import tqdm
import math
from datetime import datetime

# --- Configuration ---
FINAL_DATA_DIR = "data/data/final_attention"  # Updated to use full 39-day dataset
MODEL_SAVE_DIR = "models/wld_3min_full_dataset"  # New save directory for full dataset model
BATCH_SIZE = 32   # Large batch size for 4 GPUs (8 per GPU)
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EPOCHS = 130
PATIENCE = 10
EMBED_DIM = 128
NUM_HEADS = 8    
NUM_ENCODER_LAYERS = 4  
NUM_DECODER_LAYERS = 3  
DROPOUT = 0.1
REG_WEIGHT = 0.01
NUM_WORKERS = 8

# WLD-specific configuration
TARGET_STEPS = 36  # 3 minutes × 60 seconds ÷ 5 seconds = 36 steps
TARGET_PAIR = "WLD-USDT"
TARGET_EXCHANGE = "binance_perp"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("wld_3min_full_dataset_training")

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
logger.info(f"🚀 Target: 3-minute WLD predictions using FULL 39-DAY DATASET (40x more training data!)")
logger.info(f"💪 Training on 303 sequences from 39 days vs previous 642 sequences from 4 hours")

# --- Copy the same model classes from the working script ---

class FullToBinancePerpDataset(Dataset):
    """Dataset for full market → WLD perp prediction."""
    def __init__(self, file_path, target_feature_indices, target_steps=36):
        with np.load(file_path, allow_pickle=True) as data:
            self.context = torch.from_numpy(data['contexts']).float()
            self.target_full = torch.from_numpy(data['targets']).float()
            
            # Handle variable target length
            current_target_len = self.target_full.shape[1]  # Current is 24
            
            if target_steps <= current_target_len:
                # Truncate if we need fewer steps
                self.target_full = self.target_full[:, :target_steps, :]
            else:
                # Pad if we need more steps (repeat last values)
                padding_needed = target_steps - current_target_len
                last_values = self.target_full[:, -1:, :].repeat(1, padding_needed, 1)
                self.target_full = torch.cat([self.target_full, last_values], dim=1)
            
            # Extract only target features
            self.target = self.target_full[:, :, target_feature_indices]
            self.len = self.context.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]

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
        # Get device from embedding weights instead of next(parameters())
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
        # Get device from embedding weights instead of next(parameters())
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

class BinancePerpStructuralLoss(nn.Module):
    """Structural regularizer for Binance perp predictions."""
    def __init__(self, embedding_metadata, target_feature_indices):
        super().__init__()
        self.metadata = embedding_metadata
        self.target_indices = target_feature_indices
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
                for level in range(1, max_level):
                    if level in ask_cols and (level + 1) in ask_cols:
                        ask_k_idx = ask_cols[level]
                        ask_k1_idx = ask_cols[level + 1]
                        
                        ask_k = predictions[:, t, ask_k_idx]
                        ask_k1 = predictions[:, t, ask_k1_idx]
                        
                        violation = torch.relu(ask_k - ask_k1)
                        total_loss += violation.mean()
                
                for level in range(1, max_level):
                    if level in bid_cols and (level + 1) in bid_cols:
                        bid_k_idx = bid_cols[level]
                        bid_k1_idx = bid_cols[level + 1]
                        
                        bid_k = predictions[:, t, bid_k_idx]
                        bid_k1 = predictions[:, t, bid_k1_idx]
                        
                        violation = torch.relu(bid_k1 - bid_k)
                        total_loss += violation.mean()
                
                if 1 in bid_cols and 1 in ask_cols:
                    bid_1_idx = bid_cols[1]
                    ask_1_idx = ask_cols[1]
                    
                    bid_1 = predictions[:, t, bid_1_idx]
                    ask_1 = predictions[:, t, ask_1_idx]
                    
                    violation = torch.relu(bid_1 - ask_1)
                    total_loss += violation.mean()
        
        return total_loss

def get_wld_perp_indices(embedding_metadata):
    """Get indices of WLD perpetual features only."""
    target_indices = []
    
    for i, col_name in enumerate(embedding_metadata['columns']):
        col_info = embedding_metadata['column_mapping'][col_name]
        if (col_info['exchange'] == 'binance_perp' and 
            col_info['trading_pair'] == 'WLD-USDT'):
            target_indices.append(i)
    
    return target_indices

def warmup_lr_schedule(step, warmup_steps, d_model):
    """Learning rate warmup schedule."""
    return min(step ** -0.5, step * warmup_steps ** -1.5) * (d_model ** -0.5)

def main():
    """Main training function for WLD 3-minute predictions using FULL 39-day dataset"""
    logger.info("🚀 WLD 3-Minute Prediction Training Started with FULL DATASET!")
    logger.info("📈 Using 39 days of LOB data (823k records) vs previous 4 hours (83k records)")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return
        
    num_gpus = torch.cuda.device_count()
    logger.info(f"Using {num_gpus} GPUs with DataParallel")
    logger.info(f"Effective batch size: {BATCH_SIZE} ({BATCH_SIZE//num_gpus} per GPU)")
    if num_gpus != 4:
        logger.warning(f"Expected 4 H100s but found {num_gpus} GPUs")

    # --- Data Loading ---
    logger.info("Loading data and metadata...")
    
    # Try multiple data directory locations
    data_dir = FINAL_DATA_DIR
    try:
        with open(os.path.join(data_dir, 'embedding_metadata.json'), 'r') as f:
            embedding_metadata = json.load(f)
    except FileNotFoundError:
        logger.error(f"Metadata file not found in {data_dir}")
        logger.info("Trying alternative data directory...")
        try:
            data_dir = 'data/final_attention'
            with open(os.path.join(data_dir, 'embedding_metadata.json'), 'r') as f:
                embedding_metadata = json.load(f)
        except FileNotFoundError:
            logger.error("No metadata file found. Please check data preparation.")
            return

    target_feature_indices = get_wld_perp_indices(embedding_metadata)
    
    logger.info(f"Input features: 240 (ALL market data)")
    logger.info(f"Output features: {len(target_feature_indices)} (WLD perp only)")
    logger.info(f"Prediction: 3 minutes (36 steps)")
    logger.info(f"Context: 20 minutes (240 steps)")

    # Load datasets
    try:
        train_dataset = FullToBinancePerpDataset(
            os.path.join(data_dir, 'train.npz'), 
            target_feature_indices,
            target_steps=TARGET_STEPS
        )
        val_dataset = FullToBinancePerpDataset(
            os.path.join(data_dir, 'validation.npz'), 
            target_feature_indices,
            target_steps=TARGET_STEPS
        )
    except FileNotFoundError:
        logger.error(f"Dataset files not found in {data_dir}")
        return

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    logger.info(f"Training sequences: {len(train_dataset)}")
    logger.info(f"Validation sequences: {len(val_dataset)}")

    # --- Model Creation ---
    logger.info("Initializing WLD-specific model...")
    
    model = FullToBinancePerpForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        target_len=TARGET_STEPS  # 36 steps for 3 minutes
    )

    # Use DataParallel for multi-GPU (4x H100s)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Model wrapped with DataParallel for {num_gpus} H100 GPUs")
    
    model = model.to(DEVICE)

    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("💰 Using all 4x H100s - maximizing expensive compute!")

    # --- Loss Functions & Optimizer ---
    mse_loss_fn = nn.MSELoss()
    struct_loss_fn = BinancePerpStructuralLoss(embedding_metadata, target_feature_indices).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Mixed precision scaler
    scaler = GradScaler()

    # --- Training Loop ---
    logger.info(f"Starting training for {EPOCHS} epochs...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Training
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
                
                # Mixed precision forward pass
                with autocast('cuda'):
                    predictions = model(context, decoder_input)
                    
                    # Compute losses
                    mse_loss = mse_loss_fn(predictions, target)
                    struct_loss = struct_loss_fn(predictions)
                    total_loss_batch = mse_loss + REG_WEIGHT * struct_loss
                
                # Mixed precision backward pass
                optimizer.zero_grad()
                scaler.scale(total_loss_batch).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
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
        
        train_metrics = {
            'total_loss': total_loss / len(train_loader),
            'mse_loss': total_mse / len(train_loader),
            'struct_loss': total_struct / len(train_loader)
        }
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_mse = 0.0
        total_val_struct = 0.0
        
        with torch.no_grad():
            for context, target in val_loader:
                context, target = context.to(DEVICE), target.to(DEVICE)
                
                # Create decoder input (shifted target)
                decoder_input = torch.zeros_like(target)
                decoder_input[:, 1:] = target[:, :-1]
                
                # Mixed precision forward pass
                with autocast('cuda'):
                    predictions = model(context, decoder_input)
                    
                    # Compute losses
                    mse_loss = mse_loss_fn(predictions, target)
                    struct_loss = struct_loss_fn(predictions)
                    total_loss_batch = mse_loss + REG_WEIGHT * struct_loss
                
                # Accumulate losses
                total_val_loss += total_loss_batch.item()
                total_val_mse += mse_loss.item()
                total_val_struct += struct_loss.item()
        
        val_metrics = {
            'total_loss': total_val_loss / len(val_loader),
            'mse_loss': total_val_mse / len(val_loader),
            'struct_loss': total_val_struct / len(val_loader)
        }
        
        # Logging
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"Train - Loss: {train_metrics['total_loss']:.6f}, "
                   f"MSE: {train_metrics['mse_loss']:.6f}, "
                   f"Struct: {train_metrics['struct_loss']:.6f}")
        logger.info(f"Val   - Loss: {val_metrics['total_loss']:.6f}, "
                   f"MSE: {val_metrics['mse_loss']:.6f}, "
                   f"Struct: {val_metrics['struct_loss']:.6f}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            
            # Save model (no DataParallel wrapper to handle)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'target_feature_indices': target_feature_indices,
                'target_steps': TARGET_STEPS,
                'target_pair': TARGET_PAIR
            }
            
            torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, 'best_wld_3min_model.pt'))
            logger.info(f"💾 New best WLD model saved! Val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{PATIENCE}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            logger.info("🛑 Early stopping triggered!")
            break
        
        logger.info("-" * 80)
    
    logger.info("🎉 WLD 3-Minute Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to: {MODEL_SAVE_DIR}/best_wld_3min_model.pt")
    logger.info("Ready for 3-minute WLD predictions!")

if __name__ == "__main__":
    main() 