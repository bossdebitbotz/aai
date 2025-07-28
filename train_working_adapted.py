#!/usr/bin/env python3
"""
Adapted Working Script: Uses proven architecture with available data format
Based on train_full_to_binance_perp_dataparallel.py but adapted for current data
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

# --- Configuration (from working script) ---
FINAL_DATA_DIR = "data/final_attention"  # Use available data directory
MODEL_SAVE_DIR = "models/working_adapted"
BATCH_SIZE = 4   # Start smaller than working script's 32
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EPOCHS = 50
PATIENCE = 5
EMBED_DIM = 128  # From working script
NUM_HEADS = 8    # From working script (128/8 = 16, divisible)
NUM_ENCODER_LAYERS = 4  
NUM_DECODER_LAYERS = 3  
DROPOUT = 0.1
REG_WEIGHT = 0.01
NUM_WORKERS = 4

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("working_adapted")

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
logger.info(f"Available GPUs: {torch.cuda.device_count()}")

class WorkingAdaptedDataset(Dataset):
    """Dataset adapted for contexts/targets keys."""
    def __init__(self, file_path, target_feature_indices):
        with np.load(file_path, allow_pickle=True) as data:
            # Use available keys: 'contexts' and 'targets' instead of 'x' and 'y'
            self.context = torch.from_numpy(data['contexts']).float()
            self.target_full = torch.from_numpy(data['targets']).float()
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

class WorkingForecaster(nn.Module):
    """Working model architecture adapted for available data."""
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

def get_target_indices(embedding_metadata):
    """Get indices of WLD binance_perp features."""
    target_indices = []
    
    for i, col_name in enumerate(embedding_metadata['columns']):
        col_info = embedding_metadata['column_mapping'][col_name]
        if (col_info['exchange'] == 'binance_perp' and 
            col_info['trading_pair'] == 'WLD-USDT'):
            target_indices.append(i)
    
    return target_indices

def main():
    """Main training function."""
    logger.info("🚀 Working Adapted Training Started!")
    
    # --- Data Loading ---
    logger.info("Loading data and metadata...")
    
    try:
        with open(os.path.join(FINAL_DATA_DIR, 'embedding_metadata.json'), 'r') as f:
            embedding_metadata = json.load(f)
    except FileNotFoundError:
        logger.error(f"Metadata file not found in {FINAL_DATA_DIR}")
        return

    target_feature_indices = get_target_indices(embedding_metadata)
    
    logger.info(f"Input features: {embedding_metadata['num_features']}")
    logger.info(f"Target features: {len(target_feature_indices)} (WLD binance_perp)")

    # Load datasets
    try:
        train_dataset = WorkingAdaptedDataset(
            os.path.join(FINAL_DATA_DIR, 'train.npz'), 
            target_feature_indices
        )
        val_dataset = WorkingAdaptedDataset(
            os.path.join(FINAL_DATA_DIR, 'validation.npz'), 
            target_feature_indices
        )
    except FileNotFoundError:
        logger.error(f"Dataset files not found in {FINAL_DATA_DIR}")
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
    logger.info("Initializing working model...")
    
    model = WorkingForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        target_len=12  # 1 minute = 12 steps
    ).to(DEVICE)

    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Loss Functions & Optimizer ---
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # --- Training Loop ---
    logger.info(f"Starting training for {EPOCHS} epochs...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        total_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (context, target) in enumerate(pbar):
                context, target = context.to(DEVICE), target.to(DEVICE)
                
                # Create decoder input (shifted target)
                decoder_input = torch.zeros_like(target)
                decoder_input[:, 1:] = target[:, :-1]
                
                # Forward pass
                predictions = model(context, decoder_input)
                
                # Compute loss
                loss = mse_loss_fn(predictions, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}'
                })
        
        train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for context, target in val_loader:
                context, target = context.to(DEVICE), target.to(DEVICE)
                
                decoder_input = torch.zeros_like(target)
                decoder_input[:, 1:] = target[:, :-1]
                
                predictions = model(context, decoder_input)
                loss = mse_loss_fn(predictions, target)
                
                total_val_loss += loss.item()
        
        val_loss = total_val_loss / len(val_loader)
        
        # Logging
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"Train Loss: {train_loss:.6f}")
        logger.info(f"Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'target_feature_indices': target_feature_indices
            }
            
            torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, 'best_model.pt'))
            logger.info(f"💾 New best model saved! Val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{PATIENCE}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            logger.info("🛑 Early stopping triggered!")
            break
        
        logger.info("-" * 80)
    
    logger.info("🎉 Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main() 