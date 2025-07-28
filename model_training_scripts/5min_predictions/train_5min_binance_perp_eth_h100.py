#!/usr/bin/env python3
"""
H100 Scaled 5-Minute ETH Binance Perp Forecasting Model
Uses proven multi-GPU architecture with target-specific predictions

Based on working train_working_scaled.py but adapted for 5min ETH predictions
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

# --- H100 SCALED CONFIGURATION ---
FINAL_DATA_DIR = "data/final_attention"
MODEL_SAVE_DIR = "models/5min_binance_perp_eth_h100"
BATCH_SIZE = 32  # H100 optimized (was 1-4 in original scripts)
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EPOCHS = 50
PATIENCE = 8
EMBED_DIM = 256  # H100 optimized (was 64-128 in original scripts)
NUM_HEADS = 8    # 256/8 = 32 (divisible)
NUM_ENCODER_LAYERS = 6  # Deeper for H100
NUM_DECODER_LAYERS = 4  # Deeper for H100
DROPOUT = 0.1
REG_WEIGHT = 0.01
NUM_WORKERS = 16

# Target configuration for 5-minute ETH predictions
TARGET_MINUTES = 5
TARGET_STEPS = 60  # 5 min * 60 sec/min / 5 sec/step
TARGET_PAIR = "ETH-USDT"
TARGET_EXCHANGE = "binance_perp"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("5min_eth_h100_training")

# Check H100 setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
logger.info(f"Using device: {DEVICE}")
logger.info(f"Available H100 GPUs: {num_gpus}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory/1e9:.1f}GB)")

class H100TargetSpecificDataset(Dataset):
    """H100-optimized dataset for target-specific predictions."""
    def __init__(self, file_path, target_feature_indices, target_steps):
        with np.load(file_path, allow_pickle=True) as data:
            self.context = torch.from_numpy(data['contexts']).float()
            self.target_full = torch.from_numpy(data['targets']).float()
            # Extract only target features and adjust sequence length
            self.target = self.target_full[:, :target_steps, target_feature_indices]
            self.len = self.context.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]

class H100CompoundEmbedding(nn.Module):
    """H100-optimized compound embedding."""
    def __init__(self, embedding_metadata, embed_dim):
        super().__init__()
        self.metadata = embedding_metadata
        self.embed_dim = embed_dim
        
        # Larger embeddings for H100 performance
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

class H100PositionalEncoding(nn.Module):
    """H100-optimized positional encoding."""
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

class H100TargetSpecificForecaster(nn.Module):
    """H100-optimized target-specific LOB forecaster."""
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
        self.compound_embedding = H100CompoundEmbedding(embedding_metadata, embed_dim)
        self.positional_encoding = H100PositionalEncoding(embed_dim, dropout, max_len=100000)

        # H100-optimized transformer layers
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
        
        # Disable nested tensor for H100 stability
        self.transformer_encoder.enable_nested_tensor = False
        self.transformer_decoder.enable_nested_tensor = False

        # H100-optimized output layers
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
        # Get embeddings for all input features
        input_feature_embeds = self.compound_embedding(self.num_input_features)
        # Use only target feature embeddings for decoder
        target_feature_embeds = input_feature_embeds[self.target_feature_indices]
        
        # Project input values and add feature embeddings
        src_proj = self.value_projection(src.unsqueeze(-1))
        tgt_proj = self.value_projection(tgt.unsqueeze(-1))

        src_embedded = src_proj + input_feature_embeds.unsqueeze(0).unsqueeze(0)
        tgt_embedded = tgt_proj + target_feature_embeds.unsqueeze(0).unsqueeze(0)

        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        # H100-optimized sequence flattening
        src_flat = src_embedded.reshape(batch_size, context_len * self.num_input_features, self.embed_dim)
        src_pos = self.positional_encoding(src_flat)
        memory = self.transformer_encoder(src_pos)
        
        tgt_flat = tgt_embedded.reshape(batch_size, target_len * self.num_target_features, self.embed_dim)
        tgt_pos = self.positional_encoding(tgt_flat)
        
        # Create causal mask for decoder
        combined_target_len = target_len * self.num_target_features
        tgt_mask = self.generate_square_subsequent_mask(combined_target_len).to(src.device)
        
        transformer_out = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)

        # Project to output
        output = self.output_layer(transformer_out)
        output = output.squeeze(-1)
        output = output.reshape(batch_size, target_len, self.num_target_features)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def get_target_feature_indices(embedding_metadata, target_exchange, target_pair):
    """Get indices for target exchange and pair."""
    target_indices = []
    
    for i, col_name in enumerate(embedding_metadata['columns']):
        col_info = embedding_metadata['column_mapping'][col_name]
        if (col_info['exchange'] == target_exchange and 
            col_info['trading_pair'] == target_pair):
            target_indices.append(i)
    
    return target_indices

def warmup_lr_schedule(step, warmup_steps, d_model):
    """Learning rate warmup schedule."""
    return min(step ** -0.5, step * warmup_steps ** -1.5) * (d_model ** -0.5)

def main():
    """Main H100 training function."""
    logger.info("🚀 H100 Scaled 5-Minute ETH Training Started!")
    logger.info(f"Target: {TARGET_MINUTES}-minute predictions for {TARGET_EXCHANGE} {TARGET_PAIR}")
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return
        
    effective_batch_size = BATCH_SIZE * max(1, num_gpus)
    logger.info(f"Using {num_gpus} H100 GPUs with DataParallel")
    logger.info(f"Per-GPU batch size: {BATCH_SIZE}")
    logger.info(f"Effective batch size: {effective_batch_size}")

    # --- Data Loading ---
    logger.info("Loading data and metadata...")
    
    try:
        with open(os.path.join(FINAL_DATA_DIR, 'embedding_metadata.json'), 'r') as f:
            embedding_metadata = json.load(f)
    except FileNotFoundError:
        logger.error(f"Metadata file not found in {FINAL_DATA_DIR}")
        return

    # Get target feature indices
    target_feature_indices = get_target_feature_indices(embedding_metadata, TARGET_EXCHANGE, TARGET_PAIR)
    num_target_features = len(target_feature_indices)
    
    logger.info(f"Input features: {embedding_metadata['num_features']}")
    logger.info(f"Target: {TARGET_EXCHANGE} {TARGET_PAIR}")
    logger.info(f"Target features: {num_target_features} out of {len(embedding_metadata['columns'])}")
    logger.info(f"Prediction horizon: {TARGET_MINUTES} minutes ({TARGET_STEPS} steps)")

    # Create H100-optimized datasets
    try:
        train_dataset = H100TargetSpecificDataset(
            os.path.join(FINAL_DATA_DIR, 'train.npz'), 
            target_feature_indices, 
            TARGET_STEPS
        )
        val_dataset = H100TargetSpecificDataset(
            os.path.join(FINAL_DATA_DIR, 'validation.npz'), 
            target_feature_indices, 
            TARGET_STEPS
        )
    except FileNotFoundError:
        logger.error(f"Dataset files not found in {FINAL_DATA_DIR}")
        return

    # Create H100-optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True
    )

    logger.info(f"Training sequences: {len(train_dataset)}")
    logger.info(f"Validation sequences: {len(val_dataset)}")

    # --- H100 Model Creation ---
    logger.info("Initializing H100-optimized model...")
    
    model = H100TargetSpecificForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        target_len=TARGET_STEPS,
        num_target_features=num_target_features
    )

    # Apply DataParallel for H100 multi-GPU
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Model wrapped with DataParallel for {num_gpus} H100 GPUs")
    
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Model size: ~{total_params * 4 / 1e9:.2f}GB (FP32)")

    # --- Loss Functions & Optimizer ---
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # H100-optimized mixed precision
    scaler = GradScaler()

    # --- H100 Training Loop ---
    logger.info(f"Starting H100 training for {EPOCHS} epochs...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        total_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (context, target) in enumerate(pbar):
                context, target = context.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                
                # Create decoder input (shifted target)
                decoder_input = torch.zeros_like(target)
                decoder_input[:, 1:] = target[:, :-1]
                
                # H100 mixed precision forward pass
                with autocast('cuda'):
                    predictions = model(context, decoder_input)
                    loss = mse_loss_fn(predictions, target)
                
                # H100 mixed precision backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Learning rate warmup
                step = epoch * len(train_loader) + batch_idx
                if step < WARMUP_STEPS:
                    lr = warmup_lr_schedule(step + 1, WARMUP_STEPS, EMBED_DIM)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr * LEARNING_RATE
                
                total_loss += loss.item()
                
                # Update progress bar
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'LR': f'{current_lr:.2e}',
                    'H100s': f'{num_gpus}'
                })
        
        train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for context, target in val_loader:
                context, target = context.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                
                decoder_input = torch.zeros_like(target)
                decoder_input[:, 1:] = target[:, :-1]
                
                with autocast('cuda'):
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
            
            # Handle DataParallel saving
            model_to_save = model.module if hasattr(model, 'module') else model
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'target_feature_indices': target_feature_indices,
                'model_config': {
                    'embed_dim': EMBED_DIM,
                    'num_heads': NUM_HEADS,
                    'num_encoder_layers': NUM_ENCODER_LAYERS,
                    'num_decoder_layers': NUM_DECODER_LAYERS,
                    'target_steps': TARGET_STEPS,
                    'target_pair': TARGET_PAIR,
                    'target_exchange': TARGET_EXCHANGE,
                }
            }
            
            torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, 'best_model_h100.pt'))
            logger.info(f"💾 New best H100 model saved! Val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{PATIENCE}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            logger.info("🛑 Early stopping triggered!")
            break
        
        logger.info("-" * 80)
    
    logger.info("🎉 H100 Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
