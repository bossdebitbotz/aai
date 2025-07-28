#!/usr/bin/env python3
"""
H100 EFFICIENT MODE: Smart Architecture Design
Fixes the core issue: DON'T flatten features into sequence dimension

PROBLEM: sequence_len = 120 × 240 = 28,800 → attention matrix = 28,800² = massive OOM
SOLUTION: Keep sequence_len = 120, handle 240 features with feature-wise processing
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

# --- H100 EFFICIENT MODE CONFIGURATION ---
FINAL_DATA_DIR = "data/final_attention"
MODEL_SAVE_DIR = "models/h100_efficient_wld"

# EFFICIENT SCALING - No sequence flattening!
BATCH_SIZE = 96       # Can be larger now that we fixed the architecture
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EPOCHS = 50
PATIENCE = 8

# EFFICIENT MODEL SCALING
EMBED_DIM = 512       # Good size for features
FEATURE_DIM = 256     # Separate dimension for feature processing
NUM_HEADS = 8         # 512/8 = 64 (perfectly divisible)
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 4
FEEDFORWARD_DIM = 2048

DROPOUT = 0.1
REG_WEIGHT = 0.01
NUM_WORKERS = 24

# Target configuration
TARGET_MINUTES = 1
TARGET_STEPS = 12
TARGET_PAIR = "WLD-USDT"
TARGET_EXCHANGE = "binance_perp"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("h100_efficient_mode")

# Check H100 setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
logger.info(f"⚡ H100 EFFICIENT MODE ACTIVATED ⚡")
logger.info(f"Using device: {DEVICE}")
logger.info(f"Available H100 GPUs: {num_gpus}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory/1e9:.1f}GB)")

effective_batch_size = BATCH_SIZE * max(1, num_gpus)
logger.info(f"⚡ EFFICIENT SCALING:")
logger.info(f"  Per-GPU batch size: {BATCH_SIZE}")
logger.info(f"  Effective batch size: {effective_batch_size}")
logger.info(f"  Sequence length: 120 (NOT 28,800!)")
logger.info(f"  Feature dimension: {FEATURE_DIM}")
logger.info(f"  Embed dimension: {EMBED_DIM}")
logger.info(f"  Architecture: Feature-first, then sequence")

class H100EfficientDataset(Dataset):
    """H100 Efficient Mode dataset."""
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

class FeatureProcessor(nn.Module):
    """Process 240 features efficiently without flattening into sequence."""
    def __init__(self, num_features, feature_dim):
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        
        # Feature-wise processing
        self.feature_embeddings = nn.Embedding(num_features, feature_dim)
        self.feature_projection = nn.Linear(1, feature_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)
        
        # Feature attention (compress 240 features)
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature compression (240 → 64 compressed features)
        self.feature_compressor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, feature_dim // 8)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, num_features)
        batch_size, seq_len, num_features = x.shape
        
        # Create feature embeddings
        feature_ids = torch.arange(num_features, device=x.device)
        feature_embeds = self.feature_embeddings(feature_ids)  # (num_features, feature_dim)
        
        # Project values and add feature embeddings
        x_proj = self.feature_projection(x.unsqueeze(-1))  # (batch, seq_len, num_features, feature_dim)
        x_embedded = x_proj + feature_embeds.unsqueeze(0).unsqueeze(0)  # Broadcast
        
        # Reshape for feature attention: (batch * seq_len, num_features, feature_dim)
        x_reshaped = x_embedded.reshape(batch_size * seq_len, num_features, self.feature_dim)
        
        # Apply feature attention (each timestep attends over features)
        attended_features, _ = self.feature_attention(x_reshaped, x_reshaped, x_reshaped)
        
        # Compress features: 240 → 32 compressed features
        compressed = self.feature_compressor(attended_features)  # (batch * seq_len, num_features, compressed_dim)
        
        # Take mean over features to get fixed-size representation
        compressed_mean = compressed.mean(dim=1)  # (batch * seq_len, compressed_dim)
        
        # Reshape back: (batch, seq_len, compressed_dim)
        output = compressed_mean.reshape(batch_size, seq_len, -1)
        
        return output

class H100EfficientPositionalEncoding(nn.Module):
    """Standard positional encoding for sequence length 120."""
    def __init__(self, d_model, dropout=0.1, max_len=1000):  # Much smaller!
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

class H100EfficientForecaster(nn.Module):
    """H100 Efficient Mode: Smart architecture that doesn't flatten features into sequence."""
    def __init__(self, embedding_metadata, target_feature_indices, embed_dim, feature_dim, num_heads, 
                 num_encoder_layers, num_decoder_layers, feedforward_dim, dropout, target_len, num_target_features):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_input_features = embedding_metadata['num_features']
        self.target_feature_indices = target_feature_indices
        self.num_target_features = num_target_features
        self.target_len = target_len
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim

        # Feature processing (240 features → compressed representation)
        self.input_feature_processor = FeatureProcessor(self.num_input_features, feature_dim)
        self.target_feature_processor = FeatureProcessor(self.num_target_features, feature_dim)
        
        # Project compressed features to embedding dimension
        compressed_dim = feature_dim // 8  # From FeatureProcessor
        self.input_projection = nn.Linear(compressed_dim, embed_dim)
        self.target_projection = nn.Linear(compressed_dim, embed_dim)
        
        # Positional encoding for sequence (120 steps, NOT 28,800!)
        self.positional_encoding = H100EfficientPositionalEncoding(embed_dim, dropout, max_len=1000)

        # Transformer with manageable sequence length
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=feedforward_dim,
            dropout=dropout, 
            activation='gelu', 
            batch_first=True, 
            norm_first=True
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=feedforward_dim,
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

        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, self.num_target_features)
        )

    def forward(self, src, tgt):
        # src: (batch, 120, 240), tgt: (batch, 12, 20)
        batch_size = src.shape[0]
        
        # Process features efficiently (NO sequence flattening!)
        src_processed = self.input_feature_processor(src)  # (batch, 120, compressed_dim)
        tgt_processed = self.target_feature_processor(tgt)  # (batch, 12, compressed_dim)
        
        # Project to embedding space
        src_embedded = self.input_projection(src_processed)  # (batch, 120, embed_dim)
        tgt_embedded = self.target_projection(tgt_processed)  # (batch, 12, embed_dim)
        
        # Add positional encoding
        src_pos = self.positional_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)  # (batch, 120, embed_dim)
        tgt_pos = self.positional_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)  # (batch, 12, embed_dim)
        
        # Transformer with reasonable sequence lengths
        memory = self.transformer_encoder(src_pos)  # (batch, 120, embed_dim)
        
        # Create causal mask for target sequence (12 steps, NOT 28,800!)
        tgt_mask = self.generate_square_subsequent_mask(self.target_len).to(src.device)
        
        transformer_out = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)  # (batch, 12, embed_dim)
        
        # Output projection
        output = self.output_layer(transformer_out)  # (batch, 12, num_target_features)
        
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
    """H100 Efficient Mode training function."""
    logger.info("⚡⚡⚡ H100 EFFICIENT MODE ENGAGED! ⚡⚡⚡")
    logger.info("🔧 ARCHITECTURE FIX: No more sequence flattening!")
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return

    # --- Data Loading ---
    logger.info("Loading data and metadata...")
    
    try:
        with open(os.path.join(FINAL_DATA_DIR, 'embedding_metadata.json'), 'r') as f:
            embedding_metadata = json.load(f)
    except FileNotFoundError:
        logger.error(f"Metadata file not found in {FINAL_DATA_DIR}")
        return

    target_feature_indices = get_target_feature_indices(embedding_metadata, TARGET_EXCHANGE, TARGET_PAIR)
    num_target_features = len(target_feature_indices)
    
    logger.info(f"Input features: {embedding_metadata['num_features']}")
    logger.info(f"Target: {TARGET_EXCHANGE} {TARGET_PAIR}")
    logger.info(f"Target features: {num_target_features}")

    # Create EFFICIENT datasets
    try:
        train_dataset = H100EfficientDataset(
            os.path.join(FINAL_DATA_DIR, 'train.npz'), 
            target_feature_indices, 
            TARGET_STEPS
        )
        val_dataset = H100EfficientDataset(
            os.path.join(FINAL_DATA_DIR, 'validation.npz'), 
            target_feature_indices, 
            TARGET_STEPS
        )
    except FileNotFoundError:
        logger.error(f"Dataset files not found in {FINAL_DATA_DIR}")
        return

    # Create EFFICIENT data loaders
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

    # --- H100 EFFICIENT MODEL CREATION ---
    logger.info("🏗️ Building H100 EFFICIENT MODEL...")
    
    model = H100EfficientForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=EMBED_DIM,
        feature_dim=FEATURE_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        feedforward_dim=FEEDFORWARD_DIM,
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
    model_size_gb = total_params * 4 / 1e9
    logger.info(f"⚡ EFFICIENT MODEL STATS:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Model size: ~{model_size_gb:.2f}GB (FP32)")
    logger.info(f"  Max sequence length: 120 (manageable!)")
    logger.info(f"  Expected GPU memory per card: ~{model_size_gb * BATCH_SIZE / 16:.1f}GB")

    # --- Loss Functions & Optimizer ---
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # H100 mixed precision
    scaler = GradScaler()

    # --- H100 EFFICIENT TRAINING LOOP ---
    logger.info(f"🚀 Starting EFFICIENT training for {EPOCHS} epochs...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        total_loss = 0.0
        
        with tqdm(train_loader, desc=f"⚡ EFFICIENT Epoch {epoch}") as pbar:
            for batch_idx, (context, target) in enumerate(pbar):
                context, target = context.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                
                decoder_input = torch.zeros_like(target)
                decoder_input[:, 1:] = target[:, :-1]
                
                # H100 EFFICIENT forward pass
                with autocast('cuda'):
                    predictions = model(context, decoder_input)
                    loss = mse_loss_fn(predictions, target)
                
                # H100 EFFICIENT backward pass
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
                
                # Memory usage monitoring
                if batch_idx % 10 == 0:
                    gpu_memory = torch.cuda.memory_allocated() / 1e9
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.6f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                        'GPU_GB': f'{gpu_memory:.1f}',
                        'EFFICIENT': '⚡'
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
        
        # Logging with memory stats
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        
        logger.info(f"⚡ EFFICIENT Epoch {epoch}/{EPOCHS}")
        logger.info(f"Train Loss: {train_loss:.6f}")
        logger.info(f"Val Loss: {val_loss:.6f}")
        logger.info(f"GPU Memory: {gpu_memory:.1f}GB / Max: {max_memory:.1f}GB")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
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
                    'feature_dim': FEATURE_DIM,
                    'num_heads': NUM_HEADS,
                    'num_encoder_layers': NUM_ENCODER_LAYERS,
                    'num_decoder_layers': NUM_DECODER_LAYERS,
                    'feedforward_dim': FEEDFORWARD_DIM,
                    'target_steps': TARGET_STEPS,
                    'target_pair': TARGET_PAIR,
                    'target_exchange': TARGET_EXCHANGE,
                }
            }
            
            torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, 'best_efficient_model.pt'))
            logger.info(f"💾 New EFFICIENT model saved! Val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{PATIENCE}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            logger.info("🛑 Early stopping triggered!")
            break
        
        logger.info("-" * 80)
    
    logger.info("🎉 H100 EFFICIENT TRAINING COMPLETED! 🎉")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

if __name__ == "__main__":
    main() 