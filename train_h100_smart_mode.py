#!/usr/bin/env python3
"""
H100 SMART MODE: Intelligent Memory Utilization
Uses gradient accumulation to achieve massive effective batch sizes without OOM

BEAST MODE failed: Batch size 128 → OOM (attention matrices too big)
SMART MODE solution: Batch size 64 + gradient accumulation = Same throughput, no OOM
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

# --- H100 SMART MODE CONFIGURATION ---
FINAL_DATA_DIR = "data/final_attention"
MODEL_SAVE_DIR = "models/h100_smart_wld"

# SMART SCALING - Balance memory vs performance
BATCH_SIZE = 64       # Reduced from BEAST 128 to avoid OOM
GRADIENT_ACCUMULATION_STEPS = 4  # 64 × 4 = 256 effective batch size per GPU
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EPOCHS = 50
PATIENCE = 8

# LARGE BUT SMART MODEL SCALING
EMBED_DIM = 768       # Reduced from BEAST 1024 to 768
NUM_HEADS = 12        # 768/12 = 64 (perfectly divisible)
NUM_ENCODER_LAYERS = 8  # Reduced from BEAST 12 to 8
NUM_DECODER_LAYERS = 6  # Reduced from BEAST 8 to 6
FEEDFORWARD_DIM = 3072  # Reduced from BEAST 4096 to 3072

DROPOUT = 0.1
REG_WEIGHT = 0.01
NUM_WORKERS = 24      # Between original 16 and BEAST 32

# Target configuration
TARGET_MINUTES = 1
TARGET_STEPS = 12
TARGET_PAIR = "WLD-USDT"
TARGET_EXCHANGE = "binance_perp"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("h100_smart_mode")

# Check H100 setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
logger.info(f"🧠 H100 SMART MODE ACTIVATED 🧠")
logger.info(f"Using device: {DEVICE}")
logger.info(f"Available H100 GPUs: {num_gpus}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory/1e9:.1f}GB)")

effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * max(1, num_gpus)
logger.info(f"🚀 SMART SCALING:")
logger.info(f"  Per-GPU batch size: {BATCH_SIZE}")
logger.info(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
logger.info(f"  Effective batch size: {effective_batch_size} ({BATCH_SIZE}×{GRADIENT_ACCUMULATION_STEPS}×{num_gpus})")
logger.info(f"  Embed dimension: {EMBED_DIM}")
logger.info(f"  Encoder layers: {NUM_ENCODER_LAYERS}")
logger.info(f"  Decoder layers: {NUM_DECODER_LAYERS}")
logger.info(f"  Feedforward dim: {FEEDFORWARD_DIM}")

class H100SmartDataset(Dataset):
    """H100 Smart Mode dataset."""
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

class H100SmartEmbedding(nn.Module):
    """Smart embedding for H100 optimization."""
    def __init__(self, embedding_metadata, embed_dim):
        super().__init__()
        self.metadata = embedding_metadata
        self.embed_dim = embed_dim
        
        # Smart embeddings - more than original, less than BEAST
        embed_per_type = embed_dim // 6  # 6 embedding types
        self.price_embed = nn.Embedding(1, embed_per_type)
        self.size_embed = nn.Embedding(1, embed_per_type)
        self.exchange_embed = nn.Embedding(4, embed_per_type)
        self.pair_embed = nn.Embedding(8, embed_per_type)
        self.level_embed = nn.Embedding(16, embed_per_type)
        self.time_embed = nn.Embedding(32, embed_per_type)

    def forward(self, num_features):
        embeddings = []
        device = self.price_embed.weight.device
        
        for i in range(num_features):
            feature_embed = torch.cat([
                self.price_embed(torch.tensor(0, device=device)),
                self.size_embed(torch.tensor(0, device=device)),
                self.exchange_embed(torch.tensor(i % 3, device=device)),
                self.pair_embed(torch.tensor(i % 7, device=device)),
                self.level_embed(torch.tensor(i % 15, device=device)),
                self.time_embed(torch.tensor(i % 31, device=device))
            ])
            embeddings.append(feature_embed)
        
        return torch.stack(embeddings)

class H100SmartPositionalEncoding(nn.Module):
    """Smart positional encoding for H100s."""
    def __init__(self, d_model, dropout=0.1, max_len=150000):
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

class H100SmartForecaster(nn.Module):
    """H100 Smart Mode: Optimized transformer for memory efficiency."""
    def __init__(self, embedding_metadata, target_feature_indices, embed_dim, num_heads, 
                 num_encoder_layers, num_decoder_layers, feedforward_dim, dropout, target_len, num_target_features):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_input_features = embedding_metadata['num_features']
        self.target_feature_indices = target_feature_indices
        self.num_target_features = num_target_features
        self.target_len = target_len
        self.embed_dim = embed_dim

        # Smart value projection - 3 layers instead of BEAST's 5
        self.value_projection = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        self.compound_embedding = H100SmartEmbedding(embedding_metadata, embed_dim)
        self.positional_encoding = H100SmartPositionalEncoding(embed_dim, dropout, max_len=150000)

        # Smart transformer layers
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

        # Smart output network - 3 layers instead of BEAST's 5
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1)
        )

        # 2 cross-attention layers instead of BEAST's 4
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(2)
        ])

    def forward(self, src, tgt):
        # Get embeddings for all input features
        input_feature_embeds = self.compound_embedding(self.num_input_features)
        target_feature_embeds = input_feature_embeds[self.target_feature_indices]
        
        # Smart value projection
        src_proj = self.value_projection(src.unsqueeze(-1))
        tgt_proj = self.value_projection(tgt.unsqueeze(-1))

        src_embedded = src_proj + input_feature_embeds.unsqueeze(0).unsqueeze(0)
        tgt_embedded = tgt_proj + target_feature_embeds.unsqueeze(0).unsqueeze(0)

        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        # H100 sequence processing
        src_flat = src_embedded.reshape(batch_size, context_len * self.num_input_features, self.embed_dim)
        src_pos = self.positional_encoding(src_flat)
        memory = self.transformer_encoder(src_pos)
        
        # Apply cross-attention layers
        for cross_attn in self.cross_attention_layers:
            memory, _ = cross_attn(memory, memory, memory)
        
        tgt_flat = tgt_embedded.reshape(batch_size, target_len * self.num_target_features, self.embed_dim)
        tgt_pos = self.positional_encoding(tgt_flat)
        
        # Create causal mask
        combined_target_len = target_len * self.num_target_features
        tgt_mask = self.generate_square_subsequent_mask(combined_target_len).to(src.device)
        
        transformer_out = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)

        # Smart output projection
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
    """H100 Smart Mode training function."""
    logger.info("🧠🧠🧠 H100 SMART MODE ENGAGED! 🧠🧠🧠")
    
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

    # Create SMART datasets
    try:
        train_dataset = H100SmartDataset(
            os.path.join(FINAL_DATA_DIR, 'train.npz'), 
            target_feature_indices, 
            TARGET_STEPS
        )
        val_dataset = H100SmartDataset(
            os.path.join(FINAL_DATA_DIR, 'validation.npz'), 
            target_feature_indices, 
            TARGET_STEPS
        )
    except FileNotFoundError:
        logger.error(f"Dataset files not found in {FINAL_DATA_DIR}")
        return

    # Create SMART data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    logger.info(f"Training sequences: {len(train_dataset)}")
    logger.info(f"Validation sequences: {len(val_dataset)}")

    # --- H100 SMART MODEL CREATION ---
    logger.info("🏗️ Building H100 SMART MODEL...")
    
    model = H100SmartForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=EMBED_DIM,
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
    logger.info(f"🧠 SMART MODEL STATS:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Model size: ~{model_size_gb:.2f}GB (FP32)")
    logger.info(f"  Expected GPU memory per card: ~{model_size_gb * BATCH_SIZE / 8:.1f}GB")

    # --- Loss Functions & Optimizer ---
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # H100 mixed precision
    scaler = GradScaler()

    # --- H100 SMART TRAINING LOOP ---
    logger.info(f"🚀 Starting SMART training for {EPOCHS} epochs...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Training with gradient accumulation
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        with tqdm(train_loader, desc=f"🧠 SMART Epoch {epoch}") as pbar:
            for batch_idx, (context, target) in enumerate(pbar):
                context, target = context.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                
                decoder_input = torch.zeros_like(target)
                decoder_input[:, 1:] = target[:, :-1]
                
                # H100 SMART forward pass
                with autocast('cuda'):
                    predictions = model(context, decoder_input)
                    loss = mse_loss_fn(predictions, target)
                    # Scale loss by accumulation steps
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                # H100 SMART backward pass with accumulation
                scaler.scale(loss).backward()
                
                # Only step optimizer every GRADIENT_ACCUMULATION_STEPS
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Learning rate warmup
                    step = epoch * len(train_loader) + batch_idx
                    if step < WARMUP_STEPS:
                        lr = warmup_lr_schedule(step + 1, WARMUP_STEPS, EMBED_DIM)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr * LEARNING_RATE
                
                total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # Unscale for logging
                
                # Memory usage monitoring
                if batch_idx % 10 == 0:
                    gpu_memory = torch.cuda.memory_allocated() / 1e9
                    pbar.set_postfix({
                        'Loss': f'{(loss.item() * GRADIENT_ACCUMULATION_STEPS):.6f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                        'GPU_GB': f'{gpu_memory:.1f}',
                        'SMART': '🧠'
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
        
        logger.info(f"🧠 SMART Epoch {epoch}/{EPOCHS}")
        logger.info(f"Train Loss: {train_loss:.6f}")
        logger.info(f"Val Loss: {val_loss:.6f}")
        logger.info(f"GPU Memory: {gpu_memory:.1f}GB / Max: {max_memory:.1f}GB")
        logger.info(f"Effective Batch Size: {effective_batch_size}")
        
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
                    'num_heads': NUM_HEADS,
                    'num_encoder_layers': NUM_ENCODER_LAYERS,
                    'num_decoder_layers': NUM_DECODER_LAYERS,
                    'feedforward_dim': FEEDFORWARD_DIM,
                    'target_steps': TARGET_STEPS,
                    'target_pair': TARGET_PAIR,
                    'target_exchange': TARGET_EXCHANGE,
                    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
                }
            }
            
            torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, 'best_smart_model.pt'))
            logger.info(f"💾 New SMART model saved! Val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{PATIENCE}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            logger.info("🛑 Early stopping triggered!")
            break
        
        logger.info("-" * 80)
    
    logger.info("🎉 H100 SMART TRAINING COMPLETED! 🎉")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

if __name__ == "__main__":
    main() 