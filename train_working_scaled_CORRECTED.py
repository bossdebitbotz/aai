#!/usr/bin/env python3
"""
CORRECTED: Scaled Multi-GPU Training for WLD 1-Minute Model
✅ Properly calibrated for data structure: target_len=24 (not 12)
✅ Predicts 20 WLD binance_perp features
✅ Uses 24 time steps ahead prediction as per data preparation

CRITICAL FIXES:
- target_len=24 (matches data target_length)
- Proper WLD feature filtering (indices 120-139)
- Correct model configuration saved for backtesting
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
MODEL_SAVE_DIR = "models/working_scaled_h100_smart"  # H100 smart mode directory
BATCH_SIZE = 64   # 🎯 SMART: 2x larger, won't OOM
GRADIENT_ACCUMULATION_STEPS = 2  # 🎯 EFFECTIVE BATCH = 128
LEARNING_RATE = 1.5e-4  # 🎯 SCALED: Moderate increase
WARMUP_STEPS = 1500     # 🎯 BALANCED: Stable training
EPOCHS = 50
PATIENCE = 8
EMBED_DIM = 384       # 🎯 SWEET SPOT: 1.5x larger (not 2x)
NUM_HEADS = 12        # 🎯 BALANCED: 1.5x more heads
NUM_ENCODER_LAYERS = 7  # 🎯 SMART: +1 layer (not +2)
NUM_DECODER_LAYERS = 5  # 🎯 SMART: +1 layer (not +2)
DROPOUT = 0.1
TARGET_LEN = 24  # ✅ CORRECTED: Must match data target_length (was 12)
NUM_WORKERS = 24      # 🎯 BALANCED: More workers, not crazy
COMPILE_MODEL = False  # 🚨 DISABLE: torch.compile adds memory overhead

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("corrected_scaled_training")

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
logger.info(f"Using device: {DEVICE}")
logger.info(f"Available GPUs: {num_gpus}")

class WorkingScaledDataset(Dataset):
    """Dataset properly configured for 24 time step targets with TRADING-FOCUSED preprocessing."""
    def __init__(self, file_path, target_feature_indices):
        with np.load(file_path, allow_pickle=True) as data:
            self.context = torch.from_numpy(data['contexts']).float()
            self.target_full = torch.from_numpy(data['targets']).float()
            
            # Get raw targets (absolute values)
            raw_targets = self.target_full[:, :TARGET_LEN, target_feature_indices]
            
            # 🚨 CRITICAL FIX: Convert to PRICE CHANGES for trading prediction
            # Use last context value as baseline for relative changes
            last_context_values = self.context[:, -1, target_feature_indices].unsqueeze(1)  # (N, 1, 20)
            
            # Calculate relative changes: (target - baseline) / baseline
            # This teaches the model to predict DIRECTION and MAGNITUDE of changes
            self.target = (raw_targets - last_context_values) / (last_context_values + 1e-8)
            
            # Clip extreme outliers to stabilize training
            self.target = torch.clamp(self.target, -0.1, 0.1)  # ±10% max change
            
            self.len = self.context.shape[0]
            
        logger.info(f"Dataset loaded: {self.len} sequences")
        logger.info(f"Context shape: {self.context.shape}")
        logger.info(f"Target shape (RELATIVE CHANGES): {self.target.shape}")
        logger.info(f"Target range: [{self.target.min():.6f}, {self.target.max():.6f}]")
        logger.info(f"Target std: {self.target.std():.6f}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]

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
                 num_encoder_layers, num_decoder_layers, dropout, target_len):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_input_features = embedding_metadata['num_features']
        self.target_feature_indices = target_feature_indices
        self.num_target_features = len(target_feature_indices)
        self.target_len = target_len
        self.embed_dim = embed_dim

        logger.info(f"Model Architecture:")
        logger.info(f"  Input features: {self.num_input_features}")
        logger.info(f"  Target features: {self.num_target_features}")
        logger.info(f"  Target length: {self.target_len}")
        logger.info(f"  Embed dim: {embed_dim}")

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
    
    logger.info(f"WLD Perp Features Found: {len(target_indices)}")
    logger.info(f"Indices: {target_indices[0]}-{target_indices[-1]} ({target_indices})")
    
    # Validate we have exactly 20 features
    if len(target_indices) != 20:
        raise ValueError(f"Expected 20 WLD perp features, found {len(target_indices)}")
    
    return target_indices

def validate_configuration(embedding_metadata, target_indices):
    """✅ Validate all configuration matches requirements."""
    logger.info("🔍 Validating Configuration...")
    
    # Check metadata
    assert embedding_metadata['num_features'] == 240, f"Expected 240 features, got {embedding_metadata['num_features']}"
    assert embedding_metadata['context_length'] == 120, f"Expected context 120, got {embedding_metadata['context_length']}"
    assert embedding_metadata['target_length'] == 24, f"Expected target 24, got {embedding_metadata['target_length']}"
    
    # Check target indices
    assert len(target_indices) == 20, f"Expected 20 WLD features, got {len(target_indices)}"
    assert target_indices == list(range(120, 140)), f"Expected indices 120-139, got {target_indices[0]}-{target_indices[-1]}"
    
    # Check TARGET_LEN matches data
    assert TARGET_LEN == embedding_metadata['target_length'], f"TARGET_LEN {TARGET_LEN} != data target_length {embedding_metadata['target_length']}"
    
    logger.info("✅ All configuration validated!")

def main():
    """Main training function with corrected configuration."""
    logger.info("🚀 CORRECTED Scaled Multi-GPU Training Started!")
    logger.info(f"✅ Target length: {TARGET_LEN} (corrected from 12)")
    
    # --- Data Loading ---
    logger.info("Loading data and metadata...")
    
    try:
        with open(os.path.join(FINAL_DATA_DIR, 'embedding_metadata.json'), 'r') as f:
            embedding_metadata = json.load(f)
    except FileNotFoundError:
        logger.error(f"Metadata file not found in {FINAL_DATA_DIR}")
        return

    target_feature_indices = get_wld_perp_indices(embedding_metadata)
    validate_configuration(embedding_metadata, target_feature_indices)

    # Load datasets
    try:
        train_dataset = WorkingScaledDataset(
            os.path.join(FINAL_DATA_DIR, 'train.npz'), 
            target_feature_indices
        )
        val_dataset = WorkingScaledDataset(
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
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,  # 🚀 Same large batch for validation
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True
    )

    logger.info(f"Training sequences: {len(train_dataset)}")
    logger.info(f"Validation sequences: {len(val_dataset)}")

    # --- Model Creation ---
    logger.info("Initializing CORRECTED model...")
    
    model = CorrectedMultiGPUForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        target_len=TARGET_LEN  # ✅ CORRECTED: 24 time steps
    )

    # Apply DataParallel for multi-GPU training
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Model wrapped with DataParallel for {num_gpus} GPUs")
    
    model = model.to(DEVICE)

    # 🚨 DISABLED: torch.compile to save memory (was causing OOM)
    if COMPILE_MODEL and hasattr(torch, 'compile'):
        logger.info("🚀 Compiling model for H100 optimization...")
        model = torch.compile(model, mode='max-autotune')
        logger.info("✅ Model compiled successfully")
    else:
        logger.info("🚨 torch.compile disabled to save memory")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"🎯 H100 Smart Mode: {EMBED_DIM}D model, batch {BATCH_SIZE}, effective batch {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

    # --- TRADING-FOCUSED Loss Functions & Optimizer ---
    mse_loss_fn = nn.MSELoss()
    
    def trading_loss_fn(predictions, targets):
        """Trading-focused loss that emphasizes directional accuracy."""
        # Standard MSE for magnitude
        mse_loss = mse_loss_fn(predictions, targets)
        
        # Directional loss: penalize when predicted and actual directions disagree
        pred_directions = torch.sign(predictions)
        target_directions = torch.sign(targets)
        
        # Directional accuracy bonus/penalty
        direction_match = (pred_directions == target_directions).float()
        direction_loss = 1.0 - direction_match.mean()
        
        # Combined loss: 70% direction, 30% magnitude
        total_loss = 0.7 * direction_loss + 0.3 * mse_loss
        
        return total_loss, mse_loss, direction_loss
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()

    # --- H100 SMART MODE Training Loop ---
    logger.info(f"Starting H100 SMART MODE training for {EPOCHS} epochs...")
    logger.info(f"🎯 Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Training with gradient accumulation
        model.train()
        total_loss = 0.0
        accumulation_loss = 0.0
        
        with tqdm(train_loader, desc=f"H100 Smart Epoch {epoch}") as pbar:
            for batch_idx, (context, target) in enumerate(pbar):
                context, target = context.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                
                # Create decoder input (shifted target)
                decoder_input = torch.zeros_like(target)
                decoder_input[:, 1:] = target[:, :-1]
                
                with autocast('cuda'):
                    predictions = model(context, decoder_input)
                    total_loss_val, mse_loss_val, direction_loss_val = trading_loss_fn(predictions, target)
                    # Scale loss for gradient accumulation
                    total_loss_val = total_loss_val / GRADIENT_ACCUMULATION_STEPS
                
                # Accumulate gradients
                scaler.scale(total_loss_val).backward()
                accumulation_loss += total_loss_val.item()
                
                # Update every GRADIENT_ACCUMULATION_STEPS batches
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    total_loss += accumulation_loss
                    accumulation_loss = 0.0
                
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Total': f'{total_loss_val.item() * GRADIENT_ACCUMULATION_STEPS:.4f}',
                    'MSE': f'{mse_loss_val.item():.6f}',
                    'Dir': f'{direction_loss_val.item():.4f}',
                    'LR': f'{current_lr:.2e}',
                    'EffBatch': BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
                })
        
        # 🎯 H100 Smart: Handle any remaining accumulated gradients
        if accumulation_loss > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += accumulation_loss
        
        train_loss = total_loss / max(1, len(train_loader) // GRADIENT_ACCUMULATION_STEPS)
        
        # Validation with TRADING-FOCUSED metrics
        model.eval()
        total_val_loss = 0.0
        total_val_mse = 0.0
        total_val_direction = 0.0
        
        with torch.no_grad():
            for context, target in val_loader:
                context, target = context.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                
                decoder_input = torch.zeros_like(target)
                decoder_input[:, 1:] = target[:, :-1]
                
                with autocast('cuda'):
                    predictions = model(context, decoder_input)
                    val_total_loss, val_mse_loss, val_direction_loss = trading_loss_fn(predictions, target)
                
                total_val_loss += val_total_loss.item()
                total_val_mse += val_mse_loss.item()
                total_val_direction += val_direction_loss.item()
        
        val_loss = total_val_loss / len(val_loader)
        val_mse = total_val_mse / len(val_loader)
        val_direction = total_val_direction / len(val_loader)
        
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"Train Loss: {train_loss:.6f}")
        logger.info(f"Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f}, Dir: {val_direction:.4f})")
        logger.info(f"Val Directional Accuracy: {(1.0 - val_direction)*100:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            model_to_save = model.module if hasattr(model, 'module') else model
            
            # ✅ CORRECTED: Save complete configuration for backtesting
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
                    'target_steps': TARGET_LEN,  # ✅ CORRECTED: 24
                    'dropout': DROPOUT,
                    'num_input_features': embedding_metadata['num_features'],
                    'num_target_features': len(target_feature_indices)
                }
            }
            
            torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, 'best_model_corrected.pt'))
            logger.info(f"💾 New best model saved! Val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            logger.info("🛑 Early stopping triggered!")
            break
        
        logger.info("-" * 80)
    
    logger.info("🎉 CORRECTED Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to: {MODEL_SAVE_DIR}/best_model_corrected.pt")

if __name__ == "__main__":
    main() 