#!/usr/bin/env python3
"""
WLD Model V2: Trading-Focused Transformer for H100s
====================================================

✅ LESSONS LEARNED APPLIED:
- Predicts RELATIVE PRICE CHANGES (not absolute values)
- Trading-focused loss: 70% directional + 30% magnitude
- Smart H100 utilization: 64 batch, 384D, 7/5 layers
- Gradient accumulation for effective batch 128
- Real-time directional accuracy monitoring

✅ TARGET: 60%+ directional accuracy for WLD perp trading
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

# === V2 CONFIGURATION: SMART H100 MODE ===
FINAL_DATA_DIR = "data/final_attention"
MODEL_SAVE_DIR = "models/wld_v2_trading_focused"
BATCH_SIZE = 64  # Smart: 2x original, won't OOM
GRADIENT_ACCUMULATION = 2  # Effective batch = 128
LEARNING_RATE = 1.5e-4
WARMUP_STEPS = 1500
EPOCHS = 50
PATIENCE = 10
EMBED_DIM = 384  # Sweet spot: 1.5x original
NUM_HEADS = 12
NUM_ENCODER_LAYERS = 7  # +1 from original
NUM_DECODER_LAYERS = 5  # +1 from original
DROPOUT = 0.1
TARGET_LEN = 24  # Correct: 24 seconds ahead
NUM_WORKERS = 16

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WLD_Model_V2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
logger.info(f"🚀 WLD Model V2 - Trading Focused")
logger.info(f"Device: {DEVICE}, GPUs: {num_gpus}")

class TradingDatasetV2(Dataset):
    """V2 Dataset: Converts to RELATIVE PRICE CHANGES for trading signals."""
    
    def __init__(self, file_path, target_indices):
        with np.load(file_path, allow_pickle=True) as data:
            self.context = torch.from_numpy(data['contexts']).float()
            target_full = torch.from_numpy(data['targets']).float()
            
            # Get raw targets (absolute values) 
            raw_targets = target_full[:, :TARGET_LEN, target_indices]
            
            # 🚨 KEY FIX: Convert to RELATIVE CHANGES for trading
            last_context = self.context[:, -1, target_indices].unsqueeze(1)  # (N, 1, 20)
            
            # Calculate relative changes: (future - current) / current  
            self.targets = (raw_targets - last_context) / (last_context + 1e-8)
            
            # Clip extreme outliers for stability
            self.targets = torch.clamp(self.targets, -0.05, 0.05)  # ±5% max change
            
        logger.info(f"Dataset: {len(self)} sequences")
        logger.info(f"Context: {self.context.shape}")
        logger.info(f"Targets (RELATIVE): {self.targets.shape}")
        logger.info(f"Target range: [{self.targets.min():.6f}, {self.targets.max():.6f}]")
        logger.info(f"Target std: {self.targets.std():.6f}")
    
    def __len__(self):
        return len(self.context)
    
    def __getitem__(self, idx):
        return self.context[idx], self.targets[idx]

class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformer."""
    
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :].clone()
        return self.dropout(x)

class WLDModelV2(nn.Module):
    """V2 Model: Optimized for trading signal prediction."""
    
    def __init__(self, num_input_features=240, num_target_features=20, 
                 embed_dim=384, num_heads=12, num_encoder_layers=7, 
                 num_decoder_layers=5, dropout=0.1, target_len=24):
        super().__init__()
        
        self.num_input_features = num_input_features
        self.num_target_features = num_target_features
        self.target_len = target_len
        self.embed_dim = embed_dim
        
        logger.info(f"🏗️  Model V2 Architecture:")
        logger.info(f"   Input features: {num_input_features}")
        logger.info(f"   Target features: {num_target_features}")
        logger.info(f"   Target length: {target_len}")
        logger.info(f"   Embed dim: {embed_dim}")
        logger.info(f"   Encoder layers: {num_encoder_layers}")
        logger.info(f"   Decoder layers: {num_decoder_layers}")
        
        # Embeddings
        self.value_projection = nn.Linear(1, embed_dim)
        self.input_feature_embed = nn.Embedding(num_input_features, embed_dim)
        self.target_feature_embed = nn.Embedding(num_target_features, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output head optimized for trading signals
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1)
        )
    
    def forward(self, src, tgt):
        batch_size, seq_len, _ = src.shape
        
        # Process source (context)
        src_proj = self.value_projection(src.unsqueeze(-1))  # (B, seq_len, features, embed_dim)
        
        # Add feature embeddings
        input_embeds = self.input_feature_embed(torch.arange(self.num_input_features, device=src.device))
        src_embedded = src_proj + input_embeds.unsqueeze(0).unsqueeze(0)
        
        # Flatten for transformer: (B, seq_len * features, embed_dim)
        src_flat = src_embedded.reshape(batch_size, seq_len * self.num_input_features, self.embed_dim)
        src_pos = self.pos_encoding(src_flat)
        memory = self.encoder(src_pos)
        
        # Process target
        tgt_proj = self.value_projection(tgt.unsqueeze(-1))
        target_embeds = self.target_feature_embed(torch.arange(self.num_target_features, device=tgt.device))
        tgt_embedded = tgt_proj + target_embeds.unsqueeze(0).unsqueeze(0)
        
        tgt_flat = tgt_embedded.reshape(batch_size, self.target_len * self.num_target_features, self.embed_dim)
        tgt_pos = self.pos_encoding(tgt_flat)
        
        # Causal mask for decoder
        tgt_len = self.target_len * self.num_target_features
        tgt_mask = self._generate_square_mask(tgt_len).to(src.device)
        
        # Decode
        decoder_out = self.decoder(tgt_pos, memory, tgt_mask=tgt_mask)
        
        # Output projection
        output = self.output_head(decoder_out).squeeze(-1)
        output = output.reshape(batch_size, self.target_len, self.num_target_features)
        
        return output
    
    def _generate_square_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def get_wld_indices(metadata):
    """Get WLD binance_perp feature indices."""
    indices = []
    for i, col_name in enumerate(metadata['columns']):
        col_info = metadata['column_mapping'][col_name]
        if (col_info['exchange'] == 'binance_perp' and 
            col_info['trading_pair'] == 'WLD-USDT'):
            indices.append(i)
    
    logger.info(f"🎯 WLD perp features: {len(indices)} (indices {indices[0]}-{indices[-1]})")
    return indices

def trading_loss_fn(predictions, targets):
    """Trading-focused loss: 70% directional accuracy + 30% magnitude."""
    # Standard MSE for magnitude
    mse_loss = nn.MSELoss()(predictions, targets)
    
    # Directional accuracy loss
    pred_directions = torch.sign(predictions)
    target_directions = torch.sign(targets)
    direction_matches = (pred_directions == target_directions).float()
    directional_accuracy = direction_matches.mean()
    direction_loss = 1.0 - directional_accuracy
    
    # Combined loss: prioritize direction
    total_loss = 0.7 * direction_loss + 0.3 * mse_loss
    
    return total_loss, mse_loss, direction_loss, directional_accuracy

def train_epoch(model, train_loader, optimizer, scaler, epoch):
    """Train one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_direction_acc = 0.0
    accumulation_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"V2 Epoch {epoch}")
    
    for batch_idx, (context, target) in enumerate(pbar):
        context = context.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)
        
        # Create decoder input (shifted)
        decoder_input = torch.zeros_like(target)
        decoder_input[:, 1:] = target[:, :-1]
        
        with autocast('cuda'):
            predictions = model(context, decoder_input)
            loss, mse_loss, direction_loss, direction_acc = trading_loss_fn(predictions, target)
            loss = loss / GRADIENT_ACCUMULATION  # Scale for accumulation
        
        # Accumulate gradients
        scaler.scale(loss).backward()
        accumulation_loss += loss.item()
        
        # Update every GRADIENT_ACCUMULATION steps
        if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += accumulation_loss
            total_mse += mse_loss.item()
            total_direction_acc += direction_acc.item()
            accumulation_loss = 0.0
        
        pbar.set_postfix({
            'Loss': f'{loss.item() * GRADIENT_ACCUMULATION:.4f}',
            'MSE': f'{mse_loss.item():.6f}',
            'DirAcc': f'{direction_acc.item():.3f}',
            'EffBatch': BATCH_SIZE * GRADIENT_ACCUMULATION
        })
    
    # Handle remaining gradients
    if accumulation_loss > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += accumulation_loss
    
    num_updates = max(1, len(train_loader) // GRADIENT_ACCUMULATION)
    return total_loss / num_updates, total_mse / num_updates, total_direction_acc / num_updates

def validate_epoch(model, val_loader):
    """Validate one epoch."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_direction_acc = 0.0
    
    with torch.no_grad():
        for context, target in val_loader:
            context = context.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            
            decoder_input = torch.zeros_like(target)
            decoder_input[:, 1:] = target[:, :-1]
            
            with autocast('cuda'):
                predictions = model(context, decoder_input)
                loss, mse_loss, direction_loss, direction_acc = trading_loss_fn(predictions, target)
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_direction_acc += direction_acc.item()
    
    num_batches = len(val_loader)
    return total_loss / num_batches, total_mse / num_batches, total_direction_acc / num_batches

def main():
    """Main training function for WLD Model V2."""
    logger.info("=" * 80)
    logger.info("🚀 WLD MODEL V2: TRADING-FOCUSED TRAINING")
    logger.info("=" * 80)
    
    # Load metadata and data
    logger.info("📚 Loading data...")
    with open(os.path.join(FINAL_DATA_DIR, 'embedding_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    wld_indices = get_wld_indices(metadata)
    
    # Create datasets
    train_dataset = TradingDatasetV2(os.path.join(FINAL_DATA_DIR, 'train.npz'), wld_indices)
    val_dataset = TradingDatasetV2(os.path.join(FINAL_DATA_DIR, 'validation.npz'), wld_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    
    # Create model
    logger.info("🏗️  Building model...")
    model = WLDModelV2(
        num_input_features=240,
        num_target_features=20,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        target_len=TARGET_LEN
    )
    
    # Multi-GPU setup
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"🔄 DataParallel: {num_gpus} GPUs")
    
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()
    
    # Training loop
    logger.info("🎯 Starting training...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_mse, train_dir_acc = train_epoch(model, train_loader, optimizer, scaler, epoch)
        
        # Validate  
        val_loss, val_mse, val_dir_acc = validate_epoch(model, val_loader)
        
        # Log results
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"Train - Loss: {train_loss:.4f}, MSE: {train_mse:.6f}, DirAcc: {train_dir_acc:.3f} ({train_dir_acc*100:.1f}%)")
        logger.info(f"Val   - Loss: {val_loss:.4f}, MSE: {val_mse:.6f}, DirAcc: {val_dir_acc:.3f} ({val_dir_acc*100:.1f}%)")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dir_acc': train_dir_acc,
                'val_dir_acc': val_dir_acc,
                'wld_indices': wld_indices,
                'model_config': {
                    'embed_dim': EMBED_DIM,
                    'num_heads': NUM_HEADS,
                    'num_encoder_layers': NUM_ENCODER_LAYERS,
                    'num_decoder_layers': NUM_DECODER_LAYERS,
                    'target_len': TARGET_LEN,
                    'dropout': DROPOUT,
                    'num_input_features': 240,
                    'num_target_features': 20
                }
            }
            
            torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, 'best_model_v2.pt'))
            logger.info(f"💾 New best model saved! Val DirAcc: {val_dir_acc*100:.1f}%")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            logger.info("🛑 Early stopping!")
            break
        
        logger.info("-" * 60)
    
    logger.info("🎉 Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved: {MODEL_SAVE_DIR}/best_model_v2.pt")

if __name__ == "__main__":
    main() 