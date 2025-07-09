#!/usr/bin/env python3
"""
Full Market Context to Binance Perpetual Prediction Model - 4 GPU Distributed

This model uses ALL available market data (240 features) as input context
to predict ONLY Binance perpetual futures (80 features) as output.

Strategy:
- Input: All 240 features (3 exchanges × 4 pairs × 20 features)  
- Context: 20 minutes (240 steps × 5 seconds) - matches paper
- Output: Only 80 Binance perp features (4 pairs × 20 features)
- Target: 2 minutes (24 steps × 5 seconds)
- Leverages cross-market arbitrage and leading indicators
- 4x H100 VPS optimized for maximum performance
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
import numpy as np
import json
import logging
from tqdm import tqdm
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# --- H100 VPS Configuration (4-GPU Optimized) ---
FINAL_DATA_DIR = "data/final_attention_240"  # Full 240-feature dataset with 20min context
MODEL_SAVE_DIR = "models/full_to_binance_perp_4gpu"
BATCH_SIZE = 8   # Per GPU - Effective batch size: 32 (8 × 4 GPUs)
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EPOCHS = 100
PATIENCE = 10
EMBED_DIM = 128  # Memory efficient
NUM_HEADS = 8    
NUM_ENCODER_LAYERS = 4  
NUM_DECODER_LAYERS = 3  
DROPOUT = 0.1
REG_WEIGHT = 0.01  # Structural regularizer weight
NUM_WORKERS = 4   # Per GPU workers

# Distributed Training Setup
def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def is_main_process():
    """Check if this is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0

# Set up logging
def setup_logging(rank):
    """Setup logging for distributed training"""
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("full_to_binance_perp_4gpu")
        return logger
    else:
        # Suppress logging for non-main processes
        logging.basicConfig(level=logging.WARNING)
        return logging.getLogger("full_to_binance_perp_4gpu")

# --- 1. Data Loading ---

class FullToBinancePerpDataset(Dataset):
    """Dataset for full market → Binance perp prediction."""
    def __init__(self, file_path, target_feature_indices):
        with np.load(file_path, allow_pickle=True) as data:
            # Context: ALL 240 features
            self.context = torch.from_numpy(data['x']).float()
            
            # Target: Only Binance perp features (80 features)
            self.target_full = torch.from_numpy(data['y']).float()
            self.target = self.target_full[:, :, target_feature_indices]
            
            self.len = self.context.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]

# --- 2. Embedding Layers ---

class CrossMarketCompoundEmbedding(nn.Module):
    """Compound embedding for all market features."""
    def __init__(self, embedding_metadata, embed_dim):
        super().__init__()
        self.metadata = embedding_metadata
        self.embed_dim = embed_dim
        
        # Feature type embeddings
        self.price_embed = nn.Embedding(1, embed_dim // 4)
        self.size_embed = nn.Embedding(1, embed_dim // 4)
        
        # Exchange embeddings
        self.exchange_embed = nn.Embedding(4, embed_dim // 4)  # 3 exchanges + padding
        
        # Trading pair embeddings
        self.pair_embed = nn.Embedding(5, embed_dim // 4)  # 4 pairs + padding

    def forward(self, num_features):
        embeddings = []
        
        for i in range(num_features):
            # Create compound embedding for each feature
            feature_embed = torch.cat([
                self.price_embed(torch.tensor(0)),
                self.size_embed(torch.tensor(0)),
                self.exchange_embed(torch.tensor(i % 3)),
                self.pair_embed(torch.tensor(i % 4))
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
        
        # Specialized embeddings for perpetual markets
        self.perp_price_embed = nn.Embedding(1, embed_dim // 2)
        self.perp_size_embed = nn.Embedding(1, embed_dim // 2)

    def forward(self, num_target_features):
        embeddings = []
        
        for i in range(num_target_features):
            # Binance perp specialized embedding
            feature_embed = torch.cat([
                self.perp_price_embed(torch.tensor(0)),
                self.perp_size_embed(torch.tensor(0))
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
        
        # Max sequence length: context_len * num_features = 240 * 240 = 57,600
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len=100000)

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
        
        # Enable gradient checkpointing for memory efficiency
        self.transformer_encoder.enable_nested_tensor = False
        self.transformer_decoder.enable_nested_tensor = False

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

        # Simpler approach: flatten feature dimension into sequence
        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        # Encoder: Process temporal sequences with all features
        # src_embedded: (batch_size, context_len, num_input_features, embed_dim)
        # Flatten features into sequence: (batch_size, context_len * num_input_features, embed_dim)
        src_flat = src_embedded.reshape(batch_size, context_len * self.num_input_features, self.embed_dim)
        src_pos = self.positional_encoding(src_flat)
        memory = self.transformer_encoder(src_pos)  # (batch_size, context_len * 240, embed_dim)
        
        # Decoder: Process temporal sequences with target features
        # tgt_embedded: (batch_size, target_len, num_target_features, embed_dim)
        # Flatten features into sequence: (batch_size, target_len * num_target_features, embed_dim)
        tgt_flat = tgt_embedded.reshape(batch_size, target_len * self.num_target_features, self.embed_dim)
        tgt_pos = self.positional_encoding(tgt_flat)
        
        # Create target mask for temporal causality
        combined_target_len = target_len * self.num_target_features
        tgt_mask = self.generate_square_subsequent_mask(combined_target_len).to(src.device)
        
        # Cross-market decoding
        transformer_out = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)

        # Project to output
        output = self.output_layer(transformer_out)  # (batch_size, target_len * num_target_features, 1)
        
        # Reshape back to target format: (batch_size, target_len, num_target_features)
        output = output.squeeze(-1)  # (batch_size, target_len * num_target_features)
        output = output.reshape(batch_size, target_len, self.num_target_features)  # (batch_size, target_len, num_target_features)
        
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

def train_epoch(model, train_loader, optimizer, mse_loss_fn, struct_loss_fn, epoch, scaler, rank):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_struct = 0.0
    
    # Only show progress bar on main process
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, (context, target) in enumerate(pbar):
        context, target = context.cuda(rank), target.cuda(rank)
        
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
        
        # Update progress bar only on main process
        if rank == 0:
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

def validate_epoch(model, val_loader, mse_loss_fn, struct_loss_fn, rank):
    """Validate for one epoch with mixed precision."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_struct = 0.0
    
    with torch.no_grad():
        for context, target in val_loader:
            context, target = context.cuda(rank), target.cuda(rank)
            
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
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_struct += struct_loss.item()
    
    return {
        'total_loss': total_loss / len(val_loader),
        'mse_loss': total_mse / len(val_loader),
        'struct_loss': total_struct / len(val_loader)
    }

def train_worker(rank, world_size):
    """Training function for each GPU process"""
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Setup device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Setup logging
    logger = setup_logging(rank)
    
    if rank == 0:
        logger.info("🚀 Full Market → Binance Perp 4-GPU Training Started!")
        logger.info(f"Input features: 240 (ALL market data)")
        logger.info(f"Output features: 80 (Binance perp only)")
        logger.info(f"Cross-market strategy: Leverage all data to predict Binance perp")
        logger.info(f"Effective batch size: {BATCH_SIZE * world_size} ({BATCH_SIZE} per GPU × {world_size} GPUs)")

    # --- Data Loading ---
    if rank == 0:
        logger.info("Loading data and metadata...")
    
    # Try multiple data directory locations
    data_dir = FINAL_DATA_DIR
    try:
        with open(os.path.join(data_dir, 'embedding_metadata.json'), 'r') as f:
            embedding_metadata = json.load(f)
    except FileNotFoundError:
        if rank == 0:
            logger.error(f"Metadata file not found in {data_dir}")
            logger.info("Trying alternative data directory...")
        try:
            data_dir = 'data/final_attention'
            with open(os.path.join(data_dir, 'embedding_metadata.json'), 'r') as f:
                embedding_metadata = json.load(f)
        except FileNotFoundError:
            if rank == 0:
                logger.error("No metadata file found. Please check data preparation.")
            cleanup_distributed()
            return

    # Get Binance perpetual feature indices
    target_feature_indices = get_binance_perp_indices(embedding_metadata)
    
    if rank == 0:
        logger.info(f"Total features: {embedding_metadata['num_features']}")
        logger.info(f"Target features (Binance perp): {len(target_feature_indices)}")

    # Load datasets
    try:
        train_dataset = FullToBinancePerpDataset(
            os.path.join(data_dir, 'train.npz'), 
            target_feature_indices
        )
        val_dataset = FullToBinancePerpDataset(
            os.path.join(data_dir, 'validation.npz'), 
            target_feature_indices
        )
    except FileNotFoundError:
        if rank == 0:
            logger.error(f"Dataset files not found in {data_dir}")
        cleanup_distributed()
        return

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=val_sampler, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    if rank == 0:
        logger.info(f"Training sequences: {len(train_dataset)}")
        logger.info(f"Validation sequences: {len(val_dataset)}")

    # --- Model Creation ---
    if rank == 0:
        logger.info("Initializing cross-market model...")
    
    model = FullToBinancePerpForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        target_len=24  # 2 minutes at 5-second intervals
    ).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Loss Functions & Optimizer ---
    mse_loss_fn = nn.MSELoss()
    struct_loss_fn = BinancePerpStructuralLoss(embedding_metadata, target_feature_indices).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Mixed precision scaler
    scaler = GradScaler()

    # --- Training Loop ---
    if rank == 0:
        logger.info(f"Starting training for {EPOCHS} epochs...")
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, mse_loss_fn, struct_loss_fn, epoch, scaler, rank)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, mse_loss_fn, struct_loss_fn, rank)
        
        # Gather metrics from all processes
        if dist.is_initialized():
            # Average validation loss across all processes
            val_loss_tensor = torch.tensor(val_metrics['total_loss']).cuda(rank)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / world_size
        else:
            avg_val_loss = val_metrics['total_loss']
        
        # Logging and checkpointing (main process only)
        if rank == 0:
            logger.info(f"Epoch {epoch}/{EPOCHS}")
            logger.info(f"Train - Loss: {train_metrics['total_loss']:.6f}, "
                       f"MSE: {train_metrics['mse_loss']:.6f}, "
                       f"Struct: {train_metrics['struct_loss']:.6f}")
            logger.info(f"Val   - Loss: {avg_val_loss:.6f}, "
                       f"MSE: {val_metrics['mse_loss']:.6f}, "
                       f"Struct: {val_metrics['struct_loss']:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'best_val_loss': best_val_loss,
                    'target_feature_indices': target_feature_indices
                }
                
                torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, 'best_model_4gpu.pt'))
                logger.info(f"💾 New best model saved! Val loss: {avg_val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"Patience: {patience_counter}/{PATIENCE}")
            
            # Early stopping
            if patience_counter >= PATIENCE:
                logger.info("🛑 Early stopping triggered!")
                break
            
            logger.info("-" * 80)
    
    # Cleanup
    cleanup_distributed()
    
    if rank == 0:
        logger.info("🎉 4-GPU Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")

def main():
    """Main function to launch distributed training"""
    world_size = 4  # Number of H100 GPUs
    
    # Check if we have enough GPUs
    if torch.cuda.device_count() < world_size:
        print(f"Error: Need {world_size} GPUs, but only {torch.cuda.device_count()} available")
        return
    
    print(f"🚀 Launching 4-GPU distributed training on {world_size} H100s...")
    print(f"Effective batch size: {BATCH_SIZE * world_size}")
    
    # Set environment variables for distributed training
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Launch distributed training
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main() 