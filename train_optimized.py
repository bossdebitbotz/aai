print("--- SCRIPT EXECUTION STARTED ---")
print("--- SCRIPT EXECUTION STARTED ---")
#!/usr/bin/env python3
"""
High-Performance, Distributed LOB Forecasting Model Training

Optimized for multi-GPU (NVIDIA H100) environments using PyTorch DDP.

Key Optimizations:
1. Distributed Data Parallel (DDP) for multi-GPU training.
2. Automatic Mixed Precision (AMP) for speed and memory efficiency.
3. Optimized DataLoader with DistributedSampler.
4. Flexible command-line arguments for easy configuration.
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
import pickle
import argparse

# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- DDP Setup ---
def setup_ddp():
    """Initializes the distributed process group."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def is_main_process():
    """Checks if the current process is the main one (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

# --- Configuration (now with args) ---
def parse_args():
    parser = argparse.ArgumentParser(description="High-Performance LOB Forecasting Training")
    parser.add_argument("--data_dir", type=str, default="data/final_attention", help="Path to processed data")
    parser.add_argument("--model_dir", type=str, default="models/attention_lob_optimized", help="Directory to save models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size PER GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--workers", type=int, default=16, help="Number of DataLoader workers")
    parser.add_argument("--target_pair", type=str, default="BTC-USDT", help="Target pair for training")
    parser.add_argument("--target_minutes", type=int, default=2, help="Prediction horizon in minutes")
    # Model params
    parser.add_argument("--embed_dim", type=int, default=126)
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=3)
    return parser.parse_args()

# --- Data Loading ---
class LOBDataset(Dataset):
    """PyTorch Dataset for loading LOB sequences with lazy loading, safe for multiprocessing."""
    def __init__(self, file_path):
        self.file_path = file_path
        # Reading metadata (like length) is fine, as it's small and pickle-able.
        with np.load(file_path, mmap_mode='r') as data:
            self.len = data['x'].shape[0]
        # This will be initialized lazily in each worker process.
        self.data = None

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # If self.data is not initialized, it means we are in a new worker process.
        # Each worker will create its own file handle to the memory-mapped file.
        if self.data is None:
            self.data = np.load(self.file_path, mmap_mode='r')
            
        x = self.data['x'][idx]
        y = self.data['y'][idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


# --- Model and Layers ---
# (CompoundMultivariateEmbedding, PositionalEncoding, LOBForecaster, StructuralLoss remain the same)
# ... [Paste all the model classes from the previous script here] ...
# NOTE: To save space, I will omit pasting the identical classes. The new script will contain them.
class CompoundMultivariateEmbedding(nn.Module):
    def __init__(self, embedding_metadata, embed_dim, device):
        super().__init__()
        self.embed_dim = embed_dim
        self.metadata = embedding_metadata
        self.device = device
        unique_attrs = embedding_metadata['unique_attributes']
        attr_embed_dim = embed_dim // 5
        remaining_dim = embed_dim - (attr_embed_dim * 4)
        self.level_embedding = nn.Embedding(len(unique_attrs['levels']), attr_embed_dim)
        self.type_embedding = nn.Embedding(len(unique_attrs['order_types']), attr_embed_dim)
        self.feature_embedding = nn.Embedding(len(unique_attrs['features']), attr_embed_dim)
        self.exchange_embedding = nn.Embedding(len(unique_attrs['exchanges']), attr_embed_dim)
        self.pair_embedding = nn.Embedding(len(unique_attrs['trading_pairs']), remaining_dim)
        self.projection = nn.Linear(attr_embed_dim * 4 + remaining_dim, embed_dim)
        self.level_to_idx = {level: i for i, level in enumerate(unique_attrs['levels'])}
        self.type_to_idx = {otype: i for i, otype in enumerate(unique_attrs['order_types'])}
        self.feature_to_idx = {feat: i for i, feat in enumerate(unique_attrs['features'])}
        self.exchange_to_idx = {exch: i for i, exch in enumerate(unique_attrs['exchanges'])}
        self.pair_to_idx = {pair: i for i, pair in enumerate(unique_attrs['trading_pairs'])}
    def forward(self, num_features):
        embeddings = []
        for i in range(num_features):
            col_name = self.metadata['columns'][i]
            col_info = self.metadata['column_mapping'][col_name]
            level_idx = self.level_to_idx[col_info['level']]
            type_idx = self.type_to_idx[col_info['order_type']]
            feature_idx = self.feature_to_idx[col_info['feature_type']]
            exchange_idx = self.exchange_to_idx[col_info['exchange']]
            pair_idx = self.pair_to_idx[col_info['trading_pair']]
            level_emb = self.level_embedding(torch.tensor(level_idx, device=self.device))
            type_emb = self.type_embedding(torch.tensor(type_idx, device=self.device))
            feature_emb = self.feature_embedding(torch.tensor(feature_idx, device=self.device))
            exchange_emb = self.exchange_embedding(torch.tensor(exchange_idx, device=self.device))
            pair_emb = self.pair_embedding(torch.tensor(pair_idx, device=self.device))
            combined_emb = torch.cat([level_emb, type_emb, feature_emb, exchange_emb, pair_emb], dim=0)
            embeddings.append(combined_emb)
        stacked_embeddings = torch.stack(embeddings)
        projected_embeddings = self.projection(stacked_embeddings)
        return projected_embeddings

class PositionalEncoding(nn.Module):
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
        # We clone the positional encoding tensor to avoid a subtle DDP synchronization issue
        # where views of a buffer can cause conflicts.
        x = x + self.pe[:x.size(0), :].clone()
        return self.dropout(x)

class LOBForecaster(nn.Module):
    def __init__(self, embedding_metadata, embed_dim, num_heads, num_layers, dropout, target_len, device):
        super().__init__()
        self.num_features = len(embedding_metadata['columns'])
        self.target_len = target_len
        self.embed_dim = embed_dim
        self.device = device
        self.value_projection = nn.Linear(1, embed_dim)
        self.compound_embedding = CompoundMultivariateEmbedding(embedding_metadata, embed_dim, device)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, src, tgt):
        feature_embeds = self.compound_embedding(self.num_features)
        src_proj = self.value_projection(src.unsqueeze(-1))
        tgt_proj = self.value_projection(tgt.unsqueeze(-1))
        src_embedded = src_proj + feature_embeds.unsqueeze(0).unsqueeze(0)
        tgt_embedded = tgt_proj + feature_embeds.unsqueeze(0).unsqueeze(0)
        batch_size, context_len, _, _ = src_embedded.shape
        target_len = tgt.shape[1]
        src_flat = src_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, context_len, self.embed_dim)
        tgt_flat = tgt_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, target_len, self.embed_dim)
        src_pos = self.positional_encoding(src_flat.permute(1, 0, 2)).permute(1, 0, 2)
        tgt_pos = self.positional_encoding(tgt_flat.permute(1, 0, 2)).permute(1, 0, 2)
        tgt_mask = self.transformer.generate_square_subsequent_mask(target_len).to(self.device)
        transformer_out = self.transformer(src_pos, tgt_pos, tgt_mask=tgt_mask)
        output = self.output_layer(transformer_out)
        output = output.reshape(batch_size, self.num_features, target_len).permute(0, 2, 1)
        return output

class StructuralLoss(nn.Module):
    def __init__(self, embedding_metadata):
        super().__init__()
        self.metadata = embedding_metadata
        self.columns = embedding_metadata['columns']
        self.price_indices = {}
        for i, col in enumerate(self.columns):
            col_info = self.metadata['column_mapping'][col]
            key = (col_info['exchange'], col_info['trading_pair'], col_info['order_type'])
            if key not in self.price_indices: self.price_indices[key] = {}
            if col_info['feature_type'] == 'price': self.price_indices[key][col_info['level']] = i
    def forward(self, predictions):
        total_loss = 0.0
        for key, levels in self.price_indices.items():
            max_level = max(levels.keys()) if levels else 0
            if max_level < 2: continue
            for t in range(predictions.shape[1]):
                for level in range(1, max_level):
                    if 'ask' in key[-1] and level in levels and (level + 1) in levels:
                        total_loss += torch.relu(predictions[:, t, levels[level]] - predictions[:, t, levels[level+1]]).mean()
                    if 'bid' in key[-1] and level in levels and (level + 1) in levels:
                        total_loss += torch.relu(predictions[:, t, levels[level+1]] - predictions[:, t, levels[level]]).mean()
        return total_loss

# --- Training Loop ---
def main():
    args = parse_args()
    
    # --- Environment and Device Setup ---
    is_ddp = "RANK" in os.environ
    if is_ddp:
        setup_ddp()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Use a variable for the rank for logging, defaulting to 0 if not DDP
    log_rank = int(os.environ.get("RANK", 0))
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - RANK {log_rank} - %(levelname)s - %(message)s')
    logger = logging.getLogger("optimized_training")

    if is_main_process():
        logger.info(f"Starting Training")
        logger.info(f"DDP Enabled: {is_ddp}")
        logger.info(f"Using device: {device}")
        logger.info(f"Args: {args}")

    # --- Data Loading ---
    if is_main_process():
        logger.info("Loading metadata...")
    with open(os.path.join(args.data_dir, 'embedding_metadata.json'), 'r') as f:
        embedding_metadata = json.load(f)

    train_dataset = LOBDataset(os.path.join(args.data_dir, 'train.npz'))
    val_dataset = LOBDataset(os.path.join(args.data_dir, 'validation.npz'))
    
    train_sampler = DistributedSampler(train_dataset) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=(train_sampler is None) # Shuffle only if not DDP
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True
    )

    # --- Model, Loss, Optimizer ---
    model = LOBForecaster(
        embedding_metadata=embedding_metadata,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.1,
        target_len=args.target_minutes * 12, # 5s intervals
        device=device
    ).to(device)
    
    # Wrap model with DDP if applicable
    if is_ddp:
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank])
        else: # CPU DDP with 'gloo' backend
            model = DDP(model)

    mse_loss_fn = nn.MSELoss()
    struct_loss_fn = StructuralLoss(embedding_metadata).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # AMP scaler only if on CUDA
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if is_main_process():
        os.makedirs(args.model_dir, exist_ok=True)
        logger.info(f"Model will be saved to {args.model_dir}")
        raw_model = model.module if is_ddp else model
        logger.info(f"Total parameters: {sum(p.numel() for p in raw_model.parameters()):,}")

    # --- Main Training Loop ---
    for epoch in range(args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main_process())
        for context, target in pbar:
            context, target = context.to(device), target.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                predictions = model(context, target)
                mse_loss = mse_loss_fn(predictions, target)
                struct_loss = struct_loss_fn(predictions)
                loss = mse_loss + 0.01 * struct_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if is_main_process():
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "MSE": f"{mse_loss.item():.4f}"})

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for context, target in val_loader:
                context, target = context.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    predictions = model(context, target)
                    total_val_loss += mse_loss_fn(predictions, target).item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches

        if is_ddp:
            # Collect loss from all processes
            val_loss_tensor = torch.tensor(avg_val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / dist.get_world_size()

        if is_main_process():
            logger.info(f"Epoch {epoch} | Validation MSE: {avg_val_loss:.6f}")
            # Save checkpoint
            save_model = model.module if is_ddp else model
            torch.save(save_model.state_dict(), os.path.join(args.model_dir, f"epoch_{epoch}.pth"))

    if is_ddp:
        cleanup_ddp()

if __name__ == "__main__":
    main() 