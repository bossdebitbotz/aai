#!/usr/bin/env python3
"""
Optimized Attention-Based LOB Forecasting - Maximum Hardware Utilization

This script optimizes the paper implementation for maximum hardware usage:
- Automatic batch size optimization based on GPU memory
- Multi-GPU distributed training support
- Optimized data loading with all CPU cores
- Memory-efficient training
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import json
import pickle
import math
import logging
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add spacetimeformer to path
sys.path.insert(0, os.path.join(os.getcwd(), 'spacetimeformer'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimized Configuration
def get_optimal_config():
    """Get optimized configuration based on available hardware."""
    config = {
        'data_path': 'data/final_attention',
        'model_save_dir': 'models/paper_optimized',
        'learning_rate': 5e-4,
        'init_lr': 1e-10,
        'warmup_steps': 1000,
        'decay_factor': 0.8,
        'epochs': 50,
        'patience': 10,
        'embed_dim': 126,
        'num_heads': 3,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'd_ff': 512,
        'dropout': 0.1,
        'structural_weight': 0.01,
    }
    
    # Optimize based on hardware
    num_gpus = torch.cuda.device_count()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    num_cpus = os.cpu_count()
    
    # Optimize batch size based on GPU memory
    if gpu_memory > 20:  # > 20GB
        config['batch_size'] = 16
    elif gpu_memory > 10:  # > 10GB  
        config['batch_size'] = 8
    elif gpu_memory > 6:   # > 6GB
        config['batch_size'] = 6
    else:
        config['batch_size'] = 4
    
    # Optimize data loading workers
    config['num_workers'] = min(num_cpus, 8)  # Cap at 8 to avoid overhead
    
    # Multi-GPU settings
    config['num_gpus'] = num_gpus
    config['distributed'] = num_gpus > 1
    
    logger.info(f"Hardware detected: {num_gpus} GPUs, {gpu_memory:.1f}GB GPU memory, {num_cpus} CPU cores")
    logger.info(f"Optimized settings: batch_size={config['batch_size']}, num_workers={config['num_workers']}")
    
    return config

CONFIG = get_optimal_config()

# --- Copy all the model classes from the original script ---

class LOBDataset(Dataset):
    """Dataset for LOB sequences with memory-efficient loading."""
    def __init__(self, file_path):
        self.file_path = file_path
        with np.load(file_path) as data:
            self.len = data['x'].shape[0]
            self.x_shape = data['x'].shape
            self.y_shape = data['y'].shape

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with np.load(self.file_path) as data:
            x = data['x'][idx]
            y = data['y'][idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

class Time2Vec(nn.Module):
    """Time2Vec embedding layer as described in the paper."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.w = nn.parameter.Parameter(torch.randn(1, embed_dim-1))
        self.b = nn.parameter.Parameter(torch.randn(1, embed_dim-1))
        self.f = torch.sin

    def forward(self, tau):
        tau = tau.unsqueeze(-1)
        v1 = self.f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], -1)

class CompoundMultivariateEmbedding(nn.Module):
    """Compound multivariate embedding as described in the paper."""
    def __init__(self, embedding_metadata, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.metadata = embedding_metadata
        
        unique_attrs = embedding_metadata['unique_attributes']
        attr_embed_dim = embed_dim // 5
        remaining_dim = embed_dim - (attr_embed_dim * 4)
        
        self.level_embedding = nn.Embedding(len(unique_attrs['levels']), attr_embed_dim)
        self.type_embedding = nn.Embedding(len(unique_attrs['order_types']), attr_embed_dim)
        self.feature_embedding = nn.Embedding(len(unique_attrs['features']), attr_embed_dim)
        self.exchange_embedding = nn.Embedding(len(unique_attrs['exchanges']), attr_embed_dim)
        self.pair_embedding = nn.Embedding(len(unique_attrs['trading_pairs']), remaining_dim)
        
        self.projection = nn.Linear(embed_dim, embed_dim)
        
        self.level_to_idx = {level: i for i, level in enumerate(unique_attrs['levels'])}
        self.type_to_idx = {otype: i for i, otype in enumerate(unique_attrs['order_types'])}
        self.feature_to_idx = {feat: i for i, feat in enumerate(unique_attrs['features'])}
        self.exchange_to_idx = {exch: i for i, exch in enumerate(unique_attrs['exchanges'])}
        self.pair_to_idx = {pair: i for i, pair in enumerate(unique_attrs['trading_pairs'])}
        
        self._create_feature_indices()
    
    def _create_feature_indices(self):
        columns = self.metadata['columns']
        column_mapping = self.metadata['column_mapping']
        
        level_indices = []
        type_indices = []
        feature_indices = []
        exchange_indices = []
        pair_indices = []
        
        for col in columns:
            attrs = column_mapping[col]
            level_indices.append(self.level_to_idx[attrs['level']])
            type_indices.append(self.type_to_idx[attrs['order_type']])
            feature_indices.append(self.feature_to_idx[attrs['feature_type']])
            exchange_indices.append(self.exchange_to_idx[attrs['exchange']])
            pair_indices.append(self.pair_to_idx[attrs['trading_pair']])
        
        self.register_buffer('level_indices', torch.LongTensor(level_indices))
        self.register_buffer('type_indices', torch.LongTensor(type_indices))
        self.register_buffer('feature_indices', torch.LongTensor(feature_indices))
        self.register_buffer('exchange_indices', torch.LongTensor(exchange_indices))
        self.register_buffer('pair_indices', torch.LongTensor(pair_indices))
    
    def forward(self):
        level_embeds = self.level_embedding(self.level_indices)
        type_embeds = self.type_embedding(self.type_indices)
        feature_embeds = self.feature_embedding(self.feature_indices)
        exchange_embeds = self.exchange_embedding(self.exchange_indices)
        pair_embeds = self.pair_embedding(self.pair_indices)
        
        combined_embeds = torch.cat([
            level_embeds, type_embeds, feature_embeds, 
            exchange_embeds, pair_embeds
        ], dim=-1)
        
        projected_embeds = self.projection(combined_embeds)
        return projected_embeds

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)

class StructuralLoss(nn.Module):
    """Structural regularizer to preserve LOB price ordering."""
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
        
    def forward(self, predictions, feature_metadata):
        if predictions.size(-1) < 10:
            return torch.tensor(0.0, device=predictions.device)
        
        total_loss = 0.0
        columns = feature_metadata['columns']
        column_mapping = feature_metadata['column_mapping']
        
        exchange_pairs = {}
        for idx, col in enumerate(columns):
            attrs = column_mapping[col]
            key = f"{attrs['exchange']}_{attrs['trading_pair']}"
            if key not in exchange_pairs:
                exchange_pairs[key] = {'bid_prices': {}, 'ask_prices': {}}
            
            if attrs['feature_type'] == 'price':
                level = attrs['level']
                if attrs['order_type'] == 'bid':
                    exchange_pairs[key]['bid_prices'][level] = idx
                elif attrs['order_type'] == 'ask':
                    exchange_pairs[key]['ask_prices'][level] = idx
        
        for key, price_indices in exchange_pairs.items():
            bid_prices = price_indices['bid_prices']
            ask_prices = price_indices['ask_prices']
            
            if len(bid_prices) >= 2 and len(ask_prices) >= 2:
                for level in range(1, 5):
                    if level in ask_prices and (level + 1) in ask_prices:
                        ask_k = predictions[:, :, ask_prices[level]]
                        ask_k_plus_1 = predictions[:, :, ask_prices[level + 1]]
                        violation = torch.relu(ask_k - ask_k_plus_1)
                        total_loss += violation.mean()
                
                for level in range(1, 5):
                    if level in bid_prices and (level + 1) in bid_prices:
                        bid_k = predictions[:, :, bid_prices[level]]
                        bid_k_plus_1 = predictions[:, :, bid_prices[level + 1]]
                        violation = torch.relu(bid_k_plus_1 - bid_k)
                        total_loss += violation.mean()
                
                if 1 in bid_prices and 1 in ask_prices:
                    bid_1 = predictions[:, :, bid_prices[1]]
                    ask_1 = predictions[:, :, ask_prices[1]]
                    violation = torch.relu(bid_1 - ask_1)
                    total_loss += violation.mean()
        
        return self.weight * total_loss

class LOBForecaster(nn.Module):
    """Main attention-based LOB forecaster."""
    def __init__(self, embedding_metadata, embed_dim, num_heads, num_encoder_layers, 
                 num_decoder_layers, d_ff, dropout, target_len):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_features = len(embedding_metadata['columns'])
        self.target_len = target_len
        self.embed_dim = embed_dim
        
        self.value_projection = nn.Linear(1, embed_dim)
        self.compound_embedding = CompoundMultivariateEmbedding(embedding_metadata, embed_dim)
        self.time2vec = Time2Vec(embed_dim)
        self.context_target_embedding = nn.Embedding(2, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads, 
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, 1)
        
    def forward(self, src, tgt, src_times=None, tgt_times=None):
        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        feature_embeds = self.compound_embedding()
        
        src_values = self.value_projection(src.unsqueeze(-1))
        tgt_values = self.value_projection(tgt.unsqueeze(-1))
        
        src_embedded = src_values + feature_embeds.unsqueeze(0).unsqueeze(0)
        tgt_embedded = tgt_values + feature_embeds.unsqueeze(0).unsqueeze(0)
        
        if src_times is not None:
            time_embeds_src = self.time2vec(src_times)
            src_embedded = src_embedded + time_embeds_src.unsqueeze(2)
        
        if tgt_times is not None:
            time_embeds_tgt = self.time2vec(tgt_times)
            tgt_embedded = tgt_embedded + time_embeds_tgt.unsqueeze(2)
        
        context_embed = self.context_target_embedding(torch.zeros(batch_size, context_len, dtype=torch.long, device=src.device))
        target_embed = self.context_target_embedding(torch.ones(batch_size, target_len, dtype=torch.long, device=src.device))
        
        src_embedded = src_embedded + context_embed.unsqueeze(2)
        tgt_embedded = tgt_embedded + target_embed.unsqueeze(2)
        
        src_flat = src_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, context_len, self.embed_dim)
        tgt_flat = tgt_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, target_len, self.embed_dim)
        
        src_pos = self.positional_encoding(src_flat)
        tgt_pos = self.positional_encoding(tgt_flat)
        
        memory = self.transformer_encoder(src_pos)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_len).to(src.device)
        
        output = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)
        
        output = self.output_norm(output)
        output = self.output_projection(output)
        
        output = output.reshape(batch_size, self.num_features, target_len).permute(0, 2, 1)
        
        return output

# --- Distributed Training Functions ---

def setup_distributed(rank, world_size):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def create_model(embedding_metadata, device):
    """Create the LOB forecaster model."""
    model = LOBForecaster(
        embedding_metadata=embedding_metadata,
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        num_encoder_layers=CONFIG['num_encoder_layers'],
        num_decoder_layers=CONFIG['num_decoder_layers'],
        d_ff=CONFIG['d_ff'],
        dropout=CONFIG['dropout'],
        target_len=24
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created - Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model

def train_epoch(model, train_loader, mse_criterion, structural_loss, optimizer, embedding_metadata, device, rank=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_struct_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training", disable=rank != 0)
    for batch_idx, (contexts, targets) in enumerate(progress_bar):
        contexts, targets = contexts.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        decoder_input = torch.cat([
            contexts[:, -1:, :],
            targets[:, :-1, :]
        ], dim=1)
        
        predictions = model(contexts, decoder_input)
        
        mse_loss = mse_criterion(predictions, targets)
        struct_loss = structural_loss(predictions, embedding_metadata)
        total_batch_loss = mse_loss + struct_loss
        
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        total_mse_loss += mse_loss.item()
        total_struct_loss += struct_loss.item()
        num_batches += 1
        
        if rank == 0:
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.6f}',
                'MSE': f'{mse_loss.item():.6f}',
                'Struct': f'{struct_loss.item():.6f}'
            })
    
    return total_loss / num_batches, total_mse_loss / num_batches, total_struct_loss / num_batches

def validate(model, val_loader, mse_criterion, structural_loss, embedding_metadata, device, rank=0):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_struct_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for contexts, targets in tqdm(val_loader, desc="Validation", disable=rank != 0):
            contexts, targets = contexts.to(device), targets.to(device)
            
            decoder_input = torch.cat([
                contexts[:, -1:, :],
                targets[:, :-1, :]
            ], dim=1)
            
            predictions = model(contexts, decoder_input)
            
            mse_loss = mse_criterion(predictions, targets)
            struct_loss = structural_loss(predictions, embedding_metadata)
            total_batch_loss = mse_loss + struct_loss
            
            total_loss += total_batch_loss.item()
            total_mse_loss += mse_loss.item()
            total_struct_loss += struct_loss.item()
            num_batches += 1
    
    return total_loss / num_batches, total_mse_loss / num_batches, total_struct_loss / num_batches

def train_worker(rank, world_size, embedding_metadata, data_config):
    """Worker function for distributed training."""
    if CONFIG['distributed']:
        setup_distributed(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    if rank == 0:
        logger.info(f"Starting training on {world_size} GPU(s)")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size per GPU: {CONFIG['batch_size']}")
        logger.info(f"Effective batch size: {CONFIG['batch_size'] * world_size}")
    
    # Create datasets
    train_dataset = LOBDataset(os.path.join(CONFIG['data_path'], 'train.npz'))
    val_dataset = LOBDataset(os.path.join(CONFIG['data_path'], 'validation.npz'))
    
    # Create samplers and loaders
    if CONFIG['distributed']:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=CONFIG['num_workers'], 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        sampler=val_sampler,
        shuffle=False,
        num_workers=CONFIG['num_workers'], 
        pin_memory=True
    )
    
    # Create model
    model = create_model(embedding_metadata, device)
    
    if CONFIG['distributed']:
        model = DDP(model, device_ids=[rank])
    
    # Loss functions and optimizer
    mse_criterion = nn.MSELoss()
    structural_loss = StructuralLoss(weight=CONFIG['structural_weight'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    
    # Learning rate schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['decay_factor'], patience=3, verbose=(rank == 0)
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    if rank == 0:
        logger.info("Starting optimized training...")
        logger.info(f"Paper target metrics: Total loss < 0.008, Structure loss < 0.15")
    
    for epoch in range(CONFIG['epochs']):
        if CONFIG['distributed']:
            train_sampler.set_epoch(epoch)
        
        # Training
        train_loss, train_mse, train_struct = train_epoch(
            model, train_loader, mse_criterion, structural_loss, optimizer, embedding_metadata, device, rank
        )
        
        # Validation
        val_loss, val_mse, val_struct = validate(
            model, val_loader, mse_criterion, structural_loss, embedding_metadata, device, rank
        )
        
        scheduler.step(val_loss)
        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
            logger.info(f"Train - Total: {train_loss:.6f}, MSE: {train_mse:.6f}, Struct: {train_struct:.6f}")
            logger.info(f"Val   - Total: {val_loss:.6f}, MSE: {val_mse:.6f}, Struct: {val_struct:.6f}")
            logger.info(f"Learning Rate: {current_lr:.2e}")
            
            if val_loss < 0.008 and val_struct < 0.15:
                logger.info(f"🎯 Paper target metrics achieved! Total: {val_loss:.6f} < 0.008, Struct: {val_struct:.6f} < 0.15")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                model_to_save = model.module if CONFIG['distributed'] else model
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mse': val_mse,
                    'val_struct': val_struct,
                    'config': CONFIG,
                    'embedding_metadata': embedding_metadata
                }
                
                os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
                torch.save(checkpoint, os.path.join(CONFIG['model_save_dir'], 'best_model.pt'))
                logger.info(f"💾 New best model saved! Val loss: {val_loss:.6f}")
                
            else:
                patience_counter += 1
                
            if patience_counter >= CONFIG['patience']:
                logger.info(f"🛑 Early stopping triggered after {CONFIG['patience']} epochs without improvement")
                break
                
            logger.info("-" * 80)
    
    if rank == 0:
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
    
    if CONFIG['distributed']:
        cleanup_distributed()

def main():
    """Main training function."""
    logger.info("Starting Optimized Attention-Based LOB Forecasting Training")
    logger.info(f"Configuration: {CONFIG}")
    
    # Load data and metadata
    logger.info("Loading data and metadata...")
    
    with open(os.path.join(CONFIG['data_path'], 'embedding_metadata.json'), 'r') as f:
        embedding_metadata = json.load(f)
    
    with open(os.path.join(CONFIG['data_path'], 'config.json'), 'r') as f:
        data_config = json.load(f)
    
    logger.info(f"Dataset: {embedding_metadata['num_features']} features, "
                f"Context: {data_config['context_length']}, Target: {data_config['target_length']}")
    
    if CONFIG['distributed'] and CONFIG['num_gpus'] > 1:
        logger.info(f"Starting distributed training on {CONFIG['num_gpus']} GPUs")
        mp.spawn(train_worker, args=(CONFIG['num_gpus'], embedding_metadata, data_config), 
                nprocs=CONFIG['num_gpus'], join=True)
    else:
        logger.info("Starting single GPU training")
        train_worker(0, 1, embedding_metadata, data_config)

if __name__ == "__main__":
    main() 