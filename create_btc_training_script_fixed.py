#!/usr/bin/env python3

import os

def create_training_script():
    script_content = '''#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add spacetimeformer to path and import directly
sys.path.insert(0, os.path.join(os.getcwd(), 'spacetimeformer'))

# Direct imports to avoid pytorch-lightning dependency issues
from spacetimeformer.spacetimeformer_model.nn.model import STF
from spacetimeformer.spacetimeformer_model.nn.embed import DataEmbedding, CompoundMultivariateEmbedding

class LOBStructuralLoss(nn.Module):
    """Structural loss to preserve LOB price ordering constraints"""
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
        
    def forward(self, predictions, targets):
        """
        Ensure ask prices > bid prices in predictions
        Args:
            predictions: [batch, time, features] where features include bid/ask prices
            targets: [batch, time, features] 
        """
        if predictions.size(-1) < 4:  # Need at least bid_price, bid_size, ask_price, ask_size
            return torch.tensor(0.0, device=predictions.device)
            
        # Extract bid and ask prices (assuming they're the first 2 features)
        pred_bid_prices = predictions[:, :, 0]  # [batch, time]
        pred_ask_prices = predictions[:, :, 2]  # [batch, time] 
        
        # Structural constraint: ask_price should be >= bid_price
        violations = torch.relu(pred_bid_prices - pred_ask_prices)  # Positive when violated
        structural_loss = violations.mean()
        
        return self.weight * structural_loss

class SpacetimeformerWrapper(nn.Module):
    """Wrapper around spacetimeformer STF model for our LOB data"""
    def __init__(self, d_vars, max_seq_len, d_model=126, n_heads=3, e_layers=3, d_layers=3, 
                 d_ff=None, dropout=0.1, activation='gelu', device='cuda'):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.device = device
        self.d_vars = d_vars
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Compound multivariate embedding for LOB data
        self.compound_embedding = CompoundMultivariateEmbedding(
            d_vars=d_vars,
            d_model=d_model,
            max_seq_len=max_seq_len,
            # LOB-specific embedding configuration
            level_dims=10,      # 10 price levels
            type_dims=2,        # bid/ask types  
            feature_dims=3,     # price/size/count features
            exchange_dims=1,    # single exchange (Binance)
            pair_dims=1,        # single pair (BTC/USDT)
            dropout=dropout
        )
        
        # Core spacetimeformer model
        self.stf = STF(
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            output_attention=False
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_vars)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_vars] input sequences
        Returns:
            [batch, pred_len, d_vars] predictions
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply compound embedding
        embedded = self.compound_embedding(x)  # [batch, seq_len, d_model]
        
        # Apply spacetimeformer
        stf_output = self.stf(embedded)  # [batch, seq_len, d_model]
        
        # For forecasting, we typically want the last timestep's representation
        # or we can use the full sequence - here we'll use the last timestep
        forecast_repr = stf_output[:, -1:, :]  # [batch, 1, d_model]
        
        # Project to output space
        predictions = self.output_projection(forecast_repr)  # [batch, 1, d_vars]
        
        return predictions

def setup(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def load_data():
    """Load and prepare the LOB data"""
    print("Loading LOB data...")
    
    # Load the processed sequences
    data_path = os.path.join(os.getcwd(), 'data/lob_sequences_btc_1min.pkl')
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please ensure the data has been processed first.")
        return None, None, None, None
        
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']  # [N, total_len, features]
    context_length = data['context_length']  # 120
    target_length = data['target_length']    # 24
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Context length: {context_length}, Target length: {target_length}")
    print(f"Feature dimension: {sequences.shape[-1]}")
    
    # Split into train/val (80/20)
    n_train = int(0.8 * len(sequences))
    
    train_sequences = sequences[:n_train]
    val_sequences = sequences[n_train:]
    
    # Split sequences into context and target
    train_X = train_sequences[:, :context_length, :]  # [N, 120, features]
    train_y = train_sequences[:, context_length:context_length+1, :]  # [N, 1, features] - predict next step
    
    val_X = val_sequences[:, :context_length, :]
    val_y = val_sequences[:, context_length:context_length+1, :]
    
    print(f"Training set: {train_X.shape[0]} samples")
    print(f"Validation set: {val_X.shape[0]} samples")
    
    return train_X, train_y, val_X, val_y

def train_epoch(model, train_loader, criterion, structural_loss, optimizer, device, rank):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_struct_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(data)
        
        # Main prediction loss
        mse_loss = criterion(predictions, targets)
        
        # Structural loss to preserve LOB constraints
        struct_loss = structural_loss(predictions, targets)
        
        # Combined loss
        loss = mse_loss + struct_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_struct_loss += struct_loss.item()
        num_batches += 1
        
        if rank == 0 and batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, "
                  f"MSE: {mse_loss.item():.6f}, Struct: {struct_loss.item():.6f}")
    
    return total_loss / num_batches, total_mse_loss / num_batches, total_struct_loss / num_batches

def validate(model, val_loader, criterion, structural_loss, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_struct_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            
            predictions = model(data)
            
            mse_loss = criterion(predictions, targets)
            struct_loss = structural_loss(predictions, targets)
            loss = mse_loss + struct_loss
            
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_struct_loss += struct_loss.item()
            num_batches += 1
    
    return total_loss / num_batches, total_mse_loss / num_batches, total_struct_loss / num_batches

def train_worker(rank, world_size):
    """Main training function for each GPU"""
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Starting distributed training on {world_size} GPUs")
        print(f"Device: {device}")
    
    # Load data
    train_X, train_y, val_X, val_y = load_data()
    if train_X is None:
        cleanup()
        return
    
    # Convert to tensors
    train_X = torch.FloatTensor(train_X)
    train_y = torch.FloatTensor(train_y)
    val_X = torch.FloatTensor(val_X)
    val_y = torch.FloatTensor(val_y)
    
    # Create datasets
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    # Create data loaders
    batch_size = 8  # Per GPU batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    
    # Model parameters
    d_vars = train_X.shape[-1]  # Number of features
    max_seq_len = train_X.shape[1]  # Sequence length (120)
    
    if rank == 0:
        print(f"Model input dimensions: {d_vars} features, {max_seq_len} sequence length")
    
    # Create model
    model = SpacetimeformerWrapper(
        d_vars=d_vars,
        max_seq_len=max_seq_len,
        d_model=126,
        n_heads=3,
        e_layers=3,
        d_layers=3,
        dropout=0.1,
        device=device
    ).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    # Loss functions and optimizer
    criterion = nn.MSELoss()
    structural_loss = LOBStructuralLoss(weight=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Training loop
    num_epochs = 350
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    if rank == 0:
        print(f"Starting training for {num_epochs} epochs")
        print(f"Target metrics: Total loss < 0.008, Structure loss < 0.15")
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_mse, train_struct = train_epoch(
            model, train_loader, criterion, structural_loss, optimizer, device, rank
        )
        
        # Validate
        val_loss, val_mse, val_struct = validate(
            model, val_loader, criterion, structural_loss, device
        )
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train - Total: {train_loss:.6f}, MSE: {train_mse:.6f}, Struct: {train_struct:.6f}")
            print(f"Val   - Total: {val_loss:.6f}, MSE: {val_mse:.6f}, Struct: {val_struct:.6f}")
            
            # Check target metrics
            if val_loss < 0.008 and val_struct < 0.15:
                print(f"🎯 Target metrics achieved! Total: {val_loss:.6f} < 0.008, Struct: {val_struct:.6f} < 0.15")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mse': val_mse,
                    'val_struct': val_struct,
                }, f'best_model_btc_1min_rank_{rank}.pt')
                
                print(f"💾 New best model saved! Val loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
                break
                
            print("-" * 80)
    
    if rank == 0:
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
    
    cleanup()

def main():
    """Main function"""
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")
    
    if world_size < 2:
        print("This script requires at least 2 GPUs for distributed training")
        return
    
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
'''
    
    # Write the script
    script_path = 'train_btc_1min_distributed_fixed.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Created training script: {script_path}")
    print("This version imports spacetimeformer components directly to avoid pytorch-lightning issues")

if __name__ == "__main__":
    create_training_script() 