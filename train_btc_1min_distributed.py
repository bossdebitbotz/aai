import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add spacetimeformer to path
sys.path.append('/root/aai/spacetimeformer')

from spacetimeformer.spacetimeformer_model import Spacetimeformer_Model
from spacetimeformer.data.datamodule import DataModule
from spacetimeformer.data.csv_dataset import CSVDataset, CSVTorchDset
from spacetimeformer.callbacks import EarlyStopping

def setup(rank, world_size):
    """Initialize the process group for distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the process group"""
    dist.destroy_process_group()

def create_model():
    """Create the attention-based LOB forecasting model"""
    model = Spacetimeformer_Model(
        d_model=126,
        n_heads=3,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.1,
        activation='gelu',
        output_attention=False,
        
        # Data configuration
        enc_in=240,  # 240 features
        dec_in=240,
        c_out=240,
        seq_len=120,
        label_len=24,
        pred_len=24,
        
        # Embedding configuration
        embed='compound_multivariate',
        freq='t',
        
        # LOB-specific parameters
        n_levels=5,
        n_types=2,
        n_features=2,
        n_exchanges=3,
        n_pairs=4,
        
        # Loss configuration
        use_structural_loss=True,
        structural_loss_weight=0.01,
        
        # Training parameters
        learning_rate=1e-4,
        patience=5,
        max_epochs=350
    )
    return model

class LOBDataset(torch.utils.data.Dataset):
    def __init__(self, sequences_path):
        print(f"Loading sequences from {sequences_path}")
        with open(sequences_path, 'rb') as f:
            self.sequences = pickle.load(f)
        print(f"Loaded {len(self.sequences)} sequences")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Extract context (120 steps) and target (24 steps)
        context = seq[:120]  # Shape: (120, 240)
        target = seq[120:144]  # Shape: (24, 240)
        
        return {
            'past_values': torch.FloatTensor(context),
            'future_values': torch.FloatTensor(target),
            'past_time_features': torch.zeros(120, 1),  # Placeholder
            'future_time_features': torch.zeros(24, 1)  # Placeholder
        }

def train_epoch(model, dataloader, optimizer, criterion, device, rank):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Move data to device
        past_values = batch['past_values'].to(device)
        future_values = batch['future_values'].to(device)
        past_time_features = batch['past_time_features'].to(device)
        future_time_features = batch['future_time_features'].to(device)
        
        # Forward pass
        outputs = model(
            past_values=past_values,
            past_time_features=past_time_features,
            future_values=future_values,
            future_time_features=future_time_features
        )
        
        # Calculate loss
        loss = criterion(outputs, future_values)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0 and batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            past_values = batch['past_values'].to(device)
            future_values = batch['future_values'].to(device)
            past_time_features = batch['past_time_features'].to(device)
            future_time_features = batch['future_time_features'].to(device)
            
            # Forward pass
            outputs = model(
                past_values=past_values,
                past_time_features=past_time_features,
                future_values=future_values,
                future_time_features=future_time_features
            )
            
            # Calculate loss
            loss = criterion(outputs, future_values)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_model(rank, world_size):
    """Main training function for each process"""
    print(f"Running on rank {rank}, world_size {world_size}")
    
    # Setup distributed training
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Create model
    model = create_model().to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create datasets
    train_dataset = LOBDataset('/root/aai/data/attention_sequences_btc_1min_train.pkl')
    val_dataset = LOBDataset('/root/aai/data/attention_sequences_btc_1min_val.pkl')
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=8, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)
    
    # Training loop
    if rank == 0:
        print("Starting training...")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    
    for epoch in range(350):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, rank)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Synchronize validation loss across all processes
        val_loss_tensor = torch.tensor(val_loss).to(device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"Epoch {epoch+1}/350 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Check early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, f'/root/aai/models/btc_1min_checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model
    if rank == 0:
        final_checkpoint = {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
        }
        torch.save(final_checkpoint, '/root/aai/models/btc_1min_final_model.pt')
        print("Training completed!")
    
    cleanup()

if __name__ == "__main__":
    world_size = 4  # Number of GPUs
    mp.spawn(train_model, args=(world_size,), nprocs=world_size, join=True)
