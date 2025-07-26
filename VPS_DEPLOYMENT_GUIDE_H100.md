# ⚡ ULTRA HIGH-PERFORMANCE H100 Deployment Guide

**Optimized for: 4x NVIDIA H100 + 48-Core CPU + 1TB RAM**
**Timeline: 5-day intensive training period**

## 🎯 **Performance Overview**

With your **4x NVIDIA H100** setup, you have:
- **~2.5 PetaFLOPS** of FP16 compute power
- **320GB total GPU memory** (80GB per H100)
- **Potential 10-20x faster training** than consumer GPUs

**Expected Results**:
- **Complete training**: 2-4 hours instead of days
- **Multiple model variations**: Train 5-10 different configurations
- **Hyperparameter sweeps**: Test dozens of combinations
- **Advanced experiments**: Multi-scale, ensemble training

---

## 🛠️ **Step 1: H100 Environment Setup**

### **1.1 Connect and Verify Hardware**
```bash
# Connect via SSH
ssh your-username@your-h100-server

# Verify GPU setup
nvidia-smi

# Should show 4x H100 with ~80GB each
# Expected output: 4 NVIDIA H100 80GB HBM3
```

### **1.2 Install CUDA 11.8+ for H100**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y wget curl git unzip htop build-essential

# Install CUDA 11.8 (required for H100)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA
nvcc --version  # Should show CUDA 11.8
```

### **1.3 Install Python 3.8**
```bash
# Install Python 3.8 (required for spacetimeformer)
sudo apt install -y python3.8 python3.8-dev python3.8-venv python3-pip
python3.8 --version
```

---

## 📦 **Step 2: High-Performance Project Setup**

### **2.1 Create Project Directory**
```bash
# Create project directory
mkdir -p /opt/aai-lob-h100
cd /opt/aai-lob-h100
sudo chown -R $USER:$USER /opt/aai-lob-h100
```

### **2.2 Upload and Extract (Optimized)**
```bash
# Upload via SCP (from your local machine)
scp aai-lob-model-20250724.zip your-username@your-h100-server:/opt/aai-lob-h100/

# Extract with parallel processing
cd /opt/aai-lob-h100
unzip -q aai-lob-model-20250724.zip
rm aai-lob-model-20250724.zip
```

---

## 🐍 **Step 3: High-Performance Python Environment**

### **3.1 Create Virtual Environment**
```bash
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### **3.2 Install PyTorch for H100**
```bash
# Install PyTorch 2.0+ with CUDA 11.8 support for H100
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Verify H100 detection
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

### **3.3 Install Optimized Dependencies**
```bash
# Install high-performance requirements
pip install -r requirements.txt

# Install spacetimeformer with optimizations
cd spacetimeformer
pip install -e .
cd ..

# Install additional performance libraries
pip install accelerate==0.20.3  # For multi-GPU optimization
pip install transformers==4.30.0  # Latest transformer optimizations
pip install datasets==2.13.0  # Fast data loading
```

---

## ⚡ **Step 4: H100 Configuration**

### **4.1 Create H100-Optimized Environment**
```bash
# Create .env file optimized for H100
cat > .env << EOL
# H100 High-Performance Configuration
DEVICE=cuda
CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 H100s

# Model Configuration (maximized for H100)
MODEL_SAVE_DIR=models/attention_lob_h100
FINAL_DATA_DIR=data/final_attention

# H100-Optimized Training Parameters
BATCH_SIZE=32  # 8x larger due to massive GPU memory
LEARNING_RATE=2e-4  # Slightly higher for faster convergence
EMBED_DIM=256  # Larger model for better performance
NUM_HEADS=8  # More attention heads
NUM_ENCODER_LAYERS=6  # Deeper model
NUM_DECODER_LAYERS=4
DROPOUT=0.05  # Reduced dropout for powerful model

# H100 Performance Optimization
NUM_WORKERS=24  # Use half of CPU cores for data loading
PIN_MEMORY=true
MIXED_PRECISION=true  # Enable FP16 for 2x speed boost
COMPILE_MODEL=true  # PyTorch 2.0 compilation for speed
DISTRIBUTED_BACKEND=nccl
GRADIENT_ACCUMULATION_STEPS=1

# Memory Optimization
MAX_MEMORY_ALLOCATED=75  # Use 75% of 80GB per GPU
DATALOADER_NUM_WORKERS=24
PREFETCH_FACTOR=4

# Paths
PROJECT_ROOT=/opt/aai-lob-h100
DATA_PATH=/opt/aai-lob-h100/data
MODEL_PATH=/opt/aai-lob-h100/models

# Logging
LOG_LEVEL=INFO
LOG_FILE=/opt/aai-lob-h100/training.log
EOL
```

### **4.2 Create H100 Training Script**
```bash
# Create high-performance training script
cat > train_h100_distributed.py << 'EOL'
#!/usr/bin/env python3
"""
Ultra High-Performance H100 Training Script
Optimized for 4x NVIDIA H100 GPUs
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
import time
from datetime import datetime
import json

# Import your attention model
from train_attention_model import *

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def load_h100_optimized_data():
    """Load data with H100 optimizations"""
    print("Loading data with H100 optimizations...")
    
    # Load training data
    train_data = np.load('data/final_attention/train.npz')
    val_data = np.load('data/final_attention/validation.npz')
    
    train_contexts = torch.FloatTensor(train_data['contexts'])
    train_targets = torch.FloatTensor(train_data['targets'])
    val_contexts = torch.FloatTensor(val_data['contexts'])
    val_targets = torch.FloatTensor(val_data['targets'])
    
    print(f"Training data: {train_contexts.shape}")
    print(f"Validation data: {val_contexts.shape}")
    print(f"Features: {train_contexts.shape[-1]}")
    
    return train_contexts, train_targets, val_contexts, val_targets

def create_h100_model(feature_dim, context_length):
    """Create model optimized for H100"""
    model = AttentionLOBModel(
        feature_dim=feature_dim,
        context_length=context_length,
        target_length=24,
        embed_dim=256,  # Larger for H100
        num_heads=8,    # More heads
        num_encoder_layers=6,  # Deeper
        num_decoder_layers=4,
        dropout=0.05
    )
    
    # Enable torch.compile for 2x speed boost on H100
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("✅ Model compiled with PyTorch 2.0")
    
    return model

def train_epoch_h100(model, dataloader, optimizer, criterion, scaler, rank):
    """H100-optimized training epoch"""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (contexts, targets) in enumerate(dataloader):
        contexts = contexts.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training for 2x speedup
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(contexts)
            loss = criterion(predictions, targets)
        
        # Scale loss and backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if rank == 0 and batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * contexts.size(0) * 4 / elapsed  # 4 GPUs
            print(f"Batch {batch_idx}, Loss: {loss.item():.6f}, "
                  f"Speed: {samples_per_sec:.1f} samples/sec")
    
    return total_loss / len(dataloader)

def validate_h100(model, dataloader, criterion):
    """H100-optimized validation"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts = contexts.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                predictions = model(contexts)
                loss = criterion(predictions, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_h100(rank, world_size):
    """Main H100 training function"""
    print(f"Starting H100 training on rank {rank}/{world_size}")
    setup(rank, world_size)
    
    # Load data
    train_contexts, train_targets, val_contexts, val_targets = load_h100_optimized_data()
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_contexts, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_contexts, val_targets)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    # Create data loaders with H100 optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Large batch size for H100
        sampler=train_sampler,
        num_workers=24,  # High parallelism
        pin_memory=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        sampler=val_sampler,
        num_workers=24,
        pin_memory=True,
        prefetch_factor=4
    )
    
    # Create model
    feature_dim = train_contexts.shape[-1]
    context_length = train_contexts.shape[1]
    model = create_h100_model(feature_dim, context_length).cuda()
    model = DDP(model, device_ids=[rank])
    
    # H100-optimized optimizer and training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    if rank == 0:
        print("🚀 H100 Ultra High-Performance Training Started!")
        print(f"Model parameters: ~{sum(p.numel() for p in model.parameters()):,}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Batch size per GPU: 32 (Total: 128)")
        print(f"Mixed precision: Enabled")
        print(f"Model compilation: {'Enabled' if hasattr(torch, 'compile') else 'Disabled'}")
    
    # Training loop
    best_val_loss = float('inf')
    training_start = time.time()
    
    for epoch in range(50):  # More epochs possible due to speed
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch_h100(model, train_loader, optimizer, criterion, scaler, rank)
        
        # Validate
        val_loss = validate_h100(model, val_loader, criterion)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        if rank == 0:
            print(f"\n🔥 EPOCH {epoch+1}/50 - {epoch_time:.1f}s")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Check if target achieved
            if val_loss < 0.005:  # Even lower target for H100
                print(f"🎯 ULTRA TARGET ACHIEVED! Val loss: {val_loss:.6f} < 0.005")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'training_time': time.time() - training_start,
                }, f'models/attention_lob_h100/best_h100_model.pt')
                print(f"💎 NEW BEST MODEL! Val loss: {val_loss:.6f}")
            
            print("-" * 80)
    
    total_time = time.time() - training_start
    if rank == 0:
        print(f"🏆 H100 TRAINING COMPLETED!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Average time per epoch: {total_time/50:.1f} seconds")
    
    cleanup()

if __name__ == "__main__":
    world_size = 4  # 4x H100 GPUs
    mp.spawn(train_h100, args=(world_size,), nprocs=world_size, join=True)
EOL
```

---

## 🚀 **Step 5: Ultra-Fast Training Launch**

### **5.1 Final Verification**
```bash
# Activate environment
source venv/bin/activate
source load_env.sh

# Verify H100 setup
python3 -c "
import torch
print('🔥 H100 VERIFICATION:')
print(f'GPUs available: {torch.cuda.device_count()}')
print(f'Memory per GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Verify training data
ls -la data/final_attention/
```

### **5.2 Launch Ultra-High-Performance Training**
```bash
# Create model directory
mkdir -p models/attention_lob_h100

# Start 4-GPU distributed training
python3 train_h100_distributed.py

# Alternative: Monitor with screen
screen -S h100-training
python3 train_h100_distributed.py
# Detach: Ctrl+A, D
# Reattach: screen -r h100-training
```

---

## 📊 **Step 6: Performance Monitoring**

### **6.1 H100 Monitoring Script**
```bash
cat > monitor_h100.sh << 'EOL'
#!/bin/bash
echo "🔥 H100 PERFORMANCE MONITOR"
echo "=========================="
echo "Timestamp: $(date)"
echo ""

# GPU utilization
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
    echo "  GPU $line"
done
echo ""

# System resources
echo "System Resources:"
echo "  CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "  Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}') ($(free | awk '/^Mem:/ {printf "%.1f%%", $3/$2 * 100}'))"
echo "  Disk: $(df -h /opt | awk 'NR==2 {print $3 "/" $2}') ($(df /opt | awk 'NR==2 {print $5}'))"
echo ""

# Training process
training_count=$(ps aux | grep -c "train_h100_distributed.py")
if [ $training_count -gt 1 ]; then
    echo "✅ Training is RUNNING"
    echo "  Processes: $training_count"
else
    echo "❌ Training is NOT running"
fi

# GPU memory efficiency
echo ""
echo "GPU Memory Efficiency:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{
    used = $1; total = $2; efficiency = (used/total)*100
    printf "  GPU Memory: %.1f%% (%d/%d MB)\n", efficiency, used, total
}'
EOL

chmod +x monitor_h100.sh
```

### **6.2 Real-time Performance Dashboard**
```bash
# Run monitoring every 10 seconds
watch -n 10 ./monitor_h100.sh

# Or create a continuous log
while true; do
    ./monitor_h100.sh >> h100_performance.log
    echo "--- $(date) ---" >> h100_performance.log
    sleep 30
done &
```

---

## ⚡ **Step 7: Maximum Utilization Strategy**

### **7.1 5-Day Training Plan**

**Day 1-2: Base Model Training**
- Train primary attention model (2-4 hours)
- Hyperparameter sweep (4-6 configurations)
- Model size variations (128, 256, 512 dims)

**Day 3: Advanced Experiments**
- Multi-timeframe models (1min, 5min, 15min)
- Ensemble training
- Different attention mechanisms

**Day 4: Optimization & Analysis**
- Model distillation
- Quantization experiments
- Performance benchmarking

**Day 5: Production & Backup**
- Final model selection
- Comprehensive backtesting
- Model export and backup

### **7.2 Multiple Simultaneous Training**
```bash
# Train multiple models in parallel (if data allows)
# Terminal 1: Main model
python3 train_h100_distributed.py --config main

# Terminal 2: Alternative configuration
python3 train_h100_distributed.py --config alternative

# Terminal 3: Smaller experimental model
python3 train_h100_distributed.py --config experimental
```

---

## 📈 **Expected H100 Performance**

### **Training Speed Estimates**
- **Per Epoch**: 30-60 seconds (vs 10-30 minutes on consumer GPUs)
- **Complete Training**: 2-4 hours (vs 1-3 days)
- **Multiple Models**: 5-10 different configurations per day
- **Hyperparameter Sweep**: 20-50 experiments in 5 days

### **Target Metrics (Enhanced for H100)**
- **Total Loss**: < 0.005 (even better than before)
- **Structure Loss**: < 0.10
- **Training Efficiency**: >95% GPU utilization
- **Memory Utilization**: 60-75GB per H100

### **Expected Results After 5 Days**
- **10+ trained models** with different configurations
- **Comprehensive hyperparameter analysis**
- **Production-ready models** with full backtesting
- **Complete performance benchmarks**

---

## 🔧 **H100-Specific Optimizations**

### **Performance Tuning**
```bash
# Enable maximum performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 2619,1980  # Set max memory and GPU clocks

# Set CPU performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Optimize system settings
echo 'vm.swappiness=1' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_max=134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max=134217728' | sudo tee -a /etc/sysctl.conf
```

### **Memory Optimization**
```bash
# Set optimal memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=12  # Quarter of CPU cores
```

---

## 🎯 **Success Indicators**

**Training is optimized when you see**:
- ✅ All 4 H100s at >90% utilization
- ✅ GPU memory usage 60-75GB per GPU
- ✅ <60 seconds per epoch
- ✅ Consistent loss reduction
- ✅ No memory errors or bottlenecks

**🚀 With this setup, you'll achieve in 5 days what would take months on regular hardware!**

Your H100 deployment is now ready for **MAXIMUM PERFORMANCE TRAINING!** 🔥 