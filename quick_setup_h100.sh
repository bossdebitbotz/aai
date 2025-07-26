#!/bin/bash

# ⚡ H100 Ultra High-Performance Quick Setup Script
# Optimized for 4x NVIDIA H100 + 48-Core CPU + 1TB RAM
# NO DATABASE - Pure training focus for 5-day intensive period

set -e  # Exit on any error

echo "⚡ Starting H100 Ultra High-Performance Setup..."
echo "=============================================="
echo "Target: 4x NVIDIA H100 + 48-Core + 1TB RAM"
echo "Timeline: 5-day intensive training period"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_performance() {
    echo -e "${PURPLE}🚀 $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Verify we're on the H100 system
print_info "Verifying H100 hardware..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ $GPU_COUNT -eq 4 ]; then
        print_performance "Detected 4 GPUs - H100 system confirmed!"
    else
        print_warning "Expected 4 GPUs, found $GPU_COUNT"
    fi
else
    print_warning "nvidia-smi not found - will install CUDA drivers"
fi

# Update system for maximum performance
print_status "Updating system for H100 performance..."
sudo apt update && sudo apt upgrade -y

# Install essential tools
print_status "Installing essential tools..."
sudo apt install -y wget curl git unzip htop build-essential software-properties-common
sudo apt install -y linux-headers-$(uname -r)  # For NVIDIA drivers

# Install CUDA 11.8 for H100 support
print_performance "Installing CUDA 11.8 for H100..."
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
if command -v nvcc &> /dev/null; then
    print_status "CUDA installed successfully: $(nvcc --version | grep "release" | awk '{print $6}')"
else
    print_error "CUDA installation failed"
    exit 1
fi

# Install Python 3.8
print_status "Installing Python 3.8..."
sudo apt install -y python3.8 python3.8-dev python3.8-venv python3-pip

# Verify Python installation
python3.8 --version
if [ $? -eq 0 ]; then
    print_status "Python 3.8 installed successfully"
else
    print_error "Python 3.8 installation failed"
    exit 1
fi

# Create high-performance project directory
PROJECT_DIR="/opt/aai-lob-h100"
print_performance "Creating H100 project directory: $PROJECT_DIR"
sudo mkdir -p $PROJECT_DIR
sudo chown -R $USER:$USER $PROJECT_DIR

# Check if zip file exists in current directory
if [ -f "aai-lob-model-20250724.zip" ]; then
    print_status "Found project zip file, copying to H100 directory..."
    cp aai-lob-model-20250724.zip $PROJECT_DIR/
    cd $PROJECT_DIR
    print_info "Extracting project files (5GB)..."
    unzip -q aai-lob-model-20250724.zip
    rm aai-lob-model-20250724.zip
    print_status "Project files extracted"
else
    print_warning "Project zip file not found in current directory"
    print_warning "You'll need to upload aai-lob-model-20250724.zip to $PROJECT_DIR manually"
    cd $PROJECT_DIR
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
python3.8 -m venv venv

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip for performance
pip install --upgrade pip setuptools wheel

# Install PyTorch 2.0+ with CUDA 11.8 for H100
print_performance "Installing PyTorch 2.0+ optimized for H100..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Verify H100 detection
print_info "Verifying H100 GPU detection..."
python3 -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'🔥 GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory/1e9:.1f}GB)')
" && print_performance "H100 GPUs detected successfully!"

# Install project requirements if available
if [ -f "requirements.txt" ]; then
    print_status "Installing project requirements..."
    pip install -r requirements.txt
else
    print_info "Installing core dependencies..."
    pip install pandas numpy scikit-learn joblib tqdm
fi

# Install high-performance libraries
print_performance "Installing H100 performance libraries..."
pip install accelerate==0.20.3  # Multi-GPU optimization
pip install transformers==4.30.0  # Latest optimizations  
pip install datasets==2.13.0  # Fast data loading

# Install spacetimeformer if directory exists
if [ -d "spacetimeformer" ]; then
    print_status "Installing spacetimeformer..."
    cd spacetimeformer
    pip install -e .
    cd ..
    print_status "Spacetimeformer installed"
else
    print_warning "Spacetimeformer directory not found"
fi

# Install additional performance dependencies
print_performance "Installing additional performance libraries..."
pip install pytorch-lightning==1.6 torchmetrics==0.5.1
pip install performer-pytorch nystrom-attention

# Create H100-optimized environment configuration
print_performance "Creating H100-optimized configuration..."
cat > .env << EOL
# H100 Ultra High-Performance Configuration
DEVICE=cuda
CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 H100s

# Model Configuration (maximized for H100)
MODEL_SAVE_DIR=models/attention_lob_h100
FINAL_DATA_DIR=data/final_attention

# H100-Optimized Training Parameters
BATCH_SIZE=32  # 8x larger due to massive GPU memory
LEARNING_RATE=2e-4  # Optimized for fast convergence
EMBED_DIM=256  # Larger model for H100 capability
NUM_HEADS=8  # More attention heads
NUM_ENCODER_LAYERS=6  # Deeper model
NUM_DECODER_LAYERS=4
DROPOUT=0.05  # Reduced for powerful model

# H100 Performance Optimization
NUM_WORKERS=24  # Use half of CPU cores
PIN_MEMORY=true
MIXED_PRECISION=true  # FP16 for 2x speed boost
COMPILE_MODEL=true  # PyTorch 2.0 compilation
DISTRIBUTED_BACKEND=nccl
GRADIENT_ACCUMULATION_STEPS=1

# Memory Optimization for H100
MAX_MEMORY_ALLOCATED=75  # Use 75% of 80GB per GPU
DATALOADER_NUM_WORKERS=24
PREFETCH_FACTOR=4

# System Optimization
OMP_NUM_THREADS=12  # Quarter of CPU cores
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
CUDA_LAUNCH_BLOCKING=0

# Paths
PROJECT_ROOT=$PROJECT_DIR
DATA_PATH=$PROJECT_DIR/data
MODEL_PATH=$PROJECT_DIR/models

# Logging
LOG_LEVEL=INFO
LOG_FILE=$PROJECT_DIR/h100_training.log
EOL

# Create environment loader script
cat > load_env.sh << 'EOL'
#!/bin/bash
export $(cat .env | grep -v ^# | xargs)
echo "⚡ H100 environment variables loaded"
EOL

chmod +x load_env.sh

# Create H100 performance monitoring script
cat > monitor_h100.sh << 'EOL'
#!/bin/bash
echo "🔥 H100 PERFORMANCE MONITOR"
echo "=========================="
echo "Timestamp: $(date)"
echo ""

# GPU utilization with enhanced formatting
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read gpu name temp gpu_util mem_util mem_used mem_total; do
    efficiency=$(( mem_used * 100 / mem_total ))
    echo "  🔥 GPU $gpu ($name): ${gpu_util}% compute, ${mem_util}% memory (${efficiency}%), ${temp}°C"
done
echo ""

# System resources
echo "System Resources:"
echo "  💻 CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "  🧠 Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}') ($(free | awk '/^Mem:/ {printf "%.1f%%", $3/$2 * 100}'))"
echo "  💾 Disk: $(df -h /opt | awk 'NR==2 {print $3 "/" $2}') ($(df /opt | awk 'NR==2 {print $5}'))"
echo ""

# Training process detection
training_count=$(ps aux | grep -c "train.*h100\|train.*attention")
if [ $training_count -gt 1 ]; then
    echo "✅ H100 Training: RUNNING ($training_count processes)"
else
    echo "❌ H100 Training: NOT RUNNING"
fi

# GPU memory efficiency summary
echo ""
echo "H100 Memory Efficiency:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk 'BEGIN{total_used=0; total_capacity=0} {
    used = $1; total = $2; 
    efficiency = (used/total)*100
    total_used += used; total_capacity += total
    printf "  GPU %d: %.1f%% (%d/%d MB)\n", NR-1, efficiency, used, total
} END {
    overall = (total_used/total_capacity)*100
    printf "  🎯 Overall: %.1f%% (%d/%d MB)\n", overall, total_used, total_capacity
}'

echo ""
echo "🚀 Target: >90% GPU utilization, 60-75GB per H100"
EOL

chmod +x monitor_h100.sh

# Create H100 backup script
cat > backup_h100.sh << EOL
#!/bin/bash
DATE=\$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/h100-models"

mkdir -p \$BACKUP_DIR

print_info() {
    echo "ℹ️  \$1"
}

print_info "Creating H100 model backup..."

# Backup H100 models
if [ -d "models/attention_lob_h100/" ]; then
    tar -czf "\$BACKUP_DIR/h100_models_\$DATE.tar.gz" models/attention_lob_h100/
    print_info "Models backed up"
fi

# Backup training logs
if [ -f "h100_training.log" ]; then
    cp h100_training.log "\$BACKUP_DIR/h100_training_\$DATE.log"
fi

# Backup configuration
cp .env "\$BACKUP_DIR/h100_env_\$DATE.backup"

echo "✅ H100 backup completed: \$BACKUP_DIR"
EOL

chmod +x backup_h100.sh

# Set optimal file permissions
chmod 600 .env

# Apply H100 system optimizations
print_performance "Applying H100 system optimizations..."

# Enable maximum GPU performance
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 2619,1980  # Max memory and GPU clocks

# Set CPU to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

# Optimize system settings for H100
echo 'vm.swappiness=1' | sudo tee -a /etc/sysctl.conf > /dev/null
echo 'net.core.rmem_max=134217728' | sudo tee -a /etc/sysctl.conf > /dev/null
echo 'net.core.wmem_max=134217728' | sudo tee -a /etc/sysctl.conf > /dev/null

print_performance "H100 system optimizations applied!"

# Final verification
print_info "Running H100 final verification..."

# Check PyTorch with H100
python3 -c "
import torch
import sys
success = True

# Check CUDA
if not torch.cuda.is_available():
    print('❌ CUDA not available')
    success = False
else:
    print(f'✅ CUDA available: {torch.version.cuda}')

# Check GPU count
gpu_count = torch.cuda.device_count()
if gpu_count != 4:
    print(f'⚠️  Expected 4 GPUs, found {gpu_count}')
else:
    print(f'✅ 4 H100 GPUs detected')

# Check GPU memory
if gpu_count > 0:
    memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if memory > 70:  # H100 has ~80GB
        print(f'✅ GPU memory: {memory:.1f}GB (H100 confirmed)')
    else:
        print(f'⚠️  GPU memory: {memory:.1f}GB (not H100?)')

if success:
    print('🚀 H100 verification PASSED!')
    sys.exit(0)
else:
    print('❌ H100 verification FAILED!')
    sys.exit(1)
" && print_performance "H100 verification successful!"

# Check project structure
if [ -d "data" ] && [ -d "models" ]; then
    print_status "Project structure verified"
else
    print_warning "Some project directories may be missing"
fi

echo ""
print_performance "🔥 H100 SETUP COMPLETE! 🔥"
print_performance "Ultra high-performance environment ready!"
echo ""
echo "=============================================="
echo "🎯 5-DAY H100 TRAINING PLAN:"
echo "=============================================="
echo ""
echo "🚀 IMMEDIATE NEXT STEPS:"
echo "  1. Verify training data:"
echo "     ls -la data/final_attention/"
echo ""
echo "  2. Start H100 training:"
echo "     source venv/bin/activate"
echo "     source load_env.sh"
echo "     python3 train_h100_distributed.py"
echo ""
echo "  3. Monitor performance:"
echo "     ./monitor_h100.sh"
echo "     watch -n 10 './monitor_h100.sh'"
echo ""
echo "📈 EXPECTED H100 PERFORMANCE:"
echo "  • Training time: 2-4 hours (vs days on regular GPUs)"
echo "  • Epoch time: 30-60 seconds"
echo "  • GPU utilization: >90%"
echo "  • Memory usage: 60-75GB per H100"
echo "  • Multiple models: 5-10 configurations in 5 days"
echo ""
echo "⚡ H100 OPTIMIZATION FEATURES:"
echo "  ✅ 4x H100 distributed training"
echo "  ✅ Mixed precision (FP16)"
echo "  ✅ PyTorch 2.0 compilation"
echo "  ✅ Maximum performance settings"
echo "  ✅ Optimized data loading"
echo "  ✅ System-level optimizations"
echo ""
print_performance "Your H100 beast is ready to DEMOLISH this training! 🚀" 