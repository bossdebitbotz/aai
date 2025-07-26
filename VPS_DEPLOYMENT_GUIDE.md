# 🚀 LOB Attention Model VPS Deployment Guide

**Complete step-by-step deployment guide for the Attention-Based Limit Order Book Forecasting Model**

## 📋 **Prerequisites**

### **Hardware Requirements**
- **Minimum**: Ubuntu 20.04+ VPS with 32GB RAM, 4+ CPU cores
- **Recommended**: 64GB+ RAM, 8+ CPU cores, NVIDIA GPU with 16GB+ VRAM
- **Storage**: 50GB+ free space for models and data

### **Software Requirements**
- Ubuntu 20.04 LTS or newer
- sudo access
- Internet connectivity
- SSH access

---

## 🛠️ **Step 1: Environment Setup**

### **1.1 Connect to VPS and Update System**
```bash
# Connect via SSH
ssh your-username@your-vps-ip

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y wget curl git unzip htop build-essential
```

### **1.2 Install Python 3.8**
```bash
# Install Python 3.8 (required for spacetimeformer)
sudo apt install -y python3.8 python3.8-dev python3.8-venv python3-pip

# Verify installation
python3.8 --version  # Should show Python 3.8.x

# Create symbolic link if needed
sudo ln -sf /usr/bin/python3.8 /usr/bin/python3
```

### **1.3 Install PostgreSQL 14**
```bash
# Install PostgreSQL
sudo apt install -y postgresql-14 postgresql-client-14 postgresql-contrib-14

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Check status
sudo systemctl status postgresql
```

---

## 📦 **Step 2: Project Deployment**

### **2.1 Create Project Directory**
```bash
# Create main project directory
mkdir -p /opt/aai-lob-model
cd /opt/aai-lob-model

# Set permissions (replace 'username' with your user)
sudo chown -R $USER:$USER /opt/aai-lob-model
```

### **2.2 Transfer and Extract Project Files**
```bash
# Upload the zip file to VPS (from your local machine):
scp aai-lob-model-20250724.zip your-username@your-vps-ip:/opt/aai-lob-model/

# Extract the project files
cd /opt/aai-lob-model
unzip aai-lob-model-20250724.zip
rm aai-lob-model-20250724.zip

# Verify extraction
ls -la  # Should show all project files and directories
```

---

## 🐍 **Step 3: Python Environment Setup**

### **3.1 Create Virtual Environment**
```bash
# Create virtual environment with Python 3.8
python3.8 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel
```

### **3.2 Install Core Dependencies**
```bash
# Install PyTorch 1.11.0 (CUDA version if GPU available)
# For CPU-only VPS:
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu

# For GPU VPS (CUDA 11.3):
# pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install project requirements
pip install -r requirements.txt

# Install additional dependencies for spacetimeformer
pip install pytorch-lightning==1.6 torchmetrics==0.5.1
pip install performer-pytorch nystrom-attention
pip install cmdstanpy==0.9.68 pystan==2.19.1.1
```

### **3.3 Install Spacetimeformer**
```bash
# Install spacetimeformer in editable mode
cd spacetimeformer
pip install -e .
cd ..

# Verify installation
python -c "import spacetimeformer; print('✅ Spacetimeformer installed successfully')"
```

---

## 🗄️ **Step 4: Database Setup**

### **4.1 Configure PostgreSQL**
```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user (in PostgreSQL shell)
CREATE DATABASE backtest_db;
CREATE USER backtest_user WITH PASSWORD 'your_secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE backtest_db TO backtest_user;
\q
```

### **4.2 Configure PostgreSQL Settings**
```bash
# Edit PostgreSQL configuration
sudo nano /etc/postgresql/14/main/postgresql.conf

# Add/modify these settings:
# shared_buffers = 4GB                    # 25% of RAM
# effective_cache_size = 12GB             # 75% of RAM  
# work_mem = 256MB                        # For sorting operations
# maintenance_work_mem = 1GB              # For VACUUM, CREATE INDEX
# max_connections = 100                   # Adjust based on usage
# listen_addresses = 'localhost'          # Security
```

```bash
# Edit PostgreSQL authentication
sudo nano /etc/postgresql/14/main/pg_hba.conf

# Ensure this line exists for local connections:
# local   all             all                                     md5

# Restart PostgreSQL
sudo systemctl restart postgresql
```

---

## 🔐 **Step 5: Environment Variables Setup**

### **5.1 Create Environment Configuration**
```bash
# Create .env file
nano .env
```

**Add the following content to .env:**
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5433
DB_USER=backtest_user
DB_PASSWORD=your_secure_password_here
DB_NAME=backtest_db

# Model Configuration
MODEL_SAVE_DIR=models/attention_lob
FINAL_DATA_DIR=data/final_attention
DEVICE=cuda  # or 'cpu' if no GPU

# Training Configuration
BATCH_SIZE=4
LEARNING_RATE=1e-4
EMBED_DIM=126
NUM_HEADS=3
NUM_ENCODER_LAYERS=3
NUM_DECODER_LAYERS=3

# Hardware Configuration
NUM_WORKERS=4  # Adjust based on CPU cores
CUDA_VISIBLE_DEVICES=0  # GPU device ID

# Paths
PROJECT_ROOT=/opt/aai-lob-model
DATA_PATH=/opt/aai-lob-model/data
MODEL_PATH=/opt/aai-lob-model/models

# Logging
LOG_LEVEL=INFO
LOG_FILE=/opt/aai-lob-model/training.log
```

### **5.2 Load Environment Variables**
```bash
# Create environment loader script
nano load_env.sh
```

**Add this content:**
```bash
#!/bin/bash
# Load environment variables
export $(cat .env | grep -v ^# | xargs)
echo "✅ Environment variables loaded"
```

```bash
# Make executable
chmod +x load_env.sh

# Load variables
source load_env.sh
```

---

## 📊 **Step 6: Data Verification**

### **6.1 Verify Training Data**
```bash
# Activate virtual environment
source venv/bin/activate

# Check data integrity
python3 -c "
import numpy as np
import os

# Check if training data exists
data_path = 'data/final_attention'
files = ['train.npz', 'validation.npz', 'test.npz', 'config.json', 'embedding_metadata.json']

for file in files:
    filepath = os.path.join(data_path, file)
    if os.path.exists(filepath):
        print(f'✅ {file} exists')
    else:
        print(f'❌ {file} missing')

# Check data shapes
try:
    train = np.load('data/final_attention/train.npz')
    print(f'✅ Training data shape: {train[\"contexts\"].shape}')
    print(f'✅ Training targets shape: {train[\"targets\"].shape}')
    print(f'✅ Features: {train[\"contexts\"].shape[-1]}')
except Exception as e:
    print(f'❌ Error loading training data: {e}')
"
```

### **6.2 Quick Training Test**
```bash
# Test if training can start (run for 1 epoch)
python3 -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
"

# Test training script import
python3 -c "
try:
    import train_attention_model
    print('✅ Training script imports successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
"
```

---

## 🎯 **Step 7: Start Training**

### **7.1 Verify Everything is Ready**
```bash
# Final checklist
echo "🔍 Pre-training verification:"
echo "Python environment: $(which python3)"
echo "Working directory: $(pwd)"
echo "Data exists: $(ls -la data/final_attention/)"
echo "GPU available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
```

### **7.2 Start Training Process**
```bash
# Load environment
source load_env.sh
source venv/bin/activate

# Start training with nohup (background process)
nohup python3 train_attention_model.py > training.log 2>&1 &

# Get process ID
echo $! > training.pid
echo "✅ Training started! Process ID: $(cat training.pid)"

# Monitor training progress
tail -f training.log
```

### **7.3 Alternative: Screen Session Training**
```bash
# Install screen
sudo apt install -y screen

# Start screen session
screen -S lob-training

# Inside screen session:
source venv/bin/activate
source load_env.sh
python3 train_attention_model.py

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r lob-training
```

---

## 📊 **Step 8: Monitoring & Management**

### **8.1 Monitor Training Progress**
```bash
# Check training log
tail -f training.log

# Check GPU usage (if available)
nvidia-smi

# Check system resources
htop

# Check model checkpoints
ls -la models/attention_lob/
```

### **8.2 Useful Management Commands**
```bash
# Check if training is running
ps aux | grep train_attention_model.py

# Stop training (if needed)
kill $(cat training.pid)

# Check disk usage
df -h
du -sh data/ models/

# Monitor memory usage
free -h
```

---

## 🔧 **Step 9: Model Evaluation**

### **9.1 Test Trained Model**
```bash
# After training completes, test the model
source venv/bin/activate
source load_env.sh

python3 test_trained_model.py

# Run backtesting
python3 backtest_attention_model.py
```

### **9.2 Check Model Performance**
```bash
# Check final model files
ls -la models/attention_lob/

# Verify model can load
python3 -c "
import torch
model_path = 'models/attention_lob/best_model.pt'
try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f'✅ Model loaded successfully')
    print(f'✅ Final validation loss: {checkpoint.get(\"val_loss\", \"N/A\")}')
except Exception as e:
    print(f'❌ Error loading model: {e}')
"
```

---

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. CUDA/GPU Issues**
```bash
# Check NVIDIA drivers
nvidia-smi

# Install CUDA toolkit if needed
sudo apt install -y nvidia-cuda-toolkit

# Use CPU-only if GPU issues persist
export DEVICE=cpu
```

#### **2. Memory Issues**
```bash
# Reduce batch size in training script
# Edit train_attention_model.py:
# BATCH_SIZE = 2  # Reduce from 4

# Or set environment variable
export BATCH_SIZE=2
```

#### **3. PostgreSQL Connection Issues**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Reset database password
sudo -u postgres psql
\password backtest_user

# Check connections
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"
```

#### **4. Package Import Errors**
```bash
# Reinstall problematic packages
pip uninstall torch torchvision torchaudio
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0

# Check Python path
python3 -c "import sys; print('\n'.join(sys.path))"
```

#### **5. Data Loading Issues**
```bash
# Verify data integrity
python3 -c "
import numpy as np
for file in ['train.npz', 'validation.npz', 'test.npz']:
    try:
        data = np.load(f'data/final_attention/{file}')
        print(f'✅ {file}: {list(data.keys())}')
    except Exception as e:
        print(f'❌ {file}: {e}')
"
```

---

## 📈 **Expected Results**

### **Training Timeline**
- **Setup Time**: 30-60 minutes
- **Training Time**: 2-6 hours (depending on hardware)
- **Expected Epochs**: 20-50 epochs
- **Early Stopping**: After 5 epochs without improvement

### **Target Metrics**
- **Total Loss**: < 0.008
- **Structure Loss**: < 0.15
- **Validation Loss**: Should decrease steadily

### **Model Files**
After successful training, you should have:
```
models/attention_lob/
├── best_model.pt          # Best model checkpoint
├── training_history.json  # Training metrics
├── model_config.json      # Model configuration
└── final_model.pt         # Final epoch model
```

---

## 🔒 **Security & Maintenance**

### **Security Best Practices**
```bash
# Secure PostgreSQL
sudo ufw allow from 127.0.0.1 to any port 5432

# Update environment file permissions
chmod 600 .env

# Regular security updates
sudo apt update && sudo apt upgrade -y
```

### **Backup Strategy**
```bash
# Create backup script
nano backup.sh
```

**Add this content:**
```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/aai-lob-model"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup models
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" models/

# Backup processed data (metadata only)
tar -czf "$BACKUP_DIR/metadata_$DATE.tar.gz" data/final_attention/*.json

# Backup configuration
cp .env "$BACKUP_DIR/env_$DATE.backup"

echo "✅ Backup completed: $BACKUP_DIR"
```

```bash
chmod +x backup.sh
# Run backup
./backup.sh
```

---

## 📞 **Support & Next Steps**

### **Verification Checklist**
- [ ] Environment activated: `source venv/bin/activate`
- [ ] Dependencies installed: `pip list | grep torch`
- [ ] Database running: `sudo systemctl status postgresql`
- [ ] Data loaded: `ls data/final_attention/`
- [ ] Training started: `tail training.log`
- [ ] GPU detected (if available): `nvidia-smi`

### **Performance Monitoring**
```bash
# Create monitoring script
nano monitor.sh
```

**Add this content:**
```bash
#!/bin/bash
echo "=== System Status ==="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2}')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print $1 "/" $2 " MB"}')"
fi
echo "Training Process: $(ps aux | grep -c train_attention_model.py)"
```

```bash
chmod +x monitor.sh
# Run monitoring
./monitor.sh
```

**Your LOB Attention Model is now deployed and ready for training! 🚀**

For questions or issues, check the troubleshooting section above or refer to the training logs. 