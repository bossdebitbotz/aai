#!/bin/bash

# 🚀 LOB Attention Model Quick Setup Script
# This script automates the initial deployment setup on Ubuntu VPS

set -e  # Exit on any error

echo "🚀 Starting LOB Attention Model VPS Setup..."
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential tools
print_status "Installing essential tools..."
sudo apt install -y wget curl git unzip htop build-essential software-properties-common

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

# Install PostgreSQL 14
print_status "Installing PostgreSQL 14..."
sudo apt install -y postgresql-14 postgresql-client-14 postgresql-contrib-14

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql
print_status "PostgreSQL installed and started"

# Create project directory
PROJECT_DIR="/opt/aai-lob-model"
print_status "Creating project directory: $PROJECT_DIR"
sudo mkdir -p $PROJECT_DIR
sudo chown -R $USER:$USER $PROJECT_DIR

# Check if zip file exists in current directory
if [ -f "aai-lob-model-20250724.zip" ]; then
    print_status "Found project zip file, copying to project directory..."
    cp aai-lob-model-20250724.zip $PROJECT_DIR/
    cd $PROJECT_DIR
    unzip aai-lob-model-20250724.zip
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

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version by default, can be changed later)
print_status "Installing PyTorch..."
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Install project requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    print_status "Installing project requirements..."
    pip install -r requirements.txt
else
    print_warning "requirements.txt not found, installing basic dependencies..."
    pip install pandas numpy psycopg2-binary scikit-learn joblib tqdm
fi

# Install spacetimeformer dependencies
print_status "Installing spacetimeformer dependencies..."
pip install pytorch-lightning==1.6 torchmetrics==0.5.1
pip install performer-pytorch nystrom-attention
pip install cmdstanpy==0.9.68 pystan==2.19.1.1

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

# Create environment configuration file
print_status "Creating environment configuration..."
cat > .env << EOL
# Database Configuration
DB_HOST=localhost
DB_PORT=5433
DB_USER=backtest_user
DB_PASSWORD=change_this_password_now
DB_NAME=backtest_db

# Model Configuration
MODEL_SAVE_DIR=models/attention_lob
FINAL_DATA_DIR=data/final_attention
DEVICE=cpu

# Training Configuration
BATCH_SIZE=4
LEARNING_RATE=1e-4
EMBED_DIM=126
NUM_HEADS=3
NUM_ENCODER_LAYERS=3
NUM_DECODER_LAYERS=3

# Hardware Configuration
NUM_WORKERS=4
CUDA_VISIBLE_DEVICES=0

# Paths
PROJECT_ROOT=$PROJECT_DIR
DATA_PATH=$PROJECT_DIR/data
MODEL_PATH=$PROJECT_DIR/models

# Logging
LOG_LEVEL=INFO
LOG_FILE=$PROJECT_DIR/training.log
EOL

# Create environment loader script
cat > load_env.sh << 'EOL'
#!/bin/bash
export $(cat .env | grep -v ^# | xargs)
echo "✅ Environment variables loaded"
EOL

chmod +x load_env.sh

# Create monitoring script
cat > monitor.sh << 'EOL'
#!/bin/bash
echo "=== System Status ==="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2}')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print $1 "/" $2 " MB"}')"
fi
echo "Training Process: $(ps aux | grep -c train_attention_model.py)"
EOL

chmod +x monitor.sh

# Create backup script
cat > backup.sh << EOL
#!/bin/bash
DATE=\$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/aai-lob-model"

mkdir -p \$BACKUP_DIR

# Backup models
if [ -d "models/" ]; then
    tar -czf "\$BACKUP_DIR/models_\$DATE.tar.gz" models/
fi

# Backup processed data metadata
if [ -d "data/final_attention/" ]; then
    tar -czf "\$BACKUP_DIR/metadata_\$DATE.tar.gz" data/final_attention/*.json 2>/dev/null || true
fi

# Backup configuration
cp .env "\$BACKUP_DIR/env_\$DATE.backup"

echo "✅ Backup completed: \$BACKUP_DIR"
EOL

chmod +x backup.sh

# Set proper file permissions
chmod 600 .env

print_status "Setup completed! Next steps:"
echo "================================================"
echo ""
echo "1. 🔐 Configure PostgreSQL database:"
echo "   sudo -u postgres psql"
echo "   CREATE DATABASE backtest_db;"
echo "   CREATE USER backtest_user WITH PASSWORD 'your_secure_password';"
echo "   GRANT ALL PRIVILEGES ON DATABASE backtest_db TO backtest_user;"
echo "   \\q"
echo ""
echo "2. 📝 Update database password in .env file:"
echo "   nano .env"
echo "   (Change DB_PASSWORD to your actual password)"
echo ""
echo "3. 🔍 Verify training data exists:"
echo "   ls -la data/final_attention/"
echo ""
echo "4. 🚀 Start training:"
echo "   source venv/bin/activate"
echo "   source load_env.sh"
echo "   python3 train_attention_model.py"
echo ""
echo "5. 📊 Monitor progress:"
echo "   ./monitor.sh"
echo "   tail -f training.log"
echo ""

# Final verification
print_status "Running final verification..."

# Check Python
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && print_status "PyTorch working"

# Check PostgreSQL
sudo systemctl is-active postgresql && print_status "PostgreSQL running"

# Check project structure
if [ -d "data" ] && [ -d "models" ]; then
    print_status "Project structure looks good"
else
    print_warning "Some project directories may be missing"
fi

echo ""
print_status "Setup complete! 🎉"
print_status "Your LOB Attention Model environment is ready for deployment."
echo ""
print_warning "Remember to:"
echo "  - Set a secure database password"
echo "  - Verify your training data is present"
echo "  - Adjust DEVICE=cuda in .env if you have a GPU"
echo "  - Monitor system resources during training" 