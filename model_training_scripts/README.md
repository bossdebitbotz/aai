# LOB Forecasting Model Training Scripts

This directory contains 16 specialized training scripts for attention-based Limit Order Book (LOB) forecasting models. Each script uses **all available market data** (240 features from 3 exchanges × 4 pairs) to predict **specific binance_perp targets** for different timeframes.

## 📁 Directory Structure

```
model_training_scripts/
├── 1min_predictions/          # 1-minute prediction models (12 steps)
│   ├── train_1min_binance_perp_wld.py
│   ├── train_1min_binance_perp_sol.py
│   ├── train_1min_binance_perp_eth.py
│   └── train_1min_binance_perp_btc.py
├── 2min_predictions/          # 2-minute prediction models (24 steps) 
│   ├── train_2min_binance_perp_wld.py
│   ├── train_2min_binance_perp_sol.py
│   ├── train_2min_binance_perp_eth.py
│   └── train_2min_binance_perp_btc.py
├── 3min_predictions/          # 3-minute prediction models (36 steps)
│   ├── train_3min_binance_perp_wld.py
│   ├── train_3min_binance_perp_sol.py
│   ├── train_3min_binance_perp_eth.py
│   └── train_3min_binance_perp_btc.py
├── 5min_predictions/          # 5-minute prediction models (60 steps)
│   ├── train_5min_binance_perp_wld.py
│   ├── train_5min_binance_perp_sol.py
│   ├── train_5min_binance_perp_eth.py
│   └── train_5min_binance_perp_btc.py
├── run_training.py           # Master script to run any combination
└── README.md                 # This file
```

## 🎯 Model Strategy

### Input Strategy: Full Market Context
- **Input Features**: All 240 features (3 exchanges × 4 pairs × 5 levels × 2 types × 2 features)
- **Context Length**: 120 steps (10 minutes at 5-second intervals)
- **Exchanges Used**: binance_spot, binance_perp, bybit_spot
- **Pairs Used**: BTC-USDT, ETH-USDT, SOL-USDT, WLD-USDT

### Output Strategy: Target-Specific Predictions
- **Output Features**: Only 20 features for specific binance_perp pair
  - 5 LOB levels × 2 order types (bid/ask) × 2 features (price/volume)
- **Target Exchange**: binance_perp (perpetual futures)
- **Target Pairs**: WLD-USDT, SOL-USDT, ETH-USDT, or BTC-USDT

### Prediction Timeframes
| Timeframe | Steps | Duration |
|-----------|-------|----------|
| 1 minute  | 12    | 60 seconds |
| 2 minutes | 24    | 120 seconds |
| 3 minutes | 36    | 180 seconds |
| 5 minutes | 60    | 300 seconds |

## 🚀 Usage

### Master Script (Recommended)

Use `run_training.py` for easy execution:

```bash
# Single model training
python model_training_scripts/run_training.py --timeframe 1 --pair wld

# Train all pairs for specific timeframe  
python model_training_scripts/run_training.py --timeframe 2 --pair all

# Train all timeframes for specific pair
python model_training_scripts/run_training.py --timeframe all --pair btc

# Train ALL 16 models sequentially
python model_training_scripts/run_training.py --timeframe all --pair all

# Train ALL 16 models in parallel (4 workers)
python model_training_scripts/run_training.py --timeframe all --pair all --parallel

# Preview what would be run without executing
python model_training_scripts/run_training.py --timeframe all --pair all --list-only
```

### Direct Script Execution

You can also run individual scripts directly:

```bash
# 1-minute WLD predictions
python model_training_scripts/1min_predictions/train_1min_binance_perp_wld.py

# 5-minute BTC predictions
python model_training_scripts/5min_predictions/train_5min_binance_perp_btc.py
```

## 📊 Model Outputs

Each training script creates:

### Model Files
```
models/{timeframe}min_binance_perp_{pair}/
├── best_model.pth           # Best model checkpoint
├── final_model.pth          # Final model state
└── training_history.json    # Loss curves and metrics
```

### Model Checkpoints Include:
- Model state dictionary
- Optimizer state
- Target feature indices
- Training configuration
- Validation loss history

## ⚙️ Configuration

Each script uses the same base configuration:

```python
# Model Architecture
EMBED_DIM = 126              # Divisible by 3 heads
NUM_HEADS = 3                # As per paper
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1

# Training Parameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EPOCHS = 50
PATIENCE = 5                 # Early stopping
REG_WEIGHT = 0.01           # Structural regularizer

# Target-Specific (varies by script)
TARGET_STEPS = 12/24/36/60   # Based on timeframe
TARGET_PAIR = "WLD/SOL/ETH/BTC-USDT"
TARGET_EXCHANGE = "binance_perp"
```

## 🧠 Model Architecture Details

### Input Processing
1. **All Market Features**: Uses complete 240-feature input
2. **Compound Embedding**: Attribute-based embeddings (level, type, feature, exchange, pair)
3. **Positional Encoding**: Standard transformer positional encoding
4. **Value Projection**: Projects raw values to embedding dimension

### Target Processing  
1. **Feature Filtering**: Extracts only target exchange/pair features
2. **Timeframe Adaptation**: Adjusts sequence length for prediction horizon
3. **Structural Regularizer**: Preserves LOB price ordering constraints

### Loss Function
```python
total_loss = mse_loss + 0.01 * structural_loss
```

- **MSE Loss**: Standard forecasting accuracy
- **Structural Loss**: Preserves bid/ask price ordering and spreads

## 📈 Expected Performance

Based on the paper results:

| Metric | Target |
|--------|--------|
| Total Loss | < 0.008 |
| Structure Loss | < 0.15 |
| Training Time | 2-4 hours per model |
| GPU Memory | ~8-12GB |

## 🔧 Hardware Requirements

### Minimum Requirements
- **RAM**: 16GB
- **GPU**: 8GB VRAM (GTX 3070 or better)
- **Storage**: 10GB free space for models

### Recommended for Parallel Training
- **RAM**: 32GB+
- **GPU**: 16GB+ VRAM (RTX 4080 or better)
- **CPU**: 8+ cores for parallel processing
- **Storage**: 50GB+ for all model outputs

## 📝 Training Tips

### Sequential Training (Default)
- Trains one model at a time
- Lower memory usage
- Easier to monitor and debug
- Total time: ~48-64 hours for all 16 models

### Parallel Training (--parallel)
- Trains multiple models simultaneously  
- Requires more memory and GPU resources
- Much faster total completion time
- Total time: ~12-16 hours for all 16 models

### Best Practices
1. **Start with one model** to verify setup
2. **Monitor GPU memory** usage during training
3. **Use parallel training** if you have sufficient resources
4. **Check logs** for early stopping and convergence
5. **Save intermediate results** regularly

## 🚨 Troubleshooting

### Common Issues

**GPU Out of Memory:**
```bash
# Reduce batch size in script
BATCH_SIZE = 2  # Instead of 4
```

**Training Divergence:**
```bash
# Check learning rate and warmup
LEARNING_RATE = 5e-5  # Reduce if unstable
WARMUP_STEPS = 2000   # Increase warmup
```

**Slow Training:**
```bash
# Enable mixed precision training (add to script)
from torch.cuda.amp import autocast, GradScaler
```

### Log Files
Each training script logs to console. To save logs:
```bash
python model_training_scripts/1min_predictions/train_1min_binance_perp_wld.py > logs/1min_wld.log 2>&1
```

## 🎯 Next Steps After Training

1. **Model Evaluation**: Test on held-out test set
2. **Performance Comparison**: Compare across timeframes and pairs
3. **Ensemble Methods**: Combine predictions from multiple models
4. **Live Trading Integration**: Deploy best models for real-time trading
5. **Hyperparameter Tuning**: Optimize model configurations

## 📞 Support

If you encounter issues:
1. Check this README for common solutions
2. Verify data preparation completed successfully
3. Ensure sufficient hardware resources
4. Check individual script logs for specific errors

---

## 🎉 Ready to Train!

You now have 16 specialized LOB forecasting models ready to train. Each model leverages the full market context to make precise predictions for specific targets, providing comprehensive coverage of different prediction horizons and trading pairs.

**Start with a single model to verify everything works, then scale up to parallel training for maximum efficiency!** 