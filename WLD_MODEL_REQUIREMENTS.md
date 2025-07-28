# WLD 1-Minute Binance Perp Model Requirements

## 📊 Data Structure Requirements

### Input Data (Context)
- **Shape**: `(batch_size, 120, 240)`
- **Context Length**: 120 time steps (120 seconds = 2 minutes of 1-second data)
- **Input Features**: 240 features across all exchanges and trading pairs
- **Purpose**: Historical market data for prediction context

### Output Data (Targets)
- **Shape**: `(batch_size, 24, 20)`
- **Target Length**: 24 time steps (24 seconds = 24 seconds ahead prediction)
- **Target Features**: 20 WLD binance_perp features only
- **Feature Indices**: 120-139 (from the 240 total features)

## 🎯 Target Features (20 WLD Binance Perp Features)

| Index | Feature Name | Type | Level |
|-------|--------------|------|-------|
| 120 | `binance_perp_WLD-USDT_bid_price_1` | Price | 1 |
| 121 | `binance_perp_WLD-USDT_bid_quantity_1` | Volume | 1 |
| 122 | `binance_perp_WLD-USDT_bid_price_2` | Price | 2 |
| 123 | `binance_perp_WLD-USDT_bid_quantity_2` | Volume | 2 |
| 124 | `binance_perp_WLD-USDT_bid_price_3` | Price | 3 |
| 125 | `binance_perp_WLD-USDT_bid_quantity_3` | Volume | 3 |
| 126 | `binance_perp_WLD-USDT_bid_price_4` | Price | 4 |
| 127 | `binance_perp_WLD-USDT_bid_quantity_4` | Volume | 4 |
| 128 | `binance_perp_WLD-USDT_bid_price_5` | Price | 5 |
| 129 | `binance_perp_WLD-USDT_bid_quantity_5` | Volume | 5 |
| 130 | `binance_perp_WLD-USDT_ask_price_1` | Price | 1 |
| 131 | `binance_perp_WLD-USDT_ask_quantity_1` | Volume | 1 |
| 132 | `binance_perp_WLD-USDT_ask_price_2` | Price | 2 |
| 133 | `binance_perp_WLD-USDT_ask_quantity_2` | Volume | 2 |
| 134 | `binance_perp_WLD-USDT_ask_price_3` | Price | 3 |
| 135 | `binance_perp_WLD-USDT_ask_quantity_3` | Volume | 3 |
| 136 | `binance_perp_WLD-USDT_ask_price_4` | Price | 4 |
| 137 | `binance_perp_WLD-USDT_ask_quantity_4` | Volume | 4 |
| 138 | `binance_perp_WLD-USDT_ask_price_5` | Price | 5 |
| 139 | `binance_perp_WLD-USDT_ask_quantity_5` | Volume | 5 |

## 🏗️ Model Architecture Requirements

### Model Configuration
```python
EMBED_DIM = 256
NUM_HEADS = 8 
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 4
DROPOUT = 0.1
TARGET_LEN = 24  # CRITICAL: Must match data target_length
NUM_INPUT_FEATURES = 240
NUM_TARGET_FEATURES = 20
```

### Critical Architecture Settings
- **Target Length**: 24 time steps (not 12!)
- **Input Features**: All 240 market features for context
- **Output Features**: Only 20 WLD binance_perp features
- **Context Window**: 120 time steps
- **Prediction Window**: 24 time steps ahead

## 📁 Data Files Required

### Training Data
```
data/final_attention/
├── train.npz           # Shape: contexts(N,120,240), targets(N,24,240)
├── validation.npz      # Shape: contexts(N,120,240), targets(N,24,240)  
├── test.npz           # Shape: contexts(N,120,240), targets(N,24,240)
└── embedding_metadata.json  # Feature mappings and configuration
```

### Metadata Requirements
```json
{
  "num_features": 240,
  "context_length": 120,
  "target_length": 24,  // CRITICAL: Model must use this value
  "columns": [...],      // 240 feature names
  "column_mapping": {...} // Feature index mappings
}
```

## ⚠️ Common Mistakes to Avoid

1. **❌ Wrong Target Length**: Using `target_len=12` instead of `target_len=24`
2. **❌ Wrong Feature Count**: Not filtering to exactly 20 WLD perp features
3. **❌ Wrong Feature Indices**: Using wrong indices (must be 120-139)
4. **❌ Shape Mismatch**: Model output not matching (batch, 24, 20)

## ✅ Validation Checklist

Before training, verify:
- [ ] Model `target_len = 24`
- [ ] Target feature indices = `[120, 121, ..., 139]` (20 features)
- [ ] Dataset target shape = `(N, 24, 20)` after filtering
- [ ] Model output shape = `(batch_size, 24, 20)`
- [ ] All WLD perp features are binance_perp exchange
- [ ] All WLD perp features are WLD-USDT trading pair

## 🚀 Training Script Requirements

```python
# CORRECT model instantiation
model = ScaledMultiGPUForecaster(
    embedding_metadata=embedding_metadata,
    target_feature_indices=[120, 121, ..., 139],  # 20 WLD perp features
    embed_dim=256,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=4,
    dropout=0.1,
    target_len=24,  # MUST match data target_length
    num_target_features=20  # WLD perp features only
)
```

## 📈 Expected Performance Metrics

- **Input**: Market context (2 minutes of 1-second data)
- **Output**: WLD perp LOB predictions (24 seconds ahead)
- **Use Case**: 1-minute trading decisions on WLD perpetual futures
- **Trading Frequency**: Every second with 24-second lookahead 