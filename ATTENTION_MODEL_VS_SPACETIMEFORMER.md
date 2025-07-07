# Attention-Based LOB Model vs. Spacetimeformer: Key Differences

## 📊 **Overview**

This document outlines the critical differences between the attention-based LOB forecasting model described in the paper "Attention-Based Reading, Highlighting, and Forecasting of the Limit Order Book" and the existing Spacetimeformer implementation in the codebase.

## 🔍 **Architectural Differences**

### 1. **Embedding Strategy**

#### Spacetimeformer Approach
```python
# Individual variable embeddings
# Each variable gets its own embedding ID
variable_embeddings = {
    'binance_spot_BTC-USDT_bid_price_1': 0,
    'binance_spot_BTC-USDT_bid_volume_1': 1,
    'binance_spot_BTC-USDT_ask_price_1': 2,
    # ... 160 individual embeddings
}
```

#### Our LOB Model Approach (Compound Multivariate Embedding)
```python
# Attribute-based embeddings that combine
level_embedding = [1, 2, 3, 4, 5]  # LOB levels
type_embedding = ['bid', 'ask']      # Order type
feature_embedding = ['price', 'volume']  # Feature type
exchange_embedding = ['binance_spot', 'binance_perp', 'bybit_spot']
pair_embedding = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'WLD-USDT']

# Final embedding = combine(level_emb + type_emb + feature_emb + exchange_emb + pair_emb)
```

**Key Advantage**: Reduced parameters, better attribute relationship modeling

### 2. **Loss Function**

#### Spacetimeformer
```python
# Standard MSE loss only
loss = nn.MSELoss()(predictions, targets)
```

#### Our LOB Model
```python
# Combined loss with structural regularizer
forecasting_loss = nn.MSELoss()(predictions, targets)
structure_loss = compute_lob_structure_loss(predictions)
total_loss = forecasting_loss + 0.01 * structure_loss
```

**Key Advantage**: Preserves LOB price ordering structure

### 3. **Data Transformation**

#### Spacetimeformer
```python
# General normalization
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)
```

#### Our LOB Model
```python
# Financial data specific transformations
# 1. Percent-change for prices (stationarity)
price_pct_change = (prices[t] - prices[t-1]) / prices[t-1]

# 2. Min-max scaling for both prices and volumes
price_scaled = MinMaxScaler().fit_transform(price_pct_change)
volume_scaled = MinMaxScaler().fit_transform(volumes)
```

**Key Advantage**: Better stationarity for financial time series

### 4. **Attention Mechanism**

#### Spacetimeformer
```python
# Standard quadratic attention
attention_weights = softmax(Q @ K.T / sqrt(d_k))
output = attention_weights @ V
# Complexity: O(L²d) where L is sequence length
```

#### Our LOB Model
```python
# Performer attention (linear complexity)
from performer_pytorch import Performer
# Complexity: O(Ld) - linear in sequence length
```

**Key Advantage**: Scalable to longer sequences

## 🎯 **Functional Differences**

### 1. **Sequence Structure**

| Aspect | Spacetimeformer | Our LOB Model |
|--------|-----------------|---------------|
| Context Length | Variable | Fixed 120 steps (10 min) |
| Target Length | Variable | Fixed 24 steps (2 min) |
| Overlap Strategy | Configurable | 50% overlap |
| Session Awareness | No | Yes (respects trading sessions) |

### 2. **Feature Dimensions**

| Component | Spacetimeformer | Our LOB Model |
|-----------|-----------------|---------------|
| Input Features | Variable | 160 (3 exchanges × 4 pairs × 5 levels × 2 types × 2 features) |
| Sequence Length | Variable | 144 (120 context + 24 target) |
| Embedding Params | ~160 variable embeddings | ~25 attribute embeddings |

### 3. **Training Configuration**

| Parameter | Spacetimeformer | Our LOB Model |
|-----------|-----------------|---------------|
| Learning Rate | Variable | 1e-4 with decay |
| Warmup Steps | Variable | 1000 |
| Attention Heads | Variable | 3 |
| Batch Size | Variable | 4-16 |
| Regularization | L2 optional | L2 + Structure Loss |

## 🔧 **Implementation Differences**

### 1. **Data Loading**

#### Spacetimeformer
```python
# Generic CSV/dataset loading
class SpacetimeDataset(Dataset):
    def __init__(self, data_path, context_len, target_len):
        self.data = pd.read_csv(data_path)
        # Standard preprocessing
```

#### Our LOB Model
```python
# LOB-specific data loading with session awareness
class LOBDataset(Dataset):
    def __init__(self, data_path):
        # Handle session boundaries
        # Apply LOB-specific transformations
        # Maintain price ordering constraints
```

### 2. **Model Architecture**

#### Spacetimeformer
```python
class Spacetimeformer(nn.Module):
    def __init__(self, ...):
        self.variable_embedding = nn.Embedding(num_variables, embed_dim)
        self.transformer = nn.Transformer(...)
        
    def forward(self, x):
        # Flatten to (batch, seq_len * num_vars, embed_dim)
        # Apply attention across all variables and time
```

#### Our LOB Model
```python
class LOBForecaster(nn.Module):
    def __init__(self, ...):
        self.compound_embedding = CompoundMultivariateEmbedding(...)
        self.transformer = nn.Transformer(...)
        self.structure_loss = StructuralLoss(...)
        
    def forward(self, x):
        # Apply compound embeddings
        # Enforce LOB structure in loss
```

## 📈 **Performance Expectations**

### Expected Performance Improvements

| Metric | Spacetimeformer | Our LOB Model | Improvement |
|--------|-----------------|---------------|-------------|
| Total Loss | ~0.015 | <0.008 | ~47% better |
| Structure Loss | N/A | <0.15 | LOB structure preserved |
| Mid-price MSE | Variable | <0.002 | Competitive |
| Training Speed | Slower | Faster | Linear attention |
| Parameter Count | Higher | Lower | Compound embeddings |

### Model Comparison Table (From Paper)

| Model | Mid-price MSE | Mid-price MAE | Price MSE | Volume MSE | Structure Loss | Total Loss |
|-------|---------------|---------------|-----------|------------|----------------|------------|
| Spacetimeformer | 0.0122 | 0.1425 | 0.0025 | 0.0105 | 0.5774 | 0.0123 |
| **Our LOB Model** | **0.0019** | **0.1361** | **0.0025** | **0.0105** | **0.1409** | **0.0079** |

## 🚀 **Migration Strategy**

### 1. **Use Existing Spacetimeformer Components**
- [ ] Base transformer architecture
- [ ] Positional encoding
- [ ] Attention mechanisms (upgrade to Performer)
- [ ] Training utilities

### 2. **Implement LOB-Specific Components**
- [ ] Compound multivariate embedding
- [ ] Structural regularizer
- [ ] LOB data preprocessing
- [ ] Session-aware data loading

### 3. **Training Configuration Updates**
- [ ] Update hyperparameters for LOB specifics
- [ ] Implement combined loss function
- [ ] Add LOB-specific evaluation metrics
- [ ] Enable seasonal decomposition

## 📊 **Implementation Priority**

### High Priority (Core Differences)
1. **Compound Multivariate Embedding** - Essential for capturing LOB structure
2. **Structural Regularizer** - Critical for maintaining price ordering
3. **Percent-change Transformation** - Necessary for stationarity
4. **Session-aware Data Loading** - Prevents crossing session boundaries

### Medium Priority (Performance Improvements)
1. **Performer Attention** - Better scalability
2. **Combined Loss Function** - Better training stability
3. **Enhanced Evaluation Metrics** - Better model assessment

### Low Priority (Nice-to-Have)
1. **Seasonal Decomposition** - Additional performance boost
2. **Advanced Regularization** - Fine-tuning improvements
3. **Model Ensemble** - Further performance gains

## 🎯 **Success Criteria**

### Model Performance
- [ ] Total loss < 0.008 (beat Spacetimeformer)
- [ ] Structure loss < 0.15 (maintain LOB ordering)
- [ ] Mid-price MSE < 0.002 (competitive accuracy)
- [ ] Training convergence within 50 epochs

### Implementation Quality
- [ ] Code reuses Spacetimeformer components where possible
- [ ] Clear separation of LOB-specific vs. generic components
- [ ] Comprehensive testing of compound embeddings
- [ ] Validation of structural regularizer

## 📝 **Key Takeaways**

1. **Our LOB model is NOT a replacement for Spacetimeformer** - it's a specialized adaptation
2. **Compound embeddings are the key innovation** - they reduce parameters while better capturing LOB structure
3. **Structural regularizer is crucial** - it ensures predictions maintain valid LOB properties
4. **Financial data preprocessing matters** - percent-change transformation is essential for stationarity
5. **Session awareness is important** - don't cross trading session boundaries in sequences

The result should be a model that significantly outperforms the general Spacetimeformer on LOB forecasting tasks while maintaining the flexibility and power of the attention mechanism. 