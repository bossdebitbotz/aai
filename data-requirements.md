# Data Requirements for Multi-Exchange LOB Forecasting Models

This document outlines the specific data requirements needed to train the attention-based LOB forecasting models across Binance Perpetual, Binance Spot, and Bybit Spot markets.

## Data Structure Requirements

### Basic LOB Data Structure

For each exchange and market, we need to collect and maintain:

```
X_i = [t_i, (p_i^{1b}, v_i^{1b}), ..., (p_i^{5b}, v_i^{5b}), (p_i^{1a}, v_i^{1a}), ..., (p_i^{5a}, v_i^{5a})]
```

Where:
- `t_i` is the timestamp (in milliseconds)
- `p_i^{kb}` is the level-k bid price
- `v_i^{kb}` is the level-k bid volume
- `p_i^{ka}` is the level-k ask price
- `v_i^{ka}` is the level-k ask volume
- We collect 5 levels of depth (k = 1 to 5)

### Sampling Frequency

- **Raw Data Collection**: Collect at the highest frequency possible (ideally every update)
- **Model Training Data**: Resample to 5-second intervals for standardization
- **Inference Data**: Use the same 5-second intervals for consistency

### Time Period Requirements

- **Minimum Historical Data**: 3 months for initial model training
- **Optimal Historical Data**: 6+ months covering various market conditions
- **Training/Validation/Test Split**: 60%/20%/20% chronological split

## Data Quality Requirements

### Completeness Checks

- Less than 0.1% missing data points
- No gaps longer than 5 minutes
- Continuous 24/7 coverage

### Synchronization Requirements

- Timestamps across exchanges must be aligned to within 100ms
- All exchange data must be normalized to UTC time

### Data Cleaning Requirements

1. **Outlier Detection and Handling**:
   - Identify and flag price/volume outliers (>3 standard deviations)
   - Apply rolling median filter for extreme outliers

2. **Missing Data Handling**:
   - For short gaps (<30s): Linear interpolation
   - For longer gaps: Forward-fill prices, zero-fill volumes

## Data Transformation Pipeline

### Pre-processing Steps

1. **Time Standardization**:
   - Convert all timestamps to UNIX milliseconds
   - Resample to 5-second intervals using last observation

2. **Stationary Transformations**:
   - For prices: Apply percent-change transformation
     ```
     p_{k,i}^{perc} = (p_{k,i} - p_{k,i-1})/p_{k,i-1}
     ```
   - For volumes: Use raw values (will be scaled later)

3. **Scaling**:
   - Min-max scaling for both transformed prices and raw volumes
     ```
     x_scaled = (x - min(x))/(max(x) - min(x))
     ```
   - Calculate scaling factors on training data only
   - Store scaling parameters for inference

4. **Cross-Exchange Normalization**:
   - Convert all prices to USD equivalent if trading different quote currencies
   - Normalize volumes based on typical market depth for each exchange

### Sequence Preparation

1. **Context Window**: 120 time steps (10 minutes at 5-second intervals)
2. **Prediction Horizon**: 24 time steps (2 minutes)
3. **Sliding Window**: Create training samples with 50% overlap

## Feature Engineering

### Base Features (per exchange, per level)

- Bid prices (5 levels)
- Ask prices (5 levels)
- Bid volumes (5 levels)
- Ask volumes (5 levels)

### Derived Features

1. **Price-based**:
   - Mid-price: `(p_i^{1b} + p_i^{1a})/2`
   - Spread: `p_i^{1a} - p_i^{1b}`
   - Price imbalance: `(p_i^{1a} - p_i^{1b})/(p_i^{1a} + p_i^{1b})`

2. **Volume-based**:
   - Volume imbalance: `(v_i^{1b} - v_i^{1a})/(v_i^{1b} + v_i^{1a})`
   - Cumulative volume (bid): `sum(v_i^{kb}) for k=1 to 5`
   - Cumulative volume (ask): `sum(v_i^{ka}) for k=1 to 5`

3. **Cross-exchange**:
   - Price differentials between exchanges
   - Volume ratio between exchanges
   - Spread ratio between exchanges

## Data Storage Format

### Raw Data Storage

- **Format**: Parquet files with partitioning by:
  - Exchange
  - Symbol
  - Date
- **Schema**: Timestamp (index), bid prices, bid volumes, ask prices, ask volumes
- **Compression**: Snappy compression

### Processed Data Storage

- **Format**: TFRecord or PyTorch tensors
- **Structure**: 
  - Input tensors: [batch_size, 120, features]
  - Target tensors: [batch_size, 24, features]
- **Metadata**: JSON files with scaling parameters and feature names

## Models to be Trained

1. **Single-Exchange Models**:
   - One model per exchange-symbol pair
   - Used for baseline performance evaluation

2. **Multi-Exchange Model**:
   - Combined model with exchange embedding
   - Learns cross-exchange patterns and correlations

3. **Specialized Models**:
   - Mid-price prediction model
   - Volume imbalance prediction model
   - Spread prediction model

## Evaluation Metrics

1. **Accuracy Metrics**:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Directional Accuracy

2. **Structure Preservation**:
   - Order book structure loss
   - Bid-ask relationship preservation

3. **Trading Performance**:
   - Simulated P&L on validation data
   - Sharpe ratio
   - Maximum drawdown

## Data Pipeline Implementation

The data pipeline should be implemented in C++ for performance, with the following components:

1. **Data Collectors**: Exchange-specific WebSocket clients
2. **Data Processor**: Unified LOB constructor and normalizer
3. **Feature Generator**: Transforms raw LOB data into model features
4. **Data Exporter**: Creates training datasets for Python ML components

## Monitoring Requirements

1. **Data Quality Monitoring**:
   - Real-time checks for data gaps
   - Outlier detection alerts
   - Exchange connectivity status

2. **Storage Monitoring**:
   - Disk usage tracking
   - Compression ratio monitoring
   - Read/write performance metrics

## Additional Requirements

### Symbol Selection and Coverage

1. **Trading Pair Selection Criteria**:
   - Minimum daily volume requirements (>$50M)
   - Maximum spread thresholds (<0.05%)
   - Sufficient historical data availability (>6 months)
   - Consistent 24/7 trading activity

2. **Symbol Management**:
   - Quarterly review of symbol inclusion/exclusion
   - Procedure for adding new symbols (2-week observation period)
   - Gradual phase-out for removed symbols (maintain data collection for 1 month)
   - Primary focus on BTC, ETH, and top 10 altcoins by market cap

3. **Model Allocation Strategy**:
   - Tier 1 symbols: Individual dedicated models
   - Tier 2 symbols: Grouped models by correlation clusters
   - All symbols: Inclusion in global market model

### Hardware Storage Requirements

1. **Data Volume Estimates**:
   - Raw LOB data: ~2GB per symbol per day
   - Processed features: ~500MB per symbol per day
   - Model checkpoints: ~200MB per model version
   - Total annual requirement: ~1TB per symbol

2. **Retention Policies**:
   - Raw data: 30 days online storage, 1 year archived
   - Processed features: 90 days online, 2 years archived
   - Training datasets: Permanent storage with versioning
   - Model artifacts: All versions retained indefinitely

3. **Backup Procedures**:
   - Daily incremental backups of all data
   - Weekly full backups to off-site storage
   - Monthly verification of backup integrity
   - Disaster recovery testing quarterly

### Data Validation Rules

1. **Price Validation**:
   - Valid range: ±30% of 24h moving average
   - Maximum allowed gap between levels: 5% for normal markets
   - Cross-exchange price deviation alerts: >1% for major pairs

2. **Volume Validation**:
   - Minimum volume thresholds per level
   - Maximum allowed sudden changes: 500% increase/90% decrease
   - Zero-volume level detection and handling

3. **Structural Validation**:
   - Bid/ask crossing detection
   - Level inversion detection
   - Timestamp sequence validation
   - Order book depth consistency checks

4. **Automated Data Quality Scoring**:
   - Completeness score (% of expected data points)
   - Accuracy score (based on validation rules)
   - Consistency score (temporal stability of features)
   - Overall quality score with minimum threshold for model inclusion

### Exchange-Specific Considerations

1. **API Limitations**:
   - Binance: 1200 requests/minute per IP
   - Bybit: 120 requests/minute per IP
   - Connection pooling requirements
   - IP rotation strategy for high-frequency collection

2. **Order Book Representation Differences**:
   - Binance: Price level aggregation method
   - Bybit: Treatment of hidden liquidity
   - Normalization procedures for each exchange

3. **Exchange-Specific Data Quirks**:
   - Binance: Handling of iceberg/hidden orders
   - Bybit: Market maker protection mechanisms
   - Maintenance window patterns and handling

4. **Authentication and Security**:
   - API key rotation schedule
   - IP whitelisting requirements
   - Secure credential storage

### Latency Measurement

1. **Latency Metrics**:
   - Exchange-to-server latency (ping time)
   - Data processing latency (receipt to storage)
   - End-to-end latency (exchange event to feature availability)
   - Clock synchronization methodology

2. **Latency Thresholds**:
   - Critical threshold: >500ms for any component
   - Warning threshold: >200ms for any component
   - Target average: <100ms end-to-end

3. **High-Latency Procedures**:
   - Automated fallback to secondary data sources
   - Graceful degradation of feature complexity
   - Circuit breaker implementation for extreme cases

### Market Hours Handling

1. **Exchange Maintenance Periods**:
   - Scheduled maintenance detection
   - Data gap annotation
   - Model inference adjustment during maintenance

2. **Market Volatility Events**:
   - Circuit breaker detection
   - Flash crash identification
   - Special handling of post-halt data

3. **Daily Cycle Management**:
   - UTC day boundary handling
   - Weekend vs. weekday pattern differences
   - Trading session open/close effects

### Data Versioning

1. **Dataset Versioning Schema**:
   - Major.Minor.Patch format
   - Major: Significant feature changes
   - Minor: New symbols or exchanges
   - Patch: Bug fixes or small improvements

2. **Version Control Integration**:
   - Git LFS for feature datasets
   - Metadata storage with each version
   - Compatibility matrix with model versions

3. **Model-Data Relationship**:
   - Explicit mapping of model versions to data versions
   - Automated compatibility checking
   - Version dependency documentation

### Legal and Compliance

1. **Exchange Terms Compliance**:
   - Documentation of relevant TOS sections
   - Rate limiting compliance
   - Commercial usage restrictions

2. **Data Retention Requirements**:
   - Exchange-specific data retention policies
   - Regulatory considerations by jurisdiction
   - GDPR compliance (if applicable)

3. **Audit Trail**:
   - Complete logging of data access
   - Immutable record of data transformations
   - Chain of custody documentation

### Correlation Analysis Requirements

1. **Cross-Exchange Correlation Metrics**:
   - Price correlation coefficients (Pearson, Spearman)
   - Volume imbalance correlation
   - Spread correlation
   - Lead-lag relationship measurements

2. **Correlation Storage**:
   - Time-series database for correlation metrics
   - Correlation matrix snapshots (hourly)
   - Long-term correlation trend storage

3. **Correlation Monitoring**:
   - Sudden correlation breakdown alerts
   - Correlation regime change detection
   - Visualization of correlation networks

### Feature Stability Monitoring

1. **Feature Drift Detection**:
   - Kullback-Leibler divergence monitoring
   - Statistical distribution tests (daily)
   - Feature importance stability metrics

2. **Retraining Triggers**:
   - Automatic: Feature drift exceeding thresholds
   - Scheduled: Monthly regardless of drift
   - Manual: After significant market regime changes

3. **Feature Evolution Tracking**:
   - Historical feature distribution archives
   - Feature correlation stability metrics
   - Documentation of feature definition changes
