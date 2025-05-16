# Attention-Based Reading, Highlighting, and Forecasting of the Limit Order Book
*Reference: arXiv:2409.02277v1 [q-fin.CP] 03 Sep 2024*

## License
CC BY-NC-ND 4.0

## Table of Contents
- [Introduction](#introduction)
- [Data Structure and Preparation](#data-structure-and-preparation)
- [Stationary Transformation and Scaling](#stationary-transformation-and-scaling)
- [Compound Multivariate Embedding](#compound-multivariate-embedding)
- [Training Methodology](#training-methodology)
- [Model Performance](#model-performance)
- [Implementation Notes](#implementation-notes)
- [References](#references)

## Introduction

A limit order book (LOB) is a digital record of all outstanding buy and sell orders for a particular financial instrument. It contains information about the price and quantity of each order, as well as the time at which the order was placed. The LOB continuously updates as new orders are submitted, executed, or canceled, providing an up-to-the-moment snapshot of market supply and demand.

Predictive modeling of LOBs presents considerable challenges due to the high frequency and complexity of the data. The dynamic nature of the LOB requires rapid processing and analysis of vast amounts of information. This implementation focuses on using attention-based models to forecast the entire multi-level LOB, including order prices and volumes.

## Data Structure and Preparation

### LOB Data Structure

The order book consists of two types of orders: bids (to buy) and asks (to sell). Each snapshot of the LOB data is represented as:

$X_i^{\rightarrow} = [t_i, (p_i^{1b}, v_i^{1b}), \ldots, (p_i^{Kb}, v_i^{Kb}), (p_i^{1a}, v_i^{1a}), \ldots, (p_i^{Ka}, v_i^{Ka})]$

Where:
- $t_i$ is the timestamp
- $p_i^{kb/a}$ represents the level-$k$ bid/ask price
- $v_i^{kb/a}$ represents the level-$k$ bid/ask volume
- $K$ is the number of levels (fixed at 5 in the implementation)

### Data Preparation Steps

1. **Standardize timestamps** to equal intervals (e.g., 5 seconds)
2. **Create context and target sequences**:
   - Context: 120 time steps (10 minutes)
   - Target: 24 time steps (2 minutes)
3. **Dataset split**: Training, validation, and testing in a 6:2:2 ratio

## Stationary Transformation and Scaling

### Percent-change Transformation for Prices

To address non-stationarity in price data:

$p_{k,i}^{perc} = \frac{p_{k,i} - p_{k,i-1}}{p_{k,i-1}}$

### Min-max Scaling

For both price and volume:

$p_{k,i}^{scaled} = \frac{p_{k,i}^{perc} - \min_i(p_{k,i}^{perc})}{\max_i(p_{k,i}^{perc}) - \min_i(p_{k,i}^{perc})}$

$v_{k,i}^{scaled} = \frac{v_{k,i} - \min_i(v_{k,i})}{\max_i(v_{k,i}) - \min_i(v_{k,i})}$

## Compound Multivariate Embedding

### Embedding Components

1. **Time Embedding**: Using Time2Vec to transform timestamps into frequency embeddings
2. **Context-Target Embedding**: Binary encoding to distinguish observed vs. future data points
3. **Compound Attribute Embedding**:
   - Level embedding (level-1, level-2, etc.)
   - Type embedding (bid or ask)
   - Feature embedding (price or volume)
   - Stock embedding (ticker)

### Attention Mechanism

The attention formula:

$\text{Attention}(Q, K, V) \in \mathbb{R}^{L_x \times d} = \text{softmax}(\frac{QK^T}{\sqrt{d}})V$

Where:
- Query vectors $Q = W_Q X$
- Key vectors $K = W_K Z$
- Value vectors $V = W_V Z$
- $L$ is sequence length
- $d$ is dimension

## Training Methodology

### Loss Functions

1. **Forecasting Loss** (MSE):

$\text{Forecasting Loss} = \sum_{i,j,k} (p_{k,i}^j - \hat{p}_{k,i}^j)^2 + \sum_{i,j,k} (v_{k,i}^j - \hat{v}_{k,i}^j)^2$

2. **Structure Loss** (for preserving ordinal price structure):

$\text{Structure loss}_i = \sum_{k=1}^{K-1} (ReLU({\hat{p}_{k,i}^a - \hat{p}_{k+1,i}^a}) + ReLU({\hat{p}_{k+1,i}^b - \hat{p}_{k,i}^b})) + ReLU({\hat{p}_{1,i}^b - \hat{p}_{1,i}^a})$

3. **Total Loss**:

$\text{Loss} = \text{Forecasting loss} + w_o \cdot \sum_{i=1}^{I} \text{Structure loss}_i$

Where $w_o = 0.01$ is the regularization weight.

### Training Parameters

- Learning rate decay factor: 0.8
- Warmup steps: 1000
- Attention heads: 3
- Early stopping: After 10 epochs without improvement

## Model Performance

### Error Metrics for Different Models

| Model | Mid-price MSE | Mid-price MAE | Price MSE | Volume MSE | Structure Loss | Total Loss |
|-------|---------------|---------------|-----------|------------|----------------|------------|
| Linear | 0.0125 | 0.1501 | 0.0026 | 0.0105 | 0.2430 | 0.0090 |
| LSTM | 0.0019 | 0.1369 | 0.0025 | 0.0102 | 0.2732 | 0.0091 |
| Temporal | 0.0124 | 0.1520 | 0.0027 | 0.0109 | 0.8836 | 0.0157 |
| Spacetime | 0.0122 | 0.1425 | 0.0025 | 0.0105 | 0.5774 | 0.0123 |
| Compound (ours) | 0.0019 | 0.1361 | 0.0025 | 0.0105 | 0.1409 | 0.0079 |

## Implementation Notes

### Key Libraries for Implementation

- TensorFlow or PyTorch for neural network implementation
- pandas for data manipulation
- numpy for numerical operations
- matplotlib for visualization

### Multi-level LOB Structure Constraints

For any fixed LOB snapshot:
1. $p_{k1,i}^a < p_{k2,i}^a$ for all $k1 < k2$ (ask prices increase with level)
2. $p_{k1,i}^b < p_{k2,i}^a$ for $k1 = k2 = 1$ (best bid < best ask)
3. $p_{k1,i}^b > p_{k2,i}^b$ for all $k1 < k2$ (bid prices decrease with level)

### Data Collection Resources

For implementation, consider using:
1. LOBSTER dataset (Huang and Polak, 2011)
2. [arxiv-dl](https://github.com/MarkHershey/arxiv-dl) for accessing additional research papers

## References

1. Arroyo, A., Cartea, A., Moreno-Pino, F., Zohren, S. (2024). Deep attentive survival analysis in limit order books: Estimating fill probabilities with convolutional-transformers. Quantitative Finance 24, 35–57.
2. Vaswani, A., et al. (2017). Attention is all you need. Advances in neural information processing systems 30.
3. Grigsby, J., Wang, Z., Qi, Y. (2022). Long-Range Transformers for Dynamic Spatiotemporal Forecasting.
4. Choromanski, K., et al. (2020). Rethinking attention with performers. arXiv preprint arXiv:2009.14794.
5. Huang, R., Polak, T. (2011). Lobster: Limit order book reconstruction system.
6. Zhou, H., et al. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. 