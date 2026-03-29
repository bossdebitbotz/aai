# State-of-the-Art LOB Forecasting Architectures (Late 2024 - Early 2026)

*Research compiled March 2026. Baseline: Compound Attention Model (arXiv:2409.02277, Sep 2024)*

---

## Table of Contents
- [1. LOB-Specific Models](#1-lob-specific-models)
- [2. State-Space Models (SSMs) for Financial/LOB Data](#2-state-space-models-ssms-for-financiallob-data)
- [3. Advanced Transformer Variants for LOB](#3-advanced-transformer-variants-for-lob)
- [4. Diffusion Models for LOB](#4-diffusion-models-for-lob)
- [5. Graph Neural Network Approaches](#5-graph-neural-network-approaches)
- [6. Foundation Models for Time Series](#6-foundation-models-for-time-series)
- [7. Hybrid Architectures](#7-hybrid-architectures)
- [8. Benchmarking Frameworks](#8-benchmarking-frameworks)
- [9. Key Insights and Recommendations](#9-key-insights-and-recommendations)

---

## 1. LOB-Specific Models

### 1.1 TLOB: Transformer with Dual Attention for LOB
- **Paper:** "TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data"
- **Authors:** Leonardo Berti, Gjergji Kasneci
- **Date:** February 2025 | **arXiv:** 2502.15757
- **Code:** https://github.com/LeonardoBerti00/TLOB

**Architecture:**
- Dual attention mechanism: Temporal Self-Attention + Feature Self-Attention within each TLOB block
- Captures both time-wise and spatial (cross-feature) relationships in LOB data
- Adaptively focuses on market microstructure features

**Performance:**
- Outperforms SoTA on FI-2010 benchmark by average +3.7 F1-score across four horizons
- +1.3 F1 improvement on Tesla, +7.7 on Intel (NASDAQ)
- +1.1 F1 on Bitcoin LOB dataset
- Shows adapting a simple MLP-based architecture to LOB can also surpass SoTA (important insight)

**Relevance to our use case:**
- Directly handles multi-level LOB data with spatial attention
- Focuses on mid-price trend classification rather than full LOB reconstruction (our baseline does both)
- Dual attention approach could complement our compound embedding -- Feature Self-Attention is conceptually similar to our cross-feature modeling
- Well-tested on crypto data (Bitcoin)

---

### 1.2 LiT: Limit Order Book Transformer
- **Paper:** "LiT: Limit Order Book Transformer"
- **Authors:** Xiao, Ventre, Wang, Li, Huan, Liu
- **Date:** October 2025 | **Journal:** Frontiers in Artificial Intelligence
- **DOI:** 10.3389/frai.2025.1616485

**Architecture:**
- Three-component design:
  1. Linear projection + positional embeddings for structured LOB patches
  2. Transformer layers with self-attention for spatial-temporal dependencies
  3. LSTM layers for long-term temporal dependencies
- Structured patch-based tokenization of LOB data (not pixel-level or raw features)
- Replaces CNN-based feature extraction (DeepLOB) with patch + attention

**Performance:**
- Outperforms DeepLOB, TransLOB, and traditional ML baselines
- Robust under distributional shifts via fine-tuning
- Competitive F1 vs DeepLOB across multiple horizons

**Relevance to our use case:**
- Patch-based LOB representation is an alternative to our per-level embedding
- Hybrid transformer+LSTM could be interesting -- LSTM for long-range, attention for local patterns
- Demonstrated robustness to distribution shift is valuable for live trading
- Does NOT do full LOB reconstruction (only direction classification)

---

### 1.3 HLOB: Homological CNN for LOB
- **Paper:** "HLOB -- Information Persistence and Structure in Limit Order Books"
- **Authors:** Antonio Briola, Silvia Bartolucci, Tomaso Aste
- **Date:** May 2024 | **Journal:** Expert Systems with Applications
- **arXiv:** 2405.18938

**Architecture:**
- Uses Information Filtering Networks (Triangulated Maximally Filtered Graph) to uncover dependency structures among volume levels
- Inspired by Homological Convolutional Neural Networks
- Models deeper, non-trivial relationships between LOB levels that standard CNNs miss

**Performance:**
- Tested against 9 SoTA deep learning models on 3 NASDAQ LOB datasets (15 stocks each)
- Outperforms alternatives particularly when spatial (cross-level) dependencies matter

**Relevance to our use case:**
- Topological approach to modeling LOB structure is fundamentally different from attention
- Could complement our compound embedding by revealing hidden cross-level dependencies
- Focuses on mid-price movement classification, not full LOB reconstruction
- Heavy on graph-theoretic preprocessing

---

### 1.4 LOBERT: Foundation Model for LOB Messages
- **Paper:** "LOBERT: Generative AI Foundation Model for Limit Order Book Messages"
- **Authors:** Eljas Linna, Kestutis Baltakys, Alexandros Iosifidis, Juho Kanniainen
- **Date:** November 2025 | **arXiv:** 2511.12563

**Architecture:**
- Encoder-only BERT adaptation for LOB message-level data
- Novel tokenization: treats complete multi-dimensional LOB messages as single tokens while retaining continuous price/volume/time representations
- Piecewise Linear-Geometric scaling for price/volume, avoiding the token explosion of prior generators (22-24 sub-tokens per message reduced to 1)
- Reduces effective context length by ~10x compared to prior approaches

**Performance:**
- Leading performance on mid-price movement prediction and next-message prediction
- Reduced context length requirement vs previous methods

**Relevance to our use case:**
- Foundation model approach means it could potentially be fine-tuned for our specific task
- Message-level modeling (order flow) rather than snapshot-level -- fundamentally different paradigm
- Could be adapted for multi-exchange by treating exchange as part of the message embedding
- Pre-training on large LOB corpora could provide useful representations even for our snapshot-based approach

---

### 1.5 Crypto LOB Study: Better Inputs Matter More
- **Paper:** "Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books: Better Inputs Matter More Than Stacking Another Hidden Layer"
- **Authors:** Haochuan Wang
- **Date:** June 2025 | **arXiv:** 2506.05764

**Key Findings:**
- Benchmarks models from logistic regression to DeepLOB on BTC/USDT LOB snapshots (Bybit, 100ms-multi-second intervals)
- Introduces Kalman and Savitzky-Golay filtering pipelines for LOB data
- **Critical insight:** Input features and hyperparameters (prediction horizon, LOB depth) have greater impact than model complexity
- Simpler models with good preprocessing match or exceed deep architectures
- Models train in a fraction of the time, more practical for real-time use

**Relevance to our use case:**
- Directly tested on crypto LOB data (Bybit BTC/USDT -- similar to our data sources)
- Validates importance of our data preprocessing pipeline
- Suggests we should invest heavily in feature engineering (OFI, Kalman filtering) alongside architecture
- Order flow and OFI features outperform raw order book snapshots in most cases

---

## 2. State-Space Models (SSMs) for Financial/LOB Data

### 2.1 CryptoMamba
- **Paper:** "CryptoMamba: Leveraging State Space Models for Accurate Bitcoin Price Prediction"
- **Authors:** Mohammad Shahab Sepehri, Asal Mehradfar, Mahdi Soltanolkotabi, Salman Avestimehr
- **Date:** January 2025 | **arXiv:** 2501.01010
- **Venue:** IEEE International Conference on Blockchain and Cryptocurrency (ICBC) 2025

**Architecture:**
- Mamba-based SSM designed for long-range dependencies in financial time series
- Volume-inclusive variant that incorporates trading volume features

**Performance:**
- Outperforms LSTM, Bi-LSTM, GRU, and S-Mamba in predictive accuracy
- Volume-inclusive variant achieves highest returns in real-world trading scenarios
- Effective capture of regime shifts and long-range dependencies

**Relevance to our use case:**
- O(n) complexity vs O(n^2) for attention -- potentially much faster for long sequences
- Tested on crypto data specifically
- However, operates on price-level time series, NOT multi-level LOB data
- Would need significant adaptation to handle our 5-level bid/ask structure
- No structure loss or ordinal constraint enforcement

---

### 2.2 CMDMamba: Dual-Layer Mamba for Financial Time Series
- **Paper:** "CMDMamba: dual-layer Mamba architecture with dual convolutional feed-forward networks for efficient financial time series forecasting"
- **Date:** July 2025 | **Journal:** Frontiers in Artificial Intelligence

**Architecture:**
- Dual-layer Mamba: captures price fluctuations at micro- and macro-levels
- Dual Convolutional Feedforward Network (DconvFFN) for multi-dimensional feature learning
- Channel Linear Layer for variable decoupling
- TNF Encoding Layer for convergence stability
- Near-linear time complexity

**Performance:**
- MSE reductions: -11.6% (DAX), -1.5% (DJI), -0.4% (HSI), -10.2% (S&P500)
- Consistently outperforms most baselines

**Relevance to our use case:**
- DconvFFN for multi-dimensional features is promising for LOB's multivariate structure
- Dual-layer design (micro/macro) maps well to short-term tick dynamics vs longer patterns
- Not LOB-specific but the architecture ideas are transferable
- Linear complexity makes it suitable for high-frequency inference

---

### 2.3 MambaTS: Improved SSM for Long-term Time Series
- **Paper:** "MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting"
- **Authors:** Xiongxiao Xu et al.
- **Date:** May 2024 | **arXiv:** 2405.16440

**Architecture:**
- Variable Scan along Time (VST): arranges historical information of all variables together
- Temporal Mamba Block (TMB): removes causal convolution (not needed for forecasting)
- Variable-Aware Scan along Time (VAST): discovers inter-variable relationships and optimizes scan order
- Dropout on selective parameters to prevent overfitting

**Performance:**
- SoTA on 8 public datasets for long-term forecasting

**Relevance to our use case:**
- VAST's automatic variable relationship discovery could model LOB level interactions
- VST's multi-variable arrangement is relevant for our multi-level LOB input
- Would need adaptation for the structured nature of LOB data (ordinal price levels)

---

### 2.4 FinMamba: Market-Aware Graph-Enhanced Mamba
- **Paper:** "FinMamba: Market-Aware Graph Enhanced Multi-Level Mamba for Stock Movement Prediction"
- **Date:** February 2025 | **arXiv:** 2502.06707

**Architecture:**
- Combines dynamic graph learning with multi-level Mamba
- Pruning module adapts graph to market trends
- Multi-level state space mechanism for learning patterns across random intervals at different scales
- Selective mechanism discards irrelevant information and resets states

**Performance:**
- SoTA on CSI, NASDAQ, S&P indices
- Low computational complexity maintained

**Relevance to our use case:**
- Graph component models inter-stock relationships (could model cross-exchange/cross-asset)
- Multi-level design naturally fits our multi-level LOB data
- Dynamic graph pruning is useful for adapting to regime changes

---

### Key SSM Assessment for LOB

**Advantages:**
- O(n) complexity vs O(n^2) for full attention
- Up to 5x inference throughput improvement
- Handle very long sequences (million-token in some settings)
- Good at capturing regime shifts and long-range dependencies

**Disadvantages:**
- No LOB-specific SSM exists yet with head-to-head comparison vs LOBERT/LiT on identical LOB data
- Most comparisons are against LSTM/vanilla Transformer, not LOB-specific attention models
- Lack built-in mechanism for ordinal structure preservation (our structure loss)
- Limited evidence on structured multivariate inputs like LOB

---

## 3. Advanced Transformer Variants for LOB

### 3.1 CVML: Convolutional Cross-Variate Mixing Layers (ICLR 2025 Submission)
- **Paper:** "A Benchmark Study For Limit Order Book (LOB) Models and Time Series Forecasting Models on LOB Data"
- **Venue:** OpenReview / ICLR 2025

**Architecture:**
- CVML is a plug-in module that can be added to ANY deep learning multivariate time series model
- Uses convolutional layers to capture cross-variate (cross-feature) dependencies in LOB data
- Bridges the gap between general time series models and LOB-specific models

**Performance:**
- Average improvement of 244.9% in mid-price return forecasting when added to various time series models
- Tested on proprietary futures LOB dataset

**Relevance to our use case:**
- Could potentially be added on top of our Compound Attention Model as an enhancement
- Addresses the specific challenge of modeling cross-variate dependencies in LOB
- The 244.9% improvement suggests general time series models severely underperform without cross-variate mixing
- Validates our approach of using compound embeddings for multivariate LOB data

---

### 3.2 mWDN-Transformer: Wavelet Decomposition + Transformer
- **Paper:** "mWDN-Transformer: Utilizing Cyclical Patterns for Short-Term Forecasting of Limit Order Book in China Markets"
- **Venue:** CNIOT 2024

**Architecture:**
- Multi-level wavelet decomposition network (mWDN) + Transformer
- mWDN captures short-term and long-term dependencies via multi-scale decomposition
- Transformer models long-range temporal relationships within decomposed series

**Performance:**
- Superior at predicting extreme price movements and trading activity levels
- Tested on China stock market LOB data

**Relevance to our use case:**
- Wavelet decomposition for multi-scale analysis is interesting for 5s-interval LOB data
- Could capture both tick-level noise patterns and longer-term trends
- Not directly tested on crypto markets

---

## 4. Diffusion Models for LOB

### 4.1 "Painting the Market": Diffusion Inpainting for LOB
- **Paper:** "Painting the market: generative diffusion models for financial limit order book simulation and forecasting"
- **Date:** September 2025 | **arXiv:** 2509.05107

**Architecture:**
- Converts LOB data into structured image format
- Applies diffusion model with inpainting to generate future LOB states
- LOB history serves as the "unnoised region" and future states as the "noised region to fill"
- Parallel generation of long sequences (overcomes error accumulation)

**Performance:**
- SoTA on LOB-Bench despite using lower-fidelity data as input
- Leverages spatio-temporal inductive biases inherent in LOB structure

**Relevance to our use case:**
- **Highly relevant** -- this does full LOB state forecasting, like our baseline
- Image-based representation captures the 2D structure of the order book naturally
- Parallel generation avoids autoregressive error accumulation (a known issue)
- Could be used for LOB reconstruction AND simulation
- Slower inference than attention-based models (diffusion sampling requires multiple denoising steps)

---

### 4.2 TRADES: Transformer-based Diffusion for LOB Simulation
- **Paper:** "TRADES: Generating Realistic Market Simulations with Diffusion Models"
- **Authors:** Leonardo Berti, Bardh Prenkaj, Paola Velardi
- **Date:** February 2025 | **arXiv:** 2502.07071
- **Code:** https://github.com/LeonardoBerti00/DeepMarket

**Architecture:**
- Transformer-based denoising diffusion probabilistic model (DDPM)
- Generates realistic order flows as time series conditioned on market state
- Captures temporal and spatial characteristics of high-frequency market data

**Performance:**
- 3.27x and 3.48x improvement over SoTA on predictive score
- Learns conditional data distribution and reacts to experimental agents

**Relevance to our use case:**
- More focused on simulation/generation than point forecasting
- Useful for backtesting, strategy evaluation, and synthetic data generation
- Comes with DeepMarket framework for LOB simulation
- Not directly comparable to our forecasting task but valuable for evaluation

---

### 4.3 LOBDIF: Diffusion for LOB Event Stream Prediction
- **Paper:** "Limit Order Book Event Stream Prediction with Diffusion Model"
- **Date:** December 2024 (arXiv), published February 2026 in Data Science and Engineering (Springer)
- **arXiv:** 2412.09631
- **Code:** https://github.com/zhengzetao/LOBDIF

**Architecture:**
- Learns complex time-event distribution by decomposing into sequential Gaussian steps
- Denoising network for time-event interdependence
- Skip-step sampling strategy for faster inference
- Predicts both timing and type of LOB events

**Performance:**
- Significantly outperforms current SoTA on three widely traded assets

**Relevance to our use case:**
- Event-level prediction (not snapshot-level like ours)
- Could complement our approach by predicting WHEN significant events occur
- Skip-step sampling addresses the slow inference problem of diffusion models

---

### 4.4 DiffLOB: Counterfactual LOB Generation
- **Paper:** "DiffLOB: Diffusion Models for Counterfactual Generation in Limit Order Books"
- **Date:** February 2026 | **arXiv:** 2602.03776
- **Code:** https://github.com/ZhuoHan1998/DiffLOB

**Architecture:**
- Regime-conditioned diffusion model
- Conditions on future market regime: trend, volatility, liquidity, order-flow imbalance
- Answers "what if" queries about LOB evolution under different conditions

**Performance:**
- Evaluated on: controllable realism, counterfactual validity, counterfactual usefulness
- Synthetic counterfactual data improves downstream prediction of future market regimes

**Relevance to our use case:**
- Not a direct forecasting model but extremely valuable for stress testing
- Could generate training data for rare market conditions
- Regime-conditioning concept could enhance our model's robustness

---

### 4.5 DiffVolume: Volume Generation for LOB
- **Paper:** "DiffVolume: Diffusion Models for Volume Generation in Limit Order Books"
- **Authors:** Zhuohan Wang, Carmine Ventre
- **Date:** August 2025 | **arXiv:** 2508.08698
- **Venue:** ICAIF 2025

**Architecture:**
- Conditional diffusion for future LOB volume snapshots
- 32 residual convolutional layers (WaveNet-style dilated convolutions)
- Soft gating mechanism (tanh + sigmoid)
- Conditioned on past volume history and time of day

**Performance:**
- Strong realism in marginal distribution, spatial correlation, autocorrelation
- Enables controllable generation under hypothetical liquidity scenarios

**Relevance to our use case:**
- Directly generates LOB volume profiles -- half of our forecasting target
- WaveNet-style architecture could inspire our volume modeling component
- Volume forecasting is critical for execution quality estimation

---

### 4.6 CoFinDiff: Controllable Financial Diffusion
- **Paper:** "CoFinDiff: Controllable Financial Diffusion Model for Time Series Generation"
- **Authors:** Yuki Tanaka et al.
- **Date:** March 2025 | **Venue:** IJCAI 2025
- **arXiv:** 2503.04164

**Architecture:**
- Haar wavelet transformation of input data to image format
- Cross-attention conditioning on trend and realized volatility
- Generates synthetic financial time series matching arbitrary conditions

**Relevance to our use case:**
- Wavelet-to-image approach similar to "Painting the Market"
- Cross-attention conditioning on market regime could augment training data
- Not LOB-specific but the conditioning framework is transferable

---

## 5. Graph Neural Network Approaches

### 5.1 HLOB (Graph-based LOB model -- see Section 1.3)
Uses Triangulated Maximally Filtered Graph to model inter-level dependencies.

### 5.2 FinMamba (Graph-enhanced Mamba -- see Section 2.4)
Dynamic graph learning for cross-asset relationships combined with multi-level Mamba.

### 5.3 Hybrid Temporal Fusion Transformer + GNN
- **Paper:** "A Novel Hybrid Temporal Fusion Transformer Graph Neural Network Model for Stock Market Prediction"
- **Date:** October 2025 | **Journal:** MDPI

**Architecture:**
- Transformer-based temporal encoder for non-stationary temporal dependencies
- Edge-aware graph attention network for cross-asset information propagation
- TFT-GNN hybrid incorporating relational information between assets

**Relevance to our use case:**
- Cross-asset propagation via GNN is directly relevant for multi-exchange LOB modeling
- Could model relationships between BTC-USDT across Binance, Bybit, etc.
- Edge-aware attention could capture lead-lag effects between exchanges

### 5.4 Forecasting Equity Correlations with Hybrid Transformer-GNN
- **Paper:** "Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network"
- **Date:** January 2026 | **arXiv:** 2601.04602

**Relevance to our use case:**
- Models cross-asset correlation dynamics
- Useful for our multi-asset (BTC, ETH, SOL, WLD) forecasting setup

### 5.5 Assessment of GNN for LOB

**Current state:** No dominant GNN model exists specifically for LOB forecasting. However, GNNs are promising for:
- Cross-exchange relationship modeling (Binance spot vs perp vs Bybit)
- Cross-asset dependency modeling (BTC/ETH/SOL/WLD)
- Modeling LOB levels as a graph structure (HLOB approach)

**Gap:** Most GNN financial models operate on daily data; adaptation to tick-level/5s LOB data is underexplored.

---

## 6. Foundation Models for Time Series

### 6.1 Kronos: Foundation Model for Financial Markets
- **Paper:** "Kronos: A Foundation Model for the Language of Financial Markets"
- **Authors:** Yu Shi et al. (Tsinghua University)
- **Date:** August 2025 | **arXiv:** 2508.02739
- **Venue:** Accepted at AAAI 2026
- **Models:** Available on HuggingFace

**Architecture:**
- Decoder-only transformer family, pre-trained on financial K-line (OHLCV) data
- Two-stage framework: specialized tokenizer (quantizes continuous K-line data to hierarchical discrete tokens) + autoregressive transformer
- Pre-trained on 12 billion K-line records from 45 global exchanges

**Performance:**
- +93% RankIC improvement over leading TSFM for price forecasting
- +87% over best non-pre-trained baseline
- -9% MAE in volatility forecasting
- +22% improvement in generative fidelity for synthetic K-line sequences

**Relevance to our use case:**
- First open-source foundation model specifically for financial markets
- Pre-trained on massive multi-exchange data -- could capture cross-exchange patterns
- Operates on K-line (OHLCV candle) data, NOT raw LOB snapshots
- Would require adaptation/fine-tuning for tick-level LOB data
- Tokenization approach for continuous financial data is highly relevant

---

### 6.2 Re(Visiting) TSFMs in Finance (Comprehensive Evaluation)
- **Paper:** "Re(Visiting) Time Series Foundation Models in Finance"
- **Authors:** Eghbal Rahimikia, Hao Ni, Weiguan Wang
- **Date:** November 2025 | **arXiv:** 2511.18578

**Key Findings:**
- Evaluated TSFMs on ~2 billion observations across 8 major global markets
- Three regimes tested: zero-shot, fine-tuning, pre-training from scratch
- **Off-the-shelf pre-trained TSFMs perform POORLY on financial data in zero-shot and fine-tuning**
- **Models pre-trained from scratch on financial data achieve substantial improvements**
- Domain-specific adaptation is critical -- general TSFMs do not transfer well to finance

**Critical implication for our project:**
- Do NOT expect TimesFM/Chronos/Moirai to work well on LOB data out of the box
- Pre-training from scratch on our LOB data (or financial data) is likely necessary
- This validates building a custom model rather than fine-tuning a general TSFM

---

### 6.3 TimesFM 2.5 (Google)
- **Architecture:** 200M-parameter decoder-only, 16K context length, probabilistic forecasting
- **Status:** Top of GIFT-Eval benchmark (Sep 2025)
- **Financial relevance:** Univariate only; XReg for limited covariate support; no native multivariate cross-series modeling
- **LOB applicability:** Low -- would need to forecast each LOB feature independently

### 6.4 Chronos-2 (Amazon)
- **Architecture:** 120M-parameter encoder-only, group attention for arbitrary series groups
- **Innovation:** Native multivariate support, covariate-informed forecasting
- **Financial relevance:** >90% win rate vs Chronos-Bolt; group attention enables cross-series learning
- **LOB applicability:** Moderate -- group attention could model LOB levels as a group, but not tested on HF data

### 6.5 Moirai 2.0 (Salesforce)
- **Architecture:** Decoder-only, single patch, quantile loss, 36M series pre-training
- **Innovation:** 2x faster, 30x smaller than Moirai 1.0-Large, with better performance
- **LOB applicability:** Low to moderate -- efficient but designed for general time series

### 6.6 Timer-XL (Tsinghua, ICLR 2025)
- **Architecture:** Decoder-only, causal transformer with TimeAttention mechanism
- **Innovation:** Multivariate next-token prediction, arbitrary-length + any-variable series
- **LOB applicability:** Moderate -- TimeAttention captures inter-series dependencies; could model LOB levels as variables

---

## 7. Hybrid Architectures

### 7.1 SST: Multi-Scale Hybrid Mamba-Transformer Experts
- **Paper:** "SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series Forecasting"
- **Authors:** Xiongxiao Xu, Canyu Chen et al.
- **Date:** April 2024, published CIKM 2025 | **arXiv:** 2404.14757
- **Code:** https://github.com/XiongxiaoXu/SST

**Architecture:**
- Time series decomposition into long-range and short-range components
- Mamba expert for long-range patterns (linear complexity)
- Transformer expert for short-term variations (full attention, smaller sequences)
- Multi-scale patching: low resolution for long-term, high resolution for short-term
- Addresses "information interference" from naive Mamba-Transformer stacking

**Performance:**
- SoTA with linear scalability on general time series benchmarks

**Relevance to our use case:**
- **Highly promising** -- Mamba handles long context efficiently while Transformer captures local LOB dynamics
- Multi-scale patching naturally fits LOB data (5s ticks = short-term, 10-minute context = long-term)
- Could be adapted: Mamba for the 120-step context, Transformer for recent ~24 steps
- Expert routing avoids the interference problem of simply concatenating SSM + attention

---

### 7.2 T-Mamba: Hybrid Mamba-Transformer for Stock Prediction
- **Paper:** "T-Mamba: A Hybrid Mamba-Transformer Framework for Stock Price Prediction"
- **Date:** 2025 | **Venue:** CIBDA 2025

**Architecture:**
- Multi-resolution framework: coarse (Mamba for global) + fine (Transformer for local) representations
- Mamba as expert in global patterns, capturing long-range dependencies

**Performance:**
- Surpasses CNN, Transformer, BiLSTM, XGBoost on real-world stock data

**Relevance to our use case:**
- Confirms the pattern: Mamba for global, Transformer for local works well
- Multi-resolution approach aligns with our 5s interval data having both HF noise and longer patterns

---

### 7.3 LiT (Transformer + LSTM hybrid -- see Section 1.2)
Combines transformer self-attention with LSTM for long-term temporal modeling.

---

## 8. Benchmarking Frameworks

### 8.1 LOBFrame
- **Paper:** "Deep Limit Order Book Forecasting: A Microstructural Guide"
- **Date:** July 2025 | **Journal:** Quantitative Finance
- **Code:** https://github.com/FinancialComputingUCL/LOBFrame
- Open-source, modular Python/PyTorch framework for LOB data processing and model evaluation
- NASDAQ equities, model-agnostic design
- **Key insight:** Traditional ML metrics fail for LOB; proposes operational evaluation based on complete transaction prediction probability

### 8.2 LOB-Bench (Generative Models)
- **Paper:** "LOB-Bench: Benchmarking Generative AI for Finance"
- **Date:** February 2025 | **arXiv:** 2502.09172 | **Venue:** ICML 2025
- **Code:** https://github.com/peernagy/lob_bench
- Evaluates generative LOB models on distributional statistics, adversarial scores, market impact metrics
- Autoregressive GenAI beats traditional model classes

### 8.3 LOBench (Representation Learning)
- **Paper:** "Representation Learning of Limit Order Book: A Comprehensive Study and Benchmarking"
- **Date:** May 2025 | **arXiv:** 2505.02139
- **Code:** https://github.com/financial-simulation-lab/LOBench
- Standardized benchmark with China A-share market data
- Validates that LOB-specific representations outperform general time series representations

### 8.4 LOBCAST (Deep Learning Benchmark)
- **Paper:** "LOB-based deep learning models for stock price trend prediction: a benchmark study"
- **Journal:** Artificial Intelligence Review, 2024
- 15 SoTA models benchmarked with standardized evaluation

### 8.5 DeepMarket
- **Code:** https://github.com/LeonardoBerti00/DeepMarket
- First open-source Python framework for LOB market simulation with deep learning
- Accompanies TRADES paper

---

## 9. Key Insights and Recommendations

### Architecture Ranking for Our Use Case

**Tier 1 -- Most Promising (directly applicable to multi-level LOB forecasting):**

| Architecture | Why | Risk |
|---|---|---|
| SST (Mamba-Transformer Hybrid) | Best of both worlds: linear complexity + strong local attention; multi-scale naturally fits our data | Not LOB-specific yet; needs adaptation for structure loss |
| "Painting the Market" (Diffusion Inpainting) | Full LOB reconstruction; overcomes error accumulation; SoTA on LOB-Bench | Slow inference (multiple denoising steps); may not meet 5s latency requirement |
| TLOB (Dual Attention) | LOB-specific; strong results on crypto; dual attention for spatial+temporal | Classification only, not full LOB reconstruction |

**Tier 2 -- Strong Candidates (need adaptation):**

| Architecture | Why | Risk |
|---|---|---|
| CVML Add-on | 244.9% improvement as plug-in module; could enhance our existing model | Limited public details; may need proprietary data |
| LOBERT Foundation | Pre-trained LOB representations; message-level modeling | Encoder-only; designed for classification not regression |
| CMDMamba | Dual-layer Mamba for financial data; near-linear complexity | Not LOB-specific; no structure preservation |
| LiT | Patch-based + LSTM hybrid; robust to distribution shift | Direction classification only |

**Tier 3 -- Supplementary (useful for specific aspects):**

| Architecture | Why | Risk |
|---|---|---|
| DiffLOB/DiffVolume | Synthetic data generation, stress testing | Not for live forecasting |
| Kronos Foundation | Pre-trained financial representations | K-line data, not LOB snapshots |
| FinMamba (Graph+Mamba) | Cross-asset modeling | Stock-level, not tick-level |
| HLOB | Topological LOB analysis | Complex preprocessing |

### Critical Findings

1. **Domain-specific models crush general TSFMs on financial data.** Off-the-shelf TimesFM/Chronos/Moirai perform poorly. Pre-training from scratch on financial data is essential (Re(Visiting) TSFMs in Finance).

2. **Input features matter more than model complexity.** The crypto LOB study (arXiv:2506.05764) shows simpler models with proper feature engineering (OFI, Kalman filtering) match deep architectures. We should invest in preprocessing.

3. **The Mamba-Transformer hybrid paradigm is emerging as the leading approach** for time series that have both short-term dynamics and long-range dependencies (SST, T-Mamba). This fits LOB data well.

4. **Diffusion models are the SoTA for LOB generation/simulation** but have inference latency concerns for live forecasting. They excel at full LOB reconstruction.

5. **Structure loss / ordinal constraint enforcement remains underexplored** in most new architectures. Our Compound Attention Model's structure loss is a genuine advantage.

6. **Cross-variate mixing (CVML) provides massive improvements** when added to general time series models for LOB data, validating our compound embedding approach.

7. **No single model dominates all LOB tasks.** Classification models (TLOB, LiT) lead on mid-price direction. Diffusion models (Painting the Market) lead on full LOB generation. Our full LOB forecasting task is relatively unique.

### Recommended Next Steps

1. **Immediate: Implement CVML-style cross-variate mixing** as an add-on to our Compound Attention Model
2. **Short-term: Prototype SST-style Mamba-Transformer hybrid** with our compound embedding and structure loss
3. **Medium-term: Experiment with diffusion-based LOB forecasting** (Painting the Market approach) for full LOB reconstruction
4. **Ongoing: Integrate Order Flow Imbalance (OFI) features** and Kalman filtering into our preprocessing pipeline
5. **Explore: Fine-tune Kronos or LOBERT** on our specific multi-exchange LOB data as an alternative representation

---

## Sources

### LOB-Specific Models
- [TLOB - arXiv](https://arxiv.org/abs/2502.15757)
- [LiT - Frontiers in AI](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full)
- [HLOB - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417424029452)
- [LOBERT - arXiv](https://arxiv.org/abs/2511.12563)
- [Crypto LOB Better Inputs - arXiv](https://arxiv.org/abs/2506.05764)
- [CVML Benchmark - OpenReview](https://openreview.net/forum?id=MhD9rLeU31)
- [mWDN-Transformer - ACM](https://dl.acm.org/doi/10.1145/3670105.3670134)

### State-Space Models
- [CryptoMamba - arXiv](https://arxiv.org/abs/2501.01010)
- [CMDMamba - Frontiers in AI](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1599799/full)
- [MambaTS - arXiv](https://arxiv.org/abs/2405.16440)
- [FinMamba - arXiv](https://arxiv.org/abs/2502.06707)
- [SSM for Market Microstructure Analysis](https://jonathankinlay.com/2026/03/state-space-models-for-market-microstructure-can-mamba-replace-transformers-in-high-frequency-finance/)

### Diffusion Models
- [Painting the Market - arXiv](https://arxiv.org/abs/2509.05107)
- [TRADES - arXiv](https://arxiv.org/abs/2502.07071)
- [LOBDIF - arXiv](https://arxiv.org/abs/2412.09631)
- [DiffLOB - arXiv](https://arxiv.org/abs/2602.03776)
- [DiffVolume - arXiv](https://arxiv.org/abs/2508.08698)
- [CoFinDiff - IJCAI 2025](https://arxiv.org/abs/2503.04164)

### Graph Neural Networks
- [Hybrid TFT-GNN - MDPI](https://www.mdpi.com/2673-9909/5/4/176)
- [Forecasting Equity Correlations - arXiv](https://arxiv.org/abs/2601.04602)

### Foundation Models
- [Kronos - arXiv](https://arxiv.org/abs/2508.02739)
- [Re(Visiting) TSFMs in Finance - arXiv](https://arxiv.org/abs/2511.18578)
- [Chronos-2 - Amazon Science](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
- [TimesFM 2.5 - HuggingFace](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- [Moirai 2.0 - arXiv](https://arxiv.org/abs/2511.11698)
- [Timer-XL - arXiv](https://arxiv.org/abs/2410.04803)

### Hybrid Architectures
- [SST - arXiv](https://arxiv.org/abs/2404.14757)
- [T-Mamba - ACM](https://dl.acm.org/doi/10.1145/3746709.3746715)

### Benchmarks and Frameworks
- [LOBFrame - GitHub](https://github.com/FinancialComputingUCL/LOBFrame)
- [LOB-Bench - arXiv](https://arxiv.org/abs/2502.09172)
- [LOBench - arXiv](https://arxiv.org/abs/2505.02139)
- [DeepMarket - GitHub](https://github.com/LeonardoBerti00/DeepMarket)
- [LOBFrame Paper - Quantitative Finance](https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2522911)
- [SSM Foundation Model Survey - Mamba-360](https://www.sciencedirect.com/science/article/abs/pii/S0952197625012801)
- [Foundation Models for Time Series Survey](https://arxiv.org/html/2504.04011v1)
