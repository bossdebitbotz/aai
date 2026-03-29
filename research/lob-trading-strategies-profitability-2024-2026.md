# LOB Trading Strategies, Profitability, and Execution Research (2024-2026)

*Research compiled March 2026. Companion to `sota-lob-architectures-2024-2026.md`.*
*Project context: 4 exchanges (Binance spot/perp, Bybit spot, KuCoin spot), 4 symbols (BTC, ETH, SOL, WLD), 5-level LOB, 5s intervals.*

---

## Table of Contents
- [1. What Actually Makes Money: Prediction Targets and Horizons](#1-what-actually-makes-money)
- [2. Alpha Decay and Signal Persistence](#2-alpha-decay-and-signal-persistence)
- [3. Multi-Exchange Arbitrage Strategies](#3-multi-exchange-arbitrage-strategies)
- [4. Ensemble Approaches](#4-ensemble-approaches)
- [5. Execution Layer](#5-execution-layer)
- [6. Strategy-Architecture Pairings for Our Setup](#6-strategy-architecture-pairings-for-our-setup)
- [7. Practical Recommendations](#7-practical-recommendations)

---

## 1. What Actually Makes Money

### 1.1 Prediction Targets Ranked by Profitability

Research consistently shows that the choice of prediction target matters more than the model architecture for profitability.

**Tier 1 -- Highest Alpha Potential:**

| Target | Why It Works | Key Reference |
|--------|-------------|---------------|
| **Order Flow Imbalance (OFI)** | Deep-book OFI reduces RMSE by 68-74% for imminent price prediction on NASDAQ. Multi-level OFI across 5 levels captures supply-demand asymmetries invisible in top-of-book data. | Kolm & Turiel, "Deep order flow imbalance: Extracting alpha at multiple horizons" (Mathematical Finance, 2023) |
| **Volume Imbalance (Level 1)** | Single most influential LOB component for immediate price prediction. Simple to compute, fast to act on. | Wang, "Better Inputs Matter More" (arXiv:2506.05764, 2025) |
| **Mid-Price Return Direction** | Standard classification target (up/down/stationary). TLOB achieves 81-93% F1 on FI-2010. Directly actionable. | Berti & Kasneci, TLOB (arXiv:2502.15757, 2025) |

**Tier 2 -- Moderate Alpha, Lower Decay:**

| Target | Why It Works | Key Reference |
|--------|-------------|---------------|
| **Full LOB State Reconstruction** | Enables sophisticated execution strategies. Our Compound Attention model predicts complete 5-level bid/ask prices and volumes. Unique advantage over classification-only models. | Jung & Lee (arXiv:2409.02277, 2024) |
| **Spread Prediction** | Predicting spread widening/narrowing enables market making strategy timing. Spread is more predictable than direction at longer horizons. | LOBFrame (Quantitative Finance, 2025) |
| **Cross-Exchange Price Differential** | Predicting when one exchange leads/lags creates arbitrage signals. Our multi-exchange setup makes this uniquely available. | See Section 3 |

**Tier 3 -- Strategy-Specific:**

| Target | Why It Works | Key Reference |
|--------|-------------|---------------|
| **Funding Rate Prediction** | Predicting funding rate direction for perp vs spot basis trades. Average 5.98-11.4% APR on BTC/ETH with controlled risk. | Pendle/Boros analysis (2025) |
| **VPIN (Volume-Synchronized Probability of Informed Trading)** | Predicts future price jumps. Positive serial correlation in both VPIN and jump size means momentum-style strategies work. | Bitcoin futures microstructure study (2024) |
| **Regime Classification** | Not directly tradeable but essential for model switching. Identifies when to activate/deactivate strategies. | DiffLOB (arXiv:2602.03776, 2026) |

### 1.2 Optimal Prediction Horizons for Crypto LOB Trading

Research on crypto-specific LOB data reveals a clear horizon-accuracy-profitability tradeoff:

| Horizon | Accuracy (ternary) | Transaction Cost Viability | Use Case |
|---------|--------------------|-----------------------------|----------|
| **100ms** | 37-42% | Requires sub-ms execution; impractical for most setups | Ultra-HFT market making |
| **500ms** | 42-54% | Marginal after fees on major pairs | Scalping, latency arb |
| **1-5s** | 45-55% | Viable for market making on wider spreads | **Our sweet spot (5s intervals)** |
| **5-30s** | 50-60% | Good after fees; signals more persistent | Directional + market making |
| **1-10 min** | 55-65% | Best for larger positions; lower turnover | Directional trading |
| **>10 min** | Accuracy degrades | Alpha decay overwhelms signal | Not recommended for LOB-only |

**Critical finding for our 5s interval setup:** At 5-second resolution, we sit at the boundary between HFT and medium-frequency trading. This is advantageous because:
- Signal strength is still high from LOB microstructure
- Transaction costs are manageable (especially for market making)
- Latency requirements (~100ms end-to-end) are achievable without co-location
- Enough time for model inference (even complex models run in <100ms on GPU)

### 1.3 Market Making vs Directional Strategies

**Market Making with LOB Predictions:**
- Use predicted spread trajectory to optimize quote placement
- Use predicted volume imbalance to skew quotes (inventory management)
- Use predicted mid-price direction to lean the book
- Expected Sharpe: 3-8 (before infrastructure costs)
- Requires: spread prediction, volume forecasting, inventory model
- Risk: adverse selection, inventory buildup during trends

**Directional Trading with LOB Predictions:**
- Use mid-price direction classification to take positions
- Use OFI signals for entry timing
- Use spread prediction for optimal entry/exit timing
- Expected Sharpe: 1-3
- Requires: direction classifier, confidence calibration, position sizing
- Risk: alpha decay, regime changes, false signals

**Hybrid (Recommended for Our Setup):**
- Market make passively using LOB predictions for quote optimization
- Take directional positions only on high-confidence OFI signals
- Use funding rate predictions for basis trades (Binance perp vs spot)
- Expected Sharpe: 2-5 (combined)

---

## 2. Alpha Decay and Signal Persistence

### 2.1 How Quickly Do LOB Signals Lose Value?

| Signal Type | Half-Life | Decay Pattern | Mitigation |
|-------------|-----------|---------------|------------|
| **Level-1 Volume Imbalance** | 1-5 seconds | Exponential | Act immediately; use as trigger, not predictor |
| **Multi-level OFI** | 5-30 seconds | Exponential with fat tail | Our 5s interval captures this well |
| **Spread State** | 30s-2 min | Mean-reverting | Trade the reversion; spread prediction horizon matches |
| **Cross-Exchange Price Gap** | 1-10 seconds | Step function (snap-to-parity) | Must execute within the gap window |
| **Funding Rate Signal** | 8 hours (funding period) | Sawtooth | Slow alpha; position ahead of funding |
| **Deep-Book Structural Patterns** | 1-10 minutes | Gradual linear decay | Our 2-minute prediction horizon is well-suited |

### 2.2 T-KAN Results on Alpha Decay

The T-KAN architecture (Temporal Kolmogorov-Arnold Networks, arXiv:2601.02310, Jan 2026) specifically addresses alpha decay at longer horizons:

- At k=100 ticks: T-KAN achieves F1 of 0.3995 vs DeepLOB's 0.3354 (+19.1% relative improvement)
- More critically, in backtesting with 1.0 bps transaction costs: T-KAN yields +132.48% return vs DeepLOB's -82.76%
- The key insight: spline-based activation functions (B-splines) better identify high-conviction signals that survive transaction costs
- T-KAN architecture is FPGA-deployable for sub-microsecond inference

### 2.3 Filtering and Signal Enhancement

The crypto LOB study (arXiv:2506.05764) found that preprocessing dramatically extends signal life:

- **Savitzky-Golay Smoothing** (cubic polynomial, 21-point window): Preserves trends while suppressing noise. Recommended for price data.
- **Kalman Filtering**: Effective but parameter-sensitive. Requires tuning per symbol.
- **LOB Depth**: Using 40 levels achieved 71.5% accuracy vs 57.9% with 5 levels. Our 5-level setup is a constraint -- consider increasing depth if possible.
- **Practical constraint**: "Training a model achieving 80% accuracy but requiring 2 seconds for inference defeats its purpose."

### 2.4 Dealing with Flickering Liquidity

Order Book Imbalance (OBI) is vulnerable to **flickering liquidity** -- orders rapidly submitted and cancelled without execution intent (spoofing, latency arbitrage). Filtered order book signals that remove microstructural artifacts focus on more persistent order flow patterns.

Recommendation: Implement a filtering layer that tracks order persistence (how long orders stay on the book) before feeding data to the model. Orders that survive >1 second carry more information than fleeting quotes.

---

## 3. Multi-Exchange Arbitrage Strategies

### 3.1 Cross-Exchange LOB Strategies

Our setup (Binance spot, Binance perp, Bybit spot, KuCoin spot) enables several cross-venue strategies:

**Strategy 1: Price Leadership Detection**
- One exchange often leads price discovery, typically Binance perp for BTC
- Train a model to predict which exchange leads in each 5-second window
- Trade on the lagging exchange before it catches up
- Signal half-life: 1-10 seconds (tight execution required)

**Strategy 2: Cross-Exchange OFI Divergence**
- When Binance shows strong buying OFI but Bybit shows selling OFI, a divergence often creates short-term arbitrage
- These divergences typically arise from user base differences, regional preferences, or temporary liquidity disparities
- Must account for fees: Binance maker ~0.02%, Bybit maker ~0.01%

**Strategy 3: Liquidity Migration**
- Large orders often execute across venues sequentially
- Detecting the start of a large order on Binance (via LOB impact) and front-running on Bybit
- Requires multi-exchange LOB monitoring in real-time (already in our data pipeline)

### 3.2 Spot-Perpetual Basis Trading

The Binance spot + perp combination is our highest-value pair for basis trading:

**Funding Rate Arbitrage:**
- Average funding rates in 2025: ~0.015% per 8-hour period (~19.7% annualized)
- Cross-platform opportunities add 3-5% annualized returns
- Front-month SOL and XRP futures saw annualized basis spikes to 50% in mid-2025
- Strategy: Long spot + short perp when funding is positive (and vice versa)
- Delta-neutral; risk is in funding rate reversal and margin requirements

**Basis Prediction Model:**
- Use our LOB model to predict basis widening/narrowing
- Compound Attention Model can forecast both spot and perp LOB states
- The spread between predicted mid-prices is the basis forecast
- Enter basis trades when predicted spread exceeds threshold after fees

**Key Research Finding (BIS Working Paper 1087):**
Deviations from no-arbitrage prices in crypto perpetual futures are "considerably larger than those documented in traditional currency markets." A simple carry strategy generates large Sharpe ratios even for investors paying the highest Binance trading costs. These deviations diminish over time as markets mature -- suggesting acting now captures more alpha.

### 3.3 Latency Arbitrage Assessment

Research on crypto latency arbitrage (SSRN:5143158, 2025):

- Most retail traders operate at 100-500ms latency
- Market maker professionalization compressed spreads toward zero by 2024
- Cross-exchange spread arbitrage on major pairs is near-zero by 2024 for co-located participants
- However, **our 5-second interval still captures structural arbitrage** (not latency arbitrage):
  - Funding rate dislocations
  - LOB depth imbalances across venues
  - Temporary liquidity events (large market orders on one venue)

**Assessment for our setup:** Pure latency arbitrage is not viable at 5s intervals. Focus instead on structural arbitrage (basis, funding rate) and predictive signals (cross-exchange OFI divergence).

### 3.4 Multi-Exchange Feature Engineering

Cross-venue features to compute at each 5-second snapshot:

```
# Price differential features
mid_price_diff_binance_spot_vs_perp
mid_price_diff_binance_vs_bybit
mid_price_diff_binance_vs_kucoin
max_mid_price_spread_across_4_exchanges

# Volume features
total_bid_volume_ratio_binance_vs_bybit  (per level)
total_ask_volume_ratio_binance_vs_bybit  (per level)
aggregate_volume_imbalance_all_exchanges

# Spread features
spread_ratio_binance_spot_vs_perp
spread_ratio_binance_vs_bybit
min_spread_across_exchanges  (best execution venue indicator)

# Lead-lag features
rolling_correlation_5min_binance_bybit
rolling_lead_lag_granger_binance_perp_vs_spot
price_change_sequence_across_venues  (which moved first?)

# Funding rate features (Binance perp only)
current_funding_rate
predicted_next_funding_rate
funding_rate_vs_basis_spread
```

---

## 4. Ensemble Approaches

### 4.1 Best Ensemble Combinations for Risk-Adjusted Returns

**ACM ICAIF FinRL Contest Results (2024/2025):**
- Ensemble methods using majority voting on agent actions
- Rolling-window training: 30-day train, 5-day validation, 5-day test
- Deep learning ensembles significantly enhance predictive accuracy and profitability under volatile conditions
- An ensemble of neural networks achieved 1640% total return (Jan 2018 - Jan 2024) vs 305% for individual ML and 223% for buy-and-hold

**Recommended Ensemble Architecture for Our Setup:**

**Level 1 -- Signal Generators (run in parallel):**

| Model | Target | Horizon | Role |
|-------|--------|---------|------|
| Compound Attention Model | Full LOB state (5-level prices + volumes) | 2 min (24 steps) | Primary forecaster, execution planning |
| TLOB-style Direction Classifier | Mid-price direction (up/down/flat) | 5s, 15s, 30s, 60s | Directional signal at multiple scales |
| T-KAN or DeepLOB | Mid-price direction | 30s-2min | High-conviction long-horizon signal |
| XGBoost/LightGBM on OFI features | Mid-price direction + magnitude | 5s-15s | Fast, interpretable baseline |
| Mamba-SSM | Regime classifier (trend/mean-rev/volatile) | 5min rolling | Strategy selector |

**Level 2 -- Signal Aggregation:**

| Method | Description |
|--------|-------------|
| **Weighted Voting** | Weight each model's signal by recent rolling accuracy (exponentially decayed) |
| **Meta-Learner** | Train a lightweight model (logistic regression or small NN) on Level 1 outputs to produce final signal |
| **Confidence Gating** | Only act when >= 3 of 5 models agree AND meta-learner confidence > threshold |
| **Regime-Conditional Weighting** | In trending regimes, upweight directional models; in mean-reverting, upweight market making signals |

**Level 3 -- Strategy Execution:**

| Strategy | Trigger | Model Dependency |
|----------|---------|-----------------|
| Directional (market order) | High confidence direction + strong OFI | Direction ensemble + OFI model |
| Market Making (limit orders) | Moderate confidence + predicted spread widening | Full LOB model + spread predictor |
| Basis Trade (spot vs perp) | Funding rate signal + basis prediction | LOB model on both spot and perp |
| Risk-Off | Regime classifier signals high volatility + low confidence | Regime model + confidence gating |

### 4.2 Ensemble Variance Reduction

Key principle: ensemble components should be **diverse** in their error patterns. Diversity sources:

1. **Architecture diversity**: Transformer (Compound Attention) + CNN (DeepLOB) + Gradient Boosting (XGBoost) + SSM (Mamba)
2. **Feature diversity**: Some models on raw LOB, others on engineered features (OFI, imbalance)
3. **Horizon diversity**: Different models optimized for different prediction horizons
4. **Training data diversity**: Rolling windows of different lengths; different exchanges as training data
5. **Target diversity**: Direction, magnitude, volatility, spread -- each captures different information

### 4.3 Risk-Adjusted Return Optimization

For portfolio-level risk management across the ensemble:

- **Kelly Criterion adaptation**: Size positions proportional to edge / variance, using ensemble confidence as edge estimate
- **Maximum drawdown constraint**: Hard-stop at 5% daily drawdown; 15% monthly
- **Correlation-aware sizing**: Reduce position sizes when cross-asset correlations spike (regime shift indicator)
- **Volatility targeting**: Scale all positions inversely to realized volatility (target constant vol portfolio)

---

## 5. Execution Layer

### 5.1 RL-Based Execution

**Optimal Execution with RL (arXiv:2411.06389, Nov 2024):**
- RL agent trained in ABIDES multi-agent market simulator
- Outperforms standard strategies (TWAP, VWAP) in backtesting
- Uses LOB state as input features
- Practical for real-world deployment

**Market Making with Deep RL (arXiv:2305.15821):**
- Two-stage pipeline: (1) CNN/attention encoder pretrained on mid-price classification, (2) DQN agent for quoting
- Tested on BitMEX XBTUSD (top-20 levels), achieved 0.7662 out-of-sample classification accuracy
- Discrete action space: spread width + quote skew
- Inventory management: prohibit quoting in one direction when absolute inventory exceeds threshold
- Hawkes process variant captures clustered order arrival times

**RL-Based Market Making on Non-Stationary LOB (arXiv:2509.12456, 2025):**
- Treats market making as stochastic control on non-stationary LOB dynamics
- Accounts for the fact that LOB statistics shift continuously
- Uses PPO and A2C policy gradient algorithms
- Combined raw LOB data with trade and order flow imbalance features

### 5.2 Execution Strategies Paired with LOB Predictions

For our setup, each strategy type needs a different execution approach:

**Directional Trades:**
```
IF direction_confidence > 0.7 AND ensemble_agreement >= 3/5:
    IF predicted_spread < current_spread:
        # Spread expected to narrow -- use limit order at mid + skew
        place_limit_order(side=predicted_direction, price=mid + direction_skew)
        timeout = 5s  # One prediction interval
    ELSE:
        # Spread expected to widen -- use aggressive limit or market order
        place_limit_order(side=predicted_direction, price=aggressive_limit)
        timeout = 2s, then convert_to_market()
```

**Market Making:**
```
IF regime == MEAN_REVERTING AND spread > min_profitable_spread:
    bid_price = mid - (predicted_spread/2) * inventory_skew
    ask_price = mid + (predicted_spread/2) * inventory_skew
    bid_size = base_size * (1 - inventory_ratio)  # Reduce on side with inventory
    ask_size = base_size * (1 + inventory_ratio)

    # Cancel and replace every 5s with new predictions
    # Emergency cancel if OFI signals large directional move
```

**Basis Trade:**
```
IF abs(spot_mid - perp_mid) > threshold + fees:
    IF spot_mid < perp_mid AND funding_rate > 0:
        # Buy spot, sell perp (classic carry)
        execute_simultaneously(buy_spot, sell_perp, size=basis_size)
    ELIF spot_mid > perp_mid AND funding_rate < 0:
        # Sell spot, buy perp (reverse carry)
        execute_simultaneously(sell_spot, buy_perp, size=basis_size)

    # Hold until basis converges or funding rate flips
    # Manage margin on perp position
```

### 5.3 Execution Quality Metrics

Track these to evaluate execution performance:

| Metric | Target | Description |
|--------|--------|-------------|
| **Implementation Shortfall** | < 2 bps | Slippage from signal to fill |
| **Fill Rate** | > 80% for limit orders | Percent of limit orders that execute |
| **Adverse Selection** | < signal alpha | Percentage of fills followed by unfavorable moves |
| **Inventory Turnover** | > 10x/day | How frequently inventory is recycled |
| **Net PnL per Trade** | > 0.5 bps after fees | Must exceed all costs |

---

## 6. Strategy-Architecture Pairings for Our Setup

### 6.1 Complete System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER (5s snapshots)             │
│  Binance Spot LOB ─┐                                    │
│  Binance Perp LOB ─┼─→ Cross-Exchange Feature Engine    │
│  Bybit Spot LOB   ─┤     (OFI, imbalance, spreads,     │
│  KuCoin Spot LOB  ─┘      lead-lag, funding rate)       │
│                                                          │
│  Preprocessing: Savitzky-Golay smoothing, Kalman filter  │
│  Percent-change transform, min-max scaling               │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                  MODEL LAYER (Ensemble)                   │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │ Compound Attention│  │ TLOB Direction   │             │
│  │ (Full LOB Recon) │  │ (Multi-horizon)  │             │
│  │ w/ CVML add-on   │  │ Classification   │             │
│  └────────┬─────────┘  └────────┬─────────┘             │
│           │                      │                        │
│  ┌────────┴─────────┐  ┌────────┴─────────┐             │
│  │ XGBoost on OFI   │  │ Mamba Regime     │             │
│  │ (Fast baseline)  │  │ Classifier       │             │
│  └────────┬─────────┘  └────────┬─────────┘             │
│           │                      │                        │
│  ┌────────┴──────────────────────┴─────────┐             │
│  │          Meta-Learner / Signal           │             │
│  │          Aggregation Layer               │             │
│  └────────────────────┬────────────────────┘             │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                 STRATEGY LAYER                            │
│                                                          │
│  ┌───────────┐ ┌────────────┐ ┌──────────────┐          │
│  │Directional│ │Market      │ │ Basis/       │          │
│  │Trading    │ │Making      │ │ Funding Rate │          │
│  └─────┬─────┘ └─────┬──────┘ └──────┬───────┘          │
│        │              │               │                   │
│  ┌─────┴──────────────┴───────────────┴──────┐           │
│  │    Risk Manager (Kelly sizing, vol target, │           │
│  │    drawdown limits, correlation monitor)   │           │
│  └───────────────────────┬───────────────────┘           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                 EXECUTION LAYER                           │
│                                                          │
│  RL-optimized execution agent per strategy per exchange   │
│  Smart order routing across exchanges                     │
│  Latency monitoring and circuit breakers                  │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Phased Implementation Plan

**Phase 1 (Weeks 1-4): Foundation**
- Implement Compound Attention Model with current data pipeline
- Add OFI and volume imbalance features to preprocessing
- Train single-exchange, single-symbol models as baselines
- Benchmark: directional accuracy, structure loss, simulated PnL

**Phase 2 (Weeks 5-8): Multi-Exchange**
- Add cross-exchange features (price diff, spread ratio, lead-lag)
- Extend Compound Attention with exchange embeddings
- Add CVML cross-variate mixing layer
- Implement basis prediction (spot vs perp mid-price spread)

**Phase 3 (Weeks 9-12): Ensemble**
- Add TLOB direction classifier at multiple horizons
- Add XGBoost on OFI features as fast baseline
- Implement meta-learner signal aggregation
- Add regime classifier (Mamba-based or simpler HMM)

**Phase 4 (Weeks 13-16): Execution**
- Implement market making execution with LOB predictions
- Implement directional execution with confidence gating
- Implement basis trade execution
- Paper trading on all strategies

**Phase 5 (Weeks 17+): Live and Iterate**
- Deploy to live with minimum size
- Monitor execution quality metrics
- A/B test model variants
- Scale position sizes as confidence grows

---

## 7. Practical Recommendations

### 7.1 For Our Specific Setup (4 exchanges, 4 symbols, 5-level LOB, 5s)

1. **Increase LOB depth if possible.** Research shows 40 levels achieves 71.5% accuracy vs 57.9% with 5 levels. Even going to 10 or 20 levels would help significantly. The marginal storage cost is minimal compared to the information gain.

2. **Invest in preprocessing over model complexity.** The "Better Inputs Matter More" finding is the single most impactful research result. Savitzky-Golay smoothing of prices and multi-level OFI computation should be implemented before any architecture upgrades.

3. **The Compound Attention Model is a strong choice** for our primary use case (full LOB reconstruction). Its structure loss is a genuine differentiator that most newer architectures lack. Enhance it rather than replace it.

4. **Add CVML cross-variate mixing** as a plug-in module. The 244.9% improvement in mid-price return forecasting on LOB data is the largest single architectural improvement documented in recent literature.

5. **Basis trading (spot vs perp) is likely our most reliable alpha source.** BIS research confirms that crypto perpetual futures have larger no-arbitrage deviations than any traditional market, and a simple carry strategy produces strong Sharpe ratios even after Binance's highest fees. This diminishes over time as markets mature -- act soon.

6. **At 5-second intervals, we are well-positioned for mixed strategies** (not pure HFT, not pure medium-frequency). Market making with directional overlay using LOB predictions is the highest expected Sharpe combination.

7. **Ensemble with diverse models is essential for robustness.** The FinRL contest data shows ensembles dramatically outperform individual models (1640% vs 305% over 6 years). Minimum viable ensemble: Compound Attention + XGBoost on OFI + regime classifier.

8. **Plan for alpha decay from day one.** Implement rolling retraining (30-day window) and feature drift monitoring. LOB signal half-lives in crypto range from seconds to minutes; stale models lose money fast.

9. **Cross-exchange features are underexploited in the literature** and represent a potential edge. Most academic LOB papers study single-exchange data. Our multi-exchange setup enables signals that few academic models capture.

10. **Consider T-KAN for FPGA deployment** if latency becomes a constraint. The architecture is specifically designed for hardware acceleration and shows strong results at longer horizons where other models fail.

### 7.2 Expected Revenue Potential (Conservative Estimates)

| Strategy | Expected Annual Return | Sharpe Ratio | Capital Requirement | Confidence |
|----------|----------------------|--------------|--------------------| ------------|
| Basis/Funding Rate Arb | 6-15% | 2-4 | $50K-500K | High |
| Market Making (LOB-guided) | 15-40% | 3-6 | $20K-200K | Medium |
| Directional (ensemble LOB) | 10-30% | 1-3 | $10K-100K | Medium-Low |
| Cross-Exchange Structural Arb | 5-15% | 2-3 | $50K-500K | Medium |
| **Combined Portfolio** | **20-50%** | **3-5** | **$50K-500K** | **Medium** |

Note: These are gross returns before infrastructure costs. Net returns depend heavily on execution quality and infrastructure investment.

### 7.3 Key Risks

| Risk | Mitigation |
|------|------------|
| Alpha decay / signal crowding | Rolling retraining, feature drift monitoring, diverse signal sources |
| Exchange API changes / downtime | Multi-exchange redundancy, graceful degradation |
| Market regime shift | Regime classifier, strategy switching, position limits |
| Execution slippage | RL-optimized execution, smart order routing, fill rate monitoring |
| Overfitting in backtesting | Walk-forward validation, paper trading period, conservative position sizing |
| Infrastructure failure | Redundant systems, circuit breakers, position flattening on failure |

---

## Sources

### Prediction Targets and Profitability
- [Deep order flow imbalance: Extracting alpha at multiple horizons](https://ideas.repec.org/a/bla/mathfi/v33y2023i4p1044-1081.html)
- [Better Inputs Matter More - arXiv:2506.05764](https://arxiv.org/html/2506.05764v2)
- [TLOB - arXiv:2502.15757](https://arxiv.org/abs/2502.15757)
- [LOBFrame - Quantitative Finance](https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2522911)
- [LOBFrame - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
- [Attention-Based Multi-Asset Order Flow Networks - ACM ICAIF](https://dl.acm.org/doi/10.1145/3768292.3770430)

### Alpha Decay and Signal Persistence
- [T-KAN for LOB Forecasting - arXiv:2601.02310](https://arxiv.org/html/2601.02310)
- [Order Book Filtration and Directional Signal Extraction](https://arxiv.org/html/2507.22712v1)
- [Bitcoin Futures Microstructure](https://www.sciencedirect.com/science/article/pii/S2214845025001188)
- [Bitcoin Order Flow Toxicity and Price Jumps](https://www.sciencedirect.com/science/article/pii/S0275531925004192)

### Multi-Exchange Arbitrage
- [High-Frequency Arbitrage Across Cryptocurrency Exchanges](https://medium.com/@gwrx2005/high-frequency-arbitrage-and-profit-maximization-across-cryptocurrency-exchanges-4842d7b7d4d9)
- [Latency Arbitrage in Cryptocurrency Markets - SSRN:5143158](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5143158)
- [Cross-Exchange Funding Rate Arbitrage - Pendle/Boros](https://medium.com/boros-fi/cross-exchange-funding-rate-arbitrage-a-fixed-yield-strategy-through-boros-c9e828b61215)
- [BIS Working Paper 1087: Crypto Carry](https://www.bis.org/publ/work1087.pdf)
- [Funding Rate Arbitrage on CEX and DEX](https://www.sciencedirect.com/science/article/pii/S2096720925000818)
- [Fundamentals of Perpetual Futures](https://arxiv.org/html/2212.06888v5)
- [Crypto Arbitrage 2026 Guide](https://wundertrading.com/journal/en/learn/article/crypto-arbitrage)

### Ensemble Approaches
- [FinRL Contest Ensemble Methods - arXiv:2501.10709](https://arxiv.org/html/2501.10709v1)
- [ACM ICAIF 2024 FinRL Contest](https://open-finance-lab.github.io/finrl-contest-2024.github.io/)
- [Machine Learning Integration in Cryptocurrency Trading](https://link.springer.com/article/10.1007/s44163-025-00785-w)
- [CVML Benchmark - OpenReview/ICLR 2025](https://openreview.net/forum?id=MhD9rLeU31)

### Execution Layer
- [Optimal Execution with RL - arXiv:2411.06389](https://arxiv.org/abs/2411.06389)
- [Market Making with Deep RL from LOB - arXiv:2305.15821](https://arxiv.org/abs/2305.15821)
- [RL-Based Market Making on Non-Stationary LOB - arXiv:2509.12456](https://arxiv.org/html/2509.12456v2)
- [Deep Hawkes Process for HF Market Making](https://link.springer.com/article/10.1007/s42786-024-00049-8)
- [RL for Trade Execution with Market and Limit Orders](https://arxiv.org/pdf/2507.06345)

### Market Microstructure
- [Crypto Market Microstructure - Cornell](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf)
- [SSM for Market Microstructure - Kinlay](https://jonathankinlay.com/2026/03/state-space-models-for-market-microstructure-can-mamba-replace-transformers-in-high-frequency-finance/)
- [Mamba Meets Financial Markets: Graph-Mamba](https://arxiv.org/pdf/2410.03707)
