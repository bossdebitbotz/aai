# AAI LOB Model — Phase Improvement Plan

## Problem Statement

30-day model achieves R²=0.997 on price levels but only 50.5% directional accuracy (coin-flip).
Root causes: direction-blind MSE loss, over-smoothing, no return features, equal feature weighting.

---

## Phase 1: High Impact, Simple Changes

### 1a. Direction Classification Head (Multi-Task Learning)
- Add 3-class (up/flat/down) head sharing the encoder
- Joint loss: `L = L_forecast + 0.01 * L_structure + 0.3 * L_direction`
- Cross-entropy on mid-price direction over prediction horizon
- Threshold for "flat" class: 0.01% of mid-price
- Expected: +3-8 percentage points directional accuracy

### 1b. Directional Sign Loss (DSL)
- Differentiable loss penalizing wrong-direction predictions
- `DSL = mean(|tanh(s * diff(pred)) - tanh(s * diff(target))| / 2)`
- Blend: `L_forecast = 0.92 * MSE + 0.08 * DSL`
- ~20 lines of code

### 1c. Return/Momentum Features
- Log-returns at 5s, 30s, 1min, 5min windows
- OFI rate-of-change (momentum of order flow)
- Price velocity and acceleration
- Realized volatility (rolling std of returns)

### 1d. Reduce Savitzky-Golay Window
- Cut from 21 (105s) to 11 (55s)
- Current window nearly equals prediction horizon — destroys directional signal

### 1e. Gradient Accumulation
- 4-8 mini-batches before weight update
- Effective batch 128 with physical batch 16
- Scale loss by 1/accumulation_steps before backward

---

## Phase 2: Moderate Complexity

### 2a. Multi-Horizon Direction Labels
- Predict direction at 30s, 1min, 2min simultaneously
- Richer directional supervision at each granularity

### 2b. Feature-Weighted Loss
- Weight OFI/derived features higher than deep-level prices
- Level 40 bid price is noise for direction — reduce its gradient contribution

### 2c. Causal Temporal Masking
- Add causal mask to temporal attention in DualAttentionBlock
- Prevents future-to-past information flow within context window

### 2d. Enable Cross-Exchange Features
- `compute_cross_exchange_features()` exists in features.py but is never called
- Cross-exchange OFI divergence is a strong directional signal

---

## Phase 3: Advanced

### 3a. Confidence Thresholding
- Only trade when direction confidence > 0.9
- Research: >70% accuracy on ~31% of opportunities

### 3b. Contrastive Pretraining
- Supervised contrastive learning on up-move vs down-move LOB snapshots
- Pretrain encoder, then fine-tune with multi-task loss

### 3c. OFI-Specific Attention Pathway
- Separate attention pathway for OFI features feeding direction head

---

## Expected Outcomes

| Phase | Direction Accuracy (2 min) |
|-------|---------------------------|
| Current (30-day baseline) | 50.5% |
| Phase 1 | 55-58% |
| Phase 1+2 | 57-62% |
| Phase 1+2+3 | 60-65% |

---

## Benchmark Context (Crypto LOB)

| Horizon | State of Art | Target |
|---------|-------------|--------|
| 30s | 55-65% | 58%+ |
| 1 min | 50-60% | 55%+ |
| 2 min | 48-55% | 53%+ |

## Key References

- TLOB (arXiv:2502.15757) — Dual attention direction classification
- Deep LOB Guide (arXiv:2403.09267) — Confidence thresholding
- DSL (arXiv:2504.04202) — Directional sign loss
- GMADL (arXiv:2412.18405) — Differentiable directional loss
- Kolm & Turiel (2023) — Multi-level OFI alpha extraction
- Crypto LOB (arXiv:2506.05764) — SG filtering validation
