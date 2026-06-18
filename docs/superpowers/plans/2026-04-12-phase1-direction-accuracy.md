# Phase 1: Direction Accuracy Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve mid-price directional accuracy from ~50% to 55-58% by adding a direction classification head, directional sign loss, return/momentum features, and reducing SG over-smoothing.

**Architecture:** Add a 3-class direction head sharing the existing DualAttention encoder. Blend DSL into MSE. Add 8 new momentum/return features to the engineering pipeline. Reduce SG window from 21 to 11. Use gradient accumulation for effective batch=128.

**Tech Stack:** PyTorch, NumPy, SciPy (existing deps). No new dependencies.

**CRITICAL:** All changes go into new files (`model_v2.py`, `features_v2.py`, `train_v2.py`). Original files are untouched. Previous training results in `aai_30day_results.zip` on Google Drive are preserved. New results save to `aai_30day_phase1_results.zip`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `training/model_v2.py` | Create | CompoundAttentionModelV2 (direction head), LOBLossV2 (DSL + direction CE), DSL function |
| `training/features_v2.py` | Create | `engineer_features_v2()` with return/momentum features, reduced SG window |
| `training/test_model_v2.py` | Create | Tests for V2 model, loss, direction head |
| `training/test_features_v2.py` | Create | Tests for new momentum/return features |

The Colab notebook will be updated separately to use V2 imports.

---

### Task 1: Return/Momentum Features

**Files:**
- Create: `training/features_v2.py`
- Create: `training/test_features_v2.py`

- [ ] **Step 1: Write tests for new features**

Create `training/test_features_v2.py`:

```python
"""Tests for Phase 1 feature engineering improvements."""

import sys
import numpy as np

sys.path.insert(0, ".")
from training.features_v2 import engineer_features_v2, compute_momentum_features


def _make_fake_40level(T=200):
    """Create synthetic 40-level LOB data: (T, 162) features."""
    n_levels = 40
    data = np.zeros((T, n_levels * 4 + 2), dtype=np.float64)
    base_price = 100.0
    trend = np.cumsum(np.random.randn(T) * 0.01) + base_price

    for i in range(n_levels):
        data[:, i * 2] = trend - 0.01 * (i + 1)          # bid prices (descending)
        data[:, i * 2 + 1] = np.random.rand(T) * 10 + 1   # bid volumes
        offset = n_levels * 2
        data[:, offset + i * 2] = trend + 0.01 * (i + 1)  # ask prices (ascending)
        data[:, offset + i * 2 + 1] = np.random.rand(T) * 10 + 1

    mid_idx = n_levels * 4
    data[:, mid_idx] = trend
    data[:, mid_idx + 1] = 0.02  # spread
    return data


def test_momentum_features_shape():
    """Momentum features should produce 8 new columns."""
    data = _make_fake_40level(200)
    feats = compute_momentum_features(data, n_levels=40)
    assert len(feats) == 8, f"Expected 8 momentum features, got {len(feats)}"
    for name, arr in feats.items():
        assert arr.shape == (200,), f"{name} shape {arr.shape} != (200,)"
        assert not np.isnan(arr).any(), f"{name} has NaN"
    print("PASS: test_momentum_features_shape")


def test_momentum_features_names():
    """Check that all expected features are present."""
    data = _make_fake_40level(200)
    feats = compute_momentum_features(data, n_levels=40)
    expected = [
        "log_return_1", "log_return_6", "log_return_12", "log_return_60",
        "ofi_roc_6", "price_velocity_6", "price_acceleration_6",
        "realized_vol_12",
    ]
    for name in expected:
        assert name in feats, f"Missing feature: {name}"
    print("PASS: test_momentum_features_names")


def test_engineer_features_v2_output_shape():
    """V2 should produce 219 features (211 base + 8 momentum)."""
    data = _make_fake_40level(200)
    enriched, names = engineer_features_v2(data, n_levels=40)
    assert enriched.shape == (200, 219), f"Shape {enriched.shape} != (200, 219)"
    assert not np.isnan(enriched).any(), f"NaN in enriched features"
    print("PASS: test_engineer_features_v2_output_shape")


def test_savgol_window_reduced():
    """V2 default SG window should be 11, not 21."""
    import inspect
    sig = inspect.signature(engineer_features_v2)
    default = sig.parameters['savgol_window'].default
    assert default == 11, f"Default savgol_window is {default}, expected 11"
    print("PASS: test_savgol_window_reduced")


if __name__ == "__main__":
    test_momentum_features_shape()
    test_momentum_features_names()
    test_engineer_features_v2_output_shape()
    test_savgol_window_reduced()
    print("\nAll features_v2 tests passed!")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Volumes/Docker-SSD/projects/aaiwdbback/aai && python training/test_features_v2.py`
Expected: `ModuleNotFoundError: No module named 'training.features_v2'`

- [ ] **Step 3: Implement features_v2.py**

Create `training/features_v2.py`:

```python
"""
Phase 1 Feature Engineering — adds momentum/return features and reduces SG window.

Changes from features.py:
1. Default savgol_window reduced from 21 to 11 (55s vs 105s)
2. New compute_momentum_features() adding 8 directional features
3. engineer_features_v2() calls original pipeline + momentum features

Total features: 40-level = 162 + 49 (original derived) + 8 (momentum) = 219
"""

import numpy as np
from training.features import (
    get_column_indices,
    apply_savgol_smoothing,
    compute_ofi,
    compute_aggregate_ofi,
    compute_volume_features,
    compute_price_features,
)


def compute_momentum_features(features: np.ndarray, n_levels: int) -> dict[str, np.ndarray]:
    """Compute return, momentum, and volatility features from raw LOB data.

    These features provide explicit directional signal that the model
    cannot easily learn from raw price levels.

    Returns dict with 8 features:
        - log_return_1: 1-step (5s) log return of mid-price
        - log_return_6: 6-step (30s) log return
        - log_return_12: 12-step (1min) log return
        - log_return_60: 60-step (5min) log return
        - ofi_roc_6: rate of change of aggregate OFI over 6 steps
        - price_velocity_6: 6-step price velocity (smoothed 1st derivative)
        - price_acceleration_6: 6-step price acceleration (smoothed 2nd derivative)
        - realized_vol_12: 12-step rolling std of 1-step returns
    """
    idx = get_column_indices(n_levels)
    mid = features[:, idx["mid_price"]]
    T = len(mid)

    result = {}

    # Log returns at multiple horizons
    for lag, name in [(1, "log_return_1"), (6, "log_return_6"),
                      (12, "log_return_12"), (60, "log_return_60")]:
        lr = np.zeros(T)
        safe_mid = np.where(mid > 0, mid, 1.0)
        lr[lag:] = np.log(safe_mid[lag:] / safe_mid[:-lag])
        result[name] = lr

    # OFI rate of change (momentum of order flow)
    ofi = compute_ofi(features, n_levels)
    agg_ofi = compute_aggregate_ofi(ofi)
    ofi_roc = np.zeros(T)
    ofi_roc[6:] = agg_ofi[6:] - agg_ofi[:-6]
    result["ofi_roc_6"] = ofi_roc

    # Price velocity (smoothed first derivative over 6 steps)
    velocity = np.zeros(T)
    velocity[6:] = (mid[6:] - mid[:-6]) / 6.0
    result["price_velocity_6"] = velocity

    # Price acceleration (second derivative)
    accel = np.zeros(T)
    accel[1:] = np.diff(velocity)
    result["price_acceleration_6"] = accel

    # Realized volatility (rolling std of 1-step returns)
    ret1 = result["log_return_1"]
    rv = np.zeros(T)
    for t in range(12, T):
        rv[t] = np.std(ret1[t - 12:t])
    result["realized_vol_12"] = rv

    return result


def engineer_features_v2(
    features: np.ndarray,
    n_levels: int,
    apply_smoothing: bool = True,
    savgol_window: int = 11,  # Reduced from 21 (105s -> 55s)
    savgol_poly: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Phase 1 feature engineering: original pipeline + momentum features.

    Total features for 40-level: 162 base + 49 original derived + 8 momentum = 219.
    """
    T = len(features)
    if T == 0:
        return features, []

    # Step 1: Savitzky-Golay smoothing (reduced window)
    if apply_smoothing and T >= savgol_window:
        features = apply_savgol_smoothing(features, n_levels, savgol_window, savgol_poly)

    derived_arrays = []
    derived_names = []

    # Step 2: Multi-level OFI (same as original)
    ofi = compute_ofi(features, n_levels)
    for k in range(n_levels):
        derived_arrays.append(ofi[:, k])
        derived_names.append(f"ofi_level_{k+1}")

    agg_ofi = compute_aggregate_ofi(ofi)
    derived_arrays.append(agg_ofi)
    derived_names.append("ofi_aggregate")

    # Step 3: Volume features (same as original)
    vol_feats = compute_volume_features(features, n_levels)
    for name, arr in vol_feats.items():
        derived_arrays.append(arr)
        derived_names.append(name)

    # Step 4: Price features (same as original)
    price_feats = compute_price_features(features, n_levels)
    for name, arr in price_feats.items():
        derived_arrays.append(arr)
        derived_names.append(name)

    # Step 5: NEW — Momentum/return features
    momentum_feats = compute_momentum_features(features, n_levels)
    for name, arr in momentum_feats.items():
        derived_arrays.append(arr)
        derived_names.append(name)

    derived = np.column_stack(derived_arrays) if derived_arrays else np.empty((T, 0))
    enriched = np.hstack([features, derived])
    return enriched, derived_names
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Volumes/Docker-SSD/projects/aaiwdbback/aai && python training/test_features_v2.py`
Expected: All 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai
git add training/features_v2.py training/test_features_v2.py
git commit -m "feat: Phase 1 feature engineering — momentum features + reduced SG window"
```

---

### Task 2: Direction Classification Head + DSL Loss

**Files:**
- Create: `training/model_v2.py`
- Create: `training/test_model_v2.py`

- [ ] **Step 1: Write tests for V2 model and loss**

Create `training/test_model_v2.py`:

```python
"""Tests for Phase 1 model improvements."""

import sys
import torch
import numpy as np

sys.path.insert(0, ".")
from training.model_v2 import (
    CompoundAttentionModelV2,
    LOBLossV2,
    directional_sign_loss,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def test_model_v2_forward_shape():
    """V2 model returns (predictions, direction_logits) tuple."""
    model = CompoundAttentionModelV2(
        n_levels=5, n_features=36, context_length=20, prediction_length=4,
        d_model=30, n_heads=3, n_layers=1, d_ff=60, dropout=0.1,
    ).to(DEVICE)

    B = 4
    ctx = torch.randn(B, 20, 36).to(DEVICE)
    eid = torch.zeros(B, dtype=torch.long).to(DEVICE)
    sid = torch.zeros(B, dtype=torch.long).to(DEVICE)

    pred, dir_logits = model(ctx, eid, sid)
    assert pred.shape == (B, 4, 36), f"pred shape {pred.shape}"
    assert dir_logits.shape == (B, 3), f"dir_logits shape {dir_logits.shape}"
    print("PASS: test_model_v2_forward_shape")


def test_model_v2_40level_shape():
    """V2 with 40 levels and 219 features (Phase 1 feature count)."""
    model = CompoundAttentionModelV2(
        n_levels=40, n_features=219, context_length=120, prediction_length=24,
        d_model=66, n_heads=3, n_layers=3, d_ff=264, dropout=0.2,
    ).to(DEVICE)

    B = 2
    ctx = torch.randn(B, 120, 219).to(DEVICE)
    eid = torch.zeros(B, dtype=torch.long).to(DEVICE)
    sid = torch.zeros(B, dtype=torch.long).to(DEVICE)

    pred, dir_logits = model(ctx, eid, sid)
    assert pred.shape == (B, 24, 219), f"pred shape {pred.shape}"
    assert dir_logits.shape == (B, 3), f"dir_logits shape {dir_logits.shape}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PASS: test_model_v2_40level_shape ({n_params:,} params)")


def test_directional_sign_loss():
    """DSL should be 0 when directions match, >0 when they differ."""
    # Perfect prediction — same direction
    pred = torch.tensor([[[1.0, 2.0, 3.0]]]) # increasing
    tgt  = torch.tensor([[[1.0, 2.0, 3.0]]])
    loss_match = directional_sign_loss(pred, tgt, sharpness=10.0)

    # Wrong direction
    pred_bad = torch.tensor([[[3.0, 2.0, 1.0]]]) # decreasing
    loss_bad = directional_sign_loss(pred_bad, tgt, sharpness=10.0)

    assert loss_bad > loss_match, f"DSL should penalize wrong direction: {loss_bad} vs {loss_match}"
    print(f"PASS: test_directional_sign_loss (match={loss_match:.4f}, wrong={loss_bad:.4f})")


def test_lob_loss_v2_components():
    """V2 loss returns 5 components: total, forecast, structure, direction, dsl."""
    n_levels = 5
    n_feat = 36
    B, T = 4, 4

    means = np.random.randn(n_feat).astype(np.float32)
    stds = np.abs(np.random.randn(n_feat)).astype(np.float32) + 0.01

    loss_fn = LOBLossV2(
        n_levels=n_levels, structure_weight=0.01, direction_weight=0.3,
        dsl_weight=0.08, scaler_means=means, scaler_stds=stds,
        mid_price_idx=n_levels * 4,
    ).to(DEVICE)

    pred = torch.randn(B, T, n_feat).to(DEVICE)
    target = torch.randn(B, T, n_feat).to(DEVICE)
    context_last = torch.randn(B, n_feat).to(DEVICE)
    dir_logits = torch.randn(B, 3).to(DEVICE)

    total, forecast, structure, direction, dsl = loss_fn(
        pred, target, dir_logits, context_last
    )

    for name, val in [("total", total), ("forecast", forecast),
                      ("structure", structure), ("direction", direction), ("dsl", dsl)]:
        assert not torch.isnan(val), f"{name} is NaN"
        assert val.requires_grad, f"{name} has no grad"

    print(f"PASS: test_lob_loss_v2_components (total={total.item():.4f})")


def test_direction_labels():
    """Direction labels: 0=down, 1=flat, 2=up based on mid-price change."""
    loss_fn = LOBLossV2(n_levels=5, mid_price_idx=20)

    # Mid-price goes up: context_last mid=100, target end mid=101
    B, T, F = 2, 4, 36
    target = torch.zeros(B, T, F)
    context_last = torch.zeros(B, F)

    context_last[0, 20] = 0.0   # mid at end of context
    target[0, -1, 20] = 0.5     # mid at end of target (up)
    context_last[1, 20] = 0.0
    target[1, -1, 20] = -0.5    # down

    labels = loss_fn._direction_labels(target, context_last)
    assert labels[0].item() == 2, f"Expected up (2), got {labels[0].item()}"
    assert labels[1].item() == 0, f"Expected down (0), got {labels[1].item()}"
    print("PASS: test_direction_labels")


if __name__ == "__main__":
    test_model_v2_forward_shape()
    test_model_v2_40level_shape()
    test_directional_sign_loss()
    test_lob_loss_v2_components()
    test_direction_labels()
    print("\nAll model_v2 tests passed!")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Volumes/Docker-SSD/projects/aaiwdbback/aai && python training/test_model_v2.py`
Expected: `ModuleNotFoundError: No module named 'training.model_v2'`

- [ ] **Step 3: Implement model_v2.py**

Create `training/model_v2.py`:

```python
"""
Phase 1 Model — Direction-aware Compound Attention for LOB forecasting.

Changes from model.py:
1. CompoundAttentionModelV2 adds a 3-class direction classification head
2. LOBLossV2 adds Directional Sign Loss (DSL) and cross-entropy direction loss
3. Forward returns (predictions, direction_logits) tuple

Original model.py is untouched.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.model import (
    Time2Vec,
    CompoundAttributeEmbedding,
    FeatureProjection,
    DualAttentionBlock,
    WarmupDecayScheduler,
)


# ---------------------------------------------------------------------------
# Directional Sign Loss (arXiv:2504.04202)
# ---------------------------------------------------------------------------

def directional_sign_loss(
    pred: torch.Tensor, target: torch.Tensor, sharpness: float = 10.0
) -> torch.Tensor:
    """Differentiable loss penalizing wrong-direction predictions.

    Computes finite differences (changes between timesteps), then
    penalizes when predicted direction disagrees with actual direction.
    Uses tanh as a differentiable approximation to sign().

    Args:
        pred: (B, T, F) predicted features
        target: (B, T, F) ground truth features
        sharpness: controls steepness of tanh approximation (higher = closer to sign)

    Returns:
        Scalar loss in [0, 1].
    """
    pred_diff = torch.diff(pred, dim=1)
    target_diff = torch.diff(target, dim=1)

    pred_sign = torch.tanh(sharpness * pred_diff)
    target_sign = torch.tanh(sharpness * target_diff)

    return torch.mean(torch.abs(pred_sign - target_sign) / 2.0)


# ---------------------------------------------------------------------------
# V2 Model with Direction Head
# ---------------------------------------------------------------------------

class CompoundAttentionModelV2(nn.Module):
    """Compound Attention Model with auxiliary direction classification head.

    Identical encoder to CompoundAttentionModel. Adds:
    - direction_head: 3-class (down/flat/up) classifier on mid-price
    - Returns (predictions, direction_logits) tuple

    The direction gradient flows back through the shared encoder,
    forcing it to learn direction-discriminative representations.
    """

    def __init__(
        self,
        n_levels: int = 40,
        n_features: int = 219,
        context_length: int = 120,
        prediction_length: int = 24,
        d_model: int = 66,
        n_heads: int = 3,
        n_layers: int = 3,
        d_ff: int = 264,
        dropout: float = 0.2,
        n_exchanges: int = 4,
        n_symbols: int = 4,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model

        # Shared encoder (identical to V1)
        self.time_embed = Time2Vec(d_model)
        include_derived = n_features > n_levels * 4 + 2
        self.attr_embed = CompoundAttributeEmbedding(
            d_model, n_levels, n_exchanges, n_symbols,
            include_derived=include_derived,
        )
        self.feat_proj = FeatureProjection(d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, context_length, 1, d_model) * 0.02
        )
        self.encoder_layers = nn.ModuleList([
            DualAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Regression head (same as V1)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, prediction_length),
        )

        # NEW: Direction classification head
        # Pools across all features, classifies into 3 classes
        self.direction_head = nn.Sequential(
            nn.Linear(d_model * n_features, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 3),  # down=0, flat=1, up=2
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        context: torch.Tensor,
        exchange_ids: torch.Tensor,
        symbol_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            predictions: (B, T_pred, F)
            direction_logits: (B, 3) — logits for down/flat/up
        """
        B, T, F = context.shape

        # Shared encoder (identical to V1)
        attr = self.attr_embed(context, exchange_ids, symbol_ids, is_target=False)
        x = self.feat_proj(context, attr)
        x = x + self.pos_embed[:, :T, :, :]

        t = torch.linspace(0, 1, T, device=context.device).unsqueeze(0).expand(B, -1)
        time_e = self.time_embed(t)
        x = x + time_e.unsqueeze(2)

        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)

        # Regression prediction (same as V1)
        last_state = x[:, -1, :, :]  # (B, F, d)
        pred = self.prediction_head(last_state)  # (B, F, T_pred)
        pred = pred.permute(0, 2, 1)  # (B, T_pred, F)

        # Direction classification from pooled last state
        pooled = last_state.reshape(B, -1)  # (B, F*d)
        dir_logits = self.direction_head(pooled)  # (B, 3)

        return pred, dir_logits


# ---------------------------------------------------------------------------
# V2 Loss: MSE + Structure + Direction CE + DSL
# ---------------------------------------------------------------------------

class LOBLossV2(nn.Module):
    """Combined loss for direction-aware LOB forecasting.

    L = (1 - dsl_weight) * MSE + dsl_weight * DSL
        + structure_weight * structure_loss
        + direction_weight * cross_entropy_direction

    Direction labels computed from mid-price change over prediction horizon:
        0 = down (change < -threshold)
        1 = flat (|change| <= threshold)
        2 = up (change > threshold)
    """

    def __init__(
        self,
        n_levels: int = 40,
        structure_weight: float = 0.01,
        direction_weight: float = 0.3,
        dsl_weight: float = 0.08,
        dsl_sharpness: float = 10.0,
        flat_threshold: float = 0.05,
        scaler_means=None,
        scaler_stds=None,
        mid_price_idx: int = 160,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.w_struct = structure_weight
        self.w_dir = direction_weight
        self.w_dsl = dsl_weight
        self.dsl_sharpness = dsl_sharpness
        self.flat_threshold = flat_threshold
        self.mid_price_idx = mid_price_idx

        self.bid_price_indices = [i * 2 for i in range(n_levels)]
        self.ask_price_indices = [n_levels * 2 + i * 2 for i in range(n_levels)]

        if scaler_means is not None and scaler_stds is not None:
            self.register_buffer('_means', torch.tensor(scaler_means, dtype=torch.float32))
            self.register_buffer('_stds', torch.tensor(scaler_stds, dtype=torch.float32))
        else:
            self._means = None
            self._stds = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dir_logits: torch.Tensor,
        context_last: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: (B, T_pred, F) predicted features
            target: (B, T_pred, F) ground truth
            dir_logits: (B, 3) direction logits from model
            context_last: (B, F) last context timestep (for direction label computation)

        Returns:
            (total_loss, forecast_loss, structure_loss, direction_loss, dsl_loss)
        """
        # Forecast: blended MSE + DSL
        mse_loss = F.mse_loss(pred, target)
        dsl_loss = directional_sign_loss(pred, target, self.dsl_sharpness)
        forecast_loss = (1 - self.w_dsl) * mse_loss + self.w_dsl * dsl_loss

        # Structure loss (same as V1)
        structure_loss = self._structure_loss(pred)

        # Direction classification loss
        dir_labels = self._direction_labels(target, context_last)
        direction_loss = F.cross_entropy(dir_logits, dir_labels)

        total = forecast_loss + self.w_struct * structure_loss + self.w_dir * direction_loss
        return total, forecast_loss, structure_loss, direction_loss, dsl_loss

    def _direction_labels(
        self, target: torch.Tensor, context_last: torch.Tensor
    ) -> torch.Tensor:
        """Compute 3-class direction labels from mid-price change.

        0 = down, 1 = flat, 2 = up
        """
        mid_ctx = context_last[:, self.mid_price_idx]
        mid_tgt = target[:, -1, self.mid_price_idx]
        change = mid_tgt - mid_ctx

        labels = torch.ones(change.shape[0], dtype=torch.long, device=change.device)  # flat
        labels[change > self.flat_threshold] = 2   # up
        labels[change < -self.flat_threshold] = 0  # down
        return labels

    def _structure_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Structure preservation loss (identical to V1)."""
        B, T, n_feat = pred.shape
        loss = torch.tensor(0.0, device=pred.device)

        bid_prices = pred[:, :, self.bid_price_indices]
        ask_prices = pred[:, :, self.ask_price_indices]

        if self._means is not None:
            bid_means = self._means[self.bid_price_indices]
            bid_stds = self._stds[self.bid_price_indices]
            ask_means = self._means[self.ask_price_indices]
            ask_stds = self._stds[self.ask_price_indices]
            bid_prices = bid_prices * bid_stds + bid_means
            ask_prices = ask_prices * ask_stds + ask_means
            ref_price = max(bid_means[0].item(), 1.0)
        else:
            ref_price = 1.0

        if self.n_levels > 1:
            ask_diff = ask_prices[:, :, :-1] - ask_prices[:, :, 1:]
            loss = loss + F.relu(ask_diff).sum() / (B * T * ref_price)
            bid_diff = bid_prices[:, :, 1:] - bid_prices[:, :, :-1]
            loss = loss + F.relu(bid_diff).sum() / (B * T * ref_price)

        crossed = bid_prices[:, :, 0] - ask_prices[:, :, 0]
        loss = loss + F.relu(crossed).sum() / (B * T * ref_price)
        return loss
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Volumes/Docker-SSD/projects/aaiwdbback/aai && python training/test_model_v2.py`
Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai
git add training/model_v2.py training/test_model_v2.py
git commit -m "feat: Phase 1 model — direction head + DSL + V2 loss"
```

---

### Task 3: Update Colab Notebook for Phase 1 Training

**Files:**
- Modify: Colab notebook (via MCP)

This task updates the Colab notebook cells to use V2 imports, 219 features, gradient accumulation, and save results to a separate file (`aai_30day_phase1_results.zip`).

- [ ] **Step 1: Re-zip training code including V2 files**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai
rm -f training.zip
zip -r training.zip training/ -x "training/__pycache__/*" "training/*.pyc"
```

Upload to Google Drive "training data" folder (overwrite training.zip).

- [ ] **Step 2: Update Cell 3 (DataLoaders) to use features_v2**

Key changes:
- Import `engineer_features_v2` instead of `engineer_features`
- `n_features` becomes 219 (was 211)
- Batch size stays at 32 (physical), gradient accumulation in training cell

- [ ] **Step 3: Update Cell 4 (Model) to use V2**

Key changes:
- Import `CompoundAttentionModelV2` and `LOBLossV2`
- `n_features=219`
- `dropout=0.2`
- `loss_fn = LOBLossV2(...)` with direction_weight=0.3, dsl_weight=0.08

- [ ] **Step 4: Update Cell 5 (Training loop) with gradient accumulation**

Key changes:
- `ACCUMULATION_STEPS = 4` (effective batch = 128)
- Scale loss by `1/ACCUMULATION_STEPS` before backward
- Call `optimizer.step()` and `scheduler.step()` every 4 mini-batches
- Model returns `(pred, dir_logits)` tuple
- Loss function takes `(pred, target, dir_logits, context_last)`
- Log direction accuracy alongside losses
- Pass `context_last = batch['context'][:, -1, :]` to loss

- [ ] **Step 5: Update Cell 7 (Save) to output `aai_30day_phase1_results.zip`**

Change zip name to `aai_30day_phase1_results.zip` to preserve previous results.

- [ ] **Step 6: Update Cell 8 (Accuracy metrics) for V2 model**

Key change: model returns tuple, so unpack `pred, dir_logits = model(...)`.

- [ ] **Step 7: Commit notebook changes note**

No git commit needed for Colab notebook (saved on Drive), but commit the re-zipped training.zip:
```bash
git add -A
git commit -m "feat: Phase 1 Colab training notebook — V2 model + features + grad accum"
```

---

### Task 4: Run Phase 1 Training

- [ ] **Step 1: Upload new training.zip to Google Drive**

Replace old `training.zip` in "training data" folder with the version containing V2 files.

- [ ] **Step 2: Run all notebook cells (1-5)**

Execute: GPU check → Drive mount/extract → DataLoaders (219 features) → V2 Model → Training with gradient accumulation.

- [ ] **Step 3: Run evaluation cells (6-8)**

Load best checkpoint → test eval → accuracy metrics → plot → save to `aai_30day_phase1_results.zip`.

- [ ] **Step 4: Compare Phase 1 vs baseline**

Expected comparison:

| Metric | Baseline | Phase 1 Target |
|--------|----------|---------------|
| Val loss | 0.946 | <0.95 |
| Direction 30s | 50.1% | 55%+ |
| Direction 1min | 50.8% | 53%+ |
| Direction 2min | 50.5% | 52%+ |
| Structure | 0.006 | <0.01 |
