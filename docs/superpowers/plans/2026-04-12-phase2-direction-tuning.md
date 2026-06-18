# Phase 2: Direction Accuracy Tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push directional accuracy from ~51% toward 55%+ by fixing the flat_threshold, adding multi-horizon direction labels, causal temporal masking, and feature-weighted loss.

**Architecture:** Update model_v2.py in place: add CausalDualAttentionBlock, expand direction head to multi-horizon (3 horizons × 3 classes = 9 outputs), tighten flat_threshold to 0.01, add feature weighting in loss. Update existing tests. No new files.

**Tech Stack:** PyTorch, NumPy (existing deps). No new dependencies.

**Note:** Cross-exchange features deferred to Phase 3 — requires time-aligned multi-stream dataset pipeline that doesn't exist yet.

---

## File Structure

| File | Action | Changes |
|------|--------|---------|
| `training/model_v2.py` | Modify | CausalDualAttentionBlock, multi-horizon direction head, feature-weighted MSE, flat_threshold=0.01 |
| `training/test_model_v2.py` | Modify | Add tests for causal mask, multi-horizon labels, feature weights |

---

### Task 1: Causal Temporal Masking + Flat Threshold Fix + Multi-Horizon Direction + Feature-Weighted Loss

All changes are in `model_v2.py` and `test_model_v2.py`. Since these are tightly coupled (the loss needs multi-horizon labels from the model, which uses causal blocks), they're one task.

**Files:**
- Modify: `training/model_v2.py`
- Modify: `training/test_model_v2.py`

- [ ] **Step 1: Add new tests to test_model_v2.py**

Append these tests to the existing file:

```python
def test_causal_dual_attention_block():
    """Causal block should produce different output than non-causal."""
    from training.model_v2 import CausalDualAttentionBlock
    from training.model import DualAttentionBlock

    d, h, ff = 30, 3, 60
    causal = CausalDualAttentionBlock(d, h, ff, 0.1).to(DEVICE)
    # Copy weights from causal to a non-causal block for fair comparison
    non_causal = DualAttentionBlock(d, h, ff, 0.1).to(DEVICE)
    non_causal.load_state_dict(causal.state_dict(), strict=False)

    x = torch.randn(2, 10, 5, d).to(DEVICE)
    out_causal = causal(x)
    out_non_causal = non_causal(x)

    assert out_causal.shape == (2, 10, 5, d)
    # They should differ because causal masking restricts attention
    assert not torch.allclose(out_causal, out_non_causal, atol=1e-5), \
        "Causal and non-causal outputs should differ"
    logger.info("PASS: test_causal_dual_attention_block")


def test_multi_horizon_direction_labels():
    """Multi-horizon labels should return (B, 3) tensor with labels at 3 horizons."""
    loss_fn = LOBLossV2(n_levels=5, mid_price_idx=20, flat_threshold=0.01)

    B, T, F = 2, 24, 36
    target = torch.zeros(B, T, F)
    context_last = torch.zeros(B, F)

    # Stream 0: price goes up at all horizons
    context_last[0, 20] = 0.0
    target[0, 5, 20] = 0.5    # 30s: up
    target[0, 11, 20] = 0.8   # 1min: up
    target[0, 23, 20] = 1.0   # 2min: up

    # Stream 1: price goes down
    context_last[1, 20] = 0.0
    target[1, 5, 20] = -0.5
    target[1, 11, 20] = -0.8
    target[1, 23, 20] = -1.0

    labels = loss_fn._direction_labels(target, context_last)
    assert labels.shape == (B, 3), f"Expected (2, 3), got {labels.shape}"
    # All up for stream 0
    assert (labels[0] == 2).all(), f"Expected all up, got {labels[0]}"
    # All down for stream 1
    assert (labels[1] == 0).all(), f"Expected all down, got {labels[1]}"
    logger.info("PASS: test_multi_horizon_direction_labels")


def test_feature_weighted_mse():
    """Feature-weighted MSE should weight mid-price/OFI higher than deep levels."""
    loss_fn = LOBLossV2(n_levels=5, mid_price_idx=20, use_feature_weights=True)

    B, T, F = 2, 4, 36
    pred = torch.randn(B, T, F, requires_grad=True)
    target = torch.randn(B, T, F)
    dir_logits = torch.randn(B, 9, requires_grad=True)  # 3 horizons * 3 classes
    context_last = torch.randn(B, F)

    total, forecast, structure, direction, dsl = loss_fn(pred, target, dir_logits, context_last)
    assert not torch.isnan(total), "Total loss is NaN with feature weights"
    total.backward()
    assert pred.grad is not None
    logger.info(f"PASS: test_feature_weighted_mse (total={total.item():.4f})")


def test_flat_threshold_tightened():
    """Default flat_threshold should be 0.01, not 0.05."""
    import inspect
    sig = inspect.signature(LOBLossV2.__init__)
    default = sig.parameters['flat_threshold'].default
    assert default == 0.01, f"flat_threshold default is {default}, expected 0.01"
    logger.info("PASS: test_flat_threshold_tightened")
```

Also update the `__main__` block to run the new tests.

- [ ] **Step 2: Run tests to confirm new tests fail**

Run: `cd /Volumes/Docker-SSD/projects/aaiwdbback/aai && .venv/bin/python training/test_model_v2.py`
Expected: ImportError for `CausalDualAttentionBlock`, or AttributeError for missing multi-horizon labels.

- [ ] **Step 3: Implement CausalDualAttentionBlock in model_v2.py**

Add after the `directional_sign_loss` function, before `CompoundAttentionModelV2`:

```python
class CausalDualAttentionBlock(nn.Module):
    """DualAttentionBlock with causal masking on temporal attention.

    Feature attention (across features per timestep) is unchanged.
    Temporal attention (across timesteps per feature) uses a causal mask
    so timestep t can only attend to timesteps <= t.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.feat_norm = nn.LayerNorm(d_model)
        self.feat_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.temp_norm = nn.LayerNorm(d_model)
        self.temp_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, d = x.shape

        # 1. Feature attention (no mask — features are unordered)
        x_feat = x.reshape(B * T, F, d)
        normed = self.feat_norm(x_feat)
        attn_out, _ = self.feat_attn(normed, normed, normed)
        x_feat = x_feat + attn_out
        x = x_feat.reshape(B, T, F, d)

        # 2. Temporal attention WITH causal mask
        x_temp = x.permute(0, 2, 1, 3).reshape(B * F, T, d)
        normed = self.temp_norm(x_temp)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        attn_out, _ = self.temp_attn(normed, normed, normed, attn_mask=causal_mask, is_causal=True)
        x_temp = x_temp + attn_out
        x = x_temp.reshape(B, F, T, d).permute(0, 2, 1, 3)

        # 3. Feedforward
        normed = self.ff_norm(x)
        x = x + self.ff(normed)
        return x
```

- [ ] **Step 4: Update CompoundAttentionModelV2 to use CausalDualAttentionBlock**

Change the encoder_layers initialization from:
```python
self.encoder_layers = nn.ModuleList([
    DualAttentionBlock(d_model, n_heads, d_ff, dropout)
    for _ in range(n_layers)
])
```
To:
```python
self.encoder_layers = nn.ModuleList([
    CausalDualAttentionBlock(d_model, n_heads, d_ff, dropout)
    for _ in range(n_layers)
])
```

Also update the direction head output from 3 to 9 (3 horizons × 3 classes):
```python
self.direction_head = nn.Sequential(
    nn.Linear(d_model * n_features, d_ff),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_ff, 9),  # 3 horizons (6/12/24 steps) × 3 classes
)
```

And remove the `DualAttentionBlock` import since we no longer use it directly.

- [ ] **Step 5: Update LOBLossV2 for multi-horizon + feature weights + threshold**

Changes to `LOBLossV2.__init__`:
- `flat_threshold` default: `0.05` → `0.01`
- Add `use_feature_weights: bool = False` parameter
- Add `direction_horizons: list = [5, 11, 23]` parameter (step indices for 30s, 1min, 2min)
- Build feature weight vector in `__init__` when `use_feature_weights=True`

Changes to `_direction_labels`:
- Return `(B, 3)` tensor instead of `(B,)` — one label per horizon
- Each column uses the target at the corresponding horizon step

Changes to `forward`:
- `dir_logits` is now `(B, 9)`, reshaped to `(B, 3, 3)` for per-horizon CE
- Direction loss = mean of CE across 3 horizons
- When feature weights are enabled, MSE uses weighted mean instead of uniform mean

Here's the updated `LOBLossV2` class:

```python
class LOBLossV2(nn.Module):
    """Combined loss for V2: forecast + structure + direction + DSL.

    Phase 2 changes:
    - flat_threshold tightened to 0.01 (from 0.05)
    - Multi-horizon direction labels at 30s/1min/2min
    - Optional feature-weighted MSE (downweights deep levels)
    """

    def __init__(
        self,
        n_levels: int = 40,
        structure_weight: float = 0.01,
        direction_weight: float = 0.3,
        dsl_weight: float = 0.08,
        mid_price_idx: int = 160,
        flat_threshold: float = 0.01,
        direction_horizons: tuple = (5, 11, 23),
        use_feature_weights: bool = False,
        scaler_means=None,
        scaler_stds=None,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.w_struct = structure_weight
        self.w_dir = direction_weight
        self.dsl_weight = dsl_weight
        self.mid_price_idx = mid_price_idx
        self.flat_threshold = flat_threshold
        self.direction_horizons = direction_horizons
        self.use_feature_weights = use_feature_weights

        self._base_loss = LOBLoss(
            n_levels=n_levels, structure_weight=0.0,
            scaler_means=scaler_means, scaler_stds=scaler_stds,
        )

        # Feature weights: higher for mid/spread/OFI, lower for deep levels
        if use_feature_weights:
            # Will be set after we know n_features from first forward call
            self._feature_weights = None
        else:
            self._feature_weights = None

    def _build_feature_weights(self, n_features: int, device: torch.device) -> torch.Tensor:
        """Build per-feature weights. Called once on first forward."""
        w = torch.ones(n_features, device=device)
        n = self.n_levels
        # Levels 1-5: weight 2.0 (most important for trading)
        # Levels 6-20: weight 1.0 (default)
        # Levels 21-40: weight 0.5 (deep book, less relevant for direction)
        for i in range(n):
            level = i + 1
            for col_offset in [i * 2, i * 2 + 1, n * 2 + i * 2, n * 2 + i * 2 + 1]:
                if col_offset < n_features:
                    if level <= 5:
                        w[col_offset] = 2.0
                    elif level > 20:
                        w[col_offset] = 0.5
        # Mid-price and spread: weight 3.0
        if self.mid_price_idx < n_features:
            w[self.mid_price_idx] = 3.0
        if self.mid_price_idx + 1 < n_features:
            w[self.mid_price_idx + 1] = 2.0  # spread
        # OFI features (after base + mid + spread): weight 2.0
        ofi_start = n * 4 + 2
        ofi_end = min(ofi_start + n + 1, n_features)  # n levels + aggregate
        w[ofi_start:ofi_end] = 2.0
        # Normalize so mean weight = 1.0
        w = w / w.mean()
        return w

    def _direction_labels(
        self, target: torch.Tensor, context_last: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-horizon direction labels.

        Returns:
            labels: (B, n_horizons) — 0=down, 1=flat, 2=up per horizon
        """
        mid_start = context_last[:, self.mid_price_idx]
        labels_list = []
        for h in self.direction_horizons:
            h_idx = min(h, target.shape[1] - 1)
            mid_end = target[:, h_idx, self.mid_price_idx]
            change = mid_end - mid_start
            h_labels = torch.ones(change.shape[0], dtype=torch.long, device=change.device)
            h_labels[change > self.flat_threshold] = 2
            h_labels[change < -self.flat_threshold] = 0
            labels_list.append(h_labels)
        return torch.stack(labels_list, dim=1)  # (B, n_horizons)

    def forward(
        self, pred, target, dir_logits, context_last,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, F = pred.shape

        # MSE (optionally feature-weighted)
        if self.use_feature_weights:
            if self._feature_weights is None:
                self._feature_weights = self._build_feature_weights(F, pred.device)
            weights = self._feature_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, F)
            mse = (weights * (pred - target) ** 2).mean()
        else:
            mse = F.mse_loss(pred, target)

        dsl = directional_sign_loss(pred, target)
        forecast_loss = (1 - self.dsl_weight) * mse + self.dsl_weight * dsl

        structure_loss = self._base_loss._structure_loss(pred)

        # Multi-horizon direction loss
        labels = self._direction_labels(target, context_last)  # (B, n_horizons)
        n_horizons = len(self.direction_horizons)
        logits_reshaped = dir_logits.reshape(B, n_horizons, 3)  # (B, H, 3)
        dir_loss = sum(
            F.cross_entropy(logits_reshaped[:, h, :], labels[:, h])
            for h in range(n_horizons)
        ) / n_horizons

        total = forecast_loss + self.w_struct * structure_loss + self.w_dir * dir_loss
        return total, forecast_loss, structure_loss, dir_loss, dsl
```

- [ ] **Step 6: Run all tests**

Run: `cd /Volumes/Docker-SSD/projects/aaiwdbback/aai && .venv/bin/python training/test_model_v2.py`
Expected: All 9 tests pass (5 original + 4 new).

Note: The original `test_lob_loss_v2_components` and `test_direction_labels` tests will need minor updates:
- `dir_logits` shape changes from `(B, 3)` to `(B, 9)` 
- `_direction_labels` returns `(B, 3)` instead of `(B,)`

Update these in-place when implementing.

- [ ] **Step 7: Commit**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai
git add training/model_v2.py training/test_model_v2.py
git commit -m "feat: Phase 2 — causal masking, multi-horizon direction, feature weights, threshold=0.01"
```

---

### Task 2: Update Colab Notebook + Run Phase 2 Training

- [ ] **Step 1: Re-zip training code and upload to Drive**

```bash
cd /Volumes/Docker-SSD/projects/aaiwdbback/aai
rm -f training.zip
zip -r training.zip training/ -x "training/__pycache__/*" "training/*.pyc"
```

Upload to Google Drive "training data" folder.

- [ ] **Step 2: Update notebook Cell 4 (Model)**

Key changes:
- `dir_logits` is now `(B, 9)` not `(B, 3)`
- Add `use_feature_weights=True` to LOBLossV2 constructor
- flat_threshold is already 0.01 by default

- [ ] **Step 3: Update notebook Cell 5 (Training loop)**

Key change in direction accuracy tracking:
```python
# Multi-horizon direction accuracy
with torch.no_grad():
    dir_labels = loss_fn._direction_labels(target, context_last)  # (B, 3)
    dir_preds = dir_logits.reshape(-1, 3, 3).argmax(dim=2)  # (B, 3)
    dir_correct += (dir_preds == dir_labels).sum().item()
    dir_total += dir_labels.numel()
```

- [ ] **Step 4: Update notebook Cell 7 (Save)**

Change zip name to `aai_30day_phase2_results.zip`.

- [ ] **Step 5: Update notebook Cell 8 (Accuracy metrics)**

Unpack `pred, dir_logits = model(...)` where `dir_logits` is `(B, 9)`.

- [ ] **Step 6: Run all cells and evaluate**

Expected comparison:

| Metric | Baseline | Phase 1 | Phase 2 Target |
|--------|----------|---------|---------------|
| Val loss | 0.946 | 0.912 | <0.92 |
| Direction 30s | 50.1% | 51.3% | 55%+ |
| Direction 1min | 50.8% | 51.3% | 53%+ |
| Direction 2min | 50.5% | 50.9% | 52%+ |
