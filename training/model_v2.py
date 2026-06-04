"""
Compound Attention Model V2 for LOB Forecasting with Directional Prediction.

Extends the base CompoundAttentionModel with:
- Direction classification head (up / flat / down)
- Directional Sign Loss (DSL) from arXiv:2504.04202
- LOBLossV2 combining forecast, structure, direction, and DSL losses

Reuses encoder components from training.model — no duplication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.model import (
    Time2Vec,
    CompoundAttributeEmbedding,
    FeatureProjection,
    DualAttentionBlock,
    LOBLoss,
)


# ---------------------------------------------------------------------------
# Causal Dual Attention Block (causal masking on temporal attention)
# ---------------------------------------------------------------------------

class CausalDualAttentionBlock(nn.Module):
    """Factored attention with causal masking on the temporal axis.

    Same structure as DualAttentionBlock but temporal attention uses a causal
    mask so each timestep can only attend to itself and earlier timesteps.
    Feature attention remains unmasked.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Feature attention (across features for each timestep) — no mask
        self.feat_norm = nn.LayerNorm(d_model)
        self.feat_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Temporal attention (across timesteps for each feature) — causal
        self.temp_norm = nn.LayerNorm(d_model)
        self.temp_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feedforward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F, d) — 4D feature tensor

        Returns:
            (B, T, F, d)
        """
        B, T, F, d = x.shape

        # 1. Feature attention: attend across F for each timestep (no mask)
        x_feat = x.reshape(B * T, F, d)
        normed = self.feat_norm(x_feat)
        attn_out, _ = self.feat_attn(normed, normed, normed)
        x_feat = x_feat + attn_out
        x = x_feat.reshape(B, T, F, d)

        # 2. Temporal attention: attend across T for each feature (causal)
        x_temp = x.permute(0, 2, 1, 3).reshape(B * F, T, d)  # (B*F, T, d)
        normed = self.temp_norm(x_temp)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        attn_out, _ = self.temp_attn(
            normed, normed, normed,
            attn_mask=causal_mask, is_causal=True,
        )
        x_temp = x_temp + attn_out
        x = x_temp.reshape(B, F, T, d).permute(0, 2, 1, 3)  # (B, T, F, d)

        # 3. Feedforward
        normed = self.ff_norm(x)
        x = x + self.ff(normed)

        return x


# ---------------------------------------------------------------------------
# Directional Sign Loss (arXiv:2504.04202)
# ---------------------------------------------------------------------------

def directional_sign_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sharpness: float = 10.0,
) -> torch.Tensor:
    """Differentiable directional sign loss.

    Computes temporal diffs along the time dimension, uses
    tanh(sharpness * diff) as a differentiable sign approximation,
    and returns the mean absolute difference / 2.

    Args:
        pred: (B, T, F) predicted sequence
        target: (B, T, F) target sequence
        sharpness: controls steepness of tanh approximation

    Returns:
        scalar DSL value
    """
    # Temporal diffs along time dimension
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    target_diff = target[:, 1:, :] - target[:, :-1, :]

    # Differentiable sign approximation
    pred_sign = torch.tanh(sharpness * pred_diff)
    target_sign = torch.tanh(sharpness * target_diff)

    # Mean absolute difference / 2
    return torch.mean(torch.abs(pred_sign - target_sign)) / 2.0


# ---------------------------------------------------------------------------
# Direction label + accuracy helpers (reused by train/eval/backtest)
# ---------------------------------------------------------------------------

def compute_direction_labels(target, context_last, mid_price_idx, direction_horizons, flat_threshold):
    """Multi-horizon up/flat/down labels from mid-price change. Returns (B, H) longs (0=down,1=flat,2=up)."""
    B = context_last.shape[0]
    mid_start = context_last[:, mid_price_idx]
    T_pred = target.shape[1]
    labels_list = []
    for h in direction_horizons:
        t_idx = min(h, T_pred - 1)
        change = target[:, t_idx, mid_price_idx] - mid_start
        lab = torch.ones(B, dtype=torch.long, device=change.device)  # flat
        lab[change > flat_threshold] = 2
        lab[change < -flat_threshold] = 0
        labels_list.append(lab)
    return torch.stack(labels_list, dim=1)


def directional_accuracy(dir_logits, labels):
    """Per-horizon accuracy. dir_logits (B,9), labels (B,H). Returns (H,) float tensor."""
    H = labels.shape[1]
    logits = dir_logits.reshape(-1, H, 3)
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean(dim=0)


# ---------------------------------------------------------------------------
# Compound Attention Model V2
# ---------------------------------------------------------------------------

class CompoundAttentionModelV2(nn.Module):
    """Compound Attention Model with directional prediction head.

    Identical encoder to CompoundAttentionModel. Adds a direction
    classification head that predicts up / flat / down from the
    last encoder state.

    Returns (predictions, direction_logits) tuple.
    """

    def __init__(
        self,
        n_levels: int = 40,
        n_features: int = 211,
        context_length: int = 120,
        prediction_length: int = 24,
        d_model: int = 66,
        n_heads: int = 3,
        n_layers: int = 3,
        d_ff: int = 264,
        dropout: float = 0.1,
        n_exchanges: int = 4,
        n_symbols: int = 4,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model

        # === Encoder (identical to CompoundAttentionModel) ===

        # Embeddings
        self.time_embed = Time2Vec(d_model)
        include_derived = n_features > n_levels * 4 + 2
        self.attr_embed = CompoundAttributeEmbedding(
            d_model, n_levels, n_exchanges, n_symbols,
            include_derived=include_derived,
        )
        self.feat_proj = FeatureProjection(d_model)

        # Temporal position embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.randn(1, context_length, 1, d_model) * 0.02
        )

        # Causal dual attention encoder (4D tensor throughout)
        self.encoder_layers = nn.ModuleList([
            CausalDualAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Output: per-feature prediction head from last context state
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, prediction_length),
        )

        # === V2 addition: direction classification head ===
        # 3 horizons x 3 classes (down/flat/up)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model * n_features, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 9),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization."""
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
        Args:
            context: (B, T_ctx, F) -- context features (scaled)
            exchange_ids: (B,) -- exchange indices
            symbol_ids: (B,) -- symbol indices

        Returns:
            predictions: (B, T_pred, F) -- predicted features for horizon
            direction_logits: (B, 9) -- logits for 3 horizons x 3 classes
        """
        B, T, F = context.shape

        # 1. Attribute embeddings: (B, T, F_attr, d)
        # The attribute buffer may cover fewer features than the input
        # (e.g. when new features are added in V2). Pad with zeros if needed.
        n_attr = len(self.attr_embed.attr_levels)
        F_attr = min(F, n_attr)
        attr = self.attr_embed(
            context[:, :, :F_attr], exchange_ids, symbol_ids, is_target=False
        )
        if F > n_attr:
            # Zero-pad attribute embedding for extra features
            pad = torch.zeros(B, T, F - n_attr, self.d_model, device=context.device)
            attr = torch.cat([attr, pad], dim=2)

        # 2. Feature projection + attribute: (B, T, F, d)
        x = self.feat_proj(context, attr)

        # 3. Add temporal position embedding
        x = x + self.pos_embed[:, :T, :, :]

        # 4. Add Time2Vec
        t = torch.linspace(0, 1, T, device=context.device).unsqueeze(0).expand(B, -1)
        time_e = self.time_embed(t)  # (B, T, d)
        x = x + time_e.unsqueeze(2)  # broadcast over features dim

        # 5. Dual attention encoder: stays (B, T, F, d)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)

        # 6. Per-feature prediction from last timestep
        last_state = x[:, -1, :, :]  # (B, F, d)
        pred = self.prediction_head(last_state)  # (B, F, T_pred)
        pred = pred.permute(0, 2, 1)  # (B, T_pred, F)

        # 7. Direction classification from last state
        dir_input = last_state.reshape(B, -1)  # (B, F * d)
        dir_logits = self.direction_head(dir_input)  # (B, 9)

        return pred, dir_logits


# ---------------------------------------------------------------------------
# LOB Loss V2
# ---------------------------------------------------------------------------

class LOBLossV2(nn.Module):
    """Combined loss for V2: forecast + structure + direction + DSL.

    Returns 5 values: (total, forecast, structure, direction, dsl).

    Args:
        n_levels: number of LOB depth levels
        structure_weight: weight for structure preservation loss
        direction_weight: weight for direction classification loss
        dsl_weight: weight of DSL within forecast loss blend
        mid_price_idx: column index for mid-price in feature vector
        flat_threshold: threshold for classifying as 'flat'
        direction_horizons: tuple of timestep indices for multi-horizon labels
        use_feature_weights: whether to apply feature-weighted MSE
        scaler_means: optional scaler means for inverse-transform
        scaler_stds: optional scaler stds for inverse-transform
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
        self._feature_weights = None  # built lazily

        # Reuse structure loss from original LOBLoss
        self._base_loss = LOBLoss(
            n_levels=n_levels,
            structure_weight=0.0,  # we handle weighting ourselves
            scaler_means=scaler_means,
            scaler_stds=scaler_stds,
        )

    def _build_feature_weights(self, n_features: int, device: torch.device) -> torch.Tensor:
        """Build per-feature weight vector lazily on first forward call.

        Weight scheme:
        - Levels 1-5 (indices 0-19, bid/ask price+size): 2.0
        - Levels 21-40 (indices 80-159, bid/ask price+size): 0.5
        - mid_price (index mid_price_idx): 3.0
        - spread (index mid_price_idx + 1): 2.0
        - OFI features (indices mid_price_idx + 2 onwards): 2.0
        - Everything else (levels 6-20): 1.0
        Normalized so mean = 1.0.
        """
        w = torch.ones(n_features, device=device)

        # Levels 1-5: 4 features each (bid_price, ask_price, bid_size, ask_size)
        top_levels_end = min(5 * 4, n_features)
        w[:top_levels_end] = 2.0

        # Levels 21-40: indices 80-159
        deep_start = 20 * 4  # level 21 starts at index 80
        deep_end = min(40 * 4, n_features)  # level 40 ends at index 160
        if deep_start < n_features:
            w[deep_start:min(deep_end, n_features)] = 0.5

        # mid_price
        if self.mid_price_idx < n_features:
            w[self.mid_price_idx] = 3.0

        # spread (next after mid_price)
        spread_idx = self.mid_price_idx + 1
        if spread_idx < n_features:
            w[spread_idx] = 2.0

        # OFI features (everything after spread)
        ofi_start = self.mid_price_idx + 2
        if ofi_start < n_features:
            w[ofi_start:] = 2.0

        # Normalize so mean = 1.0
        w = w / w.mean()

        return w

    def _direction_labels(
        self,
        target: torch.Tensor,
        context_last: torch.Tensor,
    ) -> torch.Tensor:
        """Compute direction labels from mid-price change at multiple horizons.

        Args:
            target: (B, T_pred, F) target features
            context_last: (B, F) last context timestep features

        Returns:
            labels: (B, 3) -- one label per horizon, 0=down, 1=flat, 2=up
        """
        return compute_direction_labels(
            target, context_last, self.mid_price_idx,
            self.direction_horizons, self.flat_threshold,
        )

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
            target: (B, T_pred, F) ground truth features
            dir_logits: (B, 9) direction logits (3 horizons x 3 classes)
            context_last: (B, F) last context timestep (for direction labels)

        Returns:
            total_loss, forecast_loss, structure_loss, direction_loss, dsl_value
        """
        # MSE component (optionally feature-weighted)
        if self.use_feature_weights:
            if self._feature_weights is None:
                self._feature_weights = self._build_feature_weights(
                    pred.shape[2], pred.device
                )
            fw = self._feature_weights.to(pred.device)  # (F,)
            mse = (fw * (pred - target) ** 2).mean()
        else:
            mse = F.mse_loss(pred, target)

        # DSL component
        dsl = directional_sign_loss(pred, target)

        # Blended forecast loss
        forecast_loss = (1 - self.dsl_weight) * mse + self.dsl_weight * dsl

        # Structure loss (reuse from base LOBLoss)
        structure_loss = self._base_loss._structure_loss(pred)

        # Direction classification loss (multi-horizon)
        direction_labels = self._direction_labels(target, context_last)  # (B, 3)
        dir_logits_3h = dir_logits.reshape(-1, 3, 3)  # (B, 3_horizons, 3_classes)
        direction_loss = sum(
            F.cross_entropy(dir_logits_3h[:, h, :], direction_labels[:, h])
            for h in range(3)
        ) / 3.0

        # Total
        total = forecast_loss + self.w_struct * structure_loss + self.w_dir * direction_loss

        return total, forecast_loss, structure_loss, direction_loss, dsl
