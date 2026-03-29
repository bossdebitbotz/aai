"""
Compound Attention Model for LOB Forecasting.

Implements architecture from arXiv:2409.02277 with multi-exchange extensions:
- Time2Vec temporal embedding
- Compound attribute embedding (level, side, feature type, exchange, symbol)
- Dual attention: factored feature + temporal self-attention (inspired by TLOB arXiv:2502.15757)
- Structure-preserving loss with inverse-transform ordering checks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time2Vec — learnable temporal embedding
# ---------------------------------------------------------------------------

class Time2Vec(nn.Module):
    """Time2Vec: learnable time representation.

    Converts scalar time values into d-dimensional embeddings using
    a linear term plus learned periodic (sine) components.

    t -> [w_0 * t + b_0, sin(w_1 * t + b_1), ..., sin(w_{d-1} * t + b_{d-1})]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Linear component
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        # Periodic components
        self.w = nn.Parameter(torch.randn(d_model - 1))
        self.b = nn.Parameter(torch.randn(d_model - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch, seq_len) or (batch, seq_len, 1) — time values

        Returns:
            (batch, seq_len, d_model) — time embeddings
        """
        if t.dim() == 3:
            t = t.squeeze(-1)  # (B, T)

        linear = self.w0 * t + self.b0  # (B, T)
        periodic = torch.sin(t.unsqueeze(-1) * self.w + self.b)  # (B, T, d-1)
        return torch.cat([linear.unsqueeze(-1), periodic], dim=-1)  # (B, T, d)


# ---------------------------------------------------------------------------
# Compound Attribute Embedding
# ---------------------------------------------------------------------------

class CompoundAttributeEmbedding(nn.Module):
    """Compound embedding that encodes LOB feature structure.

    Each feature is characterized by:
    - level: which depth level (0..N-1)
    - side: bid (0), ask (1), or both (2) — e.g., mid_price, OFI, spread
    - feature_type: price (0), volume (1), or flow (2) — e.g., OFI features

    Additional embeddings for exchange and symbol identity.
    """

    def __init__(self, d_model: int, n_levels: int = 40,
                 n_exchanges: int = 4, n_symbols: int = 4,
                 include_derived: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.include_derived = include_derived

        # Structural embeddings (summed together)
        self.level_embed = nn.Embedding(n_levels, d_model)
        self.side_embed = nn.Embedding(3, d_model)        # 0=bid, 1=ask, 2=both
        self.feature_embed = nn.Embedding(3, d_model)     # 0=price, 1=volume, 2=flow

        # Identity embeddings
        self.exchange_embed = nn.Embedding(n_exchanges, d_model)
        self.symbol_embed = nn.Embedding(n_symbols, d_model)

        # Context-target embedding (0=context, 1=target)
        self.ct_embed = nn.Embedding(2, d_model)

        # Build the attribute index for each feature column
        self._register_attribute_indices(n_levels, include_derived)

    def _register_attribute_indices(self, n_levels: int, include_derived: bool):
        """Pre-compute level/side/feature indices for each column.

        Column order must match dataset.py _build_feature_columns + features.py engineer_features output.
        """
        levels = []
        sides = []
        feat_types = []

        # Bid side: bid_price_1, bid_volume_1, ..., bid_price_N, bid_volume_N
        for i in range(n_levels):
            levels.extend([i, i])         # same level for price and volume
            sides.extend([0, 0])          # bid
            feat_types.extend([0, 1])     # price, volume

        # Ask side: ask_price_1, ask_volume_1, ..., ask_price_N, ask_volume_N
        for i in range(n_levels):
            levels.extend([i, i])
            sides.extend([1, 1])          # ask
            feat_types.extend([0, 1])     # price, volume

        # mid_price, spread — these span both sides
        levels.extend([0, 0])
        sides.extend([2, 2])       # both
        feat_types.extend([0, 0])  # price

        if include_derived:
            # Order must match features.py engineer_features() exactly:
            # ofi_level_1..N, ofi_aggregate,
            # cumulative_bid_volume, cumulative_ask_volume, volume_ratio, volume_imbalance_total,
            # price_imbalance, spread_ratio, depth_bid, depth_ask

            # OFI per level
            for i in range(n_levels):
                levels.append(i)
                sides.append(2)       # both
                feat_types.append(2)  # flow

            # ofi_aggregate
            levels.append(0)
            sides.append(2)           # both
            feat_types.append(2)      # flow

            # cumulative_bid_volume
            levels.append(0)
            sides.append(0)           # bid
            feat_types.append(1)      # volume

            # cumulative_ask_volume
            levels.append(0)
            sides.append(1)           # ask
            feat_types.append(1)      # volume

            # volume_ratio
            levels.append(0)
            sides.append(2)           # both
            feat_types.append(1)      # volume

            # volume_imbalance_total
            levels.append(0)
            sides.append(2)           # both
            feat_types.append(1)      # volume

            # price_imbalance
            levels.append(0)
            sides.append(2)           # both
            feat_types.append(0)      # price

            # spread_ratio
            levels.append(0)
            sides.append(2)           # both
            feat_types.append(0)      # price

            # depth_bid
            levels.append(0)
            sides.append(0)           # bid
            feat_types.append(0)      # price

            # depth_ask
            levels.append(0)
            sides.append(1)           # ask
            feat_types.append(0)      # price

        self.register_buffer("attr_levels", torch.tensor(levels, dtype=torch.long))
        self.register_buffer("attr_sides", torch.tensor(sides, dtype=torch.long))
        self.register_buffer("attr_feat_types", torch.tensor(feat_types, dtype=torch.long))

    def forward(
        self,
        x: torch.Tensor,
        exchange_ids: torch.Tensor,
        symbol_ids: torch.Tensor,
        is_target: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) — input features
            exchange_ids: (B,) — exchange indices
            symbol_ids: (B,) — symbol indices
            is_target: whether this is target sequence (vs context)

        Returns:
            (B, T, F, d_model) — per-feature embeddings
        """
        B, T, F = x.shape

        # Structural embeddings: (F, d_model)
        level_e = self.level_embed(self.attr_levels[:F])        # (F, d)
        side_e = self.side_embed(self.attr_sides[:F])            # (F, d)
        feat_e = self.feature_embed(self.attr_feat_types[:F])    # (F, d)
        struct = level_e + side_e + feat_e                       # (F, d)

        # Broadcast to (B, T, F, d)
        struct = struct.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        # Identity embeddings: (B, d)
        exch_e = self.exchange_embed(exchange_ids)  # (B, d)
        sym_e = self.symbol_embed(symbol_ids)       # (B, d)
        identity = (exch_e + sym_e).unsqueeze(1).unsqueeze(2).expand(B, T, F, -1)

        # Context-target: scalar
        ct_idx = torch.ones(1, device=x.device, dtype=torch.long) if is_target \
            else torch.zeros(1, device=x.device, dtype=torch.long)
        ct_e = self.ct_embed(ct_idx).unsqueeze(0).unsqueeze(0).expand(B, T, F, -1)

        return struct + identity + ct_e


# ---------------------------------------------------------------------------
# Feature-wise input projection
# ---------------------------------------------------------------------------

class FeatureProjection(nn.Module):
    """Projects each scalar feature value to d_model dimensions,
    then adds the compound attribute embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor, attr_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) — scalar features
            attr_embed: (B, T, F, d_model) — attribute embeddings

        Returns:
            (B, T, F, d_model) — projected + embedded features
        """
        # Project each scalar to d_model: (B, T, F, 1) -> (B, T, F, d)
        x_proj = self.proj(x.unsqueeze(-1))
        return x_proj + attr_embed


# ---------------------------------------------------------------------------
# Dual Attention Block (factored feature + temporal attention)
# ---------------------------------------------------------------------------

class DualAttentionBlock(nn.Module):
    """Factored attention: feature attention + temporal attention + FFN.

    Instead of flat attention over T*F tokens (O((TF)^2)), this factors into:
    1. Feature attention: self-attention across F features per timestep — O(T * F^2)
    2. Temporal attention: self-attention across T timesteps per feature — O(F * T^2)
    3. Feedforward network

    Total complexity: O(T*F^2 + F*T^2) instead of O((T*F)^2).
    For 40-level (T=120, F=211): ~8.3M vs ~641M ops per layer. 77x reduction.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Feature attention (across features for each timestep)
        self.feat_norm = nn.LayerNorm(d_model)
        self.feat_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Temporal attention (across timesteps for each feature)
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

        # 1. Feature attention: attend across F for each timestep
        x_feat = x.reshape(B * T, F, d)
        normed = self.feat_norm(x_feat)
        attn_out, _ = self.feat_attn(normed, normed, normed)
        x_feat = x_feat + attn_out
        x = x_feat.reshape(B, T, F, d)

        # 2. Temporal attention: attend across T for each feature
        x_temp = x.permute(0, 2, 1, 3).reshape(B * F, T, d)  # (B*F, T, d)
        normed = self.temp_norm(x_temp)
        attn_out, _ = self.temp_attn(normed, normed, normed)
        x_temp = x_temp + attn_out
        x = x_temp.reshape(B, F, T, d).permute(0, 2, 1, 3)  # (B, T, F, d)

        # 3. Feedforward
        normed = self.ff_norm(x)
        x = x + self.ff(normed)

        return x


# ---------------------------------------------------------------------------
# Transformer Encoder Block (legacy — kept for backward compatibility)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Standard transformer encoder block with pre-norm.

    Uses flat T*F attention. Replaced by DualAttentionBlock in the main model
    but kept for backward compatibility with existing tests/checkpoints.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
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
            x: (B, S, d_model) where S = T*F (flattened time x features)

        Returns:
            (B, S, d_model)
        """
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm feedforward
        normed = self.norm2(x)
        x = x + self.ff(normed)

        return x


# ---------------------------------------------------------------------------
# Compound Attention Model
# ---------------------------------------------------------------------------

class CompoundAttentionModel(nn.Module):
    """Full Compound Attention Model for LOB forecasting.

    Architecture:
    1. Feature projection: scalar -> d_model per feature
    2. Compound attribute embedding: structural + identity + temporal
    3. Dual attention encoder (feature + temporal attention per layer)
    4. Output projection: predict target sequence per feature

    The model predicts all features (prices + volumes + derived) for the prediction horizon.
    """

    def __init__(
        self,
        n_levels: int = 40,
        n_features: int = 211,     # 40*5 + 11 (enriched features)
        context_length: int = 120,
        prediction_length: int = 24,
        d_model: int = 66,       # divisible by 3 heads
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

        # Dual attention encoder (4D tensor throughout)
        self.encoder_layers = nn.ModuleList([
            DualAttentionBlock(d_model, n_heads, d_ff, dropout)
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
    ) -> torch.Tensor:
        """
        Args:
            context: (B, T_ctx, F) — context features (scaled)
            exchange_ids: (B,) — exchange indices
            symbol_ids: (B,) — symbol indices

        Returns:
            predictions: (B, T_pred, F) — predicted features for horizon
        """
        B, T, F = context.shape

        # 1. Attribute embeddings: (B, T, F, d)
        attr = self.attr_embed(context, exchange_ids, symbol_ids, is_target=False)

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
        # (B, F, d) -> prediction_head -> (B, F, T_pred)
        last_state = x[:, -1, :, :]  # (B, F, d)
        pred = self.prediction_head(last_state)  # (B, F, T_pred)
        pred = pred.permute(0, 2, 1)  # (B, T_pred, F)

        return pred


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class LOBLoss(nn.Module):
    """Combined forecasting + structure loss for LOB prediction.

    Forecasting loss: MSE on all predicted features.
    Structure loss: Penalizes violations of LOB price ordering constraints,
    operating on inverse-transformed (raw price scale) predictions when
    scaler params are provided. This ensures meaningful ordering checks
    regardless of normalization method.

    Constraints:
        1. Ask prices should increase with level (ask_1 < ask_2 < ...)
        2. Bid prices should decrease with level (bid_1 > bid_2 > ...)
        3. Best bid < best ask (no crossed book)
    """

    def __init__(self, n_levels: int = 40, structure_weight: float = 0.01,
                 scaler_means=None, scaler_stds=None):
        super().__init__()
        self.n_levels = n_levels
        self.w_o = structure_weight

        # Pre-compute column indices for price columns
        # Layout: bid_p1, bid_v1, bid_p2, bid_v2, ..., ask_p1, ask_v1, ...
        self.bid_price_indices = [i * 2 for i in range(n_levels)]
        self.ask_price_indices = [n_levels * 2 + i * 2 for i in range(n_levels)]

        # Register scaler params as buffers for GPU transfer and save/load
        if scaler_means is not None and scaler_stds is not None:
            self.register_buffer(
                '_means', torch.tensor(scaler_means, dtype=torch.float32)
            )
            self.register_buffer(
                '_stds', torch.tensor(scaler_stds, dtype=torch.float32)
            )
        else:
            self._means = None
            self._stds = None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: (B, T_pred, F) — predicted features
            target: (B, T_pred, F) — ground truth features

        Returns:
            total_loss, forecasting_loss, structure_loss
        """
        # Forecasting loss (MSE)
        forecast_loss = F.mse_loss(pred, target)

        # Structure loss
        structure_loss = self._structure_loss(pred)

        total_loss = forecast_loss + self.w_o * structure_loss
        return total_loss, forecast_loss, structure_loss

    def _structure_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute structure preservation loss on predictions.

        When scaler params are available, inverse-transforms prices to raw scale
        for meaningful ordering checks. Normalizes by reference price to keep
        loss dimensionless.
        """
        B, T, n_feat = pred.shape
        loss = torch.tensor(0.0, device=pred.device)

        bid_prices = pred[:, :, self.bid_price_indices]  # (B, T, n_levels)
        ask_prices = pred[:, :, self.ask_price_indices]   # (B, T, n_levels)

        if self._means is not None:
            # Inverse-transform to raw price scale
            bid_means = self._means[self.bid_price_indices]
            bid_stds = self._stds[self.bid_price_indices]
            ask_means = self._means[self.ask_price_indices]
            ask_stds = self._stds[self.ask_price_indices]

            bid_prices = bid_prices * bid_stds + bid_means
            ask_prices = ask_prices * ask_stds + ask_means

            # Normalize by reference price to keep loss dimensionless
            ref_price = max(bid_means[0].item(), 1.0)
        else:
            ref_price = 1.0

        # Ask prices should increase: ask_k < ask_{k+1}
        if self.n_levels > 1:
            ask_diff = ask_prices[:, :, :-1] - ask_prices[:, :, 1:]  # should be negative
            loss = loss + F.relu(ask_diff).sum() / (B * T * ref_price)

            # Bid prices should decrease: bid_k > bid_{k+1}
            bid_diff = bid_prices[:, :, 1:] - bid_prices[:, :, :-1]  # should be negative
            loss = loss + F.relu(bid_diff).sum() / (B * T * ref_price)

        # Best bid < best ask
        crossed = bid_prices[:, :, 0] - ask_prices[:, :, 0]  # should be negative
        loss = loss + F.relu(crossed).sum() / (B * T * ref_price)

        return loss


# ---------------------------------------------------------------------------
# Learning rate scheduler with warmup
# ---------------------------------------------------------------------------

class WarmupDecayScheduler:
    """Linear warmup followed by multiplicative decay.

    Per the paper: warmup_steps=1000, decay_factor=0.8
    """

    def __init__(self, optimizer, warmup_steps: int = 1000,
                 decay_factor: float = 0.8, decay_every: int = 5000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.decay_every = decay_every
        self.step_count = 0
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Decay
            n_decays = (self.step_count - self.warmup_steps) // self.decay_every
            lr = self.base_lr * (self.decay_factor ** n_decays)

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
