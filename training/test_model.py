"""Tests for the Compound Attention Model."""

import sys
import logging
import torch
import numpy as np

sys.path.insert(0, "/Users/clint/Projects/aai")
from training.model import (
    Time2Vec, CompoundAttributeEmbedding, FeatureProjection,
    TransformerBlock, DualAttentionBlock, CompoundAttentionModel,
    LOBLoss, WarmupDecayScheduler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def test_time2vec():
    """Test Time2Vec embedding."""
    d = 32
    t2v = Time2Vec(d).to(DEVICE)

    t = torch.linspace(0, 1, 120).unsqueeze(0).expand(4, -1).to(DEVICE)  # (4, 120)
    out = t2v(t)
    assert out.shape == (4, 120, d), f"Shape: {out.shape}"

    # Check gradients flow
    loss = out.sum()
    loss.backward()
    assert t2v.w0.grad is not None

    logger.info(f"PASS: time2vec (output shape: {out.shape})")


def test_compound_attribute_embedding():
    """Test compound attribute embedding with enriched features."""
    d = 32
    n_levels = 5
    n_features = n_levels * 5 + 11  # 36 enriched

    embed = CompoundAttributeEmbedding(
        d, n_levels=n_levels, include_derived=True
    ).to(DEVICE)

    # Verify attribute buffer sizes match expected feature count
    assert len(embed.attr_levels) == n_features, \
        f"Expected {n_features} attr indices, got {len(embed.attr_levels)}"

    x = torch.randn(4, 120, n_features).to(DEVICE)
    exchange_ids = torch.tensor([0, 1, 2, 3]).to(DEVICE)
    symbol_ids = torch.tensor([0, 1, 2, 3]).to(DEVICE)

    out = embed(x, exchange_ids, symbol_ids, is_target=False)
    assert out.shape == (4, 120, n_features, d), f"Shape: {out.shape}"

    # Context and target embeddings should differ
    out_ctx = embed(x, exchange_ids, symbol_ids, is_target=False)
    out_tgt = embed(x, exchange_ids, symbol_ids, is_target=True)
    assert not torch.allclose(out_ctx, out_tgt), "Context and target embeddings should differ"

    # Verify 3 side categories are used
    unique_sides = embed.attr_sides.unique().tolist()
    assert 0 in unique_sides and 1 in unique_sides and 2 in unique_sides, \
        f"Expected side categories [0,1,2], got {unique_sides}"

    # Verify 3 feature type categories are used
    unique_feat_types = embed.attr_feat_types.unique().tolist()
    assert 0 in unique_feat_types and 1 in unique_feat_types and 2 in unique_feat_types, \
        f"Expected feat_type categories [0,1,2], got {unique_feat_types}"

    logger.info(f"PASS: compound_attribute_embedding (output shape: {out.shape})")


def test_compound_attribute_base_only():
    """Test with base features only (no derived)."""
    d = 32
    n_levels = 5
    n_features = n_levels * 4 + 2  # 22 base

    embed = CompoundAttributeEmbedding(
        d, n_levels=n_levels, include_derived=False
    ).to(DEVICE)

    assert len(embed.attr_levels) == n_features

    x = torch.randn(2, 120, n_features).to(DEVICE)
    exchange_ids = torch.tensor([0, 1]).to(DEVICE)
    symbol_ids = torch.tensor([0, 1]).to(DEVICE)

    out = embed(x, exchange_ids, symbol_ids)
    assert out.shape == (2, 120, n_features, d)

    logger.info(f"PASS: compound_attribute_base_only (output shape: {out.shape})")


def test_compound_attribute_40_levels():
    """Test with 40 levels enriched."""
    d = 64
    n_levels = 40
    n_features = n_levels * 5 + 11  # 211

    embed = CompoundAttributeEmbedding(
        d, n_levels=n_levels, include_derived=True
    ).to(DEVICE)

    assert len(embed.attr_levels) == n_features

    x = torch.randn(2, 120, n_features).to(DEVICE)
    exchange_ids = torch.tensor([0, 1]).to(DEVICE)
    symbol_ids = torch.tensor([0, 1]).to(DEVICE)

    out = embed(x, exchange_ids, symbol_ids)
    assert out.shape == (2, 120, n_features, d)

    logger.info(f"PASS: compound_attribute_40_levels (output shape: {out.shape})")


def test_feature_projection():
    """Test feature projection."""
    d = 32
    proj = FeatureProjection(d).to(DEVICE)

    x = torch.randn(4, 120, 36).to(DEVICE)
    attr = torch.randn(4, 120, 36, d).to(DEVICE)

    out = proj(x, attr)
    assert out.shape == (4, 120, 36, d)

    logger.info(f"PASS: feature_projection (output shape: {out.shape})")


def test_dual_attention_block():
    """Test DualAttentionBlock with 4D input."""
    d = 32
    block = DualAttentionBlock(d, n_heads=2, d_ff=64).to(DEVICE)

    # 4D input: (B, T, F, d)
    x = torch.randn(4, 120, 36, d).to(DEVICE)
    out = block(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    # Check gradient flow
    loss = out.sum()
    loss.backward()
    assert block.feat_attn.in_proj_weight.grad is not None
    assert block.temp_attn.in_proj_weight.grad is not None

    logger.info(f"PASS: dual_attention_block (output shape: {out.shape})")


def test_transformer_block():
    """Test legacy transformer block still works."""
    d = 64
    block = TransformerBlock(d, n_heads=4, d_ff=256).to(DEVICE)

    x = torch.randn(4, 120 * 22, d).to(DEVICE)  # flattened T*F
    out = block(x)
    assert out.shape == x.shape

    logger.info(f"PASS: transformer_block (output shape: {out.shape})")


def test_full_model_5_levels():
    """Test full model with 5 levels enriched features."""
    n_levels = 5
    n_features = n_levels * 5 + 11  # 36

    model = CompoundAttentionModel(
        n_levels=n_levels,
        n_features=n_features,
        context_length=120,
        prediction_length=24,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        dropout=0.1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  5-level model: {n_params:,} parameters")

    context = torch.randn(4, 120, n_features).to(DEVICE)
    exchange_ids = torch.tensor([0, 1, 2, 3]).to(DEVICE)
    symbol_ids = torch.tensor([0, 1, 2, 3]).to(DEVICE)

    pred = model(context, exchange_ids, symbol_ids)
    assert pred.shape == (4, 24, n_features), f"Prediction shape: {pred.shape}"

    # Check gradient flow
    loss = pred.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            break

    logger.info(f"PASS: full_model_5_levels (pred shape: {pred.shape})")


def test_full_model_40_levels():
    """Test full model with 40 levels enriched features."""
    n_levels = 40
    n_features = n_levels * 5 + 11  # 211

    model = CompoundAttentionModel(
        n_levels=n_levels,
        n_features=n_features,
        context_length=120,
        prediction_length=24,
        d_model=66,
        n_heads=3,
        n_layers=3,
        d_ff=264,
        dropout=0.1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  40-level model: {n_params:,} parameters")

    # Use small batch to keep memory reasonable
    context = torch.randn(2, 120, n_features).to(DEVICE)
    exchange_ids = torch.tensor([0, 1]).to(DEVICE)
    symbol_ids = torch.tensor([0, 1]).to(DEVICE)

    pred = model(context, exchange_ids, symbol_ids)
    assert pred.shape == (2, 24, n_features), f"Prediction shape: {pred.shape}"

    logger.info(f"PASS: full_model_40_levels (pred shape: {pred.shape})")


def test_lob_loss():
    """Test LOB loss function without scaler (fallback mode)."""
    n_levels = 5
    n_features = n_levels * 5 + 11

    loss_fn = LOBLoss(n_levels=n_levels, structure_weight=0.01)

    pred = torch.randn(4, 24, n_features)
    target = torch.randn(4, 24, n_features)

    total, forecast, structure = loss_fn(pred, target)

    assert total.shape == ()  # scalar
    assert forecast.shape == ()
    assert structure.shape == ()
    assert total > 0
    assert forecast > 0
    assert structure >= 0

    # With perfectly ordered predictions, structure loss should be 0
    ordered_pred = torch.zeros(4, 24, n_features)
    for i in range(n_levels):
        ordered_pred[:, :, i * 2] = 100 - i  # bid prices descending
    offset = n_levels * 2
    for i in range(n_levels):
        ordered_pred[:, :, offset + i * 2] = 101 + i  # ask prices ascending

    _, _, struct_perfect = loss_fn(ordered_pred, target)
    assert struct_perfect == 0.0, f"Perfect order should have 0 structure loss, got {struct_perfect}"

    logger.info(f"PASS: lob_loss (total={total:.4f}, forecast={forecast:.4f}, structure={structure:.4f})")


def test_lob_loss_with_scaler():
    """Test LOB loss with scaler params for inverse-transform structure loss."""
    n_levels = 5
    n_features = n_levels * 5 + 11

    # Simulate scaler params (means and stds for all features)
    means = np.zeros(n_features, dtype=np.float64)
    stds = np.ones(n_features, dtype=np.float64)

    # Set realistic price means/stds
    for i in range(n_levels):
        means[i * 2] = 50000 - i * 10       # bid prices ~50000
        stds[i * 2] = 100.0
    offset = n_levels * 2
    for i in range(n_levels):
        means[offset + i * 2] = 50010 + i * 10  # ask prices ~50010
        stds[offset + i * 2] = 100.0

    loss_fn = LOBLoss(
        n_levels=n_levels, structure_weight=0.01,
        scaler_means=means, scaler_stds=stds,
    )

    # Verify buffers are registered
    assert loss_fn._means is not None
    assert loss_fn._stds is not None

    pred = torch.randn(4, 24, n_features)
    target = torch.randn(4, 24, n_features)

    total, forecast, structure = loss_fn(pred, target)
    assert total > 0
    assert structure >= 0

    # Structure loss should be finite and reasonable (normalized by ref_price)
    assert torch.isfinite(structure), f"Structure loss is not finite: {structure}"

    logger.info(f"PASS: lob_loss_with_scaler (structure={structure:.6f})")


def test_lob_loss_40_levels():
    """Test LOB loss with 40 levels."""
    n_levels = 40
    n_features = n_levels * 5 + 11

    loss_fn = LOBLoss(n_levels=n_levels)
    pred = torch.randn(2, 24, n_features)
    target = torch.randn(2, 24, n_features)

    total, forecast, structure = loss_fn(pred, target)
    assert total > 0

    logger.info(f"PASS: lob_loss_40_levels (total={total:.4f})")


def test_training_step():
    """Test a complete training step (forward + backward + optimizer step)."""
    n_levels = 5
    n_features = n_levels * 5 + 11

    model = CompoundAttentionModel(
        n_levels=n_levels, n_features=n_features,
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
    ).to(DEVICE)

    loss_fn = LOBLoss(n_levels=n_levels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic batch
    context = torch.randn(8, 120, n_features).to(DEVICE)
    target = torch.randn(8, 24, n_features).to(DEVICE)
    exchange_ids = torch.randint(0, 4, (8,)).to(DEVICE)
    symbol_ids = torch.randint(0, 4, (8,)).to(DEVICE)

    # Training step
    model.train()
    pred = model(context, exchange_ids, symbol_ids)
    total_loss, forecast_loss, structure_loss = loss_fn(pred, target)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    logger.info(
        f"PASS: training_step (loss={total_loss.item():.4f}, "
        f"forecast={forecast_loss.item():.4f}, structure={structure_loss.item():.4f})"
    )


def test_warmup_scheduler():
    """Test learning rate scheduler."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = WarmupDecayScheduler(optimizer, warmup_steps=100, decay_factor=0.8, decay_every=500)

    # During warmup, LR should increase linearly
    lrs = []
    for _ in range(200):
        scheduler.step()
        lrs.append(scheduler.current_lr)

    assert lrs[0] < lrs[50] < lrs[99], "LR should increase during warmup"
    assert abs(lrs[99] - 1e-3) < 1e-6, f"LR at end of warmup should be base_lr, got {lrs[99]}"

    # After warmup, LR should stay at base_lr then decay
    assert lrs[100] <= 1e-3, f"LR after warmup should be <= base_lr, got {lrs[100]}"

    logger.info(f"PASS: warmup_scheduler (warmup end LR={lrs[99]:.6f})")


def test_model_overfits_single_batch():
    """Test that model can overfit a single batch (sanity check)."""
    n_levels = 5
    n_features = n_levels * 5 + 11

    model = CompoundAttentionModel(
        n_levels=n_levels, n_features=n_features,
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
    ).to(DEVICE)

    loss_fn = LOBLoss(n_levels=n_levels, structure_weight=0.0)  # pure MSE for this test
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Single batch
    context = torch.randn(4, 120, n_features).to(DEVICE)
    target = torch.randn(4, 24, n_features).to(DEVICE)
    exchange_ids = torch.zeros(4, dtype=torch.long).to(DEVICE)
    symbol_ids = torch.zeros(4, dtype=torch.long).to(DEVICE)

    initial_loss = None
    for step in range(50):
        model.train()
        pred = model(context, exchange_ids, symbol_ids)
        total_loss, _, _ = loss_fn(pred, target)

        if step == 0:
            initial_loss = total_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    final_loss = total_loss.item()
    assert final_loss < initial_loss, \
        f"Model didn't learn: initial={initial_loss:.4f}, final={final_loss:.4f}"

    improvement = (initial_loss - final_loss) / initial_loss * 100
    logger.info(
        f"PASS: overfit_single_batch (loss: {initial_loss:.4f} -> {final_loss:.4f}, "
        f"{improvement:.1f}% reduction)"
    )


if __name__ == "__main__":
    logger.info(f"Device: {DEVICE}")

    test_time2vec()
    test_compound_attribute_embedding()
    test_compound_attribute_base_only()
    test_compound_attribute_40_levels()
    test_feature_projection()
    test_dual_attention_block()
    test_transformer_block()
    test_full_model_5_levels()
    test_full_model_40_levels()
    test_lob_loss()
    test_lob_loss_with_scaler()
    test_lob_loss_40_levels()
    test_training_step()
    test_warmup_scheduler()
    test_model_overfits_single_batch()

    logger.info("\n=== ALL MODEL TESTS PASSED ===")
