"""Tests for Compound Attention Model V2 with directional prediction."""

import sys
import inspect
import logging
import torch

sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.model_v2 import (
    directional_sign_loss,
    CausalDualAttentionBlock,
    CompoundAttentionModelV2,
    LOBLossV2,
)
from training.model import DualAttentionBlock

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def test_model_v2_forward_shape():
    """V2 model returns (predictions, direction_logits) tuple."""
    n_levels = 5
    n_features = 36
    context_len = 20
    pred_len = 4

    model = CompoundAttentionModelV2(
        n_levels=n_levels,
        n_features=n_features,
        context_length=context_len,
        prediction_length=pred_len,
        d_model=30,
        n_heads=3,
        n_layers=1,
        d_ff=120,
        dropout=0.1,
    ).to(DEVICE)

    B = 4
    context = torch.randn(B, context_len, n_features).to(DEVICE)
    exchange_ids = torch.zeros(B, dtype=torch.long).to(DEVICE)
    symbol_ids = torch.zeros(B, dtype=torch.long).to(DEVICE)

    pred, dir_logits = model(context, exchange_ids, symbol_ids)

    assert pred.shape == (B, 4, 36), f"Pred shape: {pred.shape}"
    assert dir_logits.shape == (B, 9), f"Dir logits shape: {dir_logits.shape}"

    logger.info(f"PASS: test_model_v2_forward_shape (pred={pred.shape}, dir={dir_logits.shape})")


def test_model_v2_40level_shape():
    """Full size 40-level V2 model."""
    n_levels = 40
    n_features = 219
    context_len = 120
    pred_len = 24

    model = CompoundAttentionModelV2(
        n_levels=n_levels,
        n_features=n_features,
        context_length=context_len,
        prediction_length=pred_len,
        d_model=66,
        n_heads=3,
        n_layers=3,
        d_ff=264,
        dropout=0.1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  40-level V2 model: {n_params:,} parameters")

    B = 2
    context = torch.randn(B, context_len, n_features).to(DEVICE)
    exchange_ids = torch.zeros(B, dtype=torch.long).to(DEVICE)
    symbol_ids = torch.zeros(B, dtype=torch.long).to(DEVICE)

    pred, dir_logits = model(context, exchange_ids, symbol_ids)

    assert pred.shape == (B, pred_len, n_features), f"Pred shape: {pred.shape}"
    assert dir_logits.shape == (B, 9), f"Dir logits shape: {dir_logits.shape}"

    logger.info(f"PASS: test_model_v2_40level_shape (pred={pred.shape}, dir={dir_logits.shape})")


def test_directional_sign_loss():
    """DSL should be lower when predicted and target directions match."""
    # Matching directions: both increasing [1, 2, 3] along time
    # Shape: (B=1, T=3, F=1)
    pred_match = torch.tensor([[[1.0], [2.0], [3.0]]])
    target_match = torch.tensor([[[1.0], [2.0], [3.0]]])
    dsl_match = directional_sign_loss(pred_match, target_match)

    # Mismatched directions: pred increasing, target decreasing along time
    pred_wrong = torch.tensor([[[1.0], [2.0], [3.0]]])
    target_wrong = torch.tensor([[[3.0], [2.0], [1.0]]])
    dsl_wrong = directional_sign_loss(pred_wrong, target_wrong)

    assert dsl_match < dsl_wrong, (
        f"DSL should be lower for matching directions: match={dsl_match:.4f}, wrong={dsl_wrong:.4f}"
    )

    logger.info(
        f"PASS: test_directional_sign_loss (match={dsl_match:.4f}, wrong={dsl_wrong:.4f})"
    )


def test_lob_loss_v2_components():
    """V2 loss returns 5 components, all non-NaN and with grad."""
    n_levels = 5
    n_features = 36
    pred_len = 4

    loss_fn = LOBLossV2(n_levels=n_levels, mid_price_idx=20)

    pred = torch.randn(4, pred_len, n_features, requires_grad=True)
    target = torch.randn(4, pred_len, n_features)
    dir_logits = torch.randn(4, 9, requires_grad=True)
    context_last = torch.randn(4, n_features)

    total, forecast, structure, direction, dsl = loss_fn(
        pred, target, dir_logits, context_last
    )

    # All should be scalar and non-NaN
    for name, val in [
        ("total", total), ("forecast", forecast), ("structure", structure),
        ("direction", direction), ("dsl", dsl),
    ]:
        assert val.shape == (), f"{name} should be scalar, got {val.shape}"
        assert not torch.isnan(val), f"{name} is NaN"

    # total should require grad (it flows from pred and dir_logits)
    total.backward()
    assert pred.grad is not None, "pred should have gradient"
    assert dir_logits.grad is not None, "dir_logits should have gradient"

    logger.info(
        f"PASS: test_lob_loss_v2_components "
        f"(total={total:.4f}, forecast={forecast:.4f}, "
        f"structure={structure:.4f}, direction={direction:.4f}, dsl={dsl:.4f})"
    )


def test_direction_labels():
    """Labels should be (B, 3): all 2 (up) when mid-price goes up at all horizons, all 0 (down) when down."""
    n_levels = 5
    n_features = 36
    mid_price_idx = 20
    pred_len = 24  # needs to cover horizons 5, 11, 23

    loss_fn = LOBLossV2(
        n_levels=n_levels, mid_price_idx=mid_price_idx,
        direction_horizons=(5, 11, 23),
    )

    # Target where mid-price goes up by 1.0 at all horizon steps (exceeds 0.01)
    context_last_up = torch.zeros(1, n_features)
    context_last_up[0, mid_price_idx] = 100.0

    target_up = torch.zeros(1, pred_len, n_features)
    for h in [5, 11, 23]:
        target_up[0, h, mid_price_idx] = 101.0  # +1.0, exceeds threshold

    labels_up = loss_fn._direction_labels(target_up, context_last_up)
    assert labels_up.shape == (1, 3), f"Expected (1, 3), got {labels_up.shape}"
    assert (labels_up[0] == 2).all(), f"Expected all 2 (up), got {labels_up[0]}"

    # Target where mid-price goes down by 1.0 at all horizon steps
    context_last_down = torch.zeros(1, n_features)
    context_last_down[0, mid_price_idx] = 100.0

    target_down = torch.zeros(1, pred_len, n_features)
    for h in [5, 11, 23]:
        target_down[0, h, mid_price_idx] = 99.0  # -1.0, exceeds threshold

    labels_down = loss_fn._direction_labels(target_down, context_last_down)
    assert labels_down.shape == (1, 3), f"Expected (1, 3), got {labels_down.shape}"
    assert (labels_down[0] == 0).all(), f"Expected all 0 (down), got {labels_down[0]}"

    logger.info(
        f"PASS: test_direction_labels (up={labels_up[0].tolist()}, down={labels_down[0].tolist()})"
    )


def test_causal_dual_attention_block():
    """CausalDualAttentionBlock and DualAttentionBlock produce same shape but different outputs."""
    d_model, n_heads, d_ff = 30, 3, 120
    B, T, F = 2, 10, 8

    causal_block = CausalDualAttentionBlock(d_model, n_heads, d_ff, dropout=0.0).to(DEVICE)
    dual_block = DualAttentionBlock(d_model, n_heads, d_ff, dropout=0.0).to(DEVICE)

    # Copy weights from causal to dual so only the mask differs
    dual_block.load_state_dict(causal_block.state_dict())

    x = torch.randn(B, T, F, d_model, device=DEVICE)
    causal_block.eval()
    dual_block.eval()

    with torch.no_grad():
        out_causal = causal_block(x)
        out_dual = dual_block(x)

    assert out_causal.shape == (B, T, F, d_model), f"Causal shape: {out_causal.shape}"
    assert out_dual.shape == (B, T, F, d_model), f"Dual shape: {out_dual.shape}"

    # Outputs should differ due to causal masking
    assert not torch.allclose(out_causal, out_dual, atol=1e-5), (
        "Causal and non-causal outputs should differ"
    )

    logger.info(
        f"PASS: test_causal_dual_attention_block "
        f"(shape={out_causal.shape}, outputs_differ=True)"
    )


def test_multi_horizon_direction_labels():
    """Verify _direction_labels returns correct per-horizon labels."""
    n_features = 36
    mid_price_idx = 20
    pred_len = 24

    loss_fn = LOBLossV2(
        n_levels=5, mid_price_idx=mid_price_idx,
        direction_horizons=(5, 11, 23),
    )

    # Create target with known mid-price at horizons:
    # step 5: up (+1.0), step 11: down (-1.0), step 23: flat (0.0)
    context_last = torch.zeros(1, n_features)
    context_last[0, mid_price_idx] = 100.0

    target = torch.zeros(1, pred_len, n_features)
    target[0, 5, mid_price_idx] = 101.0   # up
    target[0, 11, mid_price_idx] = 99.0   # down
    target[0, 23, mid_price_idx] = 100.0  # flat (within 0.01 threshold)

    labels = loss_fn._direction_labels(target, context_last)
    assert labels.shape == (1, 3), f"Expected (1, 3), got {labels.shape}"
    assert labels[0, 0].item() == 2, f"Horizon 5: expected 2 (up), got {labels[0, 0].item()}"
    assert labels[0, 1].item() == 0, f"Horizon 11: expected 0 (down), got {labels[0, 1].item()}"
    assert labels[0, 2].item() == 1, f"Horizon 23: expected 1 (flat), got {labels[0, 2].item()}"

    logger.info(
        f"PASS: test_multi_horizon_direction_labels (labels={labels[0].tolist()})"
    )


def test_feature_weighted_mse():
    """LOBLossV2 with use_feature_weights=True produces valid loss with gradients."""
    n_levels = 5
    n_features = 36
    pred_len = 24

    loss_fn = LOBLossV2(
        n_levels=n_levels, mid_price_idx=20,
        use_feature_weights=True,
        direction_horizons=(5, 11, 23),
    )

    pred = torch.randn(4, pred_len, n_features, requires_grad=True)
    target = torch.randn(4, pred_len, n_features)
    dir_logits = torch.randn(4, 9, requires_grad=True)
    context_last = torch.randn(4, n_features)

    total, forecast, structure, direction, dsl = loss_fn(
        pred, target, dir_logits, context_last
    )

    # All should be non-NaN
    for name, val in [
        ("total", total), ("forecast", forecast), ("structure", structure),
        ("direction", direction), ("dsl", dsl),
    ]:
        assert not torch.isnan(val), f"{name} is NaN with feature weights"

    # Gradients should flow
    total.backward()
    assert pred.grad is not None, "pred should have gradient with feature weights"
    assert dir_logits.grad is not None, "dir_logits should have gradient with feature weights"

    logger.info(
        f"PASS: test_feature_weighted_mse (total={total:.4f}, feature_weights_shape="
        f"{loss_fn._feature_weights.shape})"
    )


def test_flat_threshold_tightened():
    """Default flat_threshold should be 0.01 (tightened from 0.05)."""
    sig = inspect.signature(LOBLossV2.__init__)
    default = sig.parameters["flat_threshold"].default
    assert default == 0.01, f"Expected flat_threshold default 0.01, got {default}"

    # Also verify instance default
    loss_fn = LOBLossV2(n_levels=5)
    assert loss_fn.flat_threshold == 0.01, (
        f"Instance flat_threshold should be 0.01, got {loss_fn.flat_threshold}"
    )

    logger.info(f"PASS: test_flat_threshold_tightened (default={default})")


if __name__ == "__main__":
    logger.info(f"Device: {DEVICE}")

    test_model_v2_forward_shape()
    test_model_v2_40level_shape()
    test_directional_sign_loss()
    test_lob_loss_v2_components()
    test_direction_labels()
    test_causal_dual_attention_block()
    test_multi_horizon_direction_labels()
    test_feature_weighted_mse()
    test_flat_threshold_tightened()

    logger.info("\n=== ALL V2 MODEL TESTS PASSED ===")
