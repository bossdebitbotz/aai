#!/usr/bin/env python3
"""
Training script for the Compound Attention LOB Forecasting Model.

Usage:
    python training/train.py                    # defaults: 5-level dry run
    python training/train.py --levels 40        # 40-level production
    python training/train.py --epochs 5 --dry   # quick sanity check
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import DataConfig, build_dataloaders, LOBScaler
from training.features import engineer_features, get_column_indices
from training.model import CompoundAttentionModel, LOBLoss, WarmupDecayScheduler
from training.tracker import ExperimentTracker, RunConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_data_config(args) -> DataConfig:
    """V1 DataConfig from CLI args. Defaults preserve historic V1 behavior
    (db source, savgol 21); the Colab baseline run passes --source parquet
    --savgol-window 11 to match the V2 pipeline's data treatment."""
    exchanges = args.exchanges.split(",") if args.exchanges else None
    pairs = args.pairs.split(",") if args.pairs else None
    return DataConfig(
        lob_levels=args.levels,
        exchanges=exchanges or DataConfig().exchanges,
        pairs=pairs or DataConfig().pairs,
        feature_version="v1",
        savgol_window=args.savgol_window,
        source=args.source,
        parquet_dir=args.parquet_dir,
    )


def train_one_epoch(
    model, loss_fn, optimizer, scheduler, train_loader, device
) -> tuple[float, float, float]:
    """Train for one epoch. Returns (total_loss, forecast_loss, structure_loss)."""
    model.train()
    total_loss_sum = 0.0
    forecast_sum = 0.0
    structure_sum = 0.0
    n_batches = 0

    for batch in train_loader:
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        exchange_ids = batch["exchange_id"].to(device)
        symbol_ids = batch["symbol_id"].to(device)

        pred = model(context, exchange_ids, symbol_ids)
        total_loss, forecast_loss, structure_loss = loss_fn(pred, target)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss_sum += total_loss.item()
        forecast_sum += forecast_loss.item()
        structure_sum += structure_loss.item()
        n_batches += 1

    return (
        total_loss_sum / max(n_batches, 1),
        forecast_sum / max(n_batches, 1),
        structure_sum / max(n_batches, 1),
    )


@torch.no_grad()
def evaluate(model, loss_fn, loader, device) -> tuple[float, float, float]:
    """Evaluate on validation/test set. Returns (total_loss, forecast_loss, structure_loss)."""
    if loader is None:
        return float("nan"), float("nan"), float("nan")

    model.eval()
    total_loss_sum = 0.0
    forecast_sum = 0.0
    structure_sum = 0.0
    n_batches = 0

    for batch in loader:
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        exchange_ids = batch["exchange_id"].to(device)
        symbol_ids = batch["symbol_id"].to(device)

        pred = model(context, exchange_ids, symbol_ids)
        total_loss, forecast_loss, structure_loss = loss_fn(pred, target)

        total_loss_sum += total_loss.item()
        forecast_sum += forecast_loss.item()
        structure_sum += structure_loss.item()
        n_batches += 1

    return (
        total_loss_sum / max(n_batches, 1),
        forecast_sum / max(n_batches, 1),
        structure_sum / max(n_batches, 1),
    )


def main():
    parser = argparse.ArgumentParser(description="Train LOB Forecasting Model")
    parser.add_argument("--levels", type=int, default=5, help="LOB levels (5 or 40)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=66, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=3, help="Attention heads")
    parser.add_argument("--n-layers", type=int, default=3, help="Transformer layers")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--exchanges", type=str, default=None,
                        help="Comma-separated exchanges (default: all)")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Comma-separated pairs (default: all)")
    parser.add_argument("--dry", action="store_true", help="Quick dry run (small config)")
    parser.add_argument("--run-name", type=str, default=None, help="Experiment run name")
    parser.add_argument("--source", type=str, default="db", choices=["db", "parquet"],
                        help="Data source: live DB or exported parquet (Colab)")
    parser.add_argument("--parquet-dir", type=str, default="lob_data",
                        help="Directory of exported parquet files (--source parquet)")
    parser.add_argument("--savgol-window", type=int, default=21,
                        help="Savitzky-Golay window (21 = historic V1; 11 matches V2 pipeline)")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Device: {device}")

    # Adjust for dry run
    if args.dry:
        args.epochs = min(args.epochs, 3)
        args.exchanges = args.exchanges or "binance_spot"
        args.pairs = args.pairs or "BTC-USDT"
        logger.info("DRY RUN mode: limited config")

    n_features = args.levels * 5 + 11  # enriched features after engineer_features()
    d_ff = args.d_model * 4

    # Warn about memory for large batch sizes with deep LOB
    if args.levels >= 40 and args.batch_size > 32:
        logger.warning(
            f"batch_size={args.batch_size} with {args.levels} levels may exceed memory. "
            f"Consider --batch-size 16 or 32."
        )

    # Run config
    run_config = RunConfig(
        n_levels=args.levels,
        n_features=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=d_ff,
        dropout=0.1,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        early_stopping_patience=args.patience,
    )

    # Data config
    data_config = build_data_config(args)

    # --- Step 1: Build DataLoaders ---
    logger.info("Building DataLoaders...")
    train_loader, val_loader, test_loader, metadata = build_dataloaders(
        data_config, batch_size=args.batch_size,
    )

    # Defensive guard: ensure the data pipeline produces the feature dim the model expects.
    sample = next(iter(train_loader))
    assert sample["context"].shape[-1] == n_features, (
        f"feature dim {sample['context'].shape[-1]} != n_features {n_features}"
    )

    logger.info(
        f"Data: {metadata['n_train_windows']} train, "
        f"{metadata['n_val_windows']} val, "
        f"{metadata['n_test_windows']} test windows"
    )

    # --- Step 2: Create Model ---
    model = CompoundAttentionModel(
        n_levels=args.levels,
        n_features=n_features,
        context_length=data_config.context_length,
        prediction_length=data_config.prediction_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=d_ff,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    # --- Step 3: Loss, Optimizer, Scheduler ---
    # Extract scaler params for structure loss inverse-transform
    first_scaler = next(iter(metadata["scalers"].values()))
    loss_fn = LOBLoss(
        n_levels=args.levels,
        structure_weight=run_config.structure_loss_weight,
        scaler_means=first_scaler._means,
        scaler_stds=first_scaler._stds,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = WarmupDecayScheduler(
        optimizer, warmup_steps=run_config.warmup_steps,
        decay_factor=run_config.lr_decay_factor,
        decay_every=run_config.lr_decay_every,
    )

    # --- Step 4: Experiment Tracker ---
    experiment_dir = Path(__file__).parent.parent / "experiments"
    tracker = ExperimentTracker(str(experiment_dir), run_name=args.run_name)
    tracker.log_config(run_config)
    tracker.log_data_info({
        "n_train_windows": metadata["n_train_windows"],
        "n_val_windows": metadata["n_val_windows"],
        "n_test_windows": metadata["n_test_windows"],
        "stream_info": {k: v for k, v in metadata["stream_info"].items()},
    })

    # --- Step 5: Training Loop ---
    logger.info(f"Training for up to {args.epochs} epochs (patience={args.patience})...")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_forecast, train_struct = train_one_epoch(
            model, loss_fn, optimizer, scheduler, train_loader, device
        )

        # Validate
        val_loss, val_forecast, val_struct = evaluate(
            model, loss_fn, val_loader, device
        )

        elapsed = time.time() - t0

        # Log
        is_best = tracker.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_forecast_loss=train_forecast,
            train_structure_loss=train_struct,
            val_loss=val_loss if not np.isnan(val_loss) else None,
            val_forecast_loss=val_forecast if not np.isnan(val_forecast) else None,
            val_structure_loss=val_struct if not np.isnan(val_struct) else None,
            learning_rate=scheduler.current_lr,
        )

        # Checkpoint
        tracker.save_checkpoint(model, optimizer, epoch, is_best)

        # Log line
        val_str = f"val={val_loss:.6f}" if not np.isnan(val_loss) else "val=N/A"
        best_str = " *BEST*" if is_best else ""
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.6f} (f={train_forecast:.6f} s={train_struct:.4f}) | "
            f"{val_str} | lr={scheduler.current_lr:.2e} | {elapsed:.1f}s{best_str}"
        )

        # Early stopping
        if is_best:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(
                    f"Early stopping after {args.patience} epochs without improvement"
                )
                break

    # --- Step 6: Final Evaluation ---
    logger.info("Loading best model for final evaluation...")
    tracker.load_checkpoint(model, checkpoint_name="best.pt")

    test_loss, test_forecast, test_struct = evaluate(
        model, loss_fn, test_loader, device
    )

    if not np.isnan(test_loss):
        logger.info(
            f"Test Results: total={test_loss:.6f}, "
            f"forecast={test_forecast:.6f}, structure={test_struct:.6f}"
        )
    else:
        logger.info("No test data available for evaluation")

    logger.info("\n" + tracker.summary())
    logger.info(f"Experiment saved to: {tracker.run_dir}")


if __name__ == "__main__":
    main()
