#!/usr/bin/env python3
"""V2 training entry point: directional LOB model (CompoundAttentionModelV2 + LOBLossV2).

Usage:
    python training/train_v2.py --levels 40 --epochs 50 --batch-size 16 --accum-steps 8
    python training/train_v2.py --levels 40 --resume
"""
import argparse, logging, sys, time, pickle
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dataset import DataConfig, build_dataloaders
from training.model_v2 import CompoundAttentionModelV2, LOBLossV2, directional_accuracy
from training.model import WarmupDecayScheduler
from training.tracker import ExperimentTracker, RunConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_v2")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dir_acc(dir_logits, target, context_last, loss_fn):
    labels = loss_fn._direction_labels(target, context_last)   # (B,H)
    return directional_accuracy(dir_logits, labels)            # (H,)


def run_one_epoch_v2(model, loss_fn, optimizer, scheduler, loader, device, accum_steps=1):
    model.train()
    sums = {"total": 0.0, "forecast": 0.0, "structure": 0.0, "direction": 0.0, "dsl": 0.0}
    acc_sum = None
    n = 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        ex = batch["exchange_id"].to(device)
        sym = batch["symbol_id"].to(device)
        context_last = context[:, -1, :]

        pred, dir_logits = model(context, ex, sym)
        total, forecast, structure, direction, dsl = loss_fn(pred, target, dir_logits, context_last)

        (total / accum_steps).backward()
        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        sums["total"] += total.item(); sums["forecast"] += forecast.item()
        sums["structure"] += structure.item(); sums["direction"] += direction.item()
        sums["dsl"] += dsl.item()
        acc = _dir_acc(dir_logits.detach(), target, context_last, loss_fn)
        acc_sum = acc if acc_sum is None else acc_sum + acc
        n += 1

    # flush a trailing partial accumulation
    if n % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    out = {k: v / max(n, 1) for k, v in sums.items()}
    acc_mean = (acc_sum / max(n, 1)) if acc_sum is not None else torch.zeros(3)
    out["dir_acc_per_h"] = acc_mean.tolist()
    out["dir_acc_mean"] = float(acc_mean.mean().item())
    return out


@torch.no_grad()
def evaluate_v2_loader(model, loss_fn, loader, device):
    if loader is None:
        return {"total": float("nan"), "dir_acc_per_h": [float("nan")] * 3, "dir_acc_mean": float("nan")}
    model.eval()
    total_sum = 0.0; acc_sum = None; n = 0
    for batch in loader:
        context = batch["context"].to(device); target = batch["target"].to(device)
        ex = batch["exchange_id"].to(device); sym = batch["symbol_id"].to(device)
        context_last = context[:, -1, :]
        pred, dir_logits = model(context, ex, sym)
        total, *_ = loss_fn(pred, target, dir_logits, context_last)
        total_sum += total.item()
        acc = _dir_acc(dir_logits, target, context_last, loss_fn)
        acc_sum = acc if acc_sum is None else acc_sum + acc
        n += 1
    acc_mean = (acc_sum / max(n, 1)) if acc_sum is not None else torch.zeros(3)
    return {"total": total_sum / max(n, 1),
            "dir_acc_per_h": acc_mean.tolist(),
            "dir_acc_mean": float(acc_mean.mean().item())}


def main():
    p = argparse.ArgumentParser(description="Train V2 directional LOB model")
    p.add_argument("--levels", type=int, default=40)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=66)
    p.add_argument("--n-heads", type=int, default=3)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--source", type=str, default="db", choices=["db", "parquet"])
    p.add_argument("--parquet-dir", type=str, default="lob_data")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = get_device(); logger.info(f"Device: {device}")
    n_features = args.levels * 5 + 19  # V2
    d_ff = args.d_model * 4

    data_config = DataConfig(lob_levels=args.levels, feature_version="v2",
                             savgol_window=11, source=args.source, parquet_dir=args.parquet_dir)
    train_loader, val_loader, test_loader, meta = build_dataloaders(data_config, batch_size=args.batch_size)
    logger.info(f"Windows: train={meta['n_train_windows']} val={meta['n_val_windows']} test={meta['n_test_windows']}")

    model = CompoundAttentionModelV2(
        n_levels=args.levels, n_features=n_features,
        context_length=data_config.context_length, prediction_length=data_config.prediction_length,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=d_ff, dropout=0.1,
    ).to(device)
    assert model.n_features == n_features, "n_features mismatch"
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    first_scaler = next(iter(meta["scalers"].values()))
    loss_fn = LOBLossV2(n_levels=args.levels, use_feature_weights=True, mid_price_idx=args.levels * 4,
                        scaler_means=first_scaler._means, scaler_stds=first_scaler._stds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = WarmupDecayScheduler(optimizer, warmup_steps=1000, decay_factor=0.8, decay_every=5000)

    run_config = RunConfig(n_levels=args.levels, n_features=n_features, d_model=args.d_model,
                           n_heads=args.n_heads, n_layers=args.n_layers, d_ff=d_ff,
                           learning_rate=args.lr, batch_size=args.batch_size, max_epochs=args.epochs,
                           early_stopping_patience=args.patience, savgol_window=11)
    tracker = ExperimentTracker(str(Path(__file__).parent.parent / "experiments"),
                                run_name=args.run_name, monitor="val_dir_acc_mean", monitor_mode="max")
    tracker.log_config(run_config)
    # persist scalers for eval/backtest
    with open(tracker.run_dir / "scalers.pkl", "wb") as f:
        pickle.dump(meta["scalers"], f)

    start_epoch = 1
    if args.resume:
        start_epoch = tracker.load_checkpoint(model, optimizer, "latest.pt") + 1
        ck = torch.load(tracker.checkpoints_dir / "latest.pt", weights_only=False)
        scheduler.step_count = ck.get("scheduler_step_count", 0)
        logger.info(f"Resumed at epoch {start_epoch} (scheduler step {scheduler.step_count})")

    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = run_one_epoch_v2(model, loss_fn, optimizer, scheduler, train_loader, device, args.accum_steps)
        va = evaluate_v2_loader(model, loss_fn, val_loader, device)
        is_best = tracker.log_epoch(
            epoch=epoch, train_loss=tr["total"], train_forecast_loss=tr["forecast"],
            train_structure_loss=tr["structure"],
            val_loss=va["total"] if not np.isnan(va["total"]) else None,
            learning_rate=scheduler.current_lr,
            extra={"train_dir_acc_mean": tr["dir_acc_mean"], "train_dir_acc_per_h": tr["dir_acc_per_h"],
                   "val_dir_acc_mean": va["dir_acc_mean"], "val_dir_acc_per_h": va["dir_acc_per_h"]},
        )
        tracker.save_checkpoint(model, optimizer, epoch, is_best, extra={
            "n_features": n_features, "context_length": data_config.context_length,
            "prediction_length": data_config.prediction_length,
            "direction_horizons": list(loss_fn.direction_horizons),
            "mid_price_idx": loss_fn.mid_price_idx, "feature_version": "v2",
            "scheduler_step_count": scheduler.step_count,
        })
        logger.info(f"Epoch {epoch:3d} | train_total={tr['total']:.4f} dir_acc={tr['dir_acc_mean']:.3f} "
                    f"| val_total={va['total']:.4f} val_dir_acc={va['dir_acc_mean']:.3f} "
                    f"| {time.time()-t0:.1f}s {'*BEST*' if is_best else ''}")
        epochs_no_improve = 0 if is_best else epochs_no_improve + 1
        if epochs_no_improve >= args.patience:
            logger.info("Early stopping."); break

    logger.info("\n" + tracker.summary())
    logger.info(f"Run dir: {tracker.run_dir}")


if __name__ == "__main__":
    main()
