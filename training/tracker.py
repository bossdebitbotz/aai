"""
Experiment Tracker — lightweight file-based tracking for training runs.

Logs metrics, hyperparameters, and model checkpoints.
Can be replaced with W&B/MLflow later.
"""

import json
import os
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Hyperparameters and settings for a training run."""
    # Model
    n_levels: int = 40
    n_features: int = 162
    d_model: int = 66
    n_heads: int = 3
    n_layers: int = 3
    d_ff: int = 264
    dropout: float = 0.1
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    warmup_steps: int = 1000
    lr_decay_factor: float = 0.8
    lr_decay_every: int = 5000
    structure_loss_weight: float = 0.01
    # Data
    context_length: int = 120
    prediction_length: int = 24
    stride: int = 60
    apply_smoothing: bool = True
    savgol_window: int = 21
    # Feature engineering
    use_ofi: bool = True
    use_derived_features: bool = True


class ExperimentTracker:
    """Tracks training metrics and saves checkpoints."""

    def __init__(self, experiment_dir: str, run_name: Optional[str] = None,
                 monitor: str = "val_loss", monitor_mode: str = "min"):
        """
        Args:
            experiment_dir: base directory for all experiments
            run_name: name of this run (auto-generated if None)
            monitor: metric key (from log_epoch entry/extra) used to select the best epoch
            monitor_mode: "min" or "max" — direction of improvement for `monitor`
        """
        if run_name is None:
            run_name = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

        self.run_dir = Path(experiment_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.run_name = run_name
        self.metrics: list[dict] = []
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.best_metric = float("inf") if monitor_mode == "min" else float("-inf")
        self._start_time = time.time()

        logger.info(f"Experiment tracker: {self.run_dir}")

    def log_config(self, config: RunConfig):
        """Save run configuration."""
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)
        logger.info(f"Config saved to {config_path}")

    def log_data_info(self, info: dict):
        """Save data/stream information."""
        info_path = self.run_dir / "data_info.json"
        # Convert any non-serializable types
        serializable = {}
        for k, v in info.items():
            if isinstance(v, dict):
                serializable[k] = {str(kk): int(vv) if isinstance(vv, (np.integer,)) else vv
                                   for kk, vv in v.items()}
            else:
                serializable[k] = v
        with open(info_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_forecast_loss: float,
        train_structure_loss: float,
        val_loss: Optional[float] = None,
        val_forecast_loss: Optional[float] = None,
        val_structure_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        extra: Optional[dict] = None,
    ):
        """Log metrics for one epoch."""
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_forecast_loss": train_forecast_loss,
            "train_structure_loss": train_structure_loss,
            "val_loss": val_loss,
            "val_forecast_loss": val_forecast_loss,
            "val_structure_loss": val_structure_loss,
            "learning_rate": learning_rate,
            "elapsed_seconds": time.time() - self._start_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            entry.update(extra)

        self.metrics.append(entry)
        self._save_metrics()

        value = entry.get(self.monitor)
        is_best = False
        if value is not None:
            better = (value < self.best_metric) if self.monitor_mode == "min" else (value > self.best_metric)
            if better:
                self.best_metric = value
                self.best_epoch = epoch
                self.best_val_loss = entry.get("val_loss", self.best_val_loss)
                is_best = True
        return is_best

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        is_best: bool = False,
        extra: Optional[dict] = None,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
        }
        if extra:
            checkpoint.update(extra)

        # Always save latest
        latest_path = self.checkpoints_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoints_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved (epoch {epoch}, val_loss={self.best_val_loss:.6f})")

        # Periodic saves every 10 epochs
        if epoch % 10 == 0:
            epoch_path = self.checkpoints_dir / f"epoch_{epoch:04d}.pt"
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_name: str = "best.pt",
    ) -> int:
        """Load a checkpoint. Returns the epoch number."""
        path = self.checkpoints_dir / checkpoint_name
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", -1)

        logger.info(f"Loaded checkpoint {path} (epoch {checkpoint['epoch']})")
        return checkpoint["epoch"]

    def _save_metrics(self):
        """Save metrics to JSON file."""
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def summary(self) -> str:
        """Return a summary string of the run."""
        elapsed = time.time() - self._start_time
        n_epochs = len(self.metrics)
        lines = [
            f"Run: {self.run_name}",
            f"Epochs: {n_epochs}",
            f"Best val loss: {self.best_val_loss:.6f} (epoch {self.best_epoch})",
            f"Elapsed: {elapsed:.0f}s",
        ]
        if self.metrics:
            last = self.metrics[-1]
            lines.append(f"Last train loss: {last['train_loss']:.6f}")
            if last.get("val_loss") is not None:
                lines.append(f"Last val loss: {last['val_loss']:.6f}")
        return "\n".join(lines)
