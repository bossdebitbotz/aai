"""Tests for the experiment tracker."""

import sys
import json
import logging
import tempfile
import shutil
from pathlib import Path

import torch

sys.path.insert(0, "/Users/clint/Projects/aai")
from training.tracker import ExperimentTracker, RunConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def test_tracker_basic():
    """Test basic tracker functionality."""
    tmpdir = tempfile.mkdtemp()
    try:
        tracker = ExperimentTracker(tmpdir, run_name="test_run")

        # Log config
        config = RunConfig(n_levels=5, d_model=32)
        tracker.log_config(config)

        config_path = Path(tmpdir) / "test_run" / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            saved = json.load(f)
        assert saved["n_levels"] == 5
        assert saved["d_model"] == 32

        # Log data info
        tracker.log_data_info({"streams": 16, "total_samples": 50000})

        # Log epochs
        is_best = tracker.log_epoch(
            epoch=1, train_loss=1.0, train_forecast_loss=0.9,
            train_structure_loss=0.1, val_loss=0.8, val_forecast_loss=0.7,
            val_structure_loss=0.1, learning_rate=1e-3,
        )
        assert is_best  # First epoch should be best

        is_best = tracker.log_epoch(
            epoch=2, train_loss=0.9, train_forecast_loss=0.8,
            train_structure_loss=0.1, val_loss=0.9, val_forecast_loss=0.8,
            val_structure_loss=0.1,
        )
        assert not is_best  # Val loss increased

        is_best = tracker.log_epoch(
            epoch=3, train_loss=0.7, train_forecast_loss=0.6,
            train_structure_loss=0.1, val_loss=0.6, val_forecast_loss=0.5,
            val_structure_loss=0.1,
        )
        assert is_best  # New best

        assert tracker.best_val_loss == 0.6
        assert tracker.best_epoch == 3

        # Check metrics file
        metrics_path = Path(tmpdir) / "test_run" / "metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert len(metrics) == 3

        logger.info("PASS: tracker_basic")
    finally:
        shutil.rmtree(tmpdir)


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    tmpdir = tempfile.mkdtemp()
    try:
        tracker = ExperimentTracker(tmpdir, run_name="ckpt_test")

        # Create a simple model
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Save original weights
        orig_weight = model.weight.data.clone()

        # Simulate training
        tracker.log_epoch(epoch=1, train_loss=1.0, train_forecast_loss=0.9,
                          train_structure_loss=0.1, val_loss=0.8,
                          val_forecast_loss=0.7, val_structure_loss=0.1)
        tracker.save_checkpoint(model, optimizer, epoch=1, is_best=True)

        # Modify model weights
        model.weight.data.fill_(0.0)
        assert not torch.allclose(model.weight.data, orig_weight)

        # Load checkpoint
        epoch = tracker.load_checkpoint(model, optimizer, "best.pt")
        assert epoch == 1
        assert torch.allclose(model.weight.data, orig_weight)

        # Latest checkpoint should also exist
        assert (Path(tmpdir) / "ckpt_test" / "checkpoints" / "latest.pt").exists()

        logger.info("PASS: checkpoint_save_load")
    finally:
        shutil.rmtree(tmpdir)


def test_tracker_summary():
    """Test summary output."""
    tmpdir = tempfile.mkdtemp()
    try:
        tracker = ExperimentTracker(tmpdir, run_name="summary_test")
        tracker.log_epoch(epoch=1, train_loss=1.0, train_forecast_loss=0.9,
                          train_structure_loss=0.1, val_loss=0.8,
                          val_forecast_loss=0.7, val_structure_loss=0.1)

        summary = tracker.summary()
        assert "summary_test" in summary
        assert "Epochs: 1" in summary
        assert "0.800000" in summary  # best val loss

        logger.info("PASS: tracker_summary")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_tracker_basic()
    test_checkpoint_save_load()
    test_tracker_summary()

    logger.info("\n=== ALL TRACKER TESTS PASSED ===")
