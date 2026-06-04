# training/test_tracker_v2.py
import sys, logging, tempfile, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_monitor_max_metric_from_extra():
    with tempfile.TemporaryDirectory() as d:
        t = ExperimentTracker(d, run_name="t", monitor="val_dir_acc", monitor_mode="max")
        b1 = t.log_epoch(1, 1.0, 1.0, 0.0, val_loss=1.0, extra={"val_dir_acc": 0.52})
        b2 = t.log_epoch(2, 0.9, 0.9, 0.0, val_loss=1.2, extra={"val_dir_acc": 0.58})
        b3 = t.log_epoch(3, 0.8, 0.8, 0.0, val_loss=0.5, extra={"val_dir_acc": 0.55})
        assert b1 is True and b2 is True and b3 is False  # selects on max dir_acc, not val_loss
        logger.info("PASS: test_monitor_max_metric_from_extra")

def test_save_checkpoint_extra(tmp_path):
    import torch.nn as nn
    t = ExperimentTracker(str(tmp_path), run_name="c")
    m = nn.Linear(2, 2); opt = torch.optim.Adam(m.parameters())
    t.save_checkpoint(m, opt, epoch=1, is_best=True, extra={"n_features": 219, "temps": [1.1, 1.2, 1.3]})
    ck = torch.load(tmp_path / "c" / "checkpoints" / "best.pt", weights_only=False)
    assert ck["n_features"] == 219 and ck["temps"] == [1.1, 1.2, 1.3]
    logger.info("PASS: test_save_checkpoint_extra")
