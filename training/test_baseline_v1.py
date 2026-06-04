# training/test_baseline_v1.py
import sys, logging, numpy as np
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.baseline_v1 import sign_direction_labels

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_sign_direction_labels():
    # entry mid 100; future path crosses thresholds
    entry = np.array([100.0, 100.0, 100.0])
    future = np.array([[100.5, 101.0], [99.5, 99.0], [100.0, 100.005]])  # up, down, flat
    labels = sign_direction_labels(entry, future, horizon_idx=1, flat_threshold=0.01)
    assert labels.tolist() == [2, 0, 1]
    logger.info("PASS: test_sign_direction_labels")
