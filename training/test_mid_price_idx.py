# training/test_mid_price_idx.py
import sys, logging, numpy as np
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.features import get_column_indices
from training.features_v2 import engineer_features_v2
from training.model_v2 import LOBLossV2

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_mid_price_idx_matches_layout():
    n_levels = 40
    # raw base array: 4N+2 cols, mid at 4N=160, spread at 161
    T, base = 80, n_levels * 4 + 2
    raw = np.random.rand(T, base).astype(np.float64) + 1.0
    enriched, _ = engineer_features_v2(raw, n_levels, apply_smoothing=False)
    assert enriched.shape[1] == n_levels * 5 + 19  # 219
    idx = get_column_indices(n_levels)
    assert idx["mid_price"] == 160
    assert LOBLossV2(n_levels=n_levels).mid_price_idx == idx["mid_price"]
    logger.info("PASS: test_mid_price_idx_matches_layout")
