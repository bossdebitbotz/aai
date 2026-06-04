# training/test_dataset_v2.py
import sys, logging, numpy as np
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.dataset import DataConfig

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_dataconfig_v2_feature_count_and_warmup():
    cfg = DataConfig(lob_levels=40, feature_version="v2", savgol_window=11)
    assert cfg.n_enriched_features == 40 * 5 + 19   # 219
    assert cfg.warmup_trim == 60                     # max(savgol_window, 60)
    cfg1 = DataConfig(lob_levels=40, feature_version="v1")
    assert cfg1.n_enriched_features == 40 * 5 + 11   # 211
    logger.info("PASS: test_dataconfig_v2_feature_count_and_warmup")


from training.features_v2 import engineer_features_v2

def test_per_split_fe_no_cross_boundary_dependence():
    """Features in a split must not depend on data outside that split (no leakage)."""
    n_levels, base = 40, 40 * 4 + 2
    rng = np.random.default_rng(0)
    full = rng.random((400, base)) + 1.0
    split_at = 250
    # Engineer the val split alone vs. as the tail of the full series, after warmup trim.
    val_alone, _ = engineer_features_v2(full[split_at:], n_levels, apply_smoothing=False)
    val_in_full, _ = engineer_features_v2(full, n_levels, apply_smoothing=False)
    trim = 60
    a = val_alone[trim:]
    b = val_in_full[split_at + trim:]
    # momentum/OFI features within the trimmed region must be identical either way
    assert np.allclose(a, b, atol=1e-9), "cross-boundary leakage in engineered features"
    logger.info("PASS: test_per_split_fe_no_cross_boundary_dependence")
