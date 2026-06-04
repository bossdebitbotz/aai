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
