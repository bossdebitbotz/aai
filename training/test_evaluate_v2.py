# training/test_evaluate_v2.py
import sys, logging, numpy as np, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.evaluate_v2 import confusion_matrix, precision_recall_f1, confidence_coverage_curve

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_confusion_and_prf1_sum_to_n():
    preds = np.array([0, 1, 2, 2, 1, 0, 2])
    labels = np.array([0, 1, 2, 1, 1, 0, 0])
    cm = confusion_matrix(preds, labels, n_classes=3)
    assert cm.sum() == len(preds)
    prf = precision_recall_f1(cm)
    assert set(prf.keys()) == {"precision", "recall", "f1"}
    assert len(prf["precision"]) == 3
    logger.info("PASS: test_confusion_and_prf1_sum_to_n")

def test_confidence_coverage_monotone_coverage():
    probs = torch.rand(500, 3); probs = probs / probs.sum(-1, keepdim=True)
    labels = torch.randint(0, 3, (500,))
    curve = confidence_coverage_curve(probs, labels, thresholds=[0.4, 0.6, 0.8, 0.95])
    covs = [row["coverage"] for row in curve]
    assert all(covs[i] >= covs[i + 1] - 1e-9 for i in range(len(covs) - 1))  # coverage non-increasing
    for row in curve:
        assert 0.0 <= row["coverage"] <= 1.0
    logger.info("PASS: test_confidence_coverage_monotone_coverage")
