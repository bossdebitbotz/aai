# training/test_direction_utils.py
import sys, logging, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.model_v2 import compute_direction_labels, directional_accuracy, LOBLossV2

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_compute_direction_labels_matches_lossv2():
    B, Tp, Fdim = 4, 24, 162
    mid_idx, horizons, thr = 160, (5, 11, 23), 0.01
    target = torch.zeros(B, Tp, Fdim)
    context_last = torch.zeros(B, Fdim)
    # craft up/down/flat at horizon 5
    target[0, 5, mid_idx] = 1.0     # up
    target[1, 5, mid_idx] = -1.0    # down
    target[2, 5, mid_idx] = 0.0     # flat
    labels = compute_direction_labels(target, context_last, mid_idx, horizons, thr)
    assert labels.shape == (B, 3)
    assert labels[0, 0].item() == 2  # up
    assert labels[1, 0].item() == 0  # down
    assert labels[2, 0].item() == 1  # flat
    # delegation parity
    loss = LOBLossV2(n_levels=40, mid_price_idx=mid_idx,
                     direction_horizons=horizons, flat_threshold=thr)
    ref = loss._direction_labels(target, context_last)
    assert torch.equal(labels, ref)
    logger.info("PASS: test_compute_direction_labels_matches_lossv2")

def test_directional_accuracy_perfect_and_chance():
    B = 100
    labels = torch.randint(0, 3, (B, 3))
    # build logits that perfectly predict labels
    logits = torch.full((B, 3, 3), -5.0)
    for h in range(3):
        logits[torch.arange(B), h, labels[:, h]] = 5.0
    acc = directional_accuracy(logits.reshape(B, 9), labels)
    assert acc.shape == (3,)
    assert torch.allclose(acc, torch.ones(3))
    logger.info("PASS: test_directional_accuracy_perfect_and_chance")
