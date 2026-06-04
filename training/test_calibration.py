# training/test_calibration.py
import sys, logging, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.calibration import fit_temperature, expected_calibration_error, apply_temperature

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_temperature_preserves_argmax_and_reduces_ece():
    torch.manual_seed(0)
    N = 2000
    labels = torch.randint(0, 3, (N,))
    # overconfident logits: correct class favored but scaled too sharply
    logits = torch.randn(N, 3)
    logits[torch.arange(N), labels] += 1.0
    logits = logits * 4.0  # exaggerate confidence -> miscalibrated
    T = fit_temperature(logits, labels)
    assert T > 0
    pre = expected_calibration_error(torch.softmax(logits, dim=-1), labels)
    post = expected_calibration_error(apply_temperature(logits, T), labels)
    # argmax (accuracy) unchanged
    assert torch.equal(logits.argmax(-1), apply_temperature(logits, T).argmax(-1))
    assert post <= pre + 1e-6
    logger.info(f"PASS: test_temperature ... T={T:.3f} ECE {pre:.3f}->{post:.3f}")
