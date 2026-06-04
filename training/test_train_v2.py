# training/test_train_v2.py
import sys, logging, torch
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.train_v2 import run_one_epoch_v2, evaluate_v2_loader
from training.model_v2 import CompoundAttentionModelV2, LOBLossV2

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def _fake_loader(n_batches=3, B=4, T=20, Tp=24, F=36, n_levels=5):
    batches = []
    for _ in range(n_batches):
        batches.append({
            "context": torch.randn(B, T, F),
            "target": torch.randn(B, Tp, F),
            "exchange_id": torch.zeros(B, dtype=torch.long),
            "symbol_id": torch.zeros(B, dtype=torch.long),
            "timestamp": torch.zeros(B),
        })
    return batches

def test_run_one_epoch_v2_and_eval():
    n_levels, F, T, Tp = 5, 36, 20, 24
    model = CompoundAttentionModelV2(n_levels=n_levels, n_features=F, context_length=T,
                                     prediction_length=Tp, d_model=30, n_heads=3, n_layers=1, d_ff=120)
    loss_fn = LOBLossV2(n_levels=n_levels, mid_price_idx=20, use_feature_weights=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _fake_loader(F=F, T=T, Tp=Tp, n_levels=n_levels)
    metrics = run_one_epoch_v2(model, loss_fn, opt, scheduler=None, loader=loader,
                               device=torch.device("cpu"), accum_steps=2)
    assert "total" in metrics and "dir_acc_mean" in metrics
    assert 0.0 <= metrics["dir_acc_mean"] <= 1.0
    ev = evaluate_v2_loader(model, loss_fn, loader, torch.device("cpu"))
    assert "dir_acc_per_h" in ev and len(ev["dir_acc_per_h"]) == 3
    logger.info("PASS: test_run_one_epoch_v2_and_eval")
