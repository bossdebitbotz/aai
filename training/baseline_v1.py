#!/usr/bin/env python3
"""Re-measure the V1 baseline directional accuracy on the SAME test split, using the
sign of V1's predicted mid-price change over each horizon (V1 has no direction head)."""
import json, logging, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baseline_v1")


def sign_direction_labels(entry_mid, future_mid, horizon_idx, flat_threshold=0.01):
    """labels (N,) from sign of (future_mid[:,horizon_idx] - entry_mid). 0=down,1=flat,2=up.
    Note: flat_threshold is on the SCALED mid (matches LOBLossV2 which compares scaled mids)."""
    change = future_mid[:, horizon_idx] - entry_mid
    labels = np.ones(len(entry_mid), dtype=np.int64)
    labels[change > flat_threshold] = 2
    labels[change < -flat_threshold] = 0
    return labels


@torch.no_grad()
def measure(v1_run_dir: str, levels: int = 40, source: str = "db", parquet_dir: str = "lob_data",
            horizons=(5, 11, 23), flat_threshold=0.01):
    from training.dataset import DataConfig, build_dataloaders
    from training.model import CompoundAttentionModel
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    n_features = levels * 5 + 11
    mid_idx = levels * 4
    cfg = DataConfig(lob_levels=levels, feature_version="v1", savgol_window=11,
                     source=source, parquet_dir=parquet_dir)
    _, _, test_loader, _ = build_dataloaders(cfg, batch_size=64)
    model = CompoundAttentionModel(n_levels=levels, n_features=n_features,
                                   context_length=cfg.context_length, prediction_length=cfg.prediction_length,
                                   d_model=66, n_heads=3, n_layers=3, d_ff=264).to(device)
    ck = torch.load(Path(v1_run_dir) / "checkpoints" / "best.pt", weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    correct = {h: 0 for h in horizons}; total = 0
    for batch in test_loader:
        context = batch["context"].to(device); target = batch["target"].to(device)
        ex = batch["exchange_id"].to(device); sym = batch["symbol_id"].to(device)
        pred = model(context, ex, sym)  # V1 returns (B,Tp,F)
        entry = context[:, -1, mid_idx].cpu().numpy()
        pred_mid = pred[:, :, mid_idx].cpu().numpy()
        true_mid = target[:, :, mid_idx].cpu().numpy()
        for h in horizons:
            hi = min(h, pred_mid.shape[1] - 1)
            pl = sign_direction_labels(entry, pred_mid, hi, flat_threshold)
            tl = sign_direction_labels(entry, true_mid, hi, flat_threshold)
            correct[h] += int((pl == tl).sum())
        total += len(entry)
    acc = {str(h): (correct[h] / max(total, 1)) for h in horizons}
    out = Path(parquet_dir).parent / "experiments" / "v1_baseline_acc.json"
    out.write_text(json.dumps({**acc, "source": "v1-remeasured", "n": total}, indent=2))
    logger.info(f"V1 baseline accuracy: {acc} (n={total}) -> {out}")
    return acc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-run-dir", required=True)
    ap.add_argument("--levels", type=int, default=40)
    ap.add_argument("--source", default="db", choices=["db", "parquet"])
    ap.add_argument("--parquet-dir", default="lob_data")
    a = ap.parse_args()
    measure(a.v1_run_dir, a.levels, a.source, a.parquet_dir)
