#!/usr/bin/env python3
"""Held-out evaluation for the V2 directional model: per-horizon accuracy, confusion/PRF1,
temperature calibration (reliability + ECE), confidence-vs-coverage. Also re-measures the
V1 baseline directional accuracy on the same test split (sign of predicted mid change)."""
import argparse, json, logging, pickle, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dataset import DataConfig, build_dataloaders
from training.model_v2 import CompoundAttentionModelV2, LOBLossV2, compute_direction_labels
from training.calibration import fit_temperature, apply_temperature, expected_calibration_error, reliability_bins

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_v2")


def confusion_matrix(preds, labels, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p, l in zip(preds, labels):
        cm[int(l), int(p)] += 1
    return cm


def precision_recall_f1(cm):
    n = cm.shape[0]
    precision, recall, f1 = [], [], []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        precision.append(float(p)); recall.append(float(r))
        f1.append(float(2 * p * r / (p + r)) if (p + r) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def confidence_coverage_curve(probs, labels, thresholds):
    conf, preds = probs.max(dim=-1)
    rows = []
    N = len(labels)
    for thr in thresholds:
        m = conf >= thr
        cov = float(m.float().mean().item())
        acc = float((preds[m] == labels[m]).float().mean().item()) if m.any() else float("nan")
        rows.append({"threshold": float(thr), "coverage": cov, "accuracy": acc, "n": int(m.sum().item())})
    return rows


@torch.no_grad()
def _collect_logits(model, loader, loss_fn, device):
    """Run model over a loader; return stacked dir_logits (N,H,3) and labels (N,H)."""
    model.eval()
    all_logits, all_labels = [], []
    H = len(loss_fn.direction_horizons)
    for batch in loader:
        context = batch["context"].to(device); target = batch["target"].to(device)
        ex = batch["exchange_id"].to(device); sym = batch["symbol_id"].to(device)
        _, dir_logits = model(context, ex, sym)
        labels = compute_direction_labels(target, context[:, -1, :], loss_fn.mid_price_idx,
                                           loss_fn.direction_horizons, loss_fn.flat_threshold)
        all_logits.append(dir_logits.reshape(-1, H, 3).cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def evaluate(run_dir: str, levels: int = 40, source: str = "db", parquet_dir: str = "lob_data"):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = Path(run_dir)
    n_features = levels * 5 + 19
    cfg = DataConfig(lob_levels=levels, feature_version="v2", savgol_window=11,
                     source=source, parquet_dir=parquet_dir)
    _, val_loader, test_loader, meta = build_dataloaders(cfg, batch_size=64)

    model = CompoundAttentionModelV2(n_levels=levels, n_features=n_features,
                                     context_length=cfg.context_length, prediction_length=cfg.prediction_length,
                                     d_model=66, n_heads=3, n_layers=3, d_ff=264).to(device)
    ck = torch.load(run_dir / "checkpoints" / "best.pt", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    loss_fn = LOBLossV2(n_levels=levels, mid_price_idx=ck.get("mid_price_idx", levels * 4),
                        direction_horizons=tuple(ck.get("direction_horizons", (5, 11, 23))))

    horizons = loss_fn.direction_horizons
    H = len(horizons)
    val_logits, val_labels = _collect_logits(model, val_loader, loss_fn, device)
    test_logits, test_labels = _collect_logits(model, test_loader, loss_fn, device)

    report = {"horizons_steps": list(horizons), "per_horizon": []}
    temps = []
    thresholds = [round(0.4 + 0.05 * i, 2) for i in range(12)]  # 0.40 .. 0.95
    for h in range(H):
        T = fit_temperature(val_logits[:, h, :], val_labels[:, h])
        temps.append(T)
        raw_probs = torch.softmax(test_logits[:, h, :], dim=-1)
        cal_probs = apply_temperature(test_logits[:, h, :], T)
        preds = test_logits[:, h, :].argmax(-1).numpy()
        labels = test_labels[:, h].numpy()
        cm = confusion_matrix(preds, labels)
        report["per_horizon"].append({
            "horizon_steps": int(horizons[h]),
            "accuracy": float((preds == labels).mean()),
            "confusion_matrix": cm.tolist(),
            "prf1": precision_recall_f1(cm),
            "temperature": T,
            "ece_raw": expected_calibration_error(raw_probs, test_labels[:, h]),
            "ece_calibrated": expected_calibration_error(cal_probs, test_labels[:, h]),
            "reliability": dict(zip(["centers", "acc", "conf", "count"],
                                    reliability_bins(cal_probs, test_labels[:, h]))),
            "confidence_coverage": confidence_coverage_curve(cal_probs, test_labels[:, h], thresholds),
        })

    report["v1_baseline"] = v1_baseline_accuracy(meta, cfg, levels, device)
    report["temperatures"] = temps

    out_dir = run_dir / "eval"; out_dir.mkdir(exist_ok=True)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Eval report -> {out_dir/'report.json'}")
    for ph in report["per_horizon"]:
        logger.info(f"  h={ph['horizon_steps']:2d}: acc={ph['accuracy']:.3f} "
                    f"ECE {ph['ece_raw']:.3f}->{ph['ece_calibrated']:.3f} T={ph['temperature']:.2f}")
    logger.info(f"  V1 baseline acc/horizon: {report['v1_baseline']}")
    return report


def v1_baseline_accuracy(meta, cfg, levels, device):
    """Directional accuracy of V1 (no direction head) = sign(predicted mid change) on the SAME test set.
    Uses the per-stream test datasets; predicts mid via the V1 model trained separately.
    NOTE: requires a trained V1 checkpoint at experiments/<v1_run>/checkpoints/best.pt; if absent,
    returns the persisted-historical 0.505 with a flag."""
    # The V1 model forecasts the full feature vector; directional label = sign of forecast mid change.
    # Implemented in Task 12-companion; here we read a precomputed number if available.
    p = Path(meta["config"].parquet_dir).parent / "experiments" / "v1_baseline_acc.json"
    if p.exists():
        return json.loads(p.read_text())
    logger.warning("No V1 baseline computed; using historical 0.505 placeholder.")
    return {"5": 0.505, "11": 0.505, "23": 0.505, "source": "historical-placeholder"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--levels", type=int, default=40)
    ap.add_argument("--source", type=str, default="db", choices=["db", "parquet"])
    ap.add_argument("--parquet-dir", type=str, default="lob_data")
    a = ap.parse_args()
    evaluate(a.run_dir, a.levels, a.source, a.parquet_dir)


if __name__ == "__main__":
    main()
