#!/usr/bin/env python3
"""Offline window-level PnL backtest on the held-out test split. Gated on calibrated
confidence; reports maker-optimistic and taker-conservative cost brackets + a conf->PnL curve.
NOT a tick-level fill simulation (realistic fill model deferred per spec)."""
import argparse, json, logging, pickle, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest_v2")

PRIMARY_HORIZON_STEPS = 23  # 2-min horizon


def simulate_window_trades(entry_mid, exit_mid, pred_cls, conf, ts,
                           threshold, cost_bps_per_side, cooldown_s, hold_s):
    """One decision per window, in time order. Long if cls==2, short if cls==0, skip if flat/gated.
    Enforces cooldown between entries. Returns metrics dict."""
    order = np.argsort(ts)
    cost = 2.0 * cost_bps_per_side / 1e4  # round-trip fraction
    rets = []
    last_entry_ts = -np.inf
    eligible = 0
    for i in order:
        if pred_cls[i] == 1 or conf[i] < threshold:
            continue
        eligible += 1
        if ts[i] - last_entry_ts < cooldown_s:
            continue
        gross = (exit_mid[i] / entry_mid[i] - 1.0) if pred_cls[i] == 2 else (1.0 - exit_mid[i] / entry_mid[i])
        rets.append(gross - cost)
        last_entry_ts = ts[i]
    rets = np.array(rets, dtype=np.float64)
    n = len(rets)
    total_decisions = int((pred_cls != 1).sum())
    win_rate = float((rets > 0).mean()) if n else float("nan")
    sharpe = float(rets.mean() / rets.std()) if n > 1 and rets.std() > 0 else float("nan")
    equity = np.cumprod(1.0 + rets) if n else np.array([1.0])
    max_dd = float((1.0 - equity / np.maximum.accumulate(equity)).max()) if n else 0.0
    return {
        "n_trades": int(n),
        "coverage": float(n / total_decisions) if total_decisions else 0.0,
        "net_return": float(rets.sum()),
        "avg_edge_bps": float(rets.mean() * 1e4) if n else float("nan"),
        "win_rate": win_rate,
        "sharpe_per_trade": sharpe,
        "max_drawdown": max_dd,
    }


@torch.no_grad()
def _collect_window_signals(model, ds, loss_fn, device, temperature):
    """For one stream's test LOBDataset: return entry_mid, exit_mid (raw), pred_cls, conf, ts
    at the primary horizon, one row per window."""
    from training.calibration import apply_temperature
    mid_idx = loss_fn.mid_price_idx
    scaler = ds  # placeholder; real scaler passed separately
    entry, exitm, cls, conf, ts = [], [], [], [], []
    h_idx = min(PRIMARY_HORIZON_STEPS, ds.config.prediction_length - 1)
    horizons = list(loss_fn.direction_horizons)
    h_pos = horizons.index(PRIMARY_HORIZON_STEPS) if PRIMARY_HORIZON_STEPS in horizons else len(horizons) - 1
    for k in range(len(ds)):
        item = ds[k]
        context = item["context"].unsqueeze(0).to(device)
        target = item["target"].unsqueeze(0).to(device)
        ex = torch.tensor([ds.exchange_id]).to(device)
        sym = torch.tensor([ds.symbol_id]).to(device)
        _, dir_logits = model(context, ex, sym)
        probs = apply_temperature(dir_logits.reshape(1, len(horizons), 3)[:, h_pos, :], temperature)
        c = int(probs.argmax(-1).item())
        entry.append(context[0, -1, mid_idx].item())
        exitm.append(target[0, h_idx, mid_idx].item())
        cls.append(c); conf.append(float(probs.max().item()))
        ts.append(float(item["timestamp"]))
    return (np.array(entry), np.array(exitm), np.array(cls), np.array(conf), np.array(ts))


def backtest(run_dir: str, levels: int = 40, source: str = "db", parquet_dir: str = "lob_data",
             gate: float = 0.6, hold_s: int = 120, cooldown_s: int = 300):
    from training.dataset import DataConfig, build_dataloaders
    from training.model_v2 import CompoundAttentionModelV2, LOBLossV2
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = Path(run_dir)
    n_features = levels * 5 + 19
    cfg = DataConfig(lob_levels=levels, feature_version="v2", savgol_window=11,
                     source=source, parquet_dir=parquet_dir)
    _, _, _, meta = build_dataloaders(cfg, batch_size=64)
    model = CompoundAttentionModelV2(n_levels=levels, n_features=n_features,
                                     context_length=cfg.context_length, prediction_length=cfg.prediction_length,
                                     d_model=66, n_heads=3, n_layers=3, d_ff=264).to(device)
    ck = torch.load(run_dir / "checkpoints" / "best.pt", weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    loss_fn = LOBLossV2(n_levels=levels, mid_price_idx=ck.get("mid_price_idx", levels * 4),
                        direction_horizons=tuple(ck.get("direction_horizons", (5, 11, 23))))
    # temperature for the primary horizon from the eval report (if present), else 1.0
    eval_report = run_dir / "eval" / "report.json"
    T = 1.0
    if eval_report.exists():
        rep = json.loads(eval_report.read_text())
        for ph in rep["per_horizon"]:
            if ph["horizon_steps"] == PRIMARY_HORIZON_STEPS:
                T = ph["temperature"]

    scalers = pickle.load(open(run_dir / "scalers.pkl", "rb"))
    results = {"gate": gate, "primary_horizon_steps": PRIMARY_HORIZON_STEPS, "per_symbol": {}}
    cost_models = {"maker": 2.0, "taker": 5.0}
    for key, ds in meta["test_datasets_by_stream"].items():
        if key not in ("binance_perp_BTC-USDT", "binance_perp_ETH-USDT"):
            continue  # tradable target only
        entry_s, exit_s, cls, conf, ts = _collect_window_signals(model, ds, loss_fn, device, T)
        # de-scale mid back to raw price using the stream's scaler
        sc = scalers[key]; mid_idx = loss_fn.mid_price_idx
        mean, std = sc._means[mid_idx], sc._stds[mid_idx]
        entry = entry_s * std + mean
        exitm = exit_s * std + mean
        results["per_symbol"][key] = {
            cm: simulate_window_trades(entry, exitm, cls, conf, ts, gate, bps, cooldown_s, hold_s)
            for cm, bps in cost_models.items()
        }
        # confidence -> net PnL curve (maker cost)
        curve = []
        for thr in [round(0.4 + 0.05 * i, 2) for i in range(12)]:
            curve.append({"threshold": thr,
                          **simulate_window_trades(entry, exitm, cls, conf, ts, thr, 2.0, cooldown_s, hold_s)})
        results["per_symbol"][key]["conf_pnl_curve_maker"] = curve

    out_dir = run_dir / "backtest"; out_dir.mkdir(exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(results, indent=2))
    logger.info(f"Backtest report -> {out_dir/'report.json'}")
    for key, r in results["per_symbol"].items():
        logger.info(f"  {key}: maker net={r['maker']['net_return']:.4f} "
                    f"(win={r['maker']['win_rate']:.2f}, n={r['maker']['n_trades']}, cov={r['maker']['coverage']:.2f}) "
                    f"| taker net={r['taker']['net_return']:.4f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--levels", type=int, default=40)
    ap.add_argument("--source", default="db", choices=["db", "parquet"])
    ap.add_argument("--parquet-dir", default="lob_data")
    ap.add_argument("--gate", type=float, default=0.6)
    a = ap.parse_args()
    backtest(a.run_dir, a.levels, a.source, a.parquet_dir, a.gate)


if __name__ == "__main__":
    main()
