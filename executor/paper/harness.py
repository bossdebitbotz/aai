#!/usr/bin/env python3
"""Paper-trade harness: orchestrates per-stream decisions through the FROZEN signal
generator + Inventory + persistent ledger. Two drivers:
  - run_replay: step through historical DB buckets (validation / dry-run).
  - run_live: poll the live DB every ~2 min and trade in real time (needs collector running).
Restart-safe: inventory state is reloaded from the ledger on start.
"""
from __future__ import annotations
import sys, time
from datetime import datetime, timezone
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
from executor.paper import db_source as DBS
from executor.paper import signal_generator as SG
from executor.paper.ledger import PaperLedger

STREAMS = [("binance_perp", "BTC-USDT"), ("binance_perp", "ETH-USDT"),
           ("binance_perp", "SOL-USDT"), ("binance_perp", "WLD-USDT")]
FEE_BP = 3.0
STRIDE = SG.STRIDE


def _step(ledger: PaperLedger, exch: str, sym: str, raw_buffer: np.ndarray, ts: str,
          offset: int = SG.LAG, buckets=None):
    """One decision for one stream on a given buffer; updates inventory + ledger."""
    stream = f"{exch}_{sym}"
    sig = SG.generate_from_buffer(raw_buffer, exch, sym, decision_offset=offset, buckets=buckets)
    if sig.get("reason"):
        return sig
    inv = ledger.load_inventory(stream)
    gated = sig["signal"] != 0
    delta, cost = inv.on_decision(sig["signal"] if gated else 0, gated,
                                  sig["mid"], sig["spread"], FEE_BP)
    ledger.record(ts, stream, sig, inv, delta, cost)
    return sig


def _step_sizing(ledger: PaperLedger, exch: str, sym: str, raw_buffer: np.ndarray, ts: str,
                 offset: int = SG.LAG, buckets=None, params: dict | None = None):
    """One decision through the vol-target/signal-decay exit (executor.paper.sizing).
    Reuses the SAME signal + gate inputs from generate_from_buffer (SSOT), so the entry
    path is identical to the base book; only sizing + exit differ."""
    from executor.paper import sizing as Z
    params = params or {"target_vol": Z.TARGET_VOL, "stop_floor_bp": Z.STOP_FLOOR_BP,
                        "cooldown_n": Z.COOLDOWN_N, "decay_mode": Z.DECAY_MODE, "decay_k": Z.DECAY_K}
    stream = f"{exch}_{sym}"
    sig = SG.generate_from_buffer(raw_buffer, exch, sym, decision_offset=offset, buckets=buckets)
    if sig.get("reason"):
        return sig
    inv = ledger.load_inventory(stream)
    tr = ledger.load_sizing_tracker(stream)
    prev_pos, prev_cost = inv.position, inv.realized_cost_bp
    Z.step_with_sizing(sig["s0"], sig["s1"], sig["s2"],
                       np.asarray(sig["scaled_mid_ctx"]), np.asarray(sig["dec_mids"]),
                       sig["mid"], sig["spread"], FEE_BP, inv, tr, params)
    delta = inv.position - prev_pos
    cost = inv.realized_cost_bp - prev_cost
    ledger.record(ts, stream, sig, inv, delta, cost)
    ledger.save_sizing_tracker(stream, tr)
    return sig


def _dispatch(strategy: str):
    return _step_sizing if strategy == "sizing" else _step


def run_replay(ledger: PaperLedger, n_decisions: int = 40, streams=STREAMS, end_time=None, strategy: str = "base"):
    """Replay n_decisions historical buckets per stream (stride 23), ending at end_time
    (default: most recent). Use a pre-gap end_time to replay a contiguous window."""
    step_fn = _dispatch(strategy)
    need = SG.HIST_NEEDED + n_decisions * STRIDE + SG.LAG + 50
    for exch, sym in streams:
        buckets, raw = DBS.fetch_buffer(exch, sym, n=need, end_time=end_time)
        if len(raw) < SG.HIST_NEEDED + STRIDE:
            print(f"{exch}_{sym}: insufficient history ({len(raw)})"); continue
        # decision indices: last n_decisions at stride, each needing LAG future in-buffer
        first = max(SG.HIST_NEEDED + 5, len(raw) - 1 - SG.LAG - (n_decisions - 1) * STRIDE)
        didx = list(range(first, len(raw) - SG.LAG, STRIDE))
        for d in didx:
            ts = buckets[d].isoformat()
            step_fn(ledger, exch, sym, raw[: d + 1 + SG.LAG], ts, offset=SG.LAG,
                    buckets=buckets[: d + 1 + SG.LAG])
        print(f"{exch}_{sym}: replayed {len(didx)} decisions ({strategy})")


def run_live(ledger: PaperLedger, streams=STREAMS, poll_s: int = 115, max_iters=None, strategy: str = "base"):
    """LIVE loop: every ~2 min, generate a decision per stream off the latest DB buffer.
    Requires the collector to be running (DB advancing). Decisions are skipped for a
    stream if its latest bucket hasn't advanced >= STRIDE since the last decision.
    """
    step_fn = _dispatch(strategy)
    last_bucket = {}
    it = 0
    while max_iters is None or it < max_iters:
        statuses = []
        for exch, sym in streams:
            stream = f"{exch}_{sym}"
            try:
                buckets, raw = DBS.fetch_buffer(exch, sym, n=SG.MIN_BUFFER + 200)
            except Exception as e:
                statuses.append(f"{sym}:dberr"); continue
            if len(raw) < SG.MIN_BUFFER or not buckets:
                statuses.append(f"{sym}:nobuf({len(raw)})"); continue
            now_b = buckets[-1]
            # only act if enough new buckets have arrived since last decision (~2 min cadence)
            prev = last_bucket.get(stream)
            if prev is not None and (now_b - prev).total_seconds() < (STRIDE - 1) * 5:
                statuses.append(f"{sym}:wait"); continue
            sig = step_fn(ledger, exch, sym, raw, now_b.isoformat(), offset=SG.LAG, buckets=buckets)
            last_bucket[stream] = now_b
            if sig.get("reason"):
                statuses.append(f"{sym}:{sig['reason'][:6]}")
            else:
                statuses.append(f"{sym}:sig={sig['signal']:+d}/pos={ledger.load_inventory(stream).position:+.0f}")
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{now_iso}] iter {it} | net={ledger.portfolio_net_bp():+.1f}bp | " + " ".join(statuses), flush=True)
        it += 1
        time.sleep(poll_s)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--db", default="/Volumes/Docker-SSD/projects/aaiwdbback/aai/executor/paper/paper_ledger.db")
    ap.add_argument("--strategy", choices=["base", "sizing"], default="base",
                    help="base = FROZEN gate+inventory (live book); sizing = vol-target/signal-decay exit (A/B)")
    a = ap.parse_args()
    led = PaperLedger(a.db)
    if a.mode == "replay":
        run_replay(led, n_decisions=a.n, strategy=a.strategy)
        print("\n=== ledger summary (stream, pos, marked_bp, cost_bp, net_bp, n) ===")
        for r in led.summary():
            print(" ", r)
        print(f"PORTFOLIO NET: {led.portfolio_net_bp():+.1f} bp-units")
    else:
        run_live(led, strategy=a.strategy)
