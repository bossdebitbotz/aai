"""DECISIVE signal-parity test: the live path (decision bucket LAG=6 back from 'now',
centered-savgol window only partially in the past) must produce the SAME signal and
gate values as a reference where that bucket is interior with full future data (the
training/backtest condition). Proves the 30s-lag mechanism yields training-accurate
live signals end-to-end (features -> scale -> model -> gates).

Run: .venv/bin/python -m pytest executor/paper/test_signal_parity.py -q -s
"""
import sys
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
from executor.paper import db_source as DBS
from executor.paper import signal_generator as SG

EXCH, SYM = "binance_perp", "BTC-USDT"
REF_OFFSET = 50           # reference: decision bucket sits 50 back from end -> full future for its savgol


def test_signal_parity_live_vs_fullfuture():
    # one buffer ending at latest available DB bucket; slice in-memory for both paths
    _, raw = DBS.fetch_buffer(EXCH, SYM, n=2200)
    T = len(raw)
    print(f"\nbuffer rows: {T}")
    assert T > SG.HIST_NEEDED + REF_OFFSET + 10
    lo = SG.HIST_NEEDED + 5
    hi = T - REF_OFFSET - 2
    dec_indices = np.linspace(lo, hi, 10).astype(int)

    mismatches = 0
    checked = 0
    for d in dec_indices:
        live = SG.generate_from_buffer(raw[: d + 1 + SG.LAG], EXCH, SYM, decision_offset=SG.LAG)
        ref  = SG.generate_from_buffer(raw[: d + 1 + REF_OFFSET], EXCH, SYM, decision_offset=REF_OFFSET)
        if live.get("reason") or ref.get("reason"):
            continue
        checked += 1
        # both paths must have targeted the same decision bucket (same raw mid)
        assert abs(live["mid"] - ref["mid"]) < 1e-9, "live/ref decided on different buckets"
        same_sig = live["signal"] == ref["signal"]
        same_heads = (live["s0"], live["s1"], live["s2"]) == (ref["s0"], ref["s1"], ref["s2"])
        dvol = abs(live["vol"] - ref["vol"]); dtr = abs(live["trend_bp"] - ref["trend_bp"]); der = abs(live["er"] - ref["er"])
        ok = same_sig and same_heads and dvol < 1e-6 and dtr < 1e-6 and der < 1e-6
        if not ok:
            mismatches += 1
            print(f"  MISMATCH d={d}: live sig={live['signal']} heads={live['s0'],live['s1'],live['s2']} | "
                  f"ref sig={ref['signal']} heads={ref['s0'],ref['s1'],ref['s2']} | dvol={dvol:.1e} dtr={dtr:.1e} der={der:.1e}")
        else:
            print(f"  ok d={d}: sig={live['signal']} heads={live['s0'],live['s1'],live['s2']} vol={live['vol']:.4f} er={live['er']:.3f}")
    print(f"checked {checked}, mismatches {mismatches}")
    assert checked >= 6
    assert mismatches == 0, f"{mismatches} signal-parity mismatches (live vs full-future)"


def test_gap_guard():
    import datetime as dt
    from executor.paper import signal_generator as SG
    _, raw = DBS.fetch_buffer(EXCH, SYM, n=2000)
    T = len(raw); didx = T - 1 - SG.LAG
    base = dt.datetime(2026, 6, 15, tzinfo=dt.timezone.utc)
    good = [base + dt.timedelta(seconds=5 * i) for i in range(T)]            # contiguous 5s
    r_ok = SG.generate_from_buffer(raw, EXCH, SYM, decision_offset=SG.LAG, buckets=good)
    assert r_ok.get("reason") != "warming_up_or_gap", "contiguous buffer wrongly rejected"
    bad = list(good)                                                          # 5-day jump inside the context
    for i in range(didx - 10, T):
        bad[i] = bad[i] + dt.timedelta(days=5)
    r_gap = SG.generate_from_buffer(raw, EXCH, SYM, decision_offset=SG.LAG, buckets=bad)
    assert r_gap.get("reason") == "warming_up_or_gap", "gap in context not rejected"
    print("\ngap guard: contiguous OK, 5-day gap rejected")
