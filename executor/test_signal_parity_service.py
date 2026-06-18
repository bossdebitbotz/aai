"""DECISIVE parity gate: the freqtrade signal bridge (signal_service.service_decide)
must produce EXACTLY the same decision as the parity-proven paper path
(signal_generator.generate_from_buffer, which is bit-exact to training). Because the
service now DELEGATES to that function, this asserts total delegation (no drift) and
guards against re-introducing the old bespoke logic.

Run: cd /Volumes/Docker-SSD/projects/aaiwdbback/aai && .venv/bin/python -m pytest executor/test_signal_parity_service.py -q -s
"""
import sys, datetime as dt
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
from executor.paper import signal_generator as SG, db_source as DBS
from executor import signal_service as SVC

EXCH, SYM = "binance_perp", "BTC-USDT"
PREGAP = dt.datetime(2026, 6, 10, 13, 0, 0, tzinfo=dt.timezone.utc)   # contiguous pre-gap window


def test_service_decide_matches_paper():
    """service_decide == generate_from_buffer (signal + heads + gates) at >=6 buckets."""
    buckets, raw = DBS.fetch_buffer(EXCH, SYM, n=SG.MIN_BUFFER + 600, end_time=PREGAP)
    T = len(raw)
    lo = SG.HIST_NEEDED + 5
    hi = T - SG.LAG - 2
    assert hi > lo, (lo, hi, T)
    idxs = np.linspace(lo, hi, 8).astype(int)
    checked = 0
    for d in idxs:
        sub_raw = raw[: d + 1 + SG.LAG]
        sub_b = buckets[: d + 1 + SG.LAG]
        paper = SG.generate_from_buffer(sub_raw, EXCH, SYM, decision_offset=SG.LAG, buckets=sub_b)
        svc = SVC.service_decide(sub_raw, EXCH, SYM, buckets=sub_b)
        if paper.get("reason") or svc.get("reason"):
            assert paper.get("reason") == svc.get("reason")
            continue
        checked += 1
        assert svc["signal"] == paper["signal"]
        assert (svc["s0"], svc["s1"], svc["s2"]) == (paper["s0"], paper["s1"], paper["s2"])
        assert abs(svc["vol"] - paper["vol"]) < 1e-12
        assert abs(svc["er"] - paper["er"]) < 1e-12
    assert checked >= 6, f"only checked {checked}"


def test_service_has_no_bespoke_decision_logic():
    """Guard against regression to the divergent path: the old re-implementation
    (rolling z-score / per-horizon thresholds / logits_to_decisions) must be gone."""
    for banned in ("SignalEngine", "TradeDecision", "logits_to_decisions", "run_inference", "prepare_features"):
        assert not hasattr(SVC, banned), f"{banned} still present — signal_service has bespoke logic"


def test_position_to_orders_mapping():
    f = SVC.position_to_orders
    assert f(0, 0) == []                                  # flat -> flat: noop
    assert f(0, 2) == [{"action": "entry", "side": "long"}]    # open long
    assert f(0, -1) == [{"action": "entry", "side": "short"}]  # open short
    assert f(2, 0) == [{"action": "exit"}]                # close
    assert f(3, 1) == []                                  # same sign (3->1): freqtrade unchanged
    flip = f(2, -1)                                        # long -> short: exit then entry
    assert flip == [{"action": "exit"}, {"action": "entry", "side": "short"}]
