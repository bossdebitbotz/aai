"""Unit tests for the FROZEN paper-trade strategy logic.

Run: .venv/bin/python -m pytest executor/paper/test_strategy.py -q
"""
import sys
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
from executor.paper import strategy as S


def test_head_sign():
    assert S.head_sign(np.array([5.0, 0.0, 1.0])) == -1   # argmax=down
    assert S.head_sign(np.array([0.0, 9.0, 1.0])) == 0    # argmax=flat
    assert S.head_sign(np.array([0.0, 1.0, 9.0])) == 1    # argmax=up


def test_agreement_required():
    assert S.base_signal(1, 1, 1) == 1
    assert S.base_signal(-1, -1, -1) == -1
    assert S.base_signal(1, 1, -1) == 0     # disagree -> no trade
    assert S.base_signal(0, 0, 0) == 0      # all flat -> no trade
    assert S.base_signal(1, 0, 1) == 0      # one flat -> no trade


def test_vol_gate_threshold():
    # construct a context whose scaled-mid first-diff std is just below / above threshold
    low = np.cumsum(np.full(121, 0.0))                    # zero vol
    assert S.context_vol(low) < S.VOL_THRESHOLD
    hi = np.cumsum(np.random.default_rng(0).normal(0, 0.01, 121))
    assert S.context_vol(hi) > S.VOL_THRESHOLD


def test_trend_veto_blocks_fading_strong_moves():
    # strong downtrend over last K: returns -100bp -> a LONG signal should be vetoed
    mids = np.linspace(100.0, 99.0, S.TREND_K + 5)        # ~ -1% = -100bp downtrend
    assert S.trend_veto(+1, mids) is True                 # don't catch the falling knife
    assert S.trend_veto(-1, mids) is False                # shorting with the trend is fine
    flat = np.full(S.TREND_K + 5, 100.0)
    assert S.trend_veto(+1, flat) is False                # no strong move -> no veto


def test_efficiency_ratio_trend_vs_chop():
    trend = np.linspace(100, 110, S.ER_KER + 5)           # monotonic -> ER ~ 1
    assert S.efficiency_ratio(trend) > 0.9
    rng = np.random.default_rng(1)
    chop = 100 + np.cumsum(rng.normal(0, 1, S.ER_KER + 5))  # random walk -> low ER
    assert S.efficiency_ratio(chop) < 0.5


def test_decide_signal_full_stack():
    # all gates pass: agreement long, high vol, mild uptrend (no veto), trending regime
    ctx = np.cumsum(np.random.default_rng(2).normal(0, 0.01, 121))  # high vol
    mids = np.linspace(100, 105, S.ER_KER + 5)            # trending up, ER high, +500bp trend
    # long signal with uptrend: trend_veto only blocks longs in DOWNtrends -> ok
    assert S.decide_signal(1, 1, 1, ctx, mids) == 1
    # disagreement kills it regardless
    assert S.decide_signal(1, 1, -1, ctx, mids) == 0
    # choppy regime (low ER) kills it
    chop = 100 + np.cumsum(np.random.default_rng(3).normal(0, 0.5, S.ER_KER + 5))
    assert S.decide_signal(1, 1, 1, ctx, chop) == 0


def test_inventory_accumulates_to_cap_and_costs():
    inv = S.Inventory(cap=3)
    # three consecutive gated long signals -> accumulate to +3, no further
    for _ in range(5):
        inv.on_decision(signal=1, gated=True, mid=100.0, spread=0.1, fee_bp=3.0)
    assert inv.position == 3.0
    assert inv.realized_cost_bp > 0


def test_inventory_marks_pnl_then_trades():
    inv = S.Inventory(cap=3)
    inv.on_decision(1, True, 100.0, 0.0, 0.0)             # enter +1 at mid 100 (no spread/fee)
    inv.on_decision(0, False, 101.0, 0.0, 0.0)            # hold; mark +1 over +1% move = +100bp
    assert abs(inv.marked_pnl_bp - 100.0) < 1e-6
    assert inv.position == 1.0                            # held (ungated)


def test_inventory_ungated_holds_not_flatten():
    inv = S.Inventory(cap=3)
    inv.on_decision(1, True, 100.0, 0.0, 0.0)
    inv.on_decision(0, False, 100.0, 0.0, 0.0)           # ungated -> hold, do not flatten
    assert inv.position == 1.0
