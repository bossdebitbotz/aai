# executor/paper/test_sizing.py
import sys; sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
from executor.paper.strategy import Inventory
from executor.paper import sizing as Z


def _ctx_highvol():
    return np.cumsum(np.tile([0.01, -0.01], 60))


def test_vol_target_cap_basic():
    assert Z.vol_target_cap(0.0) == 0
    assert Z.vol_target_cap(0.01, target_vol=0.03) == 3
    assert Z.vol_target_cap(0.01, target_vol=0.006) == 1
    assert Z.vol_target_cap(0.1, target_vol=0.006) == 0
    caps = [Z.vol_target_cap(s, target_vol=0.03) for s in [0.005, 0.01, 0.02, 0.05]]
    assert caps == sorted(caps, reverse=True)


def _run(signals, target_vol, floor=1e9, cooldown_n=0, n_warm=80):
    inv, tr = Inventory(), Z.PositionTracker()
    ctx = _ctx_highvol()
    incs = 0.001 + 0.0005 * np.sin(np.arange(n_warm))
    mids = list(100.0 * np.cumprod(1 + incs))
    params = dict(target_vol=target_vol, stop_floor_bp=floor, cooldown_n=cooldown_n)
    for j, m in enumerate(mids):
        Z.step_with_sizing(1, 1, 1, ctx, np.array(mids[:j + 1]), m, 0.02, 3.0, inv, tr, params)
    out_pos = []; series = list(mids)
    for (a, b, c) in signals:
        nm = series[-1] * 1.0005; series.append(nm)
        o = Z.step_with_sizing(a, b, c, ctx, np.array(series), nm, 0.02, 3.0, inv, tr, params)
        out_pos.append(o["position"])
    return inv, tr, out_pos


def test_accumulates_to_dynamic_cap_then_capped():
    inv, tr, _ = _run([], target_vol=0.03)
    assert inv.position == 3.0
    inv2, tr2, _ = _run([], target_vol=0.01)
    assert inv2.position == 1.0


def test_signal_decay_unwinds_to_flat():
    inv, tr, pos = _run([(0, 0, 0)] * 4, target_vol=0.03)
    assert pos == [2.0, 1.0, 0.0, 0.0]


def test_opposite_signal_reduces():
    inv, tr, pos = _run([(-1, -1, -1)], target_vol=0.03)
    assert pos[0] == 2.0


def test_catastrophic_floor_flattens_and_cools():
    inv, tr = Inventory(), Z.PositionTracker()
    ctx = _ctx_highvol()
    incs = 0.001 + 0.0005 * np.sin(np.arange(80))
    mids = list(100.0 * np.cumprod(1 + incs))
    params = dict(target_vol=0.03, stop_floor_bp=150.0, cooldown_n=2)
    for j, m in enumerate(mids):
        Z.step_with_sizing(1, 1, 1, ctx, np.array(mids[:j + 1]), m, 0.02, 3.0, inv, tr, params)
    assert inv.position == 3.0
    tr.entry_net = inv.net_pnl_bp
    crash = mids + [mids[-1] * 0.97]
    out = Z.step_with_sizing(1, 1, 1, ctx, np.array(crash), crash[-1], 0.02, 3.0, inv, tr, params)
    assert out["action"] == "floor"
    assert inv.position == 0.0 and tr.in_cooldown


def test_net_pnl_finite_and_consistent():
    inv, tr, _ = _run([(0, 0, 0)] * 3, target_vol=0.03)
    assert np.isfinite(inv.net_pnl_bp)
    assert abs(inv.net_pnl_bp - (inv.marked_pnl_bp - inv.realized_cost_bp)) < 1e-9
