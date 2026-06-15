# executor/paper/test_risk.py
import sys
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from executor.paper.strategy import Inventory


def test_inventory_flatten_marks_then_closes():
    inv = Inventory(cap=3)
    # open +2 at mid 100 (two gated longs)
    inv.on_decision(1, True, 100.0, 0.02, 3.0)
    inv.on_decision(1, True, 100.0, 0.02, 3.0)
    assert inv.position == 2.0
    # price rises to 101, then flatten
    delta, cost = inv.flatten(101.0, 0.02, 3.0)
    assert inv.position == 0.0
    assert delta == -2.0
    # marked PnL: 2 units * (101-100)/100 * 1e4 = +200 bp (marking happened in flatten)
    assert abs(inv.marked_pnl_bp - 200.0) < 1e-6
    # exact taker cost of the flatten: 2 units * (half_spread_bp + 3bp), half_spread=(0.02/2)/101*1e4
    expected_flatten_cost = 2 * ((0.02 / 2) / 101 * 1e4 + 3.0)
    assert abs(cost - expected_flatten_cost) < 1e-6
    # realized cost accumulates the two opening fills (4 bp each at mid 100) + the flatten
    assert abs(inv.realized_cost_bp - (8.0 + expected_flatten_cost)) < 1e-6
    # net = marked - realized cost
    assert abs(inv.net_pnl_bp - (200.0 - (8.0 + expected_flatten_cost))) < 1e-6
    # flatten on an already-flat book is a no-op (no cost)
    d2, c2 = inv.flatten(101.0, 0.02, 3.0)
    assert d2 == 0.0 and c2 == 0.0


def test_inventory_flatten_short():
    inv = Inventory(cap=3)
    inv.on_decision(-1, True, 100.0, 0.02, 3.0)   # short 1 at 100
    inv.on_decision(-1, True, 100.0, 0.02, 3.0)   # short 2 at 100
    assert inv.position == -2.0
    # price falls to 99 (favorable for a short), then flatten
    delta, cost = inv.flatten(99.0, 0.02, 3.0)
    assert inv.position == 0.0
    assert delta == 2.0                            # closing a short buys back +2 units
    # marked: (-2) * (99-100)/100 * 1e4 = +200 bp
    assert abs(inv.marked_pnl_bp - 200.0) < 1e-6
    expected_cost = 2 * ((0.02 / 2) / 99 * 1e4 + 3.0)
    assert abs(cost - expected_cost) < 1e-6

import numpy as np
from executor.paper import risk as R


def test_sigma_returns_needs_history():
    assert R.sigma_returns(np.array([100.0, 101.0]), L=12) == 0.0   # too short -> 0
    mids = 100.0 * np.cumprod(1 + np.full(20, 0.0))                 # flat -> zero vol
    assert R.sigma_returns(mids, L=12) == 0.0


def test_hwm_ratchets_favorable_only_long():
    st = R.TrailingStop(k=2.0, L=12)
    st.update(position=1.0, mid=100.0)     # opens long -> hwm=100
    assert st.side == 1 and st.hwm == 100.0
    st.update(position=1.0, mid=102.0)     # rises -> hwm=102
    assert st.hwm == 102.0
    st.update(position=1.0, mid=101.0)     # dips -> hwm stays 102 (ratchet up only)
    assert st.hwm == 102.0


def test_hwm_ratchets_favorable_only_short():
    st = R.TrailingStop(k=2.0, L=12)
    st.update(position=-1.0, mid=100.0)
    assert st.side == -1 and st.hwm == 100.0
    st.update(position=-1.0, mid=98.0)     # favorable for short -> hwm=98
    assert st.hwm == 98.0
    st.update(position=-1.0, mid=99.0)     # adverse -> hwm stays 98
    assert st.hwm == 98.0


def test_flat_resets():
    st = R.TrailingStop()
    st.update(1.0, 100.0)
    st.update(0.0, 100.0)
    assert st.side == 0 and st.hwm is None


def test_breach_long_fires_below_trailed_level():
    # build a noisy-but-trending mid series so sigma > 0
    rng = np.linspace(100.0, 110.0, 30) + np.sin(np.arange(30))
    st = R.TrailingStop(k=2.0, L=12, d_min=0.0005)
    st.update(1.0, rng[-1])                # long, hwm = last (a peak)
    sig = R.sigma_returns(rng, 12)
    assert sig > 0
    lvl = st.stop_level(rng)
    assert lvl is not None and lvl < st.hwm           # stop sits below the peak for a long
    assert st.breached(lvl - 1e-9, rng) is True       # just below the level -> breach
    assert st.breached(st.hwm, rng) is False          # at the peak -> no breach


def test_stop_not_armed_without_vol():
    st = R.TrailingStop()
    st.update(1.0, 100.0)
    flat = np.full(20, 100.0)
    assert st.stop_level(flat) is None                # zero vol -> unarmed
    assert st.breached(50.0, flat) is False
