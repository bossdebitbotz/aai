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
