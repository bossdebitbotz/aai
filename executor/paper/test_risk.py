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
    # cost: 2 units * (half_spread_bp + 3bp); half_spread = (0.02/2)/101*1e4 ≈ 0.990 bp
    assert cost > 0
    # flatten on an already-flat book is a no-op (no cost)
    d2, c2 = inv.flatten(101.0, 0.02, 3.0)
    assert d2 == 0.0 and c2 == 0.0
