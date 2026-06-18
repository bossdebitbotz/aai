"""Unit tests for the rebuilt (thin-dispatcher) signal_service + the dry-run invariant.

Decision parity (service_decide == paper) lives in test_signal_parity_service.py.
Run: cd /Volumes/Docker-SSD/projects/aaiwdbback/aai && .venv/bin/python -m pytest executor/test_signal_service.py -q
"""
import os, sys, json
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from executor import signal_service as SVC

_HERE = os.path.dirname(os.path.abspath(__file__))


def test_ft_pair_mapping():
    assert SVC.ft_pair("BTC-USDT") == "BTC/USDT:USDT"
    assert SVC.ft_pair("ETH-USDT") == "ETH/USDT:USDT"


def test_position_to_orders_roundtrip():
    f = SVC.position_to_orders
    assert f(0, 0) == []
    assert f(0, 1) == [{"action": "entry", "side": "long"}]
    assert f(0, -2) == [{"action": "entry", "side": "short"}]
    assert f(2, 0) == [{"action": "exit"}]
    assert f(1, 3) == []                                   # same sign -> no freqtrade change
    assert f(-1, 2) == [{"action": "exit"}, {"action": "entry", "side": "long"}]


def test_config_is_dryrun():
    """LIVE-GATE invariant: freqtrade config stays dry_run until the gate is intentionally flipped."""
    cfg = json.load(open(os.path.join(_HERE, "config.json")))
    assert cfg["dry_run"] is True, "config.json dry_run must be True until the explicit live gate"
