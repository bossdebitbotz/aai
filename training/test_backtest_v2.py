# training/test_backtest_v2.py
import sys, logging, numpy as np
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
from training.backtest_v2 import simulate_window_trades

logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

def test_simulate_basic_metrics_and_cost_ordering():
    # 5 windows; entry/exit mids and predicted (class, conf)
    entry = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    exitp = np.array([101.0, 99.0, 100.0, 102.0, 98.0])  # +1%,-1%,0,+2%,-2%
    cls = np.array([2, 0, 2, 2, 0])      # up, down, up, up, down  (all "correct" directionally)
    conf = np.array([0.9, 0.9, 0.3, 0.95, 0.95])  # window 2 below gate
    ts = np.arange(5) * 1000.0           # far apart -> no cooldown blocking
    maker = simulate_window_trades(entry, exitp, cls, conf, ts, threshold=0.6,
                                    cost_bps_per_side=2.0, cooldown_s=300, hold_s=120)
    taker = simulate_window_trades(entry, exitp, cls, conf, ts, threshold=0.6,
                                   cost_bps_per_side=5.0, cooldown_s=300, hold_s=120)
    assert 0.0 <= maker["coverage"] <= 1.0
    assert maker["n_trades"] == 4               # window 2 gated out
    assert taker["net_return"] <= maker["net_return"] + 1e-12   # taker costs more
    assert np.isfinite(maker["sharpe_per_trade"])
    logger.info(f"PASS: test_simulate ... maker_net={maker['net_return']:.4f} taker_net={taker['net_return']:.4f}")
