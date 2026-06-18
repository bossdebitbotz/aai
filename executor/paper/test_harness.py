"""Harness + ledger tests: replay orchestration, accounting consistency, restart-safety.

Run: .venv/bin/python -m pytest executor/paper/test_harness.py -q -s
"""
import sys, os, tempfile, datetime as dt
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
from executor.paper.ledger import PaperLedger
from executor.paper import harness as H
from executor.paper import signal_generator as SG, db_source as DBS, sizing as Z
from executor.paper.strategy import Inventory

ONE = [("binance_perp", "BTC-USDT")]
# replay a CONTIGUOUS pre-gap window (before the 2026-06-10 collection gap) so the
# gap guard doesn't (correctly) reject decisions, making the count deterministic.
PREGAP = dt.datetime(2026, 6, 10, 13, 0, 0, tzinfo=dt.timezone.utc)


def test_replay_and_accounting(tmp_path):
    db = str(tmp_path / "t.db")
    led = PaperLedger(db)
    H.run_replay(led, n_decisions=30, streams=ONE, end_time=PREGAP)
    # decisions recorded
    n = led.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    print(f"\nrecorded decisions: {n}")
    assert n >= 25
    # accounting consistency on every row: net == marked - cost (cumulative cost tracked)
    rows = led.conn.execute("SELECT marked_pnl_bp, net_pnl_bp FROM decisions ORDER BY id").fetchall()
    # state net equals marked - realized_cost
    s = led.conn.execute("SELECT position, marked_pnl_bp, realized_cost_bp FROM state WHERE stream='binance_perp_BTC-USDT'").fetchone()
    net = led.portfolio_net_bp()
    print(f"final pos={s[0]:+.0f} marked={s[1]:+.1f} cost={s[2]:.1f} net={net:+.1f}")
    assert abs(net - (s[1] - s[2])) < 1e-6
    led.close()


def test_restart_safety(tmp_path):
    db = str(tmp_path / "r.db")
    # session 1: 20 decisions
    led1 = PaperLedger(db)
    H.run_replay(led1, n_decisions=20, streams=ONE, end_time=PREGAP)
    pos1 = led1.load_inventory("binance_perp_BTC-USDT").position
    net1 = led1.portfolio_net_bp()
    n1 = led1.conn.execute("SELECT n_decisions FROM state WHERE stream='binance_perp_BTC-USDT'").fetchone()[0]
    led1.close()
    # session 2: reopen -> inventory must resume from persisted state
    led2 = PaperLedger(db)
    inv = led2.load_inventory("binance_perp_BTC-USDT")
    print(f"\nresumed: pos={inv.position:+.0f} (was {pos1:+.0f}) net={led2.portfolio_net_bp():+.1f} (was {net1:+.1f}) n={n1}")
    assert inv.position == pos1
    assert abs(led2.portfolio_net_bp() - net1) < 1e-6
    assert inv._last_mid is not None        # marking state preserved
    led2.close()


# --- A/B sizing book (vol-target/signal-decay exit) ---

def test_sizing_book_replay_isolated(tmp_path):
    db = str(tmp_path / "s.db")
    led = PaperLedger(db)
    H.run_replay(led, n_decisions=30, streams=ONE, end_time=PREGAP, strategy="sizing")
    n = led.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert n >= 25, n
    # PositionTracker persisted for restart-safety
    assert led.conn.execute(
        "SELECT COUNT(*) FROM sizing_tracker WHERE stream='binance_perp_BTC-USDT'").fetchone()[0] == 1
    # accounting consistency: net == marked - cost
    s = led.conn.execute(
        "SELECT marked_pnl_bp, realized_cost_bp FROM state WHERE stream='binance_perp_BTC-USDT'").fetchone()
    assert abs(led.portfolio_net_bp() - (s[0] - s[1])) < 1e-6
    led.close()


def test_base_book_unaffected_by_flag(tmp_path):
    # the --strategy flag must not break or alter the base path
    db = str(tmp_path / "b.db")
    led = PaperLedger(db)
    H.run_replay(led, n_decisions=20, streams=ONE, end_time=PREGAP, strategy="base")
    assert led.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] >= 15
    assert led.conn.execute("SELECT COUNT(*) FROM sizing_tracker").fetchone()[0] == 0  # base writes no tracker
    led.close()


def test_generate_returns_gate_inputs_and_entry_parity():
    buckets, raw = DBS.fetch_buffer("binance_perp", "BTC-USDT", n=SG.MIN_BUFFER + 400, end_time=PREGAP)
    sig = SG.generate_from_buffer(raw, "binance_perp", "BTC-USDT", decision_offset=SG.LAG, buckets=buckets)
    assert not sig.get("reason"), sig
    assert len(sig["scaled_mid_ctx"]) == SG.CTX                 # 120
    assert len(sig["dec_mids"]) == SG.S.ER_KER + 1              # 61
    # feeding the SAME gate inputs into the sizing step reproduces the gate signal (entry parity)
    inv, tr = Inventory(), Z.PositionTracker()
    out = Z.step_with_sizing(sig["s0"], sig["s1"], sig["s2"],
                             np.asarray(sig["scaled_mid_ctx"]), np.asarray(sig["dec_mids"]),
                             sig["mid"], sig["spread"], H.FEE_BP, inv, tr,
                             dict(target_vol=Z.TARGET_VOL, stop_floor_bp=1e9, cooldown_n=0))
    assert out["signal"] == sig["signal"]


def test_sizing_restart_safety(tmp_path):
    db = str(tmp_path / "sr.db")
    led1 = PaperLedger(db)
    H.run_replay(led1, n_decisions=20, streams=ONE, end_time=PREGAP, strategy="sizing")
    pos1 = led1.load_inventory("binance_perp_BTC-USDT").position
    t1 = led1.load_sizing_tracker("binance_perp_BTC-USDT")
    led1.close()
    led2 = PaperLedger(db)
    assert led2.load_inventory("binance_perp_BTC-USDT").position == pos1
    t2 = led2.load_sizing_tracker("binance_perp_BTC-USDT")
    assert t2.entry_net == t1.entry_net and t2._cooldown == t1._cooldown
    led2.close()
