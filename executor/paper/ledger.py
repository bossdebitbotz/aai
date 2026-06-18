#!/usr/bin/env python3
"""Persistent paper-trade ledger (SQLite): full per-decision log + restart-safe
per-stream inventory state. The harness records every decision here and reloads
inventory state on restart so a crash/reboot never loses position or PnL.
"""
from __future__ import annotations
import sqlite3, os
from executor.paper.strategy import Inventory

DEFAULT_DB = "/Volumes/Docker-SSD/projects/aaiwdbback/aai/executor/paper/paper_ledger.db"


class PaperLedger:
    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, stream TEXT, signal INTEGER,
            s0 INTEGER, s1 INTEGER, s2 INTEGER,
            vol REAL, trend_bp REAL, er REAL,
            position REAL, delta REAL, mid REAL, spread REAL,
            cost_bp REAL, marked_pnl_bp REAL, net_pnl_bp REAL
        );
        CREATE TABLE IF NOT EXISTS state (
            stream TEXT PRIMARY KEY,
            position REAL, last_mid REAL,
            marked_pnl_bp REAL, realized_cost_bp REAL,
            last_ts TEXT, n_decisions INTEGER
        );
        CREATE TABLE IF NOT EXISTS sizing_tracker (
            stream TEXT PRIMARY KEY, entry_net REAL, cooldown INTEGER
        );
        """)
        self.conn.commit()

    def load_sizing_tracker(self, stream: str):
        """Restore the vol-target/signal-decay PositionTracker (entry_net + cooldown)."""
        from executor.paper.sizing import PositionTracker
        row = self.conn.execute(
            "SELECT entry_net, cooldown FROM sizing_tracker WHERE stream=?", (stream,)).fetchone()
        tr = PositionTracker()
        if row:
            tr.entry_net = row[0]
            tr._cooldown = int(row[1] or 0)
        return tr

    def save_sizing_tracker(self, stream: str, tr):
        self.conn.execute(
            "INSERT INTO sizing_tracker (stream, entry_net, cooldown) VALUES (?,?,?)"
            " ON CONFLICT(stream) DO UPDATE SET entry_net=excluded.entry_net, cooldown=excluded.cooldown",
            (stream, tr.entry_net, tr._cooldown))
        self.conn.commit()

    def load_inventory(self, stream: str) -> Inventory:
        row = self.conn.execute(
            "SELECT position,last_mid,marked_pnl_bp,realized_cost_bp FROM state WHERE stream=?",
            (stream,)).fetchone()
        inv = Inventory()
        if row:
            inv.position, inv._last_mid, inv.marked_pnl_bp, inv.realized_cost_bp = (
                row[0], row[1], row[2], row[3])
        return inv

    def record(self, ts: str, stream: str, sig: dict, inv: Inventory, delta: float, cost: float):
        self.conn.execute(
            "INSERT INTO decisions (ts,stream,signal,s0,s1,s2,vol,trend_bp,er,position,delta,mid,spread,cost_bp,marked_pnl_bp,net_pnl_bp)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, stream, sig.get("signal", 0), sig.get("s0"), sig.get("s1"), sig.get("s2"),
             sig.get("vol"), sig.get("trend_bp"), sig.get("er"),
             inv.position, delta, sig.get("mid"), sig.get("spread"),
             cost, inv.marked_pnl_bp, inv.net_pnl_bp))
        self.conn.execute(
            "INSERT INTO state (stream,position,last_mid,marked_pnl_bp,realized_cost_bp,last_ts,n_decisions)"
            " VALUES (?,?,?,?,?,?,1) ON CONFLICT(stream) DO UPDATE SET"
            " position=excluded.position,last_mid=excluded.last_mid,marked_pnl_bp=excluded.marked_pnl_bp,"
            " realized_cost_bp=excluded.realized_cost_bp,last_ts=excluded.last_ts,n_decisions=state.n_decisions+1",
            (stream, inv.position, inv._last_mid, inv.marked_pnl_bp, inv.realized_cost_bp, ts))
        self.conn.commit()

    def summary(self):
        rows = self.conn.execute(
            "SELECT stream, position, marked_pnl_bp, realized_cost_bp, marked_pnl_bp-realized_cost_bp AS net, n_decisions"
            " FROM state ORDER BY stream").fetchall()
        return rows

    def portfolio_net_bp(self) -> float:
        r = self.conn.execute("SELECT COALESCE(SUM(marked_pnl_bp-realized_cost_bp),0) FROM state").fetchone()
        return float(r[0])

    def close(self):
        self.conn.close()
