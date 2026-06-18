# Paper-Trade Harness — Operator Guide

Live paper trader for the FROZEN inventory-reversion strategy (see
`training data/FROZEN_strategy_spec_2026-06-14.md` on Drive). Built + rigorously
tested 2026-06-14/15. **This is an out-of-sample paper test, not a validated edge** —
the strategy failed robust walk-forward; the paper run is the honest live test.

## What's running (background processes)
- **`lob_collector.py`** — ingests 16 LOB streams (4 exch × 4 pairs) from public WS into
  TimescaleDB `lob_snapshots`; `lob_5s` aggregate auto-refreshes every 5s. Run with `.venv` python.
- **`harness.py --mode live`** — every ~2 min, per binance_perp stream: trailing DB buffer →
  features (30s lag = bit-exact to training) → saved scalers → model → 3-horizon agreement →
  vol gate → trend-veto → trending-regime gate → inventory (cap 3, taker) → ledger.
  Logs heartbeats to `paper_run.log`; state in `paper_ledger.db` (restart-safe).

## WARMUP (~2 h after collector (re)start)
Collection had a gap (stopped 2026-06-10, restarted 2026-06-15). The gap guard refuses to
trade until the trailing buffer is contiguous fresh data — the model context (~10 min) clears
fast, but the trend-efficiency lookback needs ~1.9 h of continuous fresh buckets. Until then
`paper_run.log` shows `warmin`; after, it shows real `sig=±1/pos=...` decisions.

## Monitor
    tail -f executor/paper/paper_run.log                  # live heartbeats
    .venv/bin/python -c "from executor.paper.ledger import PaperLedger as L; \
        led=L(); [print(r) for r in led.summary()]; print('NET', led.portfolio_net_bp(),'bp-units')"
    # raw decisions:
    sqlite3 executor/paper/paper_ledger.db "SELECT ts,stream,signal,position,net_pnl_bp FROM decisions ORDER BY id DESC LIMIT 20;"

## Verified (all tests green: `.venv/bin/python -m pytest executor/paper/ -q` → 14 passed)
- Feature parity: live(30s-lag) features BIT-EXACT to training (Δ≈1e-14).
- DB source: bit-exact to training parquet.
- SIGNAL parity: live signal/heads/gates == full-future (training) condition, 10/10 buckets.
- Gap guard: rejects restart-gap / large outages.
- Inventory + ledger + restart-safety: accounting consistent, state survives restart.

## Stop / restart
    pkill -f "harness.py --mode live"      # stop paper trader
    pkill -f "lob_collector.py"            # stop collector (NB: stopping creates a gap -> ~2h re-warmup)
    # restart paper trader (resumes inventory from ledger):
    cd /Volumes/Docker-SSD/projects/aaiwdbback/aai && nohup .venv/bin/python executor/paper/harness.py --mode live > executor/paper/paper_run.log 2>&1 &

## GO / NO-GO (after ≥3 weeks spanning trending+choppy regimes)
GO toward small live capital iff: net-positive after real fees, positive in a MAJORITY of weekly
sub-periods, PnL concentrates in trending (high-ER) regimes as expected, drawdown tolerable.
NO-GO if net-negative or positive in only one window. Rotate the leaked Binance keys before ANY live capital.

## Known residuals / caveats
- Winsorization (train 0.1/99.9 pct clip) is skipped live (bounds not persisted) — affects only rare
  extremes; z-score scaling is exact. Refine by recomputing+persisting bounds if desired.
- Strategy trades binance_perp ×4 (collector also feeds spot/bybit/kucoin; perp is the relevant market).
- No collector watchdog yet: if `lob_collector.py` dies, the paper trader will sit in warmup. Check both PIDs.
