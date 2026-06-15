# training/test_walkforward.py
import sys, datetime as dt
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
import pandas as pd
from training.data_source import fetch_parquet_stream


def test_parquet_time_bounds(tmp_path):
    # tiny synthetic parquet with the columns _build_feature_columns(1) expects
    from training.dataset import _build_feature_columns
    cols = _build_feature_columns(1)              # 1 level -> bid/ask price+vol + mid + spread
    n = 100
    base = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    df = pd.DataFrame({c: np.arange(n, dtype=float) + 1.0 for c in cols})
    df["bucket"] = [base + dt.timedelta(seconds=5 * i) for i in range(n)]
    p = tmp_path / "binance_perp_BTC-USDT.parquet"
    df.to_parquet(p)

    # unbounded -> all rows
    feats, ts = fetch_parquet_stream(str(tmp_path), "binance_perp", "BTC-USDT", 1)
    assert len(feats) == n
    # bounded [t10, t40) -> rows 10..39
    start = base + dt.timedelta(seconds=5 * 10)
    end = base + dt.timedelta(seconds=5 * 40)
    feats_b, ts_b = fetch_parquet_stream(str(tmp_path), "binance_perp", "BTC-USDT", 1,
                                         start_time=start, end_time=end)
    assert len(feats_b) == 30
    assert ts_b[0] == start.timestamp()
    assert ts_b[-1] == (end - dt.timedelta(seconds=5)).timestamp()


from training import walkforward as WF


def test_fold_ranges_rolling_no_leakage():
    import datetime as dt
    t0 = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
    t1 = t0 + dt.timedelta(days=88)
    folds = WF.make_folds(t0, t1, train_days=45, test_days=8, tune_days=7)
    assert len(folds) == 5
    prev_test_start = None
    for f in folds:
        # tune ⊂ train, test strictly after train, no overlap
        assert f.train_start < f.tune_start < f.train_end == f.test_start
        assert f.tune_end == f.train_end
        assert f.test_end > f.test_start
        assert (f.train_end - f.train_start) == dt.timedelta(days=45)   # constant width
        # test windows step forward and never overlap training
        assert f.test_start >= f.train_end
        if prev_test_start is not None:
            assert f.test_start > prev_test_start
        prev_test_start = f.test_start


def test_fold_ranges_assert_disjoint():
    import datetime as dt
    t0 = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
    # too-short span yields zero folds rather than overlapping ones
    folds = WF.make_folds(t0, t0 + dt.timedelta(days=40), train_days=45, test_days=8, tune_days=7)
    assert folds == []


def test_simulate_runs_and_metrics_consistent():
    rng = np.random.default_rng(0)
    N = 300
    # clear up-drift (drift ~ noise) so the ER gate admits trades after warmup
    mids = 100.0 * np.cumprod(1 + rng.normal(0.0008, 0.0008, N))
    spreads = np.full(N, 0.02)
    vol = np.full(N, 0.01)                  # clears VOL_THRESHOLD
    signs = np.tile(np.array([1, 1, 1], dtype=np.int8), (N, 1))   # always-up heads
    params = dict(k=2.0, L=12, cooldown_n=2)
    net_series, n_trades = WF.simulate(signs, mids, spreads, vol, params, fee_bp=3.0)
    assert len(net_series) == N
    assert n_trades >= 1
    m = WF.metrics(net_series, n_trades)
    assert set(m) >= {"net_bp", "sharpe", "max_dd_bp", "n_trades", "n_decisions"}
    assert abs(m["net_bp"] - net_series[-1]) < 1e-9
    assert m["n_decisions"] == N


def test_simulate_flat_when_no_signal():
    N = 200
    mids = np.full(N, 100.0)
    spreads = np.full(N, 0.02)
    vol = np.zeros(N)                       # below VOL_THRESHOLD -> never trades
    signs = np.tile(np.array([1, 1, 1], dtype=np.int8), (N, 1))
    net_series, n_trades = WF.simulate(signs, mids, spreads, vol, dict(k=2.0, L=12, cooldown_n=2))
    assert n_trades == 0
    assert abs(net_series[-1]) < 1e-9
