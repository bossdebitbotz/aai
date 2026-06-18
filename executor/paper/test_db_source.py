"""Verify the live DB data source matches the training parquet bit-for-bit
(same lob_5s source, same column order). Memory-light: reads only the parquet's
last row group instead of the full ~1.9 GB stream (this box has 8 GB).

Run: .venv/bin/python -m pytest executor/paper/test_db_source.py -q -s
"""
import sys
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from executor.paper import db_source as DBS

STREAM = ("binance_perp", "BTC-USDT")
PARQUET = "/Volumes/Docker-SSD/projects/aaiwdbback/aai/lob_data/binance_perp_BTC-USDT.parquet"


def test_db_matches_parquet():
    # parquet tail (last row group only -> light)
    pf = pq.ParquetFile(PARQUET)
    pdf = pf.read_row_group(pf.num_row_groups - 1).to_pandas().tail(150).reset_index(drop=True)
    pq_buckets = pd.to_datetime(pdf["bucket"], utc=True)
    feat_cols = [c for c in pdf.columns if c != "bucket"]
    pq_feat = pdf[feat_cols].to_numpy(np.float64)

    # DB buffer ending at the parquet's last bucket
    end_t = pq_buckets.iloc[-1].to_pydatetime()
    db_buckets, db_raw = DBS.fetch_buffer(STREAM[0], STREAM[1], n=150, end_time=end_t)
    db_bkt = pd.to_datetime(pd.Series(db_buckets), utc=True)

    # align on common buckets and compare feature matrices
    common = sorted(set(pq_buckets) & set(db_bkt))
    assert len(common) >= 50, f"too few overlapping buckets: {len(common)}"
    pq_idx = {b: i for i, b in enumerate(pq_buckets)}
    db_idx = {b: i for i, b in enumerate(db_bkt)}
    diffs = np.array([np.abs(pq_feat[pq_idx[b]] - db_raw[db_idx[b]]).max() for b in common])
    maxdiff = float(diffs.max())
    print(f"\noverlapping buckets compared: {len(common)} | max |Δ| feature = {maxdiff:.3e}")
    assert maxdiff < 1e-6, f"DB vs parquet feature mismatch: {maxdiff}"
    # also confirm column count / layout
    assert db_raw.shape[1] == len(feat_cols) == 162
