# training/test_export.py
import sys, logging
sys.path.insert(0, "/Volumes/Docker-SSD/projects/aaiwdbback/aai")
import export_training_data as ex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_build_copy_query_filters_invalid_and_orders():
    cols = ex.build_columns()
    q = ex.build_copy_query("binance_perp", "BTC-USDT", cols)
    assert "FROM lob_5s" in q
    assert "exchange = 'binance_perp'" in q
    assert "symbol = 'BTC-USDT'" in q
    assert "all_valid" in q          # invalid buckets excluded
    assert "ORDER BY bucket" in q
    assert "all_valid" not in cols   # filter column is NOT exported as a feature
    logger.info("PASS: test_build_copy_query_filters_invalid_and_orders")

def test_clear_stale_exports(tmp_path):
    f = tmp_path / "binance_perp_BTC-USDT.parquet"
    f.write_bytes(b"x" * 2000)
    z = tmp_path / "lob_training_data.zip"
    z.write_bytes(b"x" * 2000)
    removed = ex.clear_stale_exports(tmp_path, zip_path=z)
    assert not f.exists() and not z.exists()
    assert removed == 2
    logger.info("PASS: test_clear_stale_exports")

def test_strip_psql_tags_handles_multiple_sets():
    body = "bucket,bid_price_1,spread\n1.0,2.0,0.1\n3.0,4.0,0.2\n"
    # two SET tags (work_mem + max_parallel_workers_per_gather) precede the CSV
    assert ex.strip_psql_tags("SET\nSET\n" + body) == body
    assert ex.strip_psql_tags("SET\n" + body) == body      # single SET
    assert ex.strip_psql_tags(body) == body                # no prefix
    logger.info("PASS: test_strip_psql_tags_handles_multiple_sets")
