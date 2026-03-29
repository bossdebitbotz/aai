-- LOB Data Schema for TimescaleDB
-- This script is automatically run on first container start via docker-entrypoint-initdb.d

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- Primary table: lob_snapshots (columnar, queryable, 40 levels)
-- ============================================================
CREATE TABLE lob_snapshots (
    time             TIMESTAMPTZ      NOT NULL,
    exchange         TEXT             NOT NULL,
    symbol           TEXT             NOT NULL,
    -- Bid levels (price, volume) x 40
    bid_price_1 DOUBLE PRECISION, bid_volume_1 DOUBLE PRECISION,
    bid_price_2 DOUBLE PRECISION, bid_volume_2 DOUBLE PRECISION,
    bid_price_3 DOUBLE PRECISION, bid_volume_3 DOUBLE PRECISION,
    bid_price_4 DOUBLE PRECISION, bid_volume_4 DOUBLE PRECISION,
    bid_price_5 DOUBLE PRECISION, bid_volume_5 DOUBLE PRECISION,
    bid_price_6 DOUBLE PRECISION, bid_volume_6 DOUBLE PRECISION,
    bid_price_7 DOUBLE PRECISION, bid_volume_7 DOUBLE PRECISION,
    bid_price_8 DOUBLE PRECISION, bid_volume_8 DOUBLE PRECISION,
    bid_price_9 DOUBLE PRECISION, bid_volume_9 DOUBLE PRECISION,
    bid_price_10 DOUBLE PRECISION, bid_volume_10 DOUBLE PRECISION,
    bid_price_11 DOUBLE PRECISION, bid_volume_11 DOUBLE PRECISION,
    bid_price_12 DOUBLE PRECISION, bid_volume_12 DOUBLE PRECISION,
    bid_price_13 DOUBLE PRECISION, bid_volume_13 DOUBLE PRECISION,
    bid_price_14 DOUBLE PRECISION, bid_volume_14 DOUBLE PRECISION,
    bid_price_15 DOUBLE PRECISION, bid_volume_15 DOUBLE PRECISION,
    bid_price_16 DOUBLE PRECISION, bid_volume_16 DOUBLE PRECISION,
    bid_price_17 DOUBLE PRECISION, bid_volume_17 DOUBLE PRECISION,
    bid_price_18 DOUBLE PRECISION, bid_volume_18 DOUBLE PRECISION,
    bid_price_19 DOUBLE PRECISION, bid_volume_19 DOUBLE PRECISION,
    bid_price_20 DOUBLE PRECISION, bid_volume_20 DOUBLE PRECISION,
    bid_price_21 DOUBLE PRECISION, bid_volume_21 DOUBLE PRECISION,
    bid_price_22 DOUBLE PRECISION, bid_volume_22 DOUBLE PRECISION,
    bid_price_23 DOUBLE PRECISION, bid_volume_23 DOUBLE PRECISION,
    bid_price_24 DOUBLE PRECISION, bid_volume_24 DOUBLE PRECISION,
    bid_price_25 DOUBLE PRECISION, bid_volume_25 DOUBLE PRECISION,
    bid_price_26 DOUBLE PRECISION, bid_volume_26 DOUBLE PRECISION,
    bid_price_27 DOUBLE PRECISION, bid_volume_27 DOUBLE PRECISION,
    bid_price_28 DOUBLE PRECISION, bid_volume_28 DOUBLE PRECISION,
    bid_price_29 DOUBLE PRECISION, bid_volume_29 DOUBLE PRECISION,
    bid_price_30 DOUBLE PRECISION, bid_volume_30 DOUBLE PRECISION,
    bid_price_31 DOUBLE PRECISION, bid_volume_31 DOUBLE PRECISION,
    bid_price_32 DOUBLE PRECISION, bid_volume_32 DOUBLE PRECISION,
    bid_price_33 DOUBLE PRECISION, bid_volume_33 DOUBLE PRECISION,
    bid_price_34 DOUBLE PRECISION, bid_volume_34 DOUBLE PRECISION,
    bid_price_35 DOUBLE PRECISION, bid_volume_35 DOUBLE PRECISION,
    bid_price_36 DOUBLE PRECISION, bid_volume_36 DOUBLE PRECISION,
    bid_price_37 DOUBLE PRECISION, bid_volume_37 DOUBLE PRECISION,
    bid_price_38 DOUBLE PRECISION, bid_volume_38 DOUBLE PRECISION,
    bid_price_39 DOUBLE PRECISION, bid_volume_39 DOUBLE PRECISION,
    bid_price_40 DOUBLE PRECISION, bid_volume_40 DOUBLE PRECISION,
    -- Ask levels (price, volume) x 40
    ask_price_1 DOUBLE PRECISION, ask_volume_1 DOUBLE PRECISION,
    ask_price_2 DOUBLE PRECISION, ask_volume_2 DOUBLE PRECISION,
    ask_price_3 DOUBLE PRECISION, ask_volume_3 DOUBLE PRECISION,
    ask_price_4 DOUBLE PRECISION, ask_volume_4 DOUBLE PRECISION,
    ask_price_5 DOUBLE PRECISION, ask_volume_5 DOUBLE PRECISION,
    ask_price_6 DOUBLE PRECISION, ask_volume_6 DOUBLE PRECISION,
    ask_price_7 DOUBLE PRECISION, ask_volume_7 DOUBLE PRECISION,
    ask_price_8 DOUBLE PRECISION, ask_volume_8 DOUBLE PRECISION,
    ask_price_9 DOUBLE PRECISION, ask_volume_9 DOUBLE PRECISION,
    ask_price_10 DOUBLE PRECISION, ask_volume_10 DOUBLE PRECISION,
    ask_price_11 DOUBLE PRECISION, ask_volume_11 DOUBLE PRECISION,
    ask_price_12 DOUBLE PRECISION, ask_volume_12 DOUBLE PRECISION,
    ask_price_13 DOUBLE PRECISION, ask_volume_13 DOUBLE PRECISION,
    ask_price_14 DOUBLE PRECISION, ask_volume_14 DOUBLE PRECISION,
    ask_price_15 DOUBLE PRECISION, ask_volume_15 DOUBLE PRECISION,
    ask_price_16 DOUBLE PRECISION, ask_volume_16 DOUBLE PRECISION,
    ask_price_17 DOUBLE PRECISION, ask_volume_17 DOUBLE PRECISION,
    ask_price_18 DOUBLE PRECISION, ask_volume_18 DOUBLE PRECISION,
    ask_price_19 DOUBLE PRECISION, ask_volume_19 DOUBLE PRECISION,
    ask_price_20 DOUBLE PRECISION, ask_volume_20 DOUBLE PRECISION,
    ask_price_21 DOUBLE PRECISION, ask_volume_21 DOUBLE PRECISION,
    ask_price_22 DOUBLE PRECISION, ask_volume_22 DOUBLE PRECISION,
    ask_price_23 DOUBLE PRECISION, ask_volume_23 DOUBLE PRECISION,
    ask_price_24 DOUBLE PRECISION, ask_volume_24 DOUBLE PRECISION,
    ask_price_25 DOUBLE PRECISION, ask_volume_25 DOUBLE PRECISION,
    ask_price_26 DOUBLE PRECISION, ask_volume_26 DOUBLE PRECISION,
    ask_price_27 DOUBLE PRECISION, ask_volume_27 DOUBLE PRECISION,
    ask_price_28 DOUBLE PRECISION, ask_volume_28 DOUBLE PRECISION,
    ask_price_29 DOUBLE PRECISION, ask_volume_29 DOUBLE PRECISION,
    ask_price_30 DOUBLE PRECISION, ask_volume_30 DOUBLE PRECISION,
    ask_price_31 DOUBLE PRECISION, ask_volume_31 DOUBLE PRECISION,
    ask_price_32 DOUBLE PRECISION, ask_volume_32 DOUBLE PRECISION,
    ask_price_33 DOUBLE PRECISION, ask_volume_33 DOUBLE PRECISION,
    ask_price_34 DOUBLE PRECISION, ask_volume_34 DOUBLE PRECISION,
    ask_price_35 DOUBLE PRECISION, ask_volume_35 DOUBLE PRECISION,
    ask_price_36 DOUBLE PRECISION, ask_volume_36 DOUBLE PRECISION,
    ask_price_37 DOUBLE PRECISION, ask_volume_37 DOUBLE PRECISION,
    ask_price_38 DOUBLE PRECISION, ask_volume_38 DOUBLE PRECISION,
    ask_price_39 DOUBLE PRECISION, ask_volume_39 DOUBLE PRECISION,
    ask_price_40 DOUBLE PRECISION, ask_volume_40 DOUBLE PRECISION,
    -- Derived metrics (computed at insert time)
    mid_price        DOUBLE PRECISION,
    spread           DOUBLE PRECISION,
    volume_imbalance DOUBLE PRECISION,
    -- Validation
    is_valid         BOOLEAN DEFAULT TRUE,
    validation_flags TEXT[] DEFAULT '{}'
);

-- Convert to hypertable with 1-hour chunks
SELECT create_hypertable('lob_snapshots', 'time',
    chunk_time_interval => INTERVAL '1 hour'
);

-- Composite index for queries by exchange/symbol/time
CREATE INDEX idx_lob_snapshots_exchange_symbol_time
    ON lob_snapshots (exchange, symbol, time DESC);

-- Enable compression on chunks older than 1 day
ALTER TABLE lob_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'exchange, symbol',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('lob_snapshots', INTERVAL '1 day');

-- ============================================================
-- Data quality log
-- ============================================================
CREATE TABLE data_quality_log (
    time        TIMESTAMPTZ NOT NULL,
    exchange    TEXT        NOT NULL,
    symbol      TEXT        NOT NULL,
    issue_type  TEXT        NOT NULL,
    details     TEXT
);

SELECT create_hypertable('data_quality_log', 'time',
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX idx_quality_log_type
    ON data_quality_log (issue_type, time DESC);

-- ============================================================
-- Continuous aggregate: 5-second resampled view (40 levels)
-- ============================================================
CREATE MATERIALIZED VIEW lob_5s
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 seconds', time) AS bucket,
    exchange,
    symbol,
    -- Bid levels 1-40
    last(bid_price_1, time) AS bid_price_1, last(bid_volume_1, time) AS bid_volume_1,
    last(bid_price_2, time) AS bid_price_2, last(bid_volume_2, time) AS bid_volume_2,
    last(bid_price_3, time) AS bid_price_3, last(bid_volume_3, time) AS bid_volume_3,
    last(bid_price_4, time) AS bid_price_4, last(bid_volume_4, time) AS bid_volume_4,
    last(bid_price_5, time) AS bid_price_5, last(bid_volume_5, time) AS bid_volume_5,
    last(bid_price_6, time) AS bid_price_6, last(bid_volume_6, time) AS bid_volume_6,
    last(bid_price_7, time) AS bid_price_7, last(bid_volume_7, time) AS bid_volume_7,
    last(bid_price_8, time) AS bid_price_8, last(bid_volume_8, time) AS bid_volume_8,
    last(bid_price_9, time) AS bid_price_9, last(bid_volume_9, time) AS bid_volume_9,
    last(bid_price_10, time) AS bid_price_10, last(bid_volume_10, time) AS bid_volume_10,
    last(bid_price_11, time) AS bid_price_11, last(bid_volume_11, time) AS bid_volume_11,
    last(bid_price_12, time) AS bid_price_12, last(bid_volume_12, time) AS bid_volume_12,
    last(bid_price_13, time) AS bid_price_13, last(bid_volume_13, time) AS bid_volume_13,
    last(bid_price_14, time) AS bid_price_14, last(bid_volume_14, time) AS bid_volume_14,
    last(bid_price_15, time) AS bid_price_15, last(bid_volume_15, time) AS bid_volume_15,
    last(bid_price_16, time) AS bid_price_16, last(bid_volume_16, time) AS bid_volume_16,
    last(bid_price_17, time) AS bid_price_17, last(bid_volume_17, time) AS bid_volume_17,
    last(bid_price_18, time) AS bid_price_18, last(bid_volume_18, time) AS bid_volume_18,
    last(bid_price_19, time) AS bid_price_19, last(bid_volume_19, time) AS bid_volume_19,
    last(bid_price_20, time) AS bid_price_20, last(bid_volume_20, time) AS bid_volume_20,
    last(bid_price_21, time) AS bid_price_21, last(bid_volume_21, time) AS bid_volume_21,
    last(bid_price_22, time) AS bid_price_22, last(bid_volume_22, time) AS bid_volume_22,
    last(bid_price_23, time) AS bid_price_23, last(bid_volume_23, time) AS bid_volume_23,
    last(bid_price_24, time) AS bid_price_24, last(bid_volume_24, time) AS bid_volume_24,
    last(bid_price_25, time) AS bid_price_25, last(bid_volume_25, time) AS bid_volume_25,
    last(bid_price_26, time) AS bid_price_26, last(bid_volume_26, time) AS bid_volume_26,
    last(bid_price_27, time) AS bid_price_27, last(bid_volume_27, time) AS bid_volume_27,
    last(bid_price_28, time) AS bid_price_28, last(bid_volume_28, time) AS bid_volume_28,
    last(bid_price_29, time) AS bid_price_29, last(bid_volume_29, time) AS bid_volume_29,
    last(bid_price_30, time) AS bid_price_30, last(bid_volume_30, time) AS bid_volume_30,
    last(bid_price_31, time) AS bid_price_31, last(bid_volume_31, time) AS bid_volume_31,
    last(bid_price_32, time) AS bid_price_32, last(bid_volume_32, time) AS bid_volume_32,
    last(bid_price_33, time) AS bid_price_33, last(bid_volume_33, time) AS bid_volume_33,
    last(bid_price_34, time) AS bid_price_34, last(bid_volume_34, time) AS bid_volume_34,
    last(bid_price_35, time) AS bid_price_35, last(bid_volume_35, time) AS bid_volume_35,
    last(bid_price_36, time) AS bid_price_36, last(bid_volume_36, time) AS bid_volume_36,
    last(bid_price_37, time) AS bid_price_37, last(bid_volume_37, time) AS bid_volume_37,
    last(bid_price_38, time) AS bid_price_38, last(bid_volume_38, time) AS bid_volume_38,
    last(bid_price_39, time) AS bid_price_39, last(bid_volume_39, time) AS bid_volume_39,
    last(bid_price_40, time) AS bid_price_40, last(bid_volume_40, time) AS bid_volume_40,
    -- Ask levels 1-40
    last(ask_price_1, time) AS ask_price_1, last(ask_volume_1, time) AS ask_volume_1,
    last(ask_price_2, time) AS ask_price_2, last(ask_volume_2, time) AS ask_volume_2,
    last(ask_price_3, time) AS ask_price_3, last(ask_volume_3, time) AS ask_volume_3,
    last(ask_price_4, time) AS ask_price_4, last(ask_volume_4, time) AS ask_volume_4,
    last(ask_price_5, time) AS ask_price_5, last(ask_volume_5, time) AS ask_volume_5,
    last(ask_price_6, time) AS ask_price_6, last(ask_volume_6, time) AS ask_volume_6,
    last(ask_price_7, time) AS ask_price_7, last(ask_volume_7, time) AS ask_volume_7,
    last(ask_price_8, time) AS ask_price_8, last(ask_volume_8, time) AS ask_volume_8,
    last(ask_price_9, time) AS ask_price_9, last(ask_volume_9, time) AS ask_volume_9,
    last(ask_price_10, time) AS ask_price_10, last(ask_volume_10, time) AS ask_volume_10,
    last(ask_price_11, time) AS ask_price_11, last(ask_volume_11, time) AS ask_volume_11,
    last(ask_price_12, time) AS ask_price_12, last(ask_volume_12, time) AS ask_volume_12,
    last(ask_price_13, time) AS ask_price_13, last(ask_volume_13, time) AS ask_volume_13,
    last(ask_price_14, time) AS ask_price_14, last(ask_volume_14, time) AS ask_volume_14,
    last(ask_price_15, time) AS ask_price_15, last(ask_volume_15, time) AS ask_volume_15,
    last(ask_price_16, time) AS ask_price_16, last(ask_volume_16, time) AS ask_volume_16,
    last(ask_price_17, time) AS ask_price_17, last(ask_volume_17, time) AS ask_volume_17,
    last(ask_price_18, time) AS ask_price_18, last(ask_volume_18, time) AS ask_volume_18,
    last(ask_price_19, time) AS ask_price_19, last(ask_volume_19, time) AS ask_volume_19,
    last(ask_price_20, time) AS ask_price_20, last(ask_volume_20, time) AS ask_volume_20,
    last(ask_price_21, time) AS ask_price_21, last(ask_volume_21, time) AS ask_volume_21,
    last(ask_price_22, time) AS ask_price_22, last(ask_volume_22, time) AS ask_volume_22,
    last(ask_price_23, time) AS ask_price_23, last(ask_volume_23, time) AS ask_volume_23,
    last(ask_price_24, time) AS ask_price_24, last(ask_volume_24, time) AS ask_volume_24,
    last(ask_price_25, time) AS ask_price_25, last(ask_volume_25, time) AS ask_volume_25,
    last(ask_price_26, time) AS ask_price_26, last(ask_volume_26, time) AS ask_volume_26,
    last(ask_price_27, time) AS ask_price_27, last(ask_volume_27, time) AS ask_volume_27,
    last(ask_price_28, time) AS ask_price_28, last(ask_volume_28, time) AS ask_volume_28,
    last(ask_price_29, time) AS ask_price_29, last(ask_volume_29, time) AS ask_volume_29,
    last(ask_price_30, time) AS ask_price_30, last(ask_volume_30, time) AS ask_volume_30,
    last(ask_price_31, time) AS ask_price_31, last(ask_volume_31, time) AS ask_volume_31,
    last(ask_price_32, time) AS ask_price_32, last(ask_volume_32, time) AS ask_volume_32,
    last(ask_price_33, time) AS ask_price_33, last(ask_volume_33, time) AS ask_volume_33,
    last(ask_price_34, time) AS ask_price_34, last(ask_volume_34, time) AS ask_volume_34,
    last(ask_price_35, time) AS ask_price_35, last(ask_volume_35, time) AS ask_volume_35,
    last(ask_price_36, time) AS ask_price_36, last(ask_volume_36, time) AS ask_volume_36,
    last(ask_price_37, time) AS ask_price_37, last(ask_volume_37, time) AS ask_volume_37,
    last(ask_price_38, time) AS ask_price_38, last(ask_volume_38, time) AS ask_volume_38,
    last(ask_price_39, time) AS ask_price_39, last(ask_volume_39, time) AS ask_volume_39,
    last(ask_price_40, time) AS ask_price_40, last(ask_volume_40, time) AS ask_volume_40,
    -- Derived
    last(mid_price, time) AS mid_price,
    last(spread, time) AS spread,
    -- Quality
    bool_and(is_valid) AS all_valid,
    count(*) AS snapshot_count
FROM lob_snapshots
GROUP BY time_bucket('5 seconds', time), exchange, symbol
WITH NO DATA;

-- Refresh policy: keep the continuous aggregate up to date
-- Refreshes data from 1 hour ago to real-time, every 5 seconds
SELECT add_continuous_aggregate_policy('lob_5s',
    start_offset    => INTERVAL '1 hour',
    end_offset      => INTERVAL '5 seconds',
    schedule_interval => INTERVAL '5 seconds'
);
