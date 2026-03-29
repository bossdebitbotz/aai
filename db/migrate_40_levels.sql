-- Migration: Expand LOB from 5 levels to 40 levels
-- Run this against the existing database before restarting the collector.
-- Existing 5-level data is preserved; new columns default to NULL.

-- ============================================================
-- Step 1: Add bid columns 6-40
-- ============================================================
DO $$
BEGIN
    FOR i IN 6..40 LOOP
        EXECUTE format('ALTER TABLE lob_snapshots ADD COLUMN IF NOT EXISTS bid_price_%s DOUBLE PRECISION', i);
        EXECUTE format('ALTER TABLE lob_snapshots ADD COLUMN IF NOT EXISTS bid_volume_%s DOUBLE PRECISION', i);
    END LOOP;
END $$;

-- ============================================================
-- Step 2: Add ask columns 6-40
-- ============================================================
DO $$
BEGIN
    FOR i IN 6..40 LOOP
        EXECUTE format('ALTER TABLE lob_snapshots ADD COLUMN IF NOT EXISTS ask_price_%s DOUBLE PRECISION', i);
        EXECUTE format('ALTER TABLE lob_snapshots ADD COLUMN IF NOT EXISTS ask_volume_%s DOUBLE PRECISION', i);
    END LOOP;
END $$;

-- ============================================================
-- Step 3: Drop existing continuous aggregate and recreate with 40 levels
-- ============================================================
SELECT remove_continuous_aggregate_policy('lob_5s', if_exists => true);
DROP MATERIALIZED VIEW IF EXISTS lob_5s CASCADE;

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

-- Refresh policy
SELECT add_continuous_aggregate_policy('lob_5s',
    start_offset    => INTERVAL '1 hour',
    end_offset      => INTERVAL '5 seconds',
    schedule_interval => INTERVAL '5 seconds'
);
