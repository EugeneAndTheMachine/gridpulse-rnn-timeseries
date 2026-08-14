-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =====================================================
-- 1. FORECASTS — hypertable, high write volume
-- =====================================================
CREATE TABLE IF NOT EXISTS forecasts (
    time          TIMESTAMPTZ NOT NULL,
    model_name    TEXT NOT NULL,
    dataset       TEXT NOT NULL,           -- 'ETTh1', 'UCI-AirQuality', ...
    target_col    TEXT NOT NULL,           -- 'OT', 'CO(GT)', ...
    horizon_step  INT NOT NULL,            -- 1..H (step within forecast)
    predicted     DOUBLE PRECISION NOT NULL,
    actual        DOUBLE PRECISION,        -- NULL if forecast is future
    residual      DOUBLE PRECISION GENERATED ALWAYS AS (
                    CASE WHEN actual IS NOT NULL 
                         THEN predicted - actual 
                         ELSE NULL END
                  ) STORED,
    run_id        TEXT,                    -- MLflow run_id (optional)
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable, chunk by 1 week
SELECT create_hypertable(
    'forecasts', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Composite index: query "latest N predictions của model X" cực nhanh
CREATE INDEX IF NOT EXISTS idx_forecasts_model_time
    ON forecasts (model_name, time DESC);

CREATE INDEX IF NOT EXISTS idx_forecasts_dataset_time
    ON forecasts (dataset, time DESC);

-- =====================================================
-- 2. ANOMALY EVENTS — regular table, sparse
-- =====================================================
CREATE TABLE IF NOT EXISTS anomaly_events (
    id            SERIAL PRIMARY KEY,
    start_time    TIMESTAMPTZ NOT NULL,
    end_time      TIMESTAMPTZ NOT NULL,
    duration      INT NOT NULL,            -- number of timesteps
    model_name    TEXT NOT NULL,
    dataset       TEXT NOT NULL,
    target_col    TEXT NOT NULL,
    peak_score    DOUBLE PRECISION NOT NULL,
    mean_score    DOUBLE PRECISION NOT NULL,
    threshold_type TEXT NOT NULL,          -- 'percentile_95', 'rolling_3sigma', ...
    threshold_value DOUBLE PRECISION,
    is_confirmed  BOOLEAN DEFAULT FALSE,   -- human review flag
    notes         TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_anomaly_time
    ON anomaly_events (start_time DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_model
    ON anomaly_events (model_name, start_time DESC);

-- =====================================================
-- 3. MODEL REGISTRY — metadata
-- =====================================================
CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    model_name      TEXT NOT NULL,
    version         TEXT NOT NULL,
    task            TEXT NOT NULL,          -- 'forecasting', 'imputation', 'anomaly'
    architecture    TEXT NOT NULL,          -- 'LSTM', 'GRU', 'Seq2Seq', ...
    input_len       INT,
    forecast_horizon INT,
    num_features    INT,
    num_parameters  BIGINT,
    metrics         JSONB,                  -- {'mae': 0.32, 'rmse': 0.48, ...}
    hyperparameters JSONB,
    checkpoint_path TEXT NOT NULL,          -- relative path in models/
    mlflow_run_id   TEXT,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_registry_task_active
    ON model_registry (task, is_active);

-- =====================================================
-- 4. CONTINUOUS AGGREGATE — hourly forecast accuracy rollup
-- =====================================================
-- Pre-computed rollup: rất nhanh cho dashboard queries
CREATE MATERIALIZED VIEW IF NOT EXISTS forecast_accuracy_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    model_name,
    dataset,
    target_col,
    COUNT(*) AS n_predictions,
    AVG(ABS(residual)) AS mae,
    SQRT(AVG(residual * residual)) AS rmse,
    AVG(residual) AS bias
FROM forecasts
WHERE actual IS NOT NULL
GROUP BY bucket, model_name, dataset, target_col
WITH NO DATA;

-- Auto-refresh policy: cập nhật mỗi 30 phút, offset 1h back để tránh partial data
SELECT add_continuous_aggregate_policy(
    'forecast_accuracy_hourly',
    start_offset => INTERVAL '30 days',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE
);

-- =====================================================
-- 5. RETENTION POLICY (optional, safety net)
-- =====================================================
-- Auto-drop chunks older than 2 years (bạn có thể chỉnh)
SELECT add_retention_policy('forecasts', INTERVAL '2 years', if_not_exists => TRUE);