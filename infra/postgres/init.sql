CREATE EXTENSION IF NOT EXISTS timescaledb;

-- This table will be used later in the future --
CREATE TABLE IF NOT EXISTS forecasts (
    time        TIMESTAMPTZ NOT NULL,
    model_name  TEXT NOT NULL,
    horizon     INT NOT NULL,
    target_col  TEXT NOT NULL,
    predicted   DOUBLE PRECISION,
    actual      DOUBLE PRECISION,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('forecasts', 'time', if_not_exists => TRUE);