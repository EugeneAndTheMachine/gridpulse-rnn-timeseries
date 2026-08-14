"""initial schema — mirrors infra/postgres/init.sql (TimescaleDB)

Revision ID: 0001
Revises:
Create Date: 2026-07-28

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

    # 1. forecasts (hypertable, generated `residual` column)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS forecasts (
            time          TIMESTAMPTZ NOT NULL,
            model_name    TEXT NOT NULL,
            dataset       TEXT NOT NULL,
            target_col    TEXT NOT NULL,
            horizon_step  INT NOT NULL,
            predicted     DOUBLE PRECISION NOT NULL,
            actual        DOUBLE PRECISION,
            residual      DOUBLE PRECISION GENERATED ALWAYS AS (
                            CASE WHEN actual IS NOT NULL
                                 THEN predicted - actual
                                 ELSE NULL END
                          ) STORED,
            run_id        TEXT,
            created_at    TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute(
        """
        SELECT create_hypertable(
            'forecasts', 'time',
            chunk_time_interval => INTERVAL '7 days',
            if_not_exists => TRUE
        );
        """
    )
    # Unique index required by ForecastRepository.bulk_insert's ON CONFLICT
    # target. TimescaleDB requires the partitioning column (`time`) to be
    # part of any unique index on a hypertable — satisfied here.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_forecasts_time_model_step
            ON forecasts (time, model_name, horizon_step);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_forecasts_model_time
            ON forecasts (model_name, time DESC);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_forecasts_dataset_time
            ON forecasts (dataset, time DESC);
        """
    )

    # 2. anomaly_events
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly_events (
            id              SERIAL PRIMARY KEY,
            start_time      TIMESTAMPTZ NOT NULL,
            end_time        TIMESTAMPTZ NOT NULL,
            duration        INT NOT NULL,
            model_name      TEXT NOT NULL,
            dataset         TEXT NOT NULL,
            target_col      TEXT NOT NULL,
            peak_score      DOUBLE PRECISION NOT NULL,
            mean_score      DOUBLE PRECISION NOT NULL,
            threshold_type  TEXT NOT NULL,
            threshold_value DOUBLE PRECISION,
            is_confirmed    BOOLEAN DEFAULT FALSE,
            notes           TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_anomaly_time
            ON anomaly_events (start_time DESC);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_anomaly_model
            ON anomaly_events (model_name, start_time DESC);
        """
    )

    # 3. model_registry
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            id               SERIAL PRIMARY KEY,
            model_name       TEXT NOT NULL,
            version          TEXT NOT NULL,
            task             TEXT NOT NULL,
            architecture     TEXT NOT NULL,
            input_len        INT,
            forecast_horizon INT,
            num_features     INT,
            num_parameters   BIGINT,
            metrics          JSONB,
            hyperparameters  JSONB,
            checkpoint_path  TEXT NOT NULL,
            mlflow_run_id    TEXT,
            is_active        BOOLEAN DEFAULT TRUE,
            created_at       TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (model_name, version)
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_registry_task_active
            ON model_registry (task, is_active);
        """
    )

    # 4. continuous aggregate — hourly forecast accuracy rollup
    op.execute(
        """
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
        """
    )
    op.execute(
        """
        SELECT add_continuous_aggregate_policy(
            'forecast_accuracy_hourly',
            start_offset => INTERVAL '30 days',
            end_offset   => INTERVAL '1 hour',
            schedule_interval => INTERVAL '30 minutes',
            if_not_exists => TRUE
        );
        """
    )

    # 5. retention policy
    op.execute(
        """
        SELECT add_retention_policy(
            'forecasts', INTERVAL '2 years', if_not_exists => TRUE
        );
        """
    )


def downgrade() -> None:
    # Drop policies first, then the CA view, then tables.
    op.execute(
        "SELECT remove_retention_policy('forecasts', if_exists => TRUE);"
    )
    op.execute(
        "SELECT remove_continuous_aggregate_policy("
        "'forecast_accuracy_hourly', if_exists => TRUE);"
    )
    op.execute(
        "DROP MATERIALIZED VIEW IF EXISTS forecast_accuracy_hourly CASCADE;"
    )
    op.execute("DROP TABLE IF EXISTS model_registry CASCADE;")
    op.execute("DROP TABLE IF EXISTS anomaly_events CASCADE;")
    op.execute("DROP TABLE IF EXISTS forecasts CASCADE;")
    # Leave the timescaledb extension in place — it may be shared.
