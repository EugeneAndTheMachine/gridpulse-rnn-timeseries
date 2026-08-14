# GridPulse — Project Status (2026-08-14)

> ## 📌 Orientation for Claude (web) — read this first
>
> **What this repo is:** an end-to-end time-series ML platform (Forecasting +
> Imputation + Anomaly Detection) built around RNN-family models, now being
> wrapped into a **production stack**: FastAPI serving + TimescaleDB + Streamlit
> dashboard + Docker Compose.
>
> **Where things stand (Aug 2026):**
> - ✅ **Phases 1–4 DONE** — data pipeline, baselines, RNN/LSTM/GRU/Seq2Seq,
>   imputation (6 methods), anomaly detection. See §3–§4d.
> - ✅ **Phase 5 (Serving + DB + API) MOSTLY DONE** — FastAPI app with
>   forecast/anomaly/model routes, SQLAlchemy + TimescaleDB layer, Alembic
>   migrations, model cache, backfill script. See **§4e (NEW)**.
> - 🔄 **Dashboard = SCAFFOLD ONLY** — `apps/dashboard/` files all exist but are
>   **0 bytes** (empty). This is the main remaining build item for deploy.
> - ✅ **Repo now on GitHub** with `data/raw` (67MB) + trained checkpoints
>   (`models/checkpoints/*.pt`, 4.2MB) committed so the stack can run/deploy
>   without re-training or re-downloading.
>
> **Fastest way to understand the code:** start at `apps/api/main.py` →
> `apps/api/routes/` → `src/gridpulse/serving/model_loader.py` →
> `src/gridpulse/database/` (models, repositories) → `infra/postgres/init.sql`.

## 1. Overview

**GridPulse** is an end-to-end time series analysis platform covering three core tasks:
**Forecasting**, **Missing Data Imputation**, and **Anomaly Detection** — with RNN-family
models (RNN, LSTM, GRU, Seq2Seq) as the backbone, plus classical baselines for comparison.

| Item | Detail |
|------|--------|
| Language | Python 3.12 |
| Package manager | uv (with `uv.lock`) |
| Experiment tracking | MLflow 3.14+ (SQLite backend `mlflow.db`) |
| Deep learning | PyTorch 2.2+, PyTorch Lightning 2.2+ |
| CI/CD | GitHub Actions (lint, test, Docker build) |
| Database | TimescaleDB (via Docker Compose) + SQLAlchemy 2.0 ORM + Alembic |
| API | FastAPI (`apps/api/`), Pydantic v2 settings, model cache |
| Dashboard | Streamlit (`apps/dashboard/`) — scaffold only (empty files) |

---

## 2. Git History

| Date | Commit | Description |
|------|--------|-------------|
| 2026-07-04 | `4455e90` | First commit — project initialization |
| 2026-07-04 | `3f457b8` | EDA and basic environment setup |
| 2026-07-06 | `2436ba4` | Preprocessing pipeline for data |
| 2026-07-06 | `6bd4a1e` | Baseline complete testing |
| 2026-07-07 | `c03e169` | Vanilla RNN model completion |
| 2026-07-07 | `24aa1c1` | LSTM, GRU, Seq2Seq model implementation |
| 2026-07-13 | `f967af7` | Imputation methods for missing data |
| 2026-07-26 | `0d57e19` | Anomaly detection module + notebook 07, P0/P1 bug fixes, `.gitignore` for notebooks |
| 2026-08-03 | `cc54484` | **API + backend**: FastAPI serving, SQLAlchemy/TimescaleDB layer, Alembic migrations, `data/raw` committed |
| 2026-08-14 | `7b9c12f` | Fix `docker-compose.yml` (removed invalid CI `postgres` service); commit checkpoints + data for deploy; `.gitignore` for secrets |

> Branch: `main`. Remote: `github.com/EugeneAndTheMachine/gridpulse-rnn-timeseries`.

---

## 3. Phase 1 — Data & Baselines (COMPLETED)

### 3.1 Data Ingestion & Validation

| Dataset | Status | Location |
|---------|--------|----------|
| **ETT-small** (ETTh1/h2/m1/m2) | Downloaded, validated, cleaned | `data/raw/ett/` |
| **UCI Air Quality** | Downloaded, cleaned (-200 → NaN) | `data/raw/air_quality_uci/` |
| **Beijing PRSA** | Raw data present (12 stations, 2013-2017) | `data/raw/bejing_air_quality/` |
| **NAB** | 58 CSVs across 6 categories | `data/raw/nab/` |

Implemented modules:
- `data/download_ett.py` — auto-download 4 ETT CSVs from GitHub
- `data/download_uci.py` — auto-download UCI Air Quality zip
- `data/download_nab.py` — auto-download NAB dataset
- `data/split_data.py` — chronological train/val/test split
- `data/validate_schema.py` — ETT schema validation

### 3.2 Preprocessing Pipeline

Full pipeline **implemented and tested** for ETT:

```
Raw CSV → clean_ett() → build_features() → split_by_time() → ScalerWrapper → create_windows() → numpy arrays
```

| Module | File | Key Functions |
|--------|------|---------------|
| Cleaning | `preprocessing/cleaning.py` | `clean_ett()`, `clean_air_quality()` |
| Scaling | `preprocessing/scaling.py` | `ScalerWrapper` (Standard/MinMax/Robust, save/load) |
| Windowing | `preprocessing/windowing.py` | `create_windows()`, `TimeSeriesDataset` (PyTorch) |
| Synthetic missing | `preprocessing/synthetic_missing.py` | `inject_mcar()`, `inject_block_missing()` |
| Missing markers | `preprocessing/missing_markers.py` | `create_missing_mask()` |
| Resampling | `preprocessing/resampling.py` | Resampling utilities |

### 3.3 Feature Engineering

`features/feature_builder.py` → `build_features()` orchestrates:

| Feature Group | Output Columns |
|---------------|----------------|
| Calendar | `hour`, `day_of_week`, `month`, `is_weekend`, `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` |
| Lag | `OT_lag_1/2/3/6/12/24/168` |
| Rolling | `OT_rmean_6/12/24`, `OT_rstd_6/12/24` |

Input: 8 columns → Output: 31 columns (drops 168 NaN rows from lag-168).

### 3.4 Baseline Models

3 baselines **implemented and evaluated** across horizons `[1, 6, 24, 48]h`:

| Model | Class | Strategy |
|-------|-------|----------|
| Persistence | `PersistenceForecaster` | Repeat last observed value |
| Seasonal Naive | `SeasonalNaiveForecaster` | Tile last 24h pattern |
| Linear Ridge | `LinearForecaster` | Flatten input → `sklearn.Ridge` → multi-step output |

**Benchmark results (ETTh1, scaled data):**

| Horizon | Best Model | MAE | RMSE | sMAPE |
|---------|-----------|------|------|-------|
| 1h | Persistence | 0.130 | 0.191 | 28.76% |
| 6h | Linear Ridge | 0.227 | 0.324 | 44.67% |
| 24h | Linear Ridge | 0.386 | 0.519 | 68.60% |
| 48h | Linear Ridge | 0.493 | 0.647 | 83.39% |

### 3.5 Evaluation Metrics

`evaluation/metrics_forecasting.py`:
- `mae()`, `rmse()`, `smape()`, `compute_all_metrics()`

### 3.6 MLflow Integration

- Tracking URI: `sqlite:///mlflow.db`
- Experiments: `baselines-horizon-{1,6,24,48}` + `gridpulse_baselines`
- Each run logs: `model_name`, `input_shape`, `forecast_horizon`, `test_samples`, `mae`, `rmse`, `smape`

---

## 4. Phase 2 — Deep Learning Forecasting (COMPLETED)

### 4.1 Base Classes (`models/base.py`)

Two abstract base classes provide the foundation:

| Class | Purpose | Interface |
|-------|---------|-----------|
| `BaseForecaster` | Non-neural models | `fit(X, y)`, `predict(X)`, `name` |
| `BaseNNForecaster` | PyTorch models (extends `nn.Module`) | `forward(x)`, `save_checkpoint()`, `load_checkpoint()`, `count_parameters()` |

`BaseNNForecaster` convention:
- Input shape: `(batch, input_len, num_features)`
- Output shape: `(batch, forecast_horizon)`

### 4.2 Deep Learning Models — ALL IMPLEMENTED

#### RNNForecaster (`models/rnn.py`)

```
Input (batch, input_len, num_features)
  → nn.RNN(input_size, hidden_size, num_layers, dropout, batch_first=True)
  → Take last hidden state h_n[-1]: (batch, hidden_size)
  → Dropout
  → nn.Linear(hidden_size, forecast_horizon)
  → Output (batch, forecast_horizon)
```

#### LSTMForecaster (`models/lstm.py`)

```
Input (batch, input_len, num_features)
  → nn.LSTM(input_size, hidden_size, num_layers, dropout, batch_first=True)
  → Take last hidden state h_n[-1]: (batch, hidden_size)
    (c_n = cell state provides long-term memory internally)
  → Dropout
  → nn.Linear(hidden_size, forecast_horizon)
  → Output (batch, forecast_horizon)
```

#### GRUForecaster (`models/gru.py`)

```
Input (batch, input_len, num_features)
  → nn.GRU(input_size, hidden_size, num_layers, dropout, batch_first=True)
  → Take last hidden state h_n[-1]: (batch, hidden_size)
    (reset gate + update gate, simpler than LSTM, fewer parameters)
  → Dropout
  → nn.Linear(hidden_size, forecast_horizon)
  → Output (batch, forecast_horizon)
```

#### Seq2SeqForecaster (`models/seq2seq.py`)

```
Encoder:
  Input (batch, input_len, num_features)
  → nn.LSTM encoder → context vectors (h_n, c_n)

Decoder (autoregressive, runs forecast_horizon steps):
  Step 0 input: last value from encoder input x[:, -1, -1]
  For t = 0..forecast_horizon-1:
    → nn.LSTM decoder (input_size=1) with (h_n, c_n) context
    → Dropout → nn.Linear(hidden_size, 1) → one prediction
    → Next step input = this prediction (or ground truth if teacher forcing)
  
  → torch.cat(all predictions) → (batch, forecast_horizon)
```

Teacher forcing: configurable via `teacher_forcing_ratio` (default 0.0 = always use own predictions).

**Default hyperparameters for all models:**

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 128 |
| `num_layers` | 2 |
| `dropout` | 0.2 |
| `forecast_horizon` | 24 |

### 4.3 Training Infrastructure

#### TimeSeriesTrainer (`training/trainer.py`)

General-purpose trainer for any `BaseNNForecaster`:

| Component | Implementation |
|-----------|---------------|
| Device selection | Auto: CUDA → MPS → CPU |
| Loss function | `nn.MSELoss()` |
| Optimizer | `Adam` (configurable lr, weight_decay) |
| LR Scheduler | `ReduceLROnPlateau` (factor=0.5, patience=5) |
| Gradient clipping | `clip_grad_norm_` (default max_norm=1.0) |
| Experiment tracking | MLflow (auto-logs loss, lr, training time per epoch) |
| Seq2Seq support | Auto-detects `teacher_forcing_ratio` attr → passes `y_batch` to forward |

API:
```python
trainer = TimeSeriesTrainer(model, config, device)
history = trainer.fit(train_loader, val_loader)  # returns {"train_loss", "val_loss", "lr"}
predictions = trainer.predict(test_loader)        # loads best checkpoint, returns numpy
```

#### Callbacks (`training/callbacks.py`)

| Callback | Behavior |
|----------|----------|
| `EarlyStopping` | Stops training when val_loss doesn't improve for `patience` epochs (default 10, min_delta=1e-4) |
| `ModelCheckpoint` | Saves best model (lowest val_loss) to `models/checkpoints/{model_name}_best.pt` |

#### Baseline Evaluation (`training/train_forecasting.py`)

`evaluate_baseline()` — runs predict + compute metrics + log to MLflow for non-neural models.

### 4.4 Dataset Classes

| Class | File | Purpose |
|-------|------|---------|
| `BaseTimeSeriesDataset` | `datasets/base_dataset.py` | Abstract base (PyTorch Dataset) |
| `ForecastingDataset` | `datasets/forecasting_dataset.py` | Sliding-window: returns `(X, y)` with configurable stride |
| `TimeSeriesDataset` | `preprocessing/windowing.py` | Simple wrapper for pre-windowed numpy arrays |

### 4.5 Evaluation & Visualization

| Module | Functions |
|--------|-----------|
| `evaluation/metrics_forecasting.py` | `mae()`, `rmse()`, `smape()`, `compute_all_metrics()` |
| `evaluation/plots.py` | `plot_predictions_vs_actual()`, `plot_error_distribution()`, `plot_training_curves()` |
| `evaluation/error_analysis.py` | `error_by_hour()` — MAE breakdown by hour of day |

### 4.6 Tests — ALL PASSING (29/29 as of 2026-07-26)

```
tests/test_anomaly.py            12 tests   ✅ (4 original + 8 regression for P0/P1 fixes)
tests/test_data_download.py       2 tests   ✅
tests/test_imputation.py          5 tests   ✅
tests/test_models_forward.py      5 tests   ✅
tests/test_preprocessing.py       2 tests   ✅
tests/test_windowing.py           3 tests   ✅
────────────────────────────────────────
Total: 29 passed in 6.34s
```

| Test | What It Verifies |
|------|-----------------|
| `test_encoder_decoder_output_shape[RNNForecaster]` | `(8, 96, 30) → (8, 24)` |
| `test_encoder_decoder_output_shape[LSTMForecaster]` | `(8, 96, 30) → (8, 24)` |
| `test_encoder_decoder_output_shape[GRUForecaster]` | `(8, 96, 30) → (8, 24)` |
| `test_seq2seq_output_shape` | `(8, 96, 30) → (8, 24)` |
| `test_model_save_load` | Save LSTM checkpoint → load into new instance → outputs match exactly |

---

## 4b. Phase 3 — Missing Data Imputation (COMPLETED)

### 4b.1 Imputer Modules — ALL IMPLEMENTED

`src/gridpulse/imputation/` — common interface: `fit() → transform() → fit_transform()`.

| Module | Class | Strategy |
|--------|-------|----------|
| `simple_imputers.py` | `ForwardFillImputer` | Last valid observation carried forward (stateless) |
| `simple_imputers.py` | `MeanImputer` | Column mean from training set (stateful) |
| `interpolation.py` | `LinearInterpolationImputer` | Linear interpolation between two valid points |
| `knn_imputer.py` | `KNNImputer` | Weighted mean of `k=5` nearest rows in feature space |
| `mice_imputer.py` | `MICEImputer` | Multiple Imputation by Chained Equations, Bayesian Ridge, 10 iterations |
| `rnn_imputer.py` | `RNNImputer` | Bidirectional LSTM with self-supervised masking; `seq_len=96`, `hidden=64`, 20 epochs |
| `evaluate_imputation.py` | `benchmark_imputers()` | Orchestrates mask-and-recover + downstream training + MLflow logging |

### 4b.2 Notebook `06_imputation_benchmark.ipynb` — Executed

Two-part evaluation:

**Part 1 — Intrinsic Quality (mask-and-recover):** inject missing → impute → compute MAE/RMSE against ground truth at masked positions. 5 missing scenarios × 6 imputers = 30 runs.

MAE table (lower is better):

| Imputer | mcar_5 | mcar_10 | mcar_20 | block_6h_x10 | block_24h_x5 |
|---|---:|---:|---:|---:|---:|
| **LinearInterp** | **0.710** | **0.731** | **0.778** | **0.755** | **1.430** |
| BiLSTM_h64 | 0.674 | 0.752 | 0.821 | 2.608 | 6.860 |
| forward_fill | 1.159 | 1.210 | 1.296 | 0.945 | 2.091 |
| MICE_iter10 | 2.063 | 2.218 | 2.461 | 7.872 | 8.845 |
| KNN_k5 | 4.293 | 4.473 | 4.189 | 7.053 | 8.877 |
| Mean | 4.329 | 4.370 | 4.364 | 7.636 | 9.058 |

**Part 2 — Extrinsic Impact (LSTM forecasting after imputing 10% MCAR):**

Baseline (clean data) MAE = 0.2976. Each imputer's downstream degradation:

| Imputer | Test MAE | vs baseline |
|---|---:|---:|
| BiLSTM_h64 | 0.2850 | −4.23% |
| LinearInterp | 0.2903 | −2.45% |
| forward_fill | 0.2921 | −1.85% |
| KNN_k5 | 0.3446 | +15.79% |
| MICE_iter10 | 0.3567 | +19.86% |
| Mean | 0.3578 | +20.23% |

### 4b.3 Known Issues in Notebook 06 (not fixed yet)

- **MAPE blowup** (~5×10⁷) in Part 1: LUFL column crosses zero → MAPE unusable. Should replace with sMAPE / WAPE.
- **Single-seed downstream results**: sub-baseline degradations (−1% to −4%) are within stochasticity of one training run. Needs 3–5 seed replication for statistical significance.
- **Kết luận cuối notebook mâu thuẫn bảng số**: claims MICE/BiLSTM better for block missing, but LinearInterp clearly wins (block_24h: LinearInterp=1.43 vs MICE=8.85, BiLSTM=6.86).

---

## 4c. Phase 4 — Anomaly Detection (COMPLETED, with recent bug fixes)

### 4c.1 Anomaly Modules

`src/gridpulse/anomaly/`:

| Module | Purpose | Key API |
|--------|---------|---------|
| `residual_scorer.py` | Forecaster → anomaly scorer via `|actual − predicted|` | `ResidualScorer(model, aggregation="mean"/"max"/"first")`, `.score(X, y)`, `.score_per_step(X, y)` |
| `thresholding.py` | Convert scores → boolean flags | `BaseThreshold`, `PercentileThreshold(p)`, `ZScoreThreshold(k)`, `RollingThreshold(window, k)` |
| `event_grouping.py` | Group contiguous flags into events | `group_anomalies(is_anom, scores, max_gap, min_duration)`, `filter_short_runs(...)`, `AnomalyEvent` dataclass |
| `evaluate_anomaly.py` | Metrics | `point_wise_metrics()`, `event_wise_metrics()` (overlap-based) |
| `nab_loader.py` | Numenta Anomaly Benchmark loader | `list_nab_files()`, `load_nab_series()`, `load_nab_labels()`, `download_nab_labels()`, `get_anomaly_windows()` |

### 4c.2 Notebook `07_anomaly_detection.ipynb` — Executed (partial)

Sections A–F executed; sections G (LSTM vs RNN comparison) and H (NAB with ground truth) are wired but not yet run.

**Section A — Synthetic spikes on ETT** (30 injected, ±5·σ, LSTM residual + P95 threshold):

- Point-wise: precision 0.156, recall 0.867, F1 0.264
- Event-wise: precision 0.553, recall 0.867, F1 **0.675**

**Section B — Threshold comparison** (event F1):

| Strategy | Event Precision | Event Recall | Event F1 |
|---|---:|---:|---:|
| P95 | 0.553 | 0.867 | 0.675 |
| P99 | 0.857 | 0.600 | 0.706 |
| **Rolling_3σ** | **0.905** | 0.633 | **0.745** |

**Section E — Train-fitted thresholds (fix data leakage vs A/B):** Rolling_3σ still ties for best at F1 = 0.745, confirming its robustness across calibration source.

**Section F — Post-filter `min_duration=2`:** kills nearly all TPs (F1 collapses to ≤0.06). Root cause is a bug in the notebook's synthetic-injection routine (see below), **not** in `filter_short_runs`.

**Section D — NAB baseline** (rolling z-score on `ambient_temperature_system_failure.csv`, P98): 91 events detected; ground-truth eval deferred to Section H.

### 4c.3 Known Issue in Notebook 07 (not fixed)

**Bug in `inject_spike_anomalies`**: injects into flat `y.reshape(-1)` → each spike touches exactly **1 window**. Correct behavior for sliding stride=1: inject into raw time series → 1 spike touches ~24 consecutive windows. This makes Section F's `min_duration=2` filter incorrectly kill all TPs. Fix requires ~10 LOC in notebook, not in the anomaly module.

---

## 4d. Recent Bug Fixes (2026-07-26)

Audit of `anomaly/`, `models/`, `training/` triggered by notebook 07 results. Fixed all P0 (correctness) and P1 (naming/consistency) issues.

### 4d.1 P0 — `event_grouping.group_anomalies` / `filter_short_runs` — `max_gap` broken

**Symptom** (verified): `is_anomaly = [T, F, F, T]` with `max_gap=2` returned **2 events** instead of 1 (start=0,end=0 and start=3,end=3). Same bug in `[T, F, T]` with `max_gap=1`.

**Root cause**: after probing across a gap of False values, code did `j += gap` but did **not** update `end` or verify `is_anomaly[j]` on the next iteration. The reachable True was skipped.

**Fix**: extracted shared helper `_walk_run(is_anomaly, start, max_gap)` — walks forward, probing up to `max_gap` positions for the next True after each gap. Both `group_anomalies` and `filter_short_runs` now delegate to it.

**Impact on notebook 07**: Section D NAB (`max_gap=6`) event count previously inflated because contiguous flags across small gaps were counted separately.

### 4d.2 P0 — `thresholding.RollingThreshold` — data leakage

**Symptom** (verified): series `[1, 1, 1, 1, 100]` with `RollingThreshold(window=3, k=3)` did **not** flag the spike. The spike itself inflated the rolling std to 57.2 → threshold=205 > spike value 100.

**Root cause**: `pd.Series.rolling()` includes the current point by default (trailing window). The current score participates in its own threshold calculation.

**Fix**: added `s.shift(1)` before rolling → past-only statistics. Verified via new regression test that `[1,1,1,1,100]` now flags position 4.

### 4d.3 P1 — `BaseTheshold` typo

**Fix**: renamed class `BaseTheshold` → `BaseThreshold` in `thresholding.py`. Only used within one file — no external ripple. Regression test added to guard against re-introduction.

### 4d.4 P1 — `ResidualScorer` docstring/impl mismatch

**Symptom**: docstring mentioned modes `"mean, max, sum, 1step"` but code implemented only `"mean", "max", "first"` (raised on `"sum"`, `"1step"`).

**Fix**:
- Docstring now lists exactly the three valid options (`mean`, `max`, `first`) with descriptions.
- Validation moved to `__init__` (raise on invalid `aggregation`) → fail fast, not at first `.score()` call.
- Bonus: removed unused `import pandas as pd`, fixed `DataLoader (...)` spacing typo.

### 4d.5 Test Coverage Added — `tests/test_anomaly.py`

8 new regression tests, all passing:

```
test_group_anomalies_bridges_across_gap             — [T,F,F,T] max_gap=2 → 1 event dur=4
test_group_anomalies_bridges_single_gap             — [T,F,T]   max_gap=1 → 1 event
test_group_anomalies_does_not_bridge_over_max_gap   — gap=3 > max_gap=1  → 2 events
test_filter_short_runs_bridges_gap_before_length_check
test_filter_short_runs_zero_gap_default
test_rolling_threshold_flags_spike_without_leaking_into_stats
test_rolling_threshold_does_not_flag_stationary_series
test_base_threshold_name_no_typo
```

**Full suite state**: `29 passed in 6.34s` (was 21 before the audit — +8 anomaly tests).

### 4d.6 `.gitignore` update

Added pattern `notebooks/**/*.ipynb` (line 62) so new notebooks are not pushed. 6 notebooks tracked from before Phase 3 remain tracked (need explicit `git rm --cached` to fully untrack if desired):

```
notebooks/ett/00_dataset_overview.ipynb
notebooks/ett/05_multi_horizon_benchmark.ipynb
notebooks/ett/baselines.ipynb
notebooks/ett/ett_pipeline_test.ipynb
notebooks/ett/rnn_lstm_gru_forecasting.ipynb
notebooks/ett/rnn_training_test.ipynb
```

New notebooks (06, 07) verified ignored via `git check-ignore`.

### 4d.7 Follow-ups Not Addressed (deferred)

| Item | Priority | Notes |
|------|----------|-------|
| Fix `inject_spike_anomalies` in notebook 07 | P0 | Inject into time-domain, not flat matrix — will unlock correct Section F |
| Replace MAPE with sMAPE/WAPE in notebook 06 | P1 | MAPE unusable when target crosses zero |
| Multi-seed replication for notebook 06 Part 2 | P1 | Single-run degradation of ±few% is within noise |
| Run notebook 07 sections G & H | P1 | LSTM vs RNN comparison + NAB with ground truth |
| Fill empty stubs: `models/deepar.py`, `models/tcn.py`, `training/train_anomaly.py`, `training/train_imputation.py` | P2 | Currently import-safe (empty) but silently no-op |
| `ModelCheckpoint` should track `best_epoch` | P2 | Debug UX |
| Replace `hasattr(model, 'teacher_forcing_ratio')` in trainer | P2 | Fragile detection; use `isinstance` or explicit flag |

---

## 4e. Phase 5 — Serving, Database & API (NEW — mostly complete, 2026-08)

This is the production layer added after the 2026-07-26 snapshot. It turns the
trained models into a queryable service backed by TimescaleDB.

### 4e.1 Serving Layer (`src/gridpulse/serving/`)

| Module | Purpose | Key API |
|--------|---------|---------|
| `model_loader.py` | In-memory model cache (load once, serve many) | `ModelCache.get(name, num_features, forecast_horizon)`, `.list_available()`, `.clear()`, singleton `get_model_cache()` |
| `predict.py` | Pure inference functions (no HTTP) | `predict_from_window(model, window, device)` |
| `postprocess.py` | Post-processing of raw predictions | — |
| `schemas.py` | Serving-side dataclasses/schemas | — |

- `MODEL_REGISTRY` maps arch prefix → class: `VanillaRNN`, `LSTM`, `GRU`, `Seq2Seq`.
- Cache key = `{model_name}_h{forecast_horizon}`; checkpoint resolved as `CHECKPOINTS_DIR/{model_name}_best.pt`.
- Device auto-selects CUDA → CPU.

### 4e.2 Database Layer (`src/gridpulse/database/`)

TimescaleDB schema defined in **`infra/postgres/init.sql`** (also mirrored by Alembic).

| Table | Type | Notes |
|-------|------|-------|
| `forecasts` | **Hypertable** (chunk 7 days) | PK `(time, model_name, horizon_step)`; `residual` is a Postgres **generated column** (`predicted - actual`); indexed by `(model_name, time)` and `(dataset, time)` |
| `anomaly_events` | Regular table | Grouped events with `is_confirmed` human-review flag |
| `model_registry` | Regular table | Model metadata, `metrics`/`hyperparameters` as JSONB, unique `(model_name, version)` |
| `forecast_accuracy_hourly` | **Continuous aggregate** | Fast MAE/RMSE rollups used by `/forecast/accuracy` |

| Module | Contents |
|--------|----------|
| `connection.py` | `get_database_url()` (from `POSTGRES_*` env), singleton engine (pool_size=5, `pool_pre_ping`), `session_scope()` context manager, `Base` |
| `models.py` | SQLAlchemy 2.0 ORM: `Forecast`, `AnomalyEvent`, `ModelRegistry` (typed `Mapped[...]`) |
| `repositories.py` | Repository pattern: `ForecastRepository` (bulk UPSERT, filtered queries, DataFrame export, accuracy summary), `AnomalyRepository` (create/list/confirm), `ModelRegistryRepository` |

### 4e.3 Alembic Migrations (`infra/alembic/`, config `alembic.ini`)

| Revision | Purpose |
|----------|---------|
| `0001_initial_schema` | Mirrors `init.sql` (TimescaleDB hypertable + tables) |
| `466d0c1bcfe2_baseline_schema` | Follow-up baseline schema (revises `0001`) |

Run via `make db-migrate` (`alembic upgrade head`) or `make db-revision msg="..."`.

### 4e.4 FastAPI App (`apps/api/`)

Entry: **`apps/api/main.py`** — lifespan warm-up pre-loads the default model
(`LSTM_h128_L2`) into cache; CORS enabled for `localhost:8501` (dashboard) & `:3000`.
All routes under prefix **`/api/v1`**. Config via `apps/api/config.py`
(`GRIDPULSE_` env prefix).

| Route file | Endpoints | Notes |
|------------|-----------|-------|
| `routes/health.py` | `GET /health` | Checks DB (`SELECT 1`) + models_loaded count |
| `routes/forecast.py` | `GET /forecast`, `POST /forecast/predict`, `GET /forecast/accuracy` | Query stored forecasts; on-demand inference; accuracy from continuous aggregate |
| `routes/anomaly.py` | `GET /anomaly/events`, `POST /anomaly/events/{id}/confirm` | List + human-confirm events |
| `routes/model_metadata.py` | `GET /models`, `GET /models/available` | Registry listing + on-disk checkpoints |
| `routes/imputation.py` | — | **Empty stub; NOT mounted in `main.py`** |

DI wiring in `apps/api/dependencies.py` (per-request DB session + repos + model cache).
Request/response contracts in `apps/api/schemas.py` (Pydantic v2).
Container: `apps/api/Dockerfile` → service `api` in `docker-compose.yml`.

### 4e.5 Backfill Script (`scripts/backfill_forecasts.py`)

CLI to populate the `forecasts` hypertable from a trained model:
`python scripts/backfill_forecasts.py --model LSTM_h128_L2 --dataset ETTh1`.
Runs the full clean → features → split → scale → window → predict pipeline, then
UPSERTs into DB via `ForecastRepository`.

### 4e.6 Tests (as of 2026-08-14)

10 test files (2 are empty stubs: `test_metrics.py`, `test_synthetic_missing.py`).
**33 test functions** total:

| File | # | Notes |
|------|---|-------|
| `test_anomaly.py` | 12 | Phase 4 regressions (max_gap, rolling leakage, typo) |
| `test_api.py` | 5 | FastAPI routes (TestClient) |
| `test_database.py` | 3 | DB layer — marked `integration` (need live DB; skipped in CI without one) |
| `test_imputation.py` | 3 | Imputers |
| `test_models_forward.py` | 3 | Model output shapes |
| `test_windowing.py` | 3 | Windowing |
| `test_data_download.py` | 2 | Download scripts |
| `test_preprocessing.py` | 2 | Cleaning/features |

`pytest.ini`: `pythonpath = . src`; custom marker `integration` for live-DB tests.
(Pass/fail not re-run in this update — DB tests require a running TimescaleDB.)

### 4e.7 Docker Compose (fixed 2026-08-14)

`docker-compose.yml` now has two valid services:
- **`db`** — `timescale/timescaledb:latest-pg16`, port 5432, mounts `init.sql`, healthcheck.
- **`api`** — builds `apps/api/Dockerfile`, port 8000, `depends_on: db (healthy)`, mounts `./models` read-only.

> ⚠️ A previously present `postgres` service used GitHub-Actions syntax
> (`env:`/`options:`) and clashed on port 5432 — **removed**. No `dashboard`
> service yet (dashboard code is still empty).

---

## 5. Data Flow (End-to-End)

```
ETTh1.csv (17420 rows, 8 cols)
    │
    ▼ clean_ett()
17420 rows, 8 cols (parsed dates, no dupes)
    │
    ▼ build_features()
17252 rows, 31 cols (dropped 168 NaN rows from lag-168)
    │
    ▼ split_by_time(0.6 / 0.2 / 0.2)
Train: 10351 │ Val: 3450 │ Test: 3451
    │
    ▼ ScalerWrapper(StandardScaler).fit on train
Scaled DataFrames (30 feature cols, OT last)
    │
    ▼ create_windows(input_len=96, horizon=H, stride=1)
X: (N, 96, 30)  │  y: (N, H)
    │
    ▼ TimeSeriesDataset → DataLoader
    │
    ▼ TimeSeriesTrainer.fit(train_loader, val_loader)
    │  - Training loop with EarlyStopping + ModelCheckpoint
    │  - MLflow tracking
    │
    ▼ TimeSeriesTrainer.predict(test_loader)
    │  - Loads best checkpoint
    │  - Returns numpy predictions
    │
    ▼ compute_all_metrics(y_true, y_pred)
    │  - MAE, RMSE, sMAPE
    │
    ▼ Visualization
       - plot_predictions_vs_actual()
       - plot_error_distribution()
       - plot_training_curves()
       - error_by_hour()
```

---

## 6. Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **uv** as package manager | Fast, lockfile-based, replaces pip+venv |
| **MLflow 3.x + SQLite** | File store deprecated in MLflow 3.x; SQLite is simplest local option |
| **OT as last feature column** | Naive models use `X[:, -1, -1]` convention; `target_col_idx=-1` |
| **Scaler fit on train only** | Prevent data leakage; val/test use `transform()` only |
| **Features before split** | Avoid NaN at split boundaries from lag/rolling features |
| **Sliding windows with stride=1** | Maximum overlap for training data; configurable |
| **Editable install** (`uv pip install -e .`) | Notebooks can import `gridpulse` without sys.path hacks |
| **BaseNNForecaster extends nn.Module** | Checkpoint save/load + parameter counting in one base class |
| **Autoregressive decoder for Seq2Seq** | Each step feeds its own prediction as next input; teacher forcing optional |
| **Gradient clipping (max_norm=1.0)** | Prevent exploding gradients in RNN training |

---

## 7. Notebooks

| Notebook | Status | Content |
|----------|--------|---------|
| `notebooks/ett/00_dataset_overview.ipynb` | ✅ Executed | ETTh1 EDA: visualization, distributions, correlation, ACF/PACF, STL decomposition |
| `notebooks/01_uci_air_quality_eda.ipynb` | ✅ Executed | UCI Air Quality EDA: missing data analysis, MCAR/MAR/MNAR tests |
| `notebooks/ett/ett_pipeline_test.ipynb` | ✅ Executed | End-to-end pipeline test: Load → validate → clean → features → split → scale → DataLoader |
| `notebooks/ett/baselines.ipynb` | ✅ Executed | Multi-horizon baseline evaluation [1, 6, 24, 48]h → MLflow |
| `notebooks/ett/rnn_training_test.ipynb` | ✅ Executed | RNN training test on ETT |
| `notebooks/ett/rnn_lstm_gru_forecasting.ipynb` | 🔄 Modified | Training RNN/LSTM/GRU on ETT (in progress) |
| `notebooks/ett/05_multi_horizon_benchmark.ipynb` | ✅ Executed | Multi-horizon benchmark |
| `notebooks/ett/06_imputation_benchmark.ipynb` | ✅ Executed (untracked, ignored by git) | Full imputation benchmark: mask-and-recover + downstream LSTM impact. See §4b.2 |
| `notebooks/ett/07_anomaly_detection.ipynb` | 🔄 Partial (untracked, ignored by git) | Residual-based anomaly detection on ETT + NAB baseline. Sections A–F run; G, H not yet. See §4c.2 |
| `notebooks/02–04_*.ipynb` | ⬜ Placeholders | Reserved |

---

## 8. Project Structure

```
gridpulse-rnn-timeseries/
├── src/gridpulse/              # Core Python package
│   ├── data/                   # Download scripts (ETT, UCI, NAB)
│   ├── preprocessing/          # Cleaning, scaling, windowing, resampling
│   ├── features/               # Calendar, lag, rolling feature builders
│   ├── datasets/               # PyTorch Dataset classes
│   ├── models/                 # All model architectures
│   │   ├── base.py             # BaseForecaster + BaseNNForecaster
│   │   ├── naive.py            # Persistence + SeasonalNaive
│   │   ├── linear.py           # Ridge regression baseline
│   │   ├── rnn.py              # ✅ Vanilla RNN
│   │   ├── lstm.py             # ✅ LSTM
│   │   ├── gru.py              # ✅ GRU
│   │   ├── seq2seq.py          # ✅ Encoder-Decoder LSTM
│   │   ├── deepar.py           # ⬜ DeepAR (stub)
│   │   └── tcn.py              # ⬜ TCN (stub)
│   ├── training/               # Trainer, callbacks, evaluation scripts
│   │   ├── trainer.py          # ✅ TimeSeriesTrainer
│   │   ├── callbacks.py        # ✅ EarlyStopping + ModelCheckpoint
│   │   └── train_forecasting.py # ✅ Baseline evaluation
│   ├── evaluation/             # Metrics, plots, error analysis
│   ├── imputation/             # ✅ ForwardFill, Mean, LinearInterp, KNN, MICE, RNN + benchmark harness
│   ├── anomaly/                # ✅ ResidualScorer, thresholding (P95/P99/Rolling), event grouping, NAB loader
│   ├── explainability/         # Permutation importance, IG, attention (stubs)
│   ├── database/               # ✅ connection, ORM models, repositories (§4e.2)
│   ├── serving/                # ✅ model_loader (cache), predict, postprocess (§4e.1)
│   └── utils/                  # Paths, logger, seed
├── notebooks/                  # Jupyter notebooks (git-ignored)
├── tests/                      # 33 test functions across 8 files (+2 empty stubs) — §4e.6
├── configs/                    # YAML configs
├── data/raw/                   # ✅ Raw datasets committed (ETT, UCI, NAB, Beijing) — 67MB
├── models/checkpoints/         # ✅ 8 trained *.pt checkpoints committed — 4.2MB
├── apps/
│   ├── api/                    # ✅ FastAPI app: main, config, routes, schemas (§4e.4)
│   └── dashboard/              # 🔄 Streamlit scaffold — ALL FILES EMPTY (to build)
├── infra/
│   ├── postgres/init.sql       # ✅ TimescaleDB schema (hypertable + continuous aggregate)
│   └── alembic/                # ✅ migrations (0001 + baseline)
├── scripts/                    # ✅ backfill_forecasts.py + run_*.sh helpers
├── docker-compose.yml          # ✅ db + api services (dashboard TBD)
└── pipelines/prefect/          # Orchestration flows — stubs
```

---

## 9. What's Next

### Immediate follow-ups (from §4d.7)

- [ ] Fix `inject_spike_anomalies` in notebook 07 (inject into time-domain, not flat matrix)
- [ ] Replace MAPE with sMAPE/WAPE in notebook 06 Part 1
- [ ] Re-run notebook 06 Part 2 with 3–5 seeds for statistical significance
- [ ] Execute notebook 07 sections G (LSTM vs RNN residual) and H (NAB with ground truth)
- [ ] Correct the mismatched conclusion in notebook 06 (LinearInterp is winner, not MICE/BiLSTM)

### Phase 5 — Advanced Models

- [ ] Attention mechanism for Seq2Seq
- [ ] Implement TCN (`models/tcn.py` currently empty)
- [ ] Implement DeepAR (`models/deepar.py` currently empty) — probabilistic forecasting
- [ ] Transformer-based baseline (Informer / PatchTST)

### Phase 6 — Hyperparameter Tuning

- [ ] Optuna sweep over `hidden_size`, `num_layers`, `dropout`, `lr`
- [ ] Teacher forcing ratio ablation for Seq2Seq
- [ ] `input_len` scan {48, 96, 192, 336}

### Phase 7 — Production Integration

- [x] Wire up FastAPI serving endpoints (`apps/api/`) — health/forecast/anomaly/models done (§4e.4)
- [x] Database layer + TimescaleDB schema + Alembic migrations (§4e.2–4e.3)
- [x] Fix `docker-compose.yml` (db + api services valid)
- [ ] **Build the Streamlit dashboard** (`apps/dashboard/` — files exist but empty; top priority for deploy)
- [ ] Add `dashboard` service to `docker-compose.yml` (Streamlit, port 8501, `depends_on: api`)
- [ ] Implement `routes/imputation.py` + mount it in `main.py`
- [ ] Seed/backfill DB on deploy (run `scripts/backfill_forecasts.py`) so dashboard has data
- [ ] Complete DVC pipeline
- [ ] CI/CD deploy (build + push images)

---

## 10. Working Tree Status (2026-08-14)

Working tree is **clean** — everything committed and pushed to `origin/main`
(latest `7b9c12f`). Notably now **committed and on GitHub**:

- `data/raw/` — all datasets (ETT, UCI, Beijing PRSA, NAB), 67MB.
- `models/checkpoints/*.pt` — 8 trained checkpoints (RNN/LSTM/GRU/Seq2Seq, h64/h128), 4.2MB.
- Full `apps/api/`, `src/gridpulse/{serving,database}/`, `infra/alembic/`, `infra/postgres/init.sql`.

Still **git-ignored** (by design): `.env`, `mlflow.db`, `mlruns/`, `logs/`,
`notebooks/**/*.ipynb`, `models/checkpoints/*` non-`.pt`, Streamlit secrets.
`PROJECT_STATUS.md` is now **tracked** so web tools can read project state.

---

## 11. How to Run

Common tasks are wrapped in the **`Makefile`**:

```bash
make setup          # uv sync --all-extras + pre-commit install
make test           # uv run pytest tests/ -v
make lint           # ruff check + format --check
make download-data  # download ETT + UCI (already committed, usually not needed)

# --- Full stack via Docker Compose ---
make docker-up      # start db (TimescaleDB) + api (FastAPI) → API at :8000/docs
make db-migrate     # alembic upgrade head
make db-shell       # psql into the db container
make docker-down    # stop

# --- API locally (without Docker) ---
uv run uvicorn apps.api.main:app --reload   # http://localhost:8000/docs

# --- Populate DB with forecasts for the dashboard ---
python scripts/backfill_forecasts.py --model LSTM_h128_L2 --dataset ETTh1

# --- MLflow UI ---
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

**DB config** comes from `POSTGRES_*` env vars (see `.env.example`); API settings
use the `GRIDPULSE_` prefix (`apps/api/config.py`).
