// Shared response types mirroring the FastAPI backend.

export interface Health {
  status: string;
  api_version: string;
  database: string;
  models_loaded: number;
}

export interface ModelInfo {
  model_name: string;
  version?: string;
  task?: string;
  architecture?: string;
  forecast_horizon?: number;
  num_parameters?: number;
  metrics?: { mae?: number; rmse?: number };
  is_active?: boolean;
  created_at?: string;
}

export interface ForecastPoint {
  time: string;
  horizon_step: number;
  predicted: number;
  actual: number | null;
  residual: number | null;
}

export interface ForecastResponse {
  model_name: string;
  dataset: string;
  target_col: string;
  n_points: number;
  points: ForecastPoint[];
}

export interface Metrics {
  model_name: string;
  dataset: string | null;
  horizon_step: number | null;
  n_predictions: number;
  mae: number | null;
  rmse: number | null;
  smape: number | null;
}

export interface EdaColumn {
  name: string;
  role: string;
  description: string;
  dtype: string;
}

export interface EdaSummary {
  column: string;
  mean: number;
  std: number;
  min: number;
  p25: number;
  median: number;
  p75: number;
  max: number;
  missing: number;
  missing_pct: number;
}

export interface Eda {
  dataset: string;
  meta: {
    rows: number;
    cols: number;
    start: string;
    end: string;
    freq: string;
    target: string;
    blurb: string;
  };
  columns: EdaColumn[];
  summary: EdaSummary[];
  missing: { column: string; missing: number; missing_pct: number }[];
  histogram: { column: string; bins: number[]; counts: number[] };
  correlation: { columns: string[]; matrix: number[][] };
  seasonality: {
    hourly: { hour: number; mean: number; std: number }[];
    weekly: { dow: number; label: string; mean: number }[];
    monthly: { month: number; mean: number }[];
  };
  series: { time: string; value: number | null }[];
  observations: string[];
}

export interface ImputeResponse {
  method: string;
  n_missing: number;
  imputed: number[];
}
