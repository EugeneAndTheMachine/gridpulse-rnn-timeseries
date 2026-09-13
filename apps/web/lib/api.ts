// Thin client for the GridPulse FastAPI backend.
// The browser calls the API directly, so this must be a host-reachable URL
// even when the web app runs in Docker (the API port is published on the host).

import type {
  Eda,
  ForecastResponse,
  Health,
  ImputeResponse,
  Metrics,
  ModelInfo,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
  "http://localhost:8000/api/v1";

export class ApiError extends Error {}

async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
    }
  }
  let res: Response;
  try {
    res = await fetch(url.toString(), { cache: "no-store" });
  } catch (e) {
    throw new ApiError(`Cannot reach API at ${url.toString()} — is it running?`);
  }
  if (!res.ok) {
    throw new ApiError(`${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const url = `${API_BASE}${path}`;
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    throw new ApiError(`Cannot reach API at ${url} — is it running?`);
  }
  if (!res.ok) {
    throw new ApiError(`${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => get<Health>("/health"),
  models: (task?: string) => get<ModelInfo[]>("/models", { task }),
  forecasts: (model_name: string, dataset: string, limit = 2000) =>
    get<ForecastResponse>("/forecast", { model_name, dataset, limit }),
  metrics: (model_name: string, dataset?: string, horizon_step?: number) =>
    get<Metrics>("/forecast/metrics", { model_name, dataset, horizon_step }),
  edaList: () => get<{ datasets: string[] }>("/eda"),
  eda: (dataset: string) => get<Eda>(`/eda/${dataset}`),
  impute: (values: (number | null)[], method: string) =>
    post<ImputeResponse>("/imputation/impute", { values, method }),
};
