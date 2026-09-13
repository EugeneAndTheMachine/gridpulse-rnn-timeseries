"use client";

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { fmt, fmtInt } from "@/lib/format";
import {
  Card,
  Empty,
  ErrorState,
  Loading,
  PageHeader,
  StatTile,
} from "@/components/ui";

const DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"];

export default function ForecastingPage() {
  const [dataset, setDataset] = useState("ETTh1");
  const [model, setModel] = useState("LSTM_h128_L2");

  const models = useAsync(() => api.models("forecasting"), []);
  const metrics = useAsync(() => api.metrics(model, dataset), [model, dataset]);
  const fc = useAsync(() => api.forecasts(model, dataset, 3000), [model, dataset]);

  const modelNames = models.data?.map((m) => m.model_name) ?? [model];

  const horizons = useMemo(() => {
    const pts = fc.data?.points ?? [];
    if (!pts.length) return [] as number[];
    return Array.from(new Set(pts.map((p) => p.horizon_step))).sort(
      (a, b) => a - b,
    );
  }, [fc.data]);

  const [horizon, setHorizon] = useState<number | null>(null);
  const activeHorizon = horizon ?? horizons[0] ?? null;

  const series = useMemo(() => {
    const pts = fc.data?.points ?? [];
    if (activeHorizon == null) return [];
    return pts
      .filter((p) => p.horizon_step === activeHorizon)
      .map((p) => ({
        time: p.time,
        predicted: p.predicted,
        actual: p.actual,
      }))
      .sort((a, b) => a.time.localeCompare(b.time));
  }, [fc.data, activeHorizon]);

  const controls = (
    <div className="flex flex-wrap items-center gap-3">
      <select
        className="select"
        value={model}
        onChange={(e) => setModel(e.target.value)}
      >
        {modelNames.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>
      <select
        className="select"
        value={dataset}
        onChange={(e) => setDataset(e.target.value)}
      >
        {DATASETS.map((d) => (
          <option key={d} value={d}>
            {d}
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Forecasting"
        title="Predictions vs. actuals"
        subtitle="Stored multi-horizon forecasts scored against the ground-truth oil temperature. Metrics are computed over the full evaluation range."
        right={controls}
      />

      {/* Metrics */}
      {metrics.loading && <Loading label="Computing metrics…" />}
      {metrics.error && <ErrorState message={metrics.error} />}
      {metrics.data && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatTile label="MAE" value={fmt(metrics.data.mae, 4)} accent />
          <StatTile label="RMSE" value={fmt(metrics.data.rmse, 4)} />
          <StatTile label="sMAPE" value={fmt(metrics.data.smape, 2)} unit="%" />
          <StatTile
            label="Scored predictions"
            value={fmtInt(metrics.data.n_predictions)}
          />
        </div>
      )}

      {/* Chart */}
      <Card
        title={`${model} · ${dataset}`}
        hint={
          horizons.length
            ? `horizon step ${activeHorizon} of ${horizons[horizons.length - 1]}`
            : undefined
        }
      >
        {horizons.length > 1 && (
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <span className="text-xs text-ink-muted">Horizon step:</span>
            {horizons.map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                  h === activeHorizon
                    ? "bg-brand-600 text-white"
                    : "bg-slate-100 text-ink-soft hover:bg-slate-200"
                }`}
              >
                {h}
              </button>
            ))}
          </div>
        )}

        {fc.loading && <Loading label="Loading forecasts…" />}
        {fc.error && <ErrorState message={fc.error} />}
        {fc.data && series.length === 0 && (
          <Empty message={`No forecasts stored for ${model} on ${dataset}. Run the backfill script.`} />
        )}
        {series.length > 0 && (
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={series}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef2f6" vertical={false} />
              <XAxis
                dataKey="time"
                tickFormatter={(v) => String(v).slice(0, 10)}
                minTickGap={60}
                tickLine={false}
                axisLine={false}
              />
              <YAxis width={44} tickLine={false} axisLine={false} />
              <Tooltip labelFormatter={(v) => String(v).slice(0, 16).replace("T", " ")} />
              <Line
                type="monotone"
                dataKey="actual"
                name="Actual"
                stroke="#334155"
                strokeWidth={1.4}
                dot={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="predicted"
                name="Predicted"
                stroke="#16a34a"
                strokeWidth={1.8}
                dot={false}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        )}
        <div className="mt-3 flex items-center gap-5 text-xs text-ink-muted">
          <span className="flex items-center gap-2">
            <span className="inline-block h-2 w-4 rounded bg-[#334155]" /> Actual
          </span>
          <span className="flex items-center gap-2">
            <span className="inline-block h-2 w-4 rounded bg-brand-600" /> Predicted
          </span>
        </div>
      </Card>
    </div>
  );
}
