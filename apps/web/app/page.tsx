"use client";

import Link from "next/link";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { fmtInt, shortDate } from "@/lib/format";
import { Card, ErrorState, Loading, PageHeader, StatTile } from "@/components/ui";

const FEATURES = [
  {
    icon: "◔",
    title: "Forecasting",
    body: "Multi-horizon oil-temperature predictions from RNN-family models, scored against stored actuals.",
    href: "/forecasting",
  },
  {
    icon: "◇",
    title: "Imputation",
    body: "Fill missing sensor readings with linear, forward-fill or mean strategies and measure recovery error.",
    href: "/imputation",
  },
  {
    icon: "◈",
    title: "Model Comparison",
    body: "Benchmark LSTM, GRU and Vanilla RNN on MAE / RMSE / sMAPE over the full evaluation range.",
    href: "/comparison",
  },
];

export default function OverviewPage() {
  const models = useAsync(() => api.models(), []);
  const eda = useAsync(() => api.eda("ETTh1"), []);

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Overview"
        title="Electricity Transformer Analytics"
        subtitle="A production-style dashboard for forecasting, imputation and anomaly detection on the ETT benchmark, served by RNN / LSTM / GRU models."
      />

      {/* Hero + preview */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="card card-pad lg:col-span-2">
          <div className="eyebrow mb-2">Target signal · OT (Oil Temperature)</div>
          {eda.loading && <Loading label="Loading series…" />}
          {eda.error && <ErrorState message={eda.error} />}
          {eda.data && (
            <>
              <div className="mb-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 text-sm text-ink-muted">
                <span>
                  <span className="font-semibold text-ink">{eda.data.dataset}</span>{" "}
                  · {eda.data.meta.freq}
                </span>
                <span>
                  {shortDate(eda.data.meta.start)} → {shortDate(eda.data.meta.end)}
                </span>
                <span>{fmtInt(eda.data.meta.rows)} records</span>
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={eda.data.series}>
                  <defs>
                    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#16a34a" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="#16a34a" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="time"
                    tickFormatter={shortDate}
                    minTickGap={60}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis width={36} tickLine={false} axisLine={false} />
                  <Tooltip
                    labelFormatter={(v) => shortDate(String(v))}
                    formatter={(v: number) => [v, "OT"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke="#16a34a"
                    strokeWidth={1.5}
                    fill="url(#g)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </>
          )}
        </div>

        <div className="grid grid-rows-2 gap-6">
          <StatTile
            label="Registered models"
            value={models.data ? String(models.data.length) : "…"}
            accent
          />
          <StatTile
            label="Records analysed"
            value={eda.data ? fmtInt(eda.data.meta.rows) : "…"}
            unit="rows"
          />
        </div>
      </div>

      {/* Feature cards */}
      <div className="grid gap-6 md:grid-cols-3">
        {FEATURES.map((f) => (
          <Link key={f.title} href={f.href} className="group">
            <div className="card card-pad h-full transition-shadow group-hover:shadow-pop">
              <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-brand-50 text-lg text-brand-600">
                {f.icon}
              </div>
              <h3 className="text-base font-semibold text-ink">{f.title}</h3>
              <p className="mt-1 text-sm text-ink-muted">{f.body}</p>
              <span className="mt-3 inline-block text-sm font-medium text-brand-600 group-hover:text-brand-700">
                Open →
              </span>
            </div>
          </Link>
        ))}
      </div>

      {/* Models table */}
      <Card title="Registered Models" hint="from the model registry">
        {models.loading && <Loading />}
        {models.error && <ErrorState message={models.error} />}
        {models.data && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-line">
              <thead>
                <tr>
                  {["Model", "Architecture", "Task", "Horizon", "Params", "MAE"].map(
                    (h) => (
                      <th key={h} className="th">
                        {h}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {models.data.map((m) => (
                  <tr key={m.model_name} className="hover:bg-slate-50">
                    <td className="td font-medium text-ink">{m.model_name}</td>
                    <td className="td">{m.architecture ?? "—"}</td>
                    <td className="td">{m.task ?? "—"}</td>
                    <td className="td">{m.forecast_horizon ?? "—"}</td>
                    <td className="td tabular-nums">
                      {m.num_parameters ? fmtInt(m.num_parameters) : "—"}
                    </td>
                    <td className="td tabular-nums">
                      {m.metrics?.mae != null ? m.metrics.mae.toFixed(4) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
