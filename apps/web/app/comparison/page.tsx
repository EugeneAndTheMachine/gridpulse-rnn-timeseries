"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
import type { Metrics } from "@/lib/types";

const DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"];

export default function ComparisonPage() {
  const [dataset, setDataset] = useState("ETTh1");

  const state = useAsync(async () => {
    const models = await api.models("forecasting");
    const rows = await Promise.all(
      models.map((m) => api.metrics(m.model_name, dataset)),
    );
    return rows
      .filter((r) => r.n_predictions > 0)
      .sort((a, b) => (a.mae ?? Infinity) - (b.mae ?? Infinity));
  }, [dataset]);

  const rows: Metrics[] = state.data ?? [];
  const best = rows[0];

  const picker = (
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
  );

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Model Comparison"
        title="Which model wins?"
        subtitle="MAE, RMSE and sMAPE for every registered forecasting model, computed over the full stored evaluation range."
        right={picker}
      />

      {state.loading && <Loading label="Scoring models…" />}
      {state.error && <ErrorState message={state.error} />}
      {state.data && rows.length === 0 && (
        <Empty message={`No scored forecasts for ${dataset}. Backfill forecasts for at least one model first.`} />
      )}

      {rows.length > 0 && (
        <>
          {best && (
            <div className="card card-pad flex items-center gap-4 border-brand-200 bg-brand-50">
              <span className="text-2xl">🏆</span>
              <div>
                <div className="text-sm font-semibold text-brand-800">
                  Best model by MAE: {best.model_name}
                </div>
                <div className="text-sm text-brand-700">
                  MAE {fmt(best.mae, 4)} · RMSE {fmt(best.rmse, 4)} · sMAPE{" "}
                  {fmt(best.smape, 2)}%
                </div>
              </div>
            </div>
          )}

          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="MAE by model" hint="lower is better">
              <MetricBars rows={rows} field="mae" />
            </Card>
            <Card title="RMSE by model" hint="lower is better">
              <MetricBars rows={rows} field="rmse" />
            </Card>
          </div>

          <Card title="Full comparison">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-line">
                <thead>
                  <tr>
                    {["Rank", "Model", "MAE", "RMSE", "sMAPE %", "Predictions"].map(
                      (h) => (
                        <th key={h} className="th">
                          {h}
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {rows.map((r, i) => (
                    <tr
                      key={r.model_name}
                      className={i === 0 ? "bg-brand-50/60" : "hover:bg-slate-50"}
                    >
                      <td className="td tabular-nums">{i + 1}</td>
                      <td className="td font-medium text-ink">{r.model_name}</td>
                      <td className="td tabular-nums">{fmt(r.mae, 4)}</td>
                      <td className="td tabular-nums">{fmt(r.rmse, 4)}</td>
                      <td className="td tabular-nums">{fmt(r.smape, 2)}</td>
                      <td className="td tabular-nums">{fmtInt(r.n_predictions)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Research benchmark" hint="from notebooks · ETTh1">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-line">
                <thead>
                  <tr>
                    {["Horizon", "Best baseline", "MAE", "RMSE", "sMAPE"].map((h) => (
                      <th key={h} className="th">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {[
                    ["1h", "Persistence", "0.130", "0.191", "28.76%"],
                    ["6h", "Linear Ridge", "0.227", "0.324", "44.67%"],
                    ["24h", "Linear Ridge", "0.386", "0.519", "68.60%"],
                    ["48h", "Linear Ridge", "0.493", "0.647", "83.39%"],
                  ].map((r) => (
                    <tr key={r[0]} className="hover:bg-slate-50">
                      {r.map((c, j) => (
                        <td key={j} className="td tabular-nums">
                          {c}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

function MetricBars({
  rows,
  field,
}: {
  rows: Metrics[];
  field: "mae" | "rmse";
}) {
  const data = rows.map((r) => ({
    model: r.model_name.replace(/_h\d+_L\d+/, ""),
    value: r[field] ?? 0,
  }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} layout="vertical" margin={{ left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eef2f6" horizontal={false} />
        <XAxis type="number" tickLine={false} axisLine={false} />
        <YAxis
          type="category"
          dataKey="model"
          width={110}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip formatter={(v: number) => v.toFixed(4)} />
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={i === 0 ? "#16a34a" : "#86efac"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
