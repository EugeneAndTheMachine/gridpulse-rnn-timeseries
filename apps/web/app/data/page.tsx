"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { corrColor, fmt, fmtInt, shortDate } from "@/lib/format";
import {
  Card,
  Empty,
  ErrorState,
  Loading,
  PageHeader,
  RoleBadge,
  StatTile,
} from "@/components/ui";

const DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"];

export default function DataPage() {
  const [dataset, setDataset] = useState("ETTh1");
  const { data, loading, error } = useAsync(() => api.eda(dataset), [dataset]);

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
        eyebrow="Data & Exploratory Analysis"
        title="Understand the dataset"
        subtitle="The ETT (Electricity Transformer Temperature) benchmark records six power-load features and the oil temperature target. Explore its shape, distribution, correlations and seasonality below."
        right={picker}
      />

      {loading && <Loading label={`Analysing ${dataset}…`} />}
      {error && <ErrorState message={error} />}

      {data && (
        <>
          {/* Intro + stat tiles */}
          <div className="card card-pad">
            <div className="eyebrow mb-1">About this dataset</div>
            <p className="text-sm text-ink-soft">{data.meta.blurb}</p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile label="Records" value={fmtInt(data.meta.rows)} unit="rows" accent />
            <StatTile label="Columns" value={String(data.meta.cols)} />
            <StatTile
              label="Time span"
              value={`${shortDate(data.meta.start)}`}
              unit={`→ ${shortDate(data.meta.end)}`}
            />
            <StatTile
              label="Resolution"
              value={data.meta.freq}
            />
          </div>

          {/* Observations */}
          <Card title="Key observations" hint="auto-generated">
            <ul className="space-y-2">
              {data.observations.map((o, i) => (
                <li key={i} className="flex gap-3 text-sm text-ink-soft">
                  <span className="mt-0.5 flex h-5 w-5 flex-none items-center justify-center rounded-full bg-brand-100 text-[11px] font-bold text-brand-700">
                    {i + 1}
                  </span>
                  {o}
                </li>
              ))}
            </ul>
          </Card>

          {/* Column dictionary */}
          <Card title="Column dictionary">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-line">
                <thead>
                  <tr>
                    {["Column", "Role", "Description", "Type"].map((h) => (
                      <th key={h} className="th">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {data.columns.map((c) => (
                    <tr key={c.name} className="hover:bg-slate-50">
                      <td className="td font-medium text-ink">{c.name}</td>
                      <td className="td">
                        <RoleBadge role={c.role} />
                      </td>
                      <td className="td">{c.description}</td>
                      <td className="td text-ink-muted">{c.dtype}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Summary statistics */}
          <Card title="Summary statistics">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-line">
                <thead>
                  <tr>
                    {["Column", "Mean", "Std", "Min", "P25", "Median", "P75", "Max", "Missing %"].map(
                      (h) => (
                        <th key={h} className="th">
                          {h}
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {data.summary.map((s) => (
                    <tr key={s.column} className="hover:bg-slate-50">
                      <td className="td font-medium text-ink">{s.column}</td>
                      <td className="td tabular-nums">{fmt(s.mean, 2)}</td>
                      <td className="td tabular-nums">{fmt(s.std, 2)}</td>
                      <td className="td tabular-nums">{fmt(s.min, 2)}</td>
                      <td className="td tabular-nums">{fmt(s.p25, 2)}</td>
                      <td className="td tabular-nums">{fmt(s.median, 2)}</td>
                      <td className="td tabular-nums">{fmt(s.p75, 2)}</td>
                      <td className="td tabular-nums">{fmt(s.max, 2)}</td>
                      <td className="td tabular-nums">{fmt(s.missing_pct, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Distribution + correlation */}
          <div className="grid gap-6 lg:grid-cols-2">
            <Card
              title={`Distribution of ${data.histogram.column}`}
              hint="30 bins"
            >
              <ResponsiveContainer width="100%" height={280}>
                <BarChart
                  data={data.histogram.counts.map((c, i) => ({
                    bin: data.histogram.bins[i].toFixed(1),
                    count: c,
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#eef2f6" vertical={false} />
                  <XAxis dataKey="bin" minTickGap={20} tickLine={false} axisLine={false} />
                  <YAxis width={40} tickLine={false} axisLine={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#16a34a" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Feature correlation" hint="Pearson r">
              <CorrelationHeatmap
                columns={data.correlation.columns}
                matrix={data.correlation.matrix}
              />
            </Card>
          </div>

          {/* Seasonality */}
          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Daily seasonality" hint="avg OT by hour of day">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={data.seasonality.hourly}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eef2f6" vertical={false} />
                  <XAxis dataKey="hour" tickLine={false} axisLine={false} />
                  <YAxis width={40} tickLine={false} axisLine={false} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="mean"
                    stroke="#16a34a"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Weekly seasonality" hint="avg OT by day of week">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={data.seasonality.weekly}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eef2f6" vertical={false} />
                  <XAxis dataKey="label" tickLine={false} axisLine={false} />
                  <YAxis width={40} tickLine={false} axisLine={false} />
                  <Tooltip />
                  <Bar dataKey="mean" fill="#22c55e" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {data.series.length === 0 && <Empty message="No series data." />}
        </>
      )}
    </div>
  );
}

function CorrelationHeatmap({
  columns,
  matrix,
}: {
  columns: string[];
  matrix: number[][];
}) {
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-1 text-xs">
        <thead>
          <tr>
            <th className="p-1" />
            {columns.map((c) => (
              <th key={c} className="p-1 font-medium text-ink-muted">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={columns[i]}>
              <td className="p-1 pr-2 text-right font-medium text-ink-muted">
                {columns[i]}
              </td>
              {row.map((v, j) => (
                <td
                  key={j}
                  title={`${columns[i]} · ${columns[j]}: ${v.toFixed(2)}`}
                  className="h-9 w-11 rounded text-center tabular-nums"
                  style={{
                    backgroundColor: corrColor(v),
                    color: Math.abs(v) > 0.6 ? "#fff" : "#334155",
                  }}
                >
                  {v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
