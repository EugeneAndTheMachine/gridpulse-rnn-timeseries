"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "@/lib/api";
import { fmt } from "@/lib/format";
import {
  Card,
  ErrorState,
  Loading,
  PageHeader,
  StatTile,
} from "@/components/ui";

const METHODS = [
  { id: "linear", label: "Linear interpolation" },
  { id: "forward_fill", label: "Forward fill" },
  { id: "mean", label: "Mean fill" },
];

// deterministic PRNG so the demo is reproducible
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng: () => number) {
  const u = 1 - rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export default function ImputationPage() {
  const [n, setN] = useState(200);
  const [missingPct, setMissingPct] = useState(20);
  const [method, setMethod] = useState("linear");

  // Build the synthetic clean series + corrupted copy (with a mask).
  const { clean, corrupted, maskIdx } = useMemo(() => {
    const rng = mulberry32(42);
    const clean: number[] = [];
    for (let i = 0; i < n; i++) {
      clean.push(10 + 5 * Math.sin((2 * Math.PI * i) / 24) + gauss(rng));
    }
    const corrupted: (number | null)[] = [...clean];
    const count = Math.floor((n * missingPct) / 100);
    const idx = new Set<number>();
    while (idx.size < count) idx.add(Math.floor(rng() * n));
    idx.forEach((i) => (corrupted[i] = null));
    return { clean, corrupted, maskIdx: idx };
  }, [n, missingPct]);

  const [imputed, setImputed] = useState<number[] | null>(null);
  const [nMissing, setNMissing] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    api
      .impute(corrupted, method)
      .then((r) => {
        if (!alive) return;
        setImputed(r.imputed);
        setNMissing(r.n_missing);
      })
      .catch((e) => alive && setError(e instanceof ApiError ? e.message : String(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [corrupted, method]);

  const mae = useMemo(() => {
    if (!imputed) return null;
    let sum = 0;
    let c = 0;
    maskIdx.forEach((i) => {
      sum += Math.abs(clean[i] - imputed[i]);
      c++;
    });
    return c ? sum / c : null;
  }, [imputed, clean, maskIdx]);

  const chartData = useMemo(() => {
    return clean.map((truth, i) => ({
      t: i,
      truth: Number(truth.toFixed(3)),
      imputed: imputed ? Number(imputed[i].toFixed(3)) : null,
      filled: imputed && maskIdx.has(i) ? Number(imputed[i].toFixed(3)) : null,
    }));
  }, [clean, imputed, maskIdx]);

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Imputation"
        title="Recover missing readings"
        subtitle="A synthetic daily-seasonal signal with injected gaps, reconstructed by the API's imputation methods. Error is measured only at the masked positions (mask-and-recover)."
      />

      <Card title="Controls">
        <div className="grid gap-6 md:grid-cols-3">
          <Slider label="Series length" value={n} min={50} max={500} step={10} onChange={setN} />
          <Slider
            label="Missing %"
            value={missingPct}
            min={5}
            max={50}
            step={5}
            onChange={setMissingPct}
          />
          <div>
            <label className="mb-2 block text-sm font-medium text-ink-soft">
              Method
            </label>
            <select
              className="select w-full"
              value={method}
              onChange={(e) => setMethod(e.target.value)}
            >
              {METHODS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </Card>

      {error && <ErrorState message={error} />}

      <div className="grid gap-4 sm:grid-cols-2">
        <StatTile label="Missing points" value={String(nMissing)} accent />
        <StatTile label="MAE at masked positions" value={fmt(mae, 4)} />
      </div>

      <Card title="Imputation result">
        {loading && !imputed ? (
          <Loading label="Calling imputation API…" />
        ) : (
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef2f6" vertical={false} />
              <XAxis dataKey="t" tickLine={false} axisLine={false} />
              <YAxis width={40} tickLine={false} axisLine={false} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="truth"
                name="Ground truth"
                stroke="#334155"
                strokeWidth={1.2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="imputed"
                name="Imputed"
                stroke="#16a34a"
                strokeWidth={1.8}
                strokeDasharray="4 3"
                dot={false}
              />
              <Scatter dataKey="filled" name="Filled" fill="#f59e0b" />
            </ComposedChart>
          </ResponsiveContainer>
        )}
        <div className="mt-3 flex flex-wrap items-center gap-5 text-xs text-ink-muted">
          <span className="flex items-center gap-2">
            <span className="inline-block h-2 w-4 rounded bg-[#334155]" /> Ground truth
          </span>
          <span className="flex items-center gap-2">
            <span className="inline-block h-2 w-4 rounded bg-brand-600" /> Imputed
          </span>
          <span className="flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-amber-500" /> Filled points
          </span>
        </div>
      </Card>

      <Card title="Reference benchmark" hint="from research notebook · ETTh1">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-line">
            <thead>
              <tr>
                {["Method", "MCAR 10%", "Block 24h"].map((h) => (
                  <th key={h} className="th">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {[
                ["Linear interp", "0.731", "1.430"],
                ["Forward fill", "1.210", "2.091"],
                ["BiLSTM", "0.752", "6.860"],
                ["MICE", "2.218", "8.845"],
              ].map((r) => (
                <tr key={r[0]} className="hover:bg-slate-50">
                  {r.map((c, j) => (
                    <td key={j} className={`td ${j === 0 ? "font-medium text-ink" : "tabular-nums"}`}>
                      {c}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-ink-muted">
            Linear interpolation wins overall, especially on block gaps.
          </p>
        </div>
      </Card>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="mb-2 flex items-center justify-between text-sm font-medium text-ink-soft">
        {label}
        <span className="tabular-nums text-brand-700">{value}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-brand-600"
      />
    </div>
  );
}
