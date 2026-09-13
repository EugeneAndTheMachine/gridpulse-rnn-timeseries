import React from "react";

export function PageHeader({
  eyebrow,
  title,
  subtitle,
  right,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
        <h1 className="text-2xl font-bold tracking-tight text-ink">{title}</h1>
        {subtitle && (
          <p className="mt-1 max-w-2xl text-sm text-ink-muted">{subtitle}</p>
        )}
      </div>
      {right && <div className="flex items-center gap-3">{right}</div>}
    </div>
  );
}

export function Card({
  title,
  hint,
  children,
  className = "",
}: {
  title?: string;
  hint?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`card card-pad ${className}`}>
      {title && (
        <div className="mb-4 flex items-baseline justify-between">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          {hint && <span className="text-xs text-ink-faint">{hint}</span>}
        </div>
      )}
      {children}
    </section>
  );
}

export function StatTile({
  label,
  value,
  unit,
  accent = false,
}: {
  label: string;
  value: string;
  unit?: string;
  accent?: boolean;
}) {
  return (
    <div
      className={`card card-pad ${accent ? "bg-brand-600 text-white" : ""}`}
    >
      <div
        className={`text-xs font-medium uppercase tracking-wide ${
          accent ? "text-brand-100" : "text-ink-muted"
        }`}
      >
        {label}
      </div>
      <div className="mt-2 flex items-baseline gap-1">
        <span
          className={`text-2xl font-bold tabular-nums ${
            accent ? "text-white" : "text-ink"
          }`}
        >
          {value}
        </span>
        {unit && (
          <span className={accent ? "text-brand-100" : "text-ink-muted"}>
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-line bg-white px-4 py-8 text-sm text-ink-muted">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-brand-200 border-t-brand-600" />
      {label}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-4 text-sm text-red-700">
      <div className="font-semibold">Something went wrong</div>
      <div className="mt-1 text-red-600">{message}</div>
      <div className="mt-2 text-xs text-red-500">
        Make sure the API is running: <code>uv run uvicorn apps.api.main:app</code>
      </div>
    </div>
  );
}

export function Empty({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-dashed border-line bg-white px-4 py-10 text-center text-sm text-ink-muted">
      {message}
    </div>
  );
}

export function RoleBadge({ role }: { role: string }) {
  const isTarget = role === "target";
  return (
    <span
      className={`pill ${
        isTarget ? "bg-brand-100 text-brand-700" : "bg-slate-100 text-ink-soft"
      }`}
    >
      {role}
    </span>
  );
}
