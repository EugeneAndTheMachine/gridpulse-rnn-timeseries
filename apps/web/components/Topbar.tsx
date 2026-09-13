"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Health } from "@/lib/types";

export default function Topbar() {
  const [health, setHealth] = useState<Health | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .health()
      .then((h) => alive && setHealth(h))
      .catch((e) => alive && setErr(e.message));
    return () => {
      alive = false;
    };
  }, []);

  const ok = health?.status === "ok";

  return (
    <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b border-line bg-white/80 px-6 backdrop-blur">
      <div className="text-sm text-ink-muted">
        Energy analytics dashboard
      </div>
      <div className="flex items-center gap-4 text-xs">
        {err ? (
          <span className="pill bg-red-50 text-red-600">
            ● API unreachable
          </span>
        ) : health ? (
          <>
            <span
              className={`pill ${ok ? "bg-brand-50 text-brand-700" : "bg-amber-50 text-amber-700"}`}
            >
              ● API {ok ? "connected" : "degraded"}
            </span>
            <span className="text-ink-muted">DB: {health.database}</span>
            <span className="text-ink-muted">
              Models: {health.models_loaded}
            </span>
          </>
        ) : (
          <span className="pill bg-slate-100 text-ink-muted">● checking…</span>
        )}
      </div>
    </header>
  );
}
