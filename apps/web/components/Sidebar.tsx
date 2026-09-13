"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Overview", icon: "◧" },
  { href: "/data", label: "Data & EDA", icon: "▤" },
  { href: "/forecasting", label: "Forecasting", icon: "◔" },
  { href: "/comparison", label: "Model Comparison", icon: "◈" },
  { href: "/imputation", label: "Imputation", icon: "◇" },
];

export default function Sidebar() {
  const path = usePathname();
  return (
    <aside className="fixed inset-y-0 left-0 z-20 flex w-60 flex-col border-r border-line bg-white">
      <div className="flex items-center gap-2 px-5 py-5">
        <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-600 text-lg font-bold text-white">
          ⚡
        </span>
        <div className="leading-tight">
          <div className="text-base font-bold text-ink">GridPulse</div>
          <div className="text-[11px] text-ink-muted">RNN Time-Series</div>
        </div>
      </div>

      <nav className="mt-2 flex-1 space-y-1 px-3">
        {NAV.map((item) => {
          const active =
            item.href === "/" ? path === "/" : path.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                active
                  ? "bg-brand-50 text-brand-700"
                  : "text-ink-soft hover:bg-slate-50 hover:text-ink"
              }`}
            >
              <span
                className={`text-base ${active ? "text-brand-600" : "text-ink-faint"}`}
              >
                {item.icon}
              </span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-line px-5 py-4 text-[11px] text-ink-muted">
        Forecasting · Imputation · Anomaly
        <br />
        powered by LSTM / GRU / RNN
      </div>
    </aside>
  );
}
