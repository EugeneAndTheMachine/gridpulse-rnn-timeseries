export const fmt = (n: number | null | undefined, digits = 3): string =>
  n === null || n === undefined || Number.isNaN(n) ? "—" : n.toFixed(digits);

export const fmtInt = (n: number | null | undefined): string =>
  n === null || n === undefined ? "—" : n.toLocaleString("en-US");

export const shortDate = (iso: string): string => iso.slice(0, 10);

// Diverging green↔red scale for correlation cells (-1..1).
export function corrColor(v: number): string {
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    // white -> brand green
    const a = t;
    const r = Math.round(255 + (22 - 255) * a);
    const g = Math.round(255 + (163 - 255) * a);
    const b = Math.round(255 + (74 - 255) * a);
    return `rgb(${r},${g},${b})`;
  }
  // white -> red
  const a = -t;
  const r = Math.round(255 + (220 - 255) * a);
  const g = Math.round(255 + (38 - 255) * a);
  const b = Math.round(255 + (38 - 255) * a);
  return `rgb(${r},${g},${b})`;
}
