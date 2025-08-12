#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds results/uncertainty_ci.html by scanning results/*/summary.json.

- Uses mean_dt_days / std_dt_days / phase_frac written by the pipeline.
- If matched_dt is present, bootstraps a 95% CI for mean Δt.
- Robust to missing fields; shows 'N/A' gracefully.
"""

from pathlib import Path
import json
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "uncertainty_ci.html"

def fmt(x, pat, default="N/A"):
    try:
        return pat.format(x)
    except Exception:
        return default

def read_summary(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def bootstrap_mean_ci(values, reps=3000, alpha=0.05, seed=12345):
    """Basic bootstrap CI for the mean; returns (lo, hi) or None if not enough data."""
    if not values or len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        try:
            means.append(statistics.fmean(sample))
        except Exception:
            means.append(sum(sample)/n)
    means.sort()
    lo_idx = int(alpha/2 * reps)
    hi_idx = int((1 - alpha/2) * reps) - 1
    lo = means[max(0, min(reps-1, lo_idx))]
    hi = means[max(0, min(reps-1, hi_idx))]
    return (lo, hi)

def collect_rows():
    rows = []
    for summary_path in sorted(RESULTS.glob("*/summary.json")):
        planet = summary_path.parent.name
        d = read_summary(summary_path)

        # Point estimates from your pipeline’s keys
        mu  = d.get("mean_dt_days", None)
        sig = d.get("std_dt_days", None)
        ph  = d.get("phase_frac", None)

        # Try to compute a CI from matched_dt if available
        matched = d.get("matched_dt", []) or []
        ci_str = "—"
        ci = None
        try:
            if isinstance(matched, list) and len(matched) >= 2:
                ci = bootstrap_mean_ci(matched, reps=3000, alpha=0.05, seed=12345)
        except Exception:
            ci = None
        if ci:
            lo, hi = ci
            ci_str = f"[{fmt(lo, '{:+.3f}')}, {fmt(hi, '{:+.3f}')} ]"

        rows.append((planet, mu, sig, ph, ci_str))
    return rows

def build_html(rows):
    html = []
    html.append("""<!doctype html>
<meta charset="utf-8">
<style>
  :root{ --bg:#0f1520; --fg:#e8eef5; --line:#243241; --muted:#a9b5c2; }
  body{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial; background:transparent; color:var(--fg) }
  table{ width:100%; border-collapse:collapse; font-size:14px }
  th,td{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap }
  th{ color:#fff; position:sticky; top:0; background:rgba(15,21,32,0.9); backdrop-filter:saturate(120%) blur(4px) }
  td.mono{ font-variant-numeric:tabular-nums; }
  caption{ text-align:left; color:var(--muted); padding:8px 0; }
</style>
<table>
  <caption>Bootstrap CIs for mean Δt and phase fraction φ (per planet)</caption>
  <thead>
    <tr>
      <th>Planet</th>
      <th>Mean Δt (d)</th>
      <th>σ (d)</th>
      <th>φ</th>
      <th>95% CI (Δt)</th>
    </tr>
  </thead>
  <tbody>
""")
    for p, mu, sig, ph, ci in rows:
        html.append(
            "<tr>"
            f"<td>{p}</td>"
            f"<td class='mono'>{fmt(mu, '{:+.3f}')}</td>"
            f"<td class='mono'>{fmt(sig, '{:.3f}')}</td>"
            f"<td class='mono'>{fmt(ph, '{:+.4f}')}</td>"
            f"<td class='mono'>{ci}</td>"
            "</tr>\n"
        )
    html.append("""  </tbody>
</table>
""")
    return "".join(html)

def main():
    rows = collect_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(rows), encoding="utf-8")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()