#!/usr/bin/env python3
# scripts/post_build_phase_stability.py

from pathlib import Path
import json
import numpy as np
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "phase_stability.html"

def _fmt(x, pattern, default="N/A"):
    try:
        return pattern.format(x)
    except Exception:
        return default

def fit_slope(years, phi):
    if len(years) < 3:
        return None, None, None
    slope, intercept, r_val, p_val, stderr = st.linregress(years, phi)
    ci95 = (slope - 1.96*stderr, slope + 1.96*stderr)
    return slope, ci95, p_val

rows = []
for summary_path in sorted(RESULTS.glob("*/summary.json")):
    planet = summary_path.parent.name
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        data = {}

    years = data.get("matched_years", [])
    phi_series = data.get("phi_series", [])

    slope, ci95, p_val = fit_slope(np.array(years, float), np.array(phi_series, float))
    if slope is None:
        rows.append((planet, None, None, None))
    else:
        rows.append((planet, slope, ci95, p_val))

# Build HTML
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
  <caption>Linear φ drift over time (per planet)</caption>
  <thead>
    <tr>
      <th>Planet</th>
      <th>Slope (φ/year)</th>
      <th>95% CI</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
""")
for planet, slope, ci95, p_val in rows:
    if slope is None:
        html.append(f"<tr><td>{planet}</td><td colspan=3>N/A</td></tr>\n")
    else:
        ci_str = f"[{_fmt(ci95[0], '{:+.4g}')}, {_fmt(ci95[1], '{:+.4g}')}]"
        html.append(
            "<tr>"
            f"<td>{planet}</td>"
            f"<td class='mono'>{_fmt(slope, '{:+.4g}')}</td>"
            f"<td class='mono'>{ci_str}</td>"
            f"<td class='mono'>{_fmt(p_val, '{:.3g}')}</td>"
            "</tr>\n"
        )
html.append("""  </tbody>
</table>
""")
OUT.write_text("".join(html), encoding="utf-8")
print(f"Wrote {OUT}")