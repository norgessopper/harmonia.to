#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Earth-only frame/cadence variants WITHOUT touching the main pipeline.
Creates four variant folders and prints comparable stats.

Outputs:
  results_variants/
    earth_helio_1day/     (from data/earth_1900to2025.csv)
    earth_helio_2days/    (from data/earth_helio_2days.csv)
    earth_ssb_1day/       (from data/earth_bary_1day.csv)
    earth_emb_1day/       (from data/earth_emb_1day.csv)

Each folder includes the usual plots (esp. drift.png) and summary.json.
A small index HTML is written to results_variants/index.html.
"""

import os, sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # so we can import the pipeline module

from pipeline_multi_planet_v3 import run_once, autoscale_params

DATA = ROOT / "data"
OUTROOT = ROOT / "results_variants"
OUTROOT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    # label, csv_path, human label
    ("earth_helio_1day",  DATA / "earth_1900to2025.csv",      "Heliocentric (Sun@10), 1-day"),
    ("earth_helio_2days", DATA / "earth_helio_2days.csv",      "Heliocentric (Sun@10), 2-day"),
    ("earth_ssb_1day",    DATA / "earth_bary_1day.csv",        "Barycentric (SSB@0), 1-day"),
    ("earth_emb_1day",    DATA / "earth_emb_1day.csv",         "Earth–Moon Barycenter (EMB), 1-day"),
]

def main():
    rows = []
    for key, csv_path, human in VARIANTS:
        if not csv_path.exists():
            print(f"⚠️  Missing expected file: {csv_path}")
            continue
        out_dir = OUTROOT / key
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {human} ===")
        params = autoscale_params("earth")
        # same edge-guard you used before
        row = run_once(
            "earth",
            data_dir=str(DATA),
            params=params,
            signal_mode="dim0",
            out_dir=str(out_dir),
            csv_override=str(csv_path),
            edge_guard=60,
        )
        rows.append((key, human, row))

        m = row["mean"]; s = row["std"]; ph = row["phase_frac"]; cov = row["coverage"]
        print(f"→ coverage={cov:.3f}, mean Δt={m:+.3f} d, σ={s:.3f} d, φ={ph:+.4f}")
        # also mirror drift.png to a short name for quick viewing
        drift = out_dir / "drift.png"
        if drift.exists():
            (OUTROOT / f"{key}_drift.png").write_bytes(drift.read_bytes())

    # simple HTML index to compare plots
    html_lines = [
        "<!doctype html><meta charset='utf-8'><title>Earth drift — variants</title>",
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial;background:#0b0f14;color:#e8eef5;padding:24px}",
        ".grid{display:grid;gap:16px;grid-template-columns:1fr;}",
        "@media(min-width:1000px){.grid{grid-template-columns:1fr 1fr}}",
        ".card{background:#0f1520;border:1px solid #243241;border-radius:12px;padding:12px}",
        "img{width:100%;height:auto;border-radius:8px;border:1px solid #243241}",
        "h2{font-size:1.2rem;margin:.2rem 0 .6rem}",
        "table{width:100%;border-collapse:collapse;margin:12px 0}th,td{border-bottom:1px solid #243241;padding:6px 8px;text-align:left}",
        "</style>",
        "<h1>Earth — frame & cadence variants</h1>",
        "<p>Comparison of OP023–perihelion drift across frames/cadence. Each card shows <code>drift.png</code> from a fresh run.</p>",
        "<div class='grid'>"
    ]
    for key, human, row in rows:
        html_lines += [
            "<div class='card'>",
            f"<h2>{human}</h2>",
            "<table><thead><tr><th>Coverage</th><th>Mean Δt (d)</th><th>σ (d)</th><th>φ</th></tr></thead><tbody>",
            f"<tr><td>{row['coverage']:.3f}</td><td>{row['mean']:+.3f}</td><td>{row['std']:.3f}</td><td>{row['phase_frac']:+.4f}</td></tr>",
            "</tbody></table>",
            f"<img src='{key}_drift.png' alt='drift {human}'/>",
            "</div>"
        ]
    html_lines += ["</div>"]
    (OUTROOT / "index.html").write_text("\n".join(html_lines), encoding="utf-8")
    print(f"\n📄 Wrote {OUTROOT/'index.html'}")
    print("Open it in a browser to compare the four drift curves side-by-side.")

if __name__ == "__main__":
    main()