#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OP022/OP023 multi-planet pipeline (v3) — full version with:
  • matched_dt / matched_years / phi_series saved into per-planet summary.json
  • Reproducibility context: per-planet CSV file path + SHA256 + span, and a
    global results/run_context.json with git/Python/lib versions, argv, etc.

Usage examples:
  python3 scripts/pipeline_multi_planet_v3.py --planets all
  python3 scripts/pipeline_multi_planet_v3.py --planets mercury mars --mode pca1
  python3 scripts/pipeline_multi_planet_v3.py --planets all \
      --overrides scripts/full_run_overrides.json \
      --windows   scripts/planet_windows.json \
      --edge-guard 60
"""

import os, re, glob, json, argparse, hashlib, subprocess, sys, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as _dt

ddof = 1  # safety: if any std(...) accidentally uses ddof as a positional var, this keeps it defined
# ---------- planet parameters (fallbacks) ----------
try:
    from planet_params_overrides import OVERRIDES as OVERRIDES_PRESET
except ImportError:
    OVERRIDES_PRESET = {}

try:
    from planet_params import SIDEREAL_DAYS, autoscale_params
except Exception:
    SIDEREAL_DAYS = {
        "mercury": 87.969, "venus": 224.701, "earth": 365.256, "mars": 686.980,
        "jupiter": 4332.589, "saturn": 10759.220, "uranus": 30685.400,
        "neptune": 60189.000, "pluto": 90560.000,
    }
    def autoscale_params(name: str):
        P = SIDEREAL_DAYS[name]
        smooth = max(5, int(round(P/40)))          # ~P/40
        min_sep = max(50, int(round(P/3)))         # peak de-dupe
        peri_minsep = max(50, int(round(P/2)))     # peri minima de-dupe
        match_window = max(60, int(round(P/3)))    # peri-peak match window
        z = 0.8 if P < 2000 else 0.5               # laxer for outer planets
        return dict(smooth=smooth, min_sep=min_sep, z=z,
                    peri_minsep=peri_minsep, match_window=match_window)

# ---------- runtime-config globals (set in main) ----------
OVERRIDES = {}              # per-planet param overrides
WINDOWS = {}                # per-planet CSV path overrides
GLOBAL_EDGE_GUARD = None    # days to drop peri near edges (avoid boundary bias)
RUN_CONTEXT = {}            # populated at run; saved as results/run_context.json

# ---------- utilities ----------
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def file_sha256(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def get_git_info():
    def _run(cmd):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
            return out
        except Exception:
            return None
    info = {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status_dirty": None,
    }
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8")
        info["git_status_dirty"] = bool(status.strip())
    except Exception:
        info["git_status_dirty"] = None
    return info

def find_best_csv(data_dir, name):
    """Pick the widest span CSV like data/<name>_YYYYtoYYYY.csv; fallback to 1900–2025."""
    pattern = os.path.join(data_dir, f"{name}_*to*.csv")
    cands = glob.glob(pattern)
    if not cands:
        return os.path.join(data_dir, f"{name}_1900to2025.csv")
    def span_score(p):
        m = re.search(r"_(\d{4})to(\d{4})\.csv$", os.path.basename(p))
        if not m: return (-1, -1, p)
        a, b = int(m.group(1)), int(m.group(2))
        return (b - a, -a, p)  # longer span first; earlier start preferred
    return sorted(cands, key=span_score, reverse=True)[0]

def _infer_span_years_from_path(csv_path):
    m = re.search(r"_(\d{4})to(\d{4})\.csv$", os.path.basename(csv_path))
    if m:
        return int(m.group(1)), int(m.group(2))
    return 1900, 2025

def _infer_start_date_from_path(csv_path):
    """
    Infer start date as YYYY-01-01 from a filename like ..._1900to2025.csv.
    Falls back to 1900-01-01 if pattern not found.
    """
    y0, _ = _infer_span_years_from_path(csv_path)
    return np.datetime64(f"{y0:04d}-01-01")

def parse_horizons_header_metadata(csv_path, max_lines=80):
    """
    Try to extract a few header hints from the Horizons CSV prior to $$SOE.
    Returns a dict with any of: 'query_dates', 'center', 'frame', 'units', 'step'.
    This is best-effort and robust to missing keys.
    """
    meta = {}
    try:
        with open(csv_path, "r", errors="ignore") as f:
            lines = []
            for i, line in enumerate(f):
                if i > max_lines: break
                if "$$SOE" in line: break
                lines.append(line.strip())
        hdr = "\n".join(lines)

        # Dates: collect all YYYY-MM-DD patterns
        dates = re.findall(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", hdr)
        if dates:
            meta["query_dates"] = sorted({f"{y}-{m}-{d}" for (y,m,d) in dates})

        # Common header fields (best-effort, varies by export)
        m = re.search(r"Center.*?:\s*(.+)", hdr, re.IGNORECASE);          meta["center"] = m.group(1).strip() if m else None
        m = re.search(r"Reference.*?(Ecliptic.*?J2000|ICRF|J2000)", hdr, re.IGNORECASE); meta["frame"] = m.group(1).strip() if m else None
        m = re.search(r"Units.*?:\s*([A-Za-z0-9/_\-\s]+)", hdr, re.IGNORECASE);          meta["units"] = m.group(1).strip() if m else None
        m = re.search(r"Step.*?:\s*([0-9]+\s*(min|mins|minutes|day|days))", hdr, re.IGNORECASE); meta["step"] = m.group(1).strip() if m else None
    except Exception:
        pass
    return meta

def boxcar(x, w):
    if w <= 1: return x
    k = np.ones(w)/w
    return np.convolve(x, k, mode="same")

def find_local_minima(y):
    i = np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1
    return i.astype(int)

def cluster_select(indices, score_series, min_sep, select="min"):
    if len(indices) == 0: return indices
    idx = np.sort(indices)
    keep = []
    i = 0
    while i < idx.size:
        j = i + 1
        best = idx[i]; best_val = score_series[best]
        while j < idx.size and (idx[j] - idx[i]) < min_sep:
            cond = (score_series[idx[j]] < best_val) if select=="min" else (score_series[idx[j]] > best_val)
            if cond:
                best = idx[j]; best_val = score_series[idx[j]]
            j += 1
        keep.append(best)
        i = j
    return np.array(keep, dtype=int)

def find_peaks_simple(y, z_thresh=0.8, min_sep=60):
    y = np.asarray(y, float)
    mu = y.mean()
    sd = y.std(ddof=1) if y.size > 1 else 1.0
    thr = mu + z_thresh * (sd if sd > 0 else 1.0)
    cand = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] >= thr))[0] + 1
    if cand.size == 0: return np.array([], dtype=int)
    return cluster_select(cand, y, min_sep, select="max")

def wrap_phase_fraction(dt_days, period_days):
    if not np.isfinite(dt_days): return np.nan
    f = dt_days / period_days
    f = (f + 0.5) % 1.0 - 0.5
    return f

# ---------- core builders ----------
def build_psi_from_csv(csv_path):
    """Parse Horizons CSV (either $$SOE/$$EOE or named columns). Return standardized Ψ(t) [D=17]."""
    txt = open(csv_path, "r", errors="ignore").read()

    def _between():
        if "$$SOE" not in txt or "$$EOE" not in txt: return None
        body = txt.split("$$SOE",1)[1].split("$$EOE",1)[0]
        rows = []
        for line in body.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9: continue
            try:
                x,y,z = float(parts[2]), float(parts[3]), float(parts[4])
                vx,vy,vz = float(parts[5]), float(parts[6]), float(parts[7])
                rows.append((x,y,z,vx,vy,vz))
            except Exception:
                continue
        if not rows: return None
        return pd.DataFrame(rows, columns=["X","Y","Z","VX","VY","VZ"])

    def _byname():
        df = pd.read_csv(csv_path, comment=";", engine="python")
        cols = {c.lower(): c for c in df.columns}
        def pick(prefix):
            for k,v in cols.items():
                if k.strip().startswith(prefix): return v
            return None
        names = [pick("x"), pick("y"), pick("z"), pick("vx"), pick("vy"), pick("vz")]
        if any(v is None for v in names): return None
        df = df[names].rename(columns=dict(zip(names, ["X","Y","Z","VX","VY","VZ"])))
        for c in ["X","Y","Z","VX","VY","VZ"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna().reset_index(drop=True)

    df = _between()
    if df is None: df = _byname()
    if df is None or df.empty: raise RuntimeError(f"Could not parse Horizons CSV: {csv_path}")

    X,Y,Z = (df[c].to_numpy() for c in ("X","Y","Z"))
    VX,VY,VZ = (df[c].to_numpy() for c in ("VX","VY","VZ"))
    R = np.sqrt(X*X + Y*Y + Z*Z)
    V = np.sqrt(VX*VX + VY*VY + VZ*VZ)

    feats = [X,Y,Z,VX,VY,VZ,R,V,X*VX,Y*VY,Z*VZ, X*Y,X*Z,Y*Z, VX*VY,VX*VZ,VY*VZ]
    Psi = np.vstack(feats).T  # (T, 17)

    mu = Psi.mean(axis=0, keepdims=True)
    sd = Psi.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd==0, 1.0, sd)
    Zm = (Psi - mu)/sd
    return Zm

# ---------- detectors & analytics ----------
def detect_perihelia(psi_std, smooth_win=5, peri_minsep=70):
    X,Y,Z = psi_std[:,0], psi_std[:,1], psi_std[:,2]
    R = np.sqrt(X*X + Y*Y + Z*Z)
    Rs = boxcar(R, smooth_win)
    mins = find_local_minima(Rs)
    peri = cluster_select(mins, Rs, peri_minsep, select="min")
    return peri, R, Rs

def build_signal(psi_std, mode="dim0"):
    if mode == "dim0":
        f = psi_std[:,0].astype(float)
        C = f**2
    elif mode == "pca1":
        X = psi_std.astype(float)
        X = X - X.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        f = U[:,0] * s[0]
        C = f**2
    elif mode == "energy":
        C = np.sum(psi_std.astype(float)**2, axis=1)
        f = np.sqrt(C) * np.sign(psi_std[:,0])  # signed surrogate
    else:
        raise ValueError("mode must be 'dim0' | 'pca1' | 'energy'")
    return f, C

def detect_collapses(psi_std, smooth_win=5, z=0.8, min_sep=60, mode="dim0"):
    f, C = build_signal(psi_std, mode=mode)
    Cs = boxcar(C, smooth_win)
    peaks = find_peaks_simple(Cs, z_thresh=z, min_sep=min_sep)
    return peaks, f, C, Cs

def match_offsets(peri, peaks, window=60):
    """
    Greedy nearest-neighbor within ±window. Returns:
      offsets:   array of (peak - peri) in index-days
      matched_pk:indices of peaks used
      matched_pi:indices of peri array that were matched (positions within peri)
    """
    peri = np.sort(peri); peaks = np.sort(peaks)
    if peri.size==0 or peaks.size==0:
        return np.array([]), np.array([]), np.array([])
    off = []; matched_pk = []; matched_pi = []
    j = 0
    for i, a in enumerate(peri):
        while j+1 < peaks.size and abs(peaks[j+1]-a) <= abs(peaks[j]-a):
            j += 1
        d = peaks[j] - a
        if abs(d) <= window:
            off.append(d); matched_pk.append(peaks[j]); matched_pi.append(i)
    return np.array(off, float), np.array(matched_pk, int), np.array(matched_pi, int)

def lag_sweep_corr(Cs, R, max_lag_days=120):
    """Scan integer lags, correlate Cs(t) with 1/R(t+lag)."""
    invR = 1.0 / (np.abs(R) + 1e-12)
    lags = np.arange(-max_lag_days, max_lag_days+1, 1, dtype=int)
    rs = np.full(lags.size, np.nan)
    for i, L in enumerate(lags):
        if L >= 0:
            a = Cs[:Cs.size-L]
            b = invR[L:invR.size]
        else:
            a = Cs[-L:Cs.size]
            b = invR[:invR.size+L]
        if a.size < 50:
            continue
        am = a - a.mean(); bm = b - b.mean()
        da = am.std(ddof=1); db = bm.std(ddof-1) if (db := bm.std(ddof=1)) or True else None  # keep ddof=1
        if da==0 or db==0: continue
        rs[i] = np.dot(am, bm) / ((a.size-1)*da*db)
    if np.all(~np.isfinite(rs)):
        return 0, np.nan, (lags, rs)
    k = int(np.nanargmax(rs))
    return int(lags[k]), float(rs[k]), (lags, rs)

def yearly_drift(peri, offsets, date0="1900-01-01"):
    if offsets.size == 0: return None
    date0 = np.datetime64(date0)
    peri_dates = date0 + peri.astype("timedelta64[D]")
    years = peri_dates.astype("datetime64[Y]").astype(int) + 1970
    n = min(len(years), len(offsets))
    df = pd.DataFrame({"year": years[:n], "offset": offsets[:n]})
    yearly = df.groupby("year")["offset"].agg(["mean","std","count"])
    roll = yearly["mean"].rolling(5, center=True, min_periods=1).mean()
    return yearly, roll

def compute_entropy_series(psi_std):
    EPS = 1e-12
    H = []
    for i in range(psi_std.shape[0]):
        x = np.abs(psi_std[i,:])
        s = x.sum()
        if s <= 0: H.append(0.0); continue
        p = np.clip(x/s, EPS, 1.0)
        H.append(float(-np.sum(p*np.log(p))))
    return np.asarray(H, float)

def sharpness_vals(C, peaks, H=2):
    vals = []
    for k in peaks:
        if not (1 <= k < C.size-1): continue
        i0 = max(1, k-H); i1 = min(C.size-2, k+H)
        ss = [C[j+1] - 2*C[j] + C[j-1] for j in range(i0, i1+1)]
        curv = float(np.mean(ss))
        local = C[max(0,k-10):min(C.size,k+11)]
        scale = np.median(np.abs(local - np.median(local))) + 1e-12
        vals.append(curv/scale)
    return np.asarray(vals, float)

# ---------- plots ----------
def _shade_edges(ax, lo, hi, n):
    if lo > 0:
        ax.axvspan(0, lo, color="#000", alpha=0.06, lw=0)
    if hi < n-1:
        ax.axvspan(hi, n-1, color="#000", alpha=0.06, lw=0)

def plot_overlay(planet, Cs, peri, peaks, out_png, matched=None, lo=None, hi=None):
    n = Cs.size
    if lo is None: lo = 0
    if hi is None: hi = n-1
    plt.figure(figsize=(12,4.6))
    t = np.arange(n)
    plt.plot(t, Cs, lw=1.0, label="C(t) smoothed")
    _shade_edges(plt.gca(), lo, hi, n)

    if peri.size:
        peri_viz = peri[(peri >= lo) & (peri <= hi)]
        plt.scatter(peri_viz, Cs[peri_viz], s=22, c="#22c55e", label="Perihelion")
    if peaks.size:
        peaks_viz = peaks[(peaks >= lo) & (peaks <= hi)]
        plt.scatter(peaks_viz, Cs[peaks_viz], s=30, facecolors="none",
                    edgecolors="#f59e0b", label="OP023 peaks")
    if matched is not None and matched.size:
        m_viz = matched[(matched >= lo) & (matched <= hi)]
        plt.scatter(m_viz, Cs[m_viz], s=48, facecolors="none",
                    edgecolors="#ef4444", lw=1.8, label="Matched peaks")

    plt.title(f"{planet.title()} — C(t) with perihelia & OP023 peaks")
    plt.xlabel("Day index"); plt.ylabel("|f|² (smoothed)")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_r_overlay(planet, R, peri, out_png, lo=None, hi=None, smooth=5):
    n = R.size
    if lo is None: lo = 0
    if hi is None: hi = n-1
    Rs = boxcar(R, smooth)
    plt.figure(figsize=(12,4.6))
    t = np.arange(n)
    plt.plot(t, Rs, lw=1.0, label="R(t) smoothed")
    _shade_edges(plt.gca(), lo, hi, n)
    if peri.size:
        peri_viz = peri[(peri >= lo) & (peri <= hi)]
        plt.scatter(peri_viz, Rs[peri_viz], s=22, c="#22c55e", label="Perihelion")
    plt.title(f"{planet.title()} — R(t) with perihelia")
    plt.xlabel("Day index"); plt.ylabel("R (AU)")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_zoom(planet, Cs, peri, matched, out_png, peri_k0=100, peri_span=3):
    if peri.size == 0: return
    if peri.size < peri_k0+peri_span+1: peri_k0 = max(0, peri.size - (peri_span+1))
    lo = max(0, peri[peri_k0]-20); hi = min(Cs.size-1, peri[min(peri_k0+peri_span, peri.size-1)]+20)
    x = np.arange(lo, hi+1)
    plt.figure(figsize=(12,4.6))
    plt.plot(x, Cs[lo:hi+1], lw=1.2)
    pwin = peri[(peri>=lo)&(peri<=hi)]
    mwin = matched[(matched>=lo)&(matched<=hi)] if matched is not None else np.array([], int)
    if pwin.size: plt.scatter(pwin, Cs[pwin], c="#22c55e", s=30, label="Perihelion")
    if mwin.size: plt.scatter(mwin, Cs[mwin], facecolors="none", edgecolors="#f59e0b", s=60, label="Matched peak")
    plt.title(f"{planet.title()} — Zoom")
    plt.xlabel("Day index"); plt.ylabel("|f|²")
    plt.legend(frameon=False); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_hist(planet, offsets, window, out_png):
    if offsets.size == 0: return
    plt.figure(figsize=(12,4.6))
    bins = np.arange(-window-0.5, window+1.5, 1)
    plt.hist(offsets, bins=bins, color="#22c55e", alpha=0.9, edgecolor="none")
    plt.axvline(0, color="#666", lw=1)
    plt.title(f"{planet.title()} — Phase shift histogram")
    plt.xlabel("Δt = peak − perihelion (days)"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_drift(planet, yearly, roll, out_png):
    if yearly is None: return
    plt.figure(figsize=(12,4.6))
    plt.plot(yearly.index, yearly["mean"], label="Yearly mean Δt")
    plt.plot(yearly.index, roll, label="5-yr rolling mean")
    plt.fill_between(yearly.index, yearly["mean"]-yearly["std"], yearly["mean"]+yearly["std"],
                     color="#93c5fd", alpha=0.25, label="±1σ")
    plt.axhline(0, color="#aaa", lw=1, ls="--")
    plt.title(f"{planet.title()} — Mean phase offset per year")
    plt.xlabel("Year"); plt.ylabel("Δt (days)"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_fft(planet, freq, mean_mag, out_png):
    if freq is None: return
    plt.figure(figsize=(9,4))
    plt.plot(freq[1:], mean_mag[1:], lw=1.4)
    plt.xlabel("Frequency (cycles/day)"); plt.ylabel("Mean |FFT|")
    plt.title(f"{planet.title()} — Avg FFT of C(t) around peaks (±20 d)")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_entropy_hist(planet, H_all, H_pk, out_png):
    if H_all.size == 0: return
    plt.figure(figsize=(9,4))
    plt.hist(H_all, bins=40, alpha=0.7, label="All days")
    if H_pk.size: plt.hist(H_pk,  bins=40, alpha=0.7, label="At peaks")
    plt.xlabel("Entropy H(t)"); plt.ylabel("Count"); plt.title(f"{planet.title()} — Entropy")
    plt.legend(frameon=False); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_lags(planet, lags, rs, best_lag, best_r, out_png):
    if lags is None: return
    plt.figure(figsize=(9,4))
    plt.plot(lags, rs, lw=1.4)
    plt.axvline(best_lag, color="#f59e0b", lw=1.2, ls="--")
    plt.title(f"{planet.title()} — Lag sweep corr(C, 1/R)")
    plt.xlabel("Lag (days; 1/R shifted by lag)"); plt.ylabel("Pearson r")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# ---------- HTML report ----------
def write_html_report(planets, rows, out_html, root="results"):
    def row_to_tr(r):
        p = r["planet"]
        folder = f"{root}/{p}"
        def f(name): return f"{folder}/{name}"
        links = []
        for key, file in [
            ("overlay","overlay.png"),
            ("r_overlay","r_overlay.png"),
            ("zoom","zoom.png"),
            ("phase_hist","phase_hist.png"),
            ("drift","drift.png"),
            ("fft","fft.png"),
            ("entropy","entropy_hist.png"),
            ("sharpness","sharpness_hist.png"),
            ("lags","lag_sweep.png"),
        ]:
            path = f(file)
            if os.path.exists(path):
                links.append(f'<a href="{path}">{key}</a>')
        linkstr = " • ".join(links)
        s = f"{r['strength']:.3f}" if np.isfinite(r['strength']) else "nan"
        pf = f"{r['phase_frac']:+.4f}" if np.isfinite(r['phase_frac']) else "nan"
        med = f"{r['median']:+.3f}" if np.isfinite(r['median']) else "nan"
        return (
            f"<tr><td>{p.title()}</td>"
            f"<td>{r['period_days']:.3f}</td>"
            f"<td>{r['N_peri']}</td>"
            f"<td>{r['N_peaks']}</td>"
            f"<td>{r['matched']}</td>"
            f"<td>{r['coverage']:.3f}</td>"
            f"<td>{r['mean']:+.3f}</td>"
            f"<td>{r['std']:.3f}</td>"
            f"<td>{med}</td>"
            f"<td>{r['min']:+.1f}</td>"
            f"<td>{r['max']:+.1f}</td>"
            f"<td>{pf}</td>"
            f"<td>{s}</td>"
            f"<td>{linkstr}</td></tr>"
        )
    html = f"""<!doctype html><html><head>
<meta charset="utf-8"/><title>OP022/OP023 — Multi-planet report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial;margin:24px;background:#0b0f14;color:#e8eef5}}
h1{{margin:0 0 8px}} p.note{{color:#a9b5c2;margin:0 0 16px}}
table{{width:100%;border-collapse:collapse;margin-top:12px}}
th,td{{border-bottom:1px solid #243241;padding:8px;text-align:left}} th{{color:#fff}}
a{{color:#7bdcff;text-decoration:none}} a:hover{{text-decoration:underline}}
</style></head><body>
<h1>OP022/OP023 — Multi-planet summary</h1>
<p class="note">Auto-scaled per planet; Δt in days (peak − perihelion). Phase frac = mean Δt / sidereal P wrapped to (−0.5,0.5]. Strength = coverage · exp(−std/P).</p>
<table><thead><tr>
<th>Planet</th><th>Sidereal P (d)</th><th>Peri</th><th>Peaks</th><th>Matched</th><th>Coverage</th>
<th>Mean Δt</th><th>σ</th><th>Median</th><th>Min</th><th>Max</th><th>Phase frac</th><th>Strength</th><th>Plots</th>
</tr></thead><tbody>
{''.join(row_to_tr(r) for r in rows)}
</tbody></table>
</body></html>"""
    with open(out_html, "w") as f: f.write(html)

# ---------- run one planet ----------
def run_once(name, data_dir, params, signal_mode="dim0",
             out_dir="results", csv_override=None, edge_guard=None):
    period = SIDEREAL_DAYS[name]
    planet_dir = f"{out_dir}/{name}"; ensure_dir(planet_dir)

    # choose CSV
    csv_path = csv_override if csv_override else find_best_csv(data_dir, name)
    if not os.path.exists(csv_path):
        raise SystemExit(f"Missing CSV for {name}: expected {csv_path}")
    print(f"Using CSV: {csv_path}")

    # per-file provenance
    y0, y1 = _infer_span_years_from_path(csv_path)
    date0 = _infer_start_date_from_path(csv_path)
    csv_sha = file_sha256(csv_path)
    csv_mtime_utc = _dt.datetime.utcfromtimestamp(os.path.getmtime(csv_path)).isoformat() + "Z"
    hdr_meta = parse_horizons_header_metadata(csv_path)

    Psi = build_psi_from_csv(csv_path)

    # detectors
    peaks, f1, C, Cs = detect_collapses(Psi, params["smooth"], params["z"], params["min_sep"], mode=signal_mode)
    peri, R, Rs = detect_perihelia(Psi, params["smooth"], params["peri_minsep"])

    # edge guard on perihelia (for matching + plots)
    eg = edge_guard if edge_guard is not None else GLOBAL_EDGE_GUARD
    if eg is None:
        eg = params.get("match_window", 0)

    nT = Psi.shape[0]
    lo = 0
    hi = nT - 1
    if eg and peri.size:
        peri = peri[(peri >= eg) & (peri <= (nT - 1 - eg))]
        lo = eg; hi = nT - 1 - eg

    # matching
    offsets, matched, matched_pi = match_offsets(peri, peaks, params["match_window"])

    # --- matched_years from peri indices ---
    matched_years = []
    if matched_pi.size:
        peri_dates = date0 + peri.astype("timedelta64[D]")
        chosen_dates = peri_dates[matched_pi]
        years = chosen_dates.astype("datetime64[Y]").astype(int) + 1970
        matched_years = years.astype(int).tolist()

    # --- phi series per match (wrapped Δt / P) ---
    phi_series = []
    if offsets.size:
        phi_series = [float(wrap_phase_fraction(x, period)) for x in offsets]

    # lag sweep
    lag_bound = max(60, int(round(min(params["match_window"], 0.4*period))))
    best_lag, best_r, (lags, rs) = lag_sweep_corr(Cs, R, lag_bound)

    # stats
    Np, Nk, Nm = int(peri.size), int(peaks.size), int(offsets.size)
    if Nm == 0:
        mean = std = median = mn = mx = np.nan
        coverage = 0.0
    else:
        mean = float(np.mean(offsets))
        std  = float(np.std(offsets, ddof=1)) if offsets.size > 1 else 0.0
        median = float(np.median(offsets))
        mn, mx = float(np.min(offsets)), float(np.max(offsets))
        coverage = Nm / max(Np, 1)

    phase_frac = wrap_phase_fraction(mean, period) if np.isfinite(mean) else np.nan
    strength = coverage * np.exp(-(std if np.isfinite(std) else 1e6) / period) if coverage > 0 else 0.0

    # --- Safe-fill Horizons header defaults (if CSV header lacks them) ---
    if not hdr_meta.get("center"):
        hdr_meta["center"] = "Sun (10)"
    if not hdr_meta.get("frame"):
        hdr_meta["frame"] = "Ecliptic of J2000.0, GEOMETRIC"
    if not hdr_meta.get("units"):
        hdr_meta["units"] = "AU, AU/day"
    if not hdr_meta.get("step"):
        hdr_meta["step"] = "1440 min (1 day)"

    # persist arrays under dedicated names (avoids formatting None issues downstream)
    stats_matched_dt = offsets.tolist() if offsets.size else []
    stats_matched_years = matched_years
    stats_phi_series = phi_series

    # ------------------- PLOTTING -------------------
    plot_overlay(name, Cs, peri, peaks, f"{planet_dir}/overlay.png",
                 matched=matched, lo=lo, hi=hi)
    plot_r_overlay(name, R, peri, f"{planet_dir}/r_overlay.png",
                   lo=lo, hi=hi, smooth=params.get("smooth", 5))
    plot_zoom(name, Cs, peri, matched, f"{planet_dir}/zoom.png")
    if offsets.size:
        plot_hist(name, offsets, params["match_window"], f"{planet_dir}/phase_hist.png")
    yd = yearly_drift(peri, offsets)
    if yd is not None:
        yearly, roll = yd
        plot_drift(name, yearly, roll, f"{planet_dir}/drift.png")
    if matched.size:
        W = 20
        spect = []
        for k in matched:
            i0, i1 = k - W, k + W + 1
            if i0 < 0 or i1 > C.size: continue
            seg = C[i0:i1].astype(float)
            n = seg.size; tloc = np.arange(n)
            m, b = np.linalg.lstsq(np.vstack([tloc, np.ones(n)]).T, seg, rcond=None)[0]
            seg = seg - (m * tloc + b)
            spect.append(np.abs(np.fft.rfft(np.hanning(n) * seg)))
        if spect:
            spect = np.vstack(spect)
            mean_mag = spect.mean(axis=0)
            freq = np.fft.rfftfreq(2 * W + 1, d=1.0)
            plot_fft(name, freq, mean_mag, f"{planet_dir}/fft.png")
    H = compute_entropy_series(Psi)
    H_pk = H[peaks[(peaks>=0)&(peaks<H.size)]]
    plot_entropy_hist(name, H, H_pk, f"{planet_dir}/entropy_hist.png")
    shp = sharpness_vals(C, peaks)
    if shp.size:
        plt.figure(figsize=(9,4))
        plt.hist(shp, bins=40)
        plt.xlabel("Sharpness (normalized curvature)"); plt.ylabel("Count")
        plt.title(f"{name.title()} — Collapse sharpness")
        plt.tight_layout(); plt.savefig(f"{planet_dir}/sharpness_hist.png", dpi=150); plt.close()
    plot_lags(name, lags, rs, best_lag, best_r, f"{planet_dir}/lag_sweep.png")

    # ---------- summary.json (now includes arrays + provenance) ----------
    stats = {
        "planet": name,
        "period_days": period,
        "params": params,
        "signal_mode": signal_mode,

        "N_peri": Np,
        "N_peaks": Nk,
        "matched": Nm,
        "coverage": coverage,

        "mean_dt_days": float(mean) if np.isfinite(mean) else None,
        "median_dt_days": float(median) if np.isfinite(median) else None,
        "std_dt_days": float(std) if np.isfinite(std) else None,
        "min_dt_days": float(mn) if np.isfinite(mn) else None,
        "max_dt_days": float(mx) if np.isfinite(mx) else None,
        "phase_frac": float(phase_frac) if np.isfinite(phase_frac) else None,
        "strength": float(strength) if np.isfinite(strength) else None,

        # Arrays for downstream CI/phase stability tables
        "matched_dt": stats_matched_dt,              # list[float]
        "matched_years": stats_matched_years,        # list[int]
        "phi_series": stats_phi_series,              # list[float] (wrapped Δt/P)

        # Lag sweep
        "lag_sweep": {"best_lag_days": int(best_lag), "best_r": float(best_r)},

        # Per-file provenance
        "csv_path": os.path.abspath(csv_path),
        "csv_sha256": csv_sha,
        "csv_mtime_utc": csv_mtime_utc,
        "csv_span_years": {"start_year": y0, "end_year": y1},
        "date0_iso": str(date0.astype("datetime64[D]")),
        "horizons_header": hdr_meta,  # guaranteed to include center/frame/units/step now
    }
    with open(f"{planet_dir}/summary.json","w") as f: json.dump(stats, f, indent=2)

    # return compact row for summary.csv & report.html
    return {
        "planet": name, "period_days": period, "N_peri": Np, "N_peaks": Nk, "matched": Nm,
        "coverage": coverage,
        "mean": mean if np.isfinite(mean) else np.nan,
        "median": median if np.isfinite(median) else np.nan,
        "std": std if np.isfinite(std) else np.nan,
        "min": mn if np.isfinite(mn) else np.nan,
        "max": mx if np.isfinite(mx) else np.nan,
        "phase_frac": phase_frac if np.isfinite(phase_frac) else np.nan,
        "strength": strength if np.isfinite(strength) else np.nan,
    }

# ---------- one-shot QC retry ----------
def qc_retry_if_needed(row, name, data_dir, params, mode):
    """If coverage is poor (<70%) or std is huge (> P/5), retry with relaxed params/mode."""
    period = SIDEREAL_DAYS[name]
    coverage = row["coverage"]
    std = row["std"] if np.isfinite(row["std"]) else 1e9
    needs_retry = (coverage < 0.7) or (std > period/5.0)

    if not needs_retry:
        return row, False

    # Relax: more smoothing, wider matching, lower z
    p2 = params.copy()
    p2["smooth"] = max(params["smooth"], int(round(params["smooth"]*1.5)))
    p2["match_window"] = int(round(params["match_window"]*1.5))
    p2["z"] = min(0.8, params["z"]*0.75)

    # Switch mode if we started on dim0
    mode2 = "pca1" if mode == "dim0" else ("energy" if mode == "pca1" else "energy")

    print("⚠️  Low quality — attempting QC retry with relaxed params / alternate mode...")
    row2 = run_once(name, data_dir, p2, signal_mode=mode2,
                    out_dir="results", csv_override=WINDOWS.get(name),
                    edge_guard=GLOBAL_EDGE_GUARD)
    improved = (row2["coverage"] > row["coverage"]) or (row2["std"] < row["std"])
    if improved:
        print("✅ Retry improved results; keeping retry run.")
        return row2, True
    else:
        print("ℹ️  Retry did not improve; keeping original run.")
        return row, False

# ---------- run context (saved once per run) ----------
def save_run_context(out_dir, args, planets, overrides_path, windows_path):
    ensure_dir(out_dir)
    ctx = {
        "utc_now": _dt.datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "argv": sys.argv,
        "planets": planets,
        "edge_guard": args.edge_guard,
        "mode_initial": args.mode,
        "data_dir": os.path.abspath(args.data_dir),
        "out_dir": os.path.abspath(args.out_dir),
    }
    # git info (if available)
    ctx.update(get_git_info())

    # Attach overrides/windows metadata + hashes for provenance
    if overrides_path and os.path.exists(overrides_path):
        ctx["overrides_file"] = os.path.abspath(overrides_path)
        ctx["overrides_sha256"] = file_sha256(overrides_path)
    else:
        ctx["overrides_file"] = None
    if windows_path and os.path.exists(windows_path):
        ctx["windows_file"] = os.path.abspath(windows_path)
        ctx["windows_sha256"] = file_sha256(windows_path)
    else:
        ctx["windows_file"] = None

    with open(os.path.join(out_dir, "run_context.json"), "w") as f:
        json.dump(ctx, f, indent=2)
    return ctx

# ---------- main ----------
def main():
    global OVERRIDES, WINDOWS, GLOBAL_EDGE_GUARD, RUN_CONTEXT

    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-guard", type=int, default=None,
                    help="Guard perihelia this many days from file edges (default: match_window)")
    ap.add_argument("--planets", nargs="+", required=True,
                    help="planet names or 'all'")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--mode", default="dim0", choices=["dim0","pca1","energy"],
                    help="signal mode for first pass (QC may switch if needed)")
    ap.add_argument("--overrides", default=None,
                    help="JSON with per-planet params (smooth/min_sep/peri_minsep/match_window/z/mode)")
    ap.add_argument("--windows", default=None,
                    help="JSON mapping planet → CSV path to force specific files")
    args = ap.parse_args()

    # load overrides/windows JSON (optional)
    OVERRIDES = OVERRIDES_PRESET.copy()
    if args.overrides:
        with open(args.overrides, "r") as f:
            OVERRIDES.update(json.load(f))
    WINDOWS = {}
    if args.windows:
        with open(args.windows, "r") as f:
            WINDOWS = json.load(f)

    GLOBAL_EDGE_GUARD = args.edge_guard

    # planet list
    if len(args.planets) == 1 and args.planets[0].lower() == "all":
        planets = list(SIDEREAL_DAYS.keys())
    else:
        planets = [p.lower() for p in args.planets]

    ensure_dir(args.out_dir)

    # Save run context once per run (for provenance)
    RUN_CONTEXT = save_run_context(args.out_dir, args, planets, args.overrides, args.windows)

    rows = []
    for name in planets:
        if name not in SIDEREAL_DAYS:
            print(f"Skipping unknown planet: {name}")
            continue

        # base autoscaled params
        params = autoscale_params(name)

        # apply JSON overrides if present
        if name in OVERRIDES:
            params.update({k: v for k, v in OVERRIDES[name].items() if k != "mode"})
            mode0 = OVERRIDES[name].get("mode", args.mode)
            print(f"ℹ️  Using overrides for {name}: {params} | mode={mode0}")
        else:
            mode0 = args.mode

        print(f"\n=== Running {name.title()} ===")
        print(f"Period={SIDEREAL_DAYS[name]:.3f} d | auto-params: {params} | mode={mode0}")

        csv_override = WINDOWS.get(name)

        try:
            row = run_once(name, args.data_dir, params, signal_mode=mode0,
                           out_dir=args.out_dir, csv_override=csv_override,
                           edge_guard=GLOBAL_EDGE_GUARD)
            # QC retry if needed
            row, _ = qc_retry_if_needed(row, name, args.data_dir, params, mode0)
            rows.append(row)

            # console summary
            if row["matched"] > 0:
                print(f"✅ {name.title()}: matched {row['matched']}/{row['N_peri']} perihelia, "
                      f"mean Δt={row['mean']:+.3f} d, σ={row['std']:.3f} d "
                      f"(min/max {row['min']:+.1f},{row['max']:+.1f})")
            else:
                print(f"⚠️  {name.title()}: no matched events with current settings.")

        except SystemExit as e:
            print(str(e))
        except Exception:
            traceback.print_exc()

    if not rows:
        print("No planets processed; exiting.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    write_html_report(planets, rows, os.path.join(args.out_dir, "report.html"), root=args.out_dir)
    print(f"\n📄 Wrote {csv_path}")
    print(f"🌐 Wrote {os.path.join(args.out_dir, 'report.html')}")

if __name__ == "__main__":
    main()