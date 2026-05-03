#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_recompute.py
===================
Standalone script that re-loads Allantis MT + TENSION daily and prints ALL
the numbers needed to populate the hardcoded sections of the
LIBERATION_SCORE_ALLANTIS dashboard.

Print format is "copy-paste-ready" — produces strings that go directly into
the HTML rules block, conditional during-trade list, year stability text,
and Section 7 callout.

Run:
    python verify_recompute.py

Cross-check against generate_evidence.py output (evidence.json) — Spearman r
at d030 + decile spread + year stability counts MUST match within rounding.

Independent of generate_evidence.py: re-loads data with different code path
(no shared imports, intentional duplication for audit purposes).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ============================== CONFIG ==============================

ALLANTIS_CSV = Path(
    r"C:\Users\Administrator\Desktop\BULK OPTIONSTRAT\ESTRATEGIAS\Allantis\LIVE"
    r"\[MAIN RANKEO MT]_combined_ALLANTIS_ALLDAYS.csv"
)
TENSION_DAILY_CSV = Path(
    r"C:\Users\Administrator\Desktop\BULK OPTIONSTRAT\ESTRATEGIAS\Skew"
    r"\SURFACE_SKEW_CONCAVITY_COMPONENTS_DAILY.csv"
)

REGIME_FAV = 80.0
REGIME_ADV = 20.0
REF_HORIZON = 30
SPX_THR = 3.0
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42

# ============================== LOAD ==============================

def stats(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return dict(N=0, mean=np.nan, median=np.nan, WR=np.nan, PF=np.nan)
    gw = s[s > 0].sum(); gl = -s[s < 0].sum()
    return dict(
        N=len(s),
        mean=float(s.mean()),
        median=float(s.median()),
        WR=100.0 * float((s > 0).mean()),
        PF=gw/gl if gl > 0 else np.nan,
    )


def main():
    print("=" * 100)
    print("LIBERATION_SCORE_ALLANTIS — verify_recompute (cross-check vs generate_evidence.py)")
    print("=" * 100)

    # Load Allantis (BOM UTF-8)
    df = pd.read_csv(ALLANTIS_CSV, encoding="utf-8-sig", low_memory=False)
    df.columns = [c.replace("﻿", "") for c in df.columns]
    df = df.rename(columns={"dia": "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trade_date"])
    print(f"\n[1] Allantis loaded: {len(df):,} rows ({df['trade_date'].min().date()} -> {df['trade_date'].max().date()})")

    # Load TENSION daily
    ten = pd.read_csv(TENSION_DAILY_CSV, low_memory=False)
    ten.columns = [c.replace("﻿", "") for c in ten.columns]
    ten = ten[["trade_date", "TENSION_3WAY_MIN"]].copy()
    ten["trade_date"] = pd.to_datetime(ten["trade_date"], errors="coerce").dt.normalize()
    ten = ten.dropna(subset=["trade_date", "TENSION_3WAY_MIN"]).drop_duplicates("trade_date")
    print(f"[2] TENSION daily: {len(ten):,} rows")

    # Join
    n_before = len(df)
    df = df.merge(ten, on="trade_date", how="left")
    n_with = df["TENSION_3WAY_MIN"].notna().sum()
    print(f"[3] Joined: {n_with:,}/{n_before:,} have TENSION ({100*n_with/n_before:.1f}%)")
    df = df.dropna(subset=["TENSION_3WAY_MIN"])
    n_unfiltered = len(df)

    # Sanity check SPX filter sign
    spx = pd.to_numeric(df["SPX_chg_pct_d030"], errors="coerce")
    print(f"[4] SPX_chg_pct_d030 stats:  min={spx.min():.2f}  max={spx.max():.2f}  std={spx.std():.2f}  (units: pp expected)")
    if spx.abs().max() < 1.0:
        print("    [WARN] looks like decimal form; convention is pp")

    # Apply filter
    mask = spx.abs() <= SPX_THR
    df_full = df.copy()
    df = df[mask].copy()
    n_filtered = len(df)
    print(f"[5] Filtered |SPX|<={SPX_THR}%: {n_filtered:,}/{n_unfiltered:,} retained ({100*n_filtered/n_unfiltered:.1f}%)")

    # =============== Baseline universe (filtered) ===============
    print("\n" + "=" * 100)
    print("BASELINE UNIVERSO (Allantis filtered |SPX|<=3%)")
    print("=" * 100)
    pnl_ref = f"PnL_d{REF_HORIZON:03d}_mediana"
    pnl_50 = "PnL_d050_mediana"
    s30 = stats(df[pnl_ref])
    s50 = stats(df[pnl_50])
    y_min = df["trade_date"].min().year
    y_max = df["trade_date"].max().year
    print(f"\n[BASELINE]  N={s30['N']:,}  range={y_min}-{y_max}")
    print(f"  d030: WR={s30['WR']:.1f}%  mean={s30['mean']:+.2f}  PF={s30['PF']:.2f}")
    print(f"  d050: WR={s50['WR']:.1f}%  mean={s50['mean']:+.2f}  PF={s50['PF']:.2f}")
    print(f"\n[HTML rules-baseline line]")
    print(f"  Baseline universo Allantis MT ({s30['N']:,} trades, {y_min}-{y_max}, |SPX|<=3%):")
    print(f"  WR {s30['WR']:.0f}% / {s50['WR']:.0f}%  *  mean +{s30['mean']:.0f} / +{s50['mean']:.0f} pts  *  PF {s30['PF']:.1f} / {s50['PF']:.1f}")
    print(f"  (d030 / d050)")

    # =============== Regime split ===============
    print("\n" + "=" * 100)
    print("REGIME SPLIT — FAV/NEU/ADV (filtered)")
    print("=" * 100)
    regime_results = {}
    for label, mask_reg in [
        ("FAVORABLE", df["TENSION_3WAY_MIN"] >= REGIME_FAV),
        ("NEUTRAL",   (df["TENSION_3WAY_MIN"] > REGIME_ADV) & (df["TENSION_3WAY_MIN"] < REGIME_FAV)),
        ("ADVERSO",   df["TENSION_3WAY_MIN"] <= REGIME_ADV),
    ]:
        sub = df[mask_reg]
        s30r = stats(sub[pnl_ref])
        s50r = stats(sub[pnl_50])
        pct_univ = 100.0 * len(sub) / s30["N"]
        regime_results[label] = (s30r, s50r, pct_univ)
        print(f"\n[{label}] N={len(sub):,} ({pct_univ:.0f}% del universo)")
        print(f"  d030: WR={s30r['WR']:.1f}%  mean={s30r['mean']:+.2f}  PF={s30r['PF']:.2f}")
        print(f"  d050: WR={s50r['WR']:.1f}%  mean={s50r['mean']:+.2f}  PF={s50r['PF']:.2f}")

    # Build vs-baseline ratios for HTML rules table
    print("\n[HTML rules-table cells (vs baseline universo)]")
    print(f"  {'Banda':<15} {'WR vs univ':<20} {'mean vs univ':<20} {'PF vs univ':<20} {'% univ':<10}")
    for label in ["FAVORABLE", "NEUTRAL", "ADVERSO"]:
        s30r, s50r, pct = regime_results[label]
        # WR delta in pp
        wr_d30 = s30r['WR'] - s30['WR']
        wr_d50 = s50r['WR'] - s50['WR']
        # mean ratio
        m_d30 = s30r['mean'] / s30['mean'] if s30['mean'] != 0 else np.nan
        m_d50 = s50r['mean'] / s50['mean'] if s50['mean'] != 0 else np.nan
        # PF ratio
        pf_d30 = s30r['PF'] / s30['PF'] if (np.isfinite(s30['PF']) and s30['PF'] > 0) else np.nan
        pf_d50 = s50r['PF'] / s50['PF'] if (np.isfinite(s50['PF']) and s50['PF'] > 0) else np.nan
        print(f"  {label:<15} {wr_d30:+.0f}pp / {wr_d50:+.0f}pp     "
              f"{m_d30:.1f}x / {m_d50:.1f}x       "
              f"{pf_d30:.1f}x / {pf_d50:.1f}x       "
              f"{pct:.0f}%")

    # =============== Conditional during-trade ===============
    print("\n" + "=" * 100)
    print("CONDITIONAL DURING-TRADE  (entry FAV)")
    print("=" * 100)
    fav_entries = df[df["TENSION_3WAY_MIN"] >= REGIME_FAV].copy()
    print(f"FAV entries (filtered): N={len(fav_entries):,}")

    # Need TENSION at entry+30 calendar days. Use asof lookup.
    ten_full = pd.read_csv(TENSION_DAILY_CSV, low_memory=False)
    ten_full.columns = [c.replace("﻿", "") for c in ten_full.columns]
    ten_full = ten_full[["trade_date", "TENSION_3WAY_MIN"]].copy()
    ten_full["trade_date"] = pd.to_datetime(ten_full["trade_date"], errors="coerce").dt.normalize()
    ten_full["TENSION_3WAY_MIN"] = pd.to_numeric(ten_full["TENSION_3WAY_MIN"], errors="coerce")
    ten_full = ten_full.dropna().drop_duplicates("trade_date").sort_values("trade_date")
    ts_dates = ten_full["trade_date"].to_numpy(dtype="datetime64[ns]")
    ts_vals = ten_full["TENSION_3WAY_MIN"].to_numpy(dtype=float)

    def asof(date):
        if pd.isna(date):
            return np.nan
        pos = np.searchsorted(ts_dates, np.datetime64(date), side="right") - 1
        if pos < 0:
            return np.nan
        return float(ts_vals[pos])

    # Of FAV entries, how many have TENSION >=80 still at entry+30 days?
    if len(fav_entries) > 0:
        future_dates = fav_entries["trade_date"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(REF_HORIZON, "D")
        ten_at_30 = np.array([asof(pd.Timestamp(d)) for d in future_dates])
        still_high = ten_at_30 >= REGIME_FAV
        pct_remain_high = 100.0 * np.mean(still_high)
        n_remain = int(np.sum(still_high))

        print(f"\n[A] Of FAV entries, {n_remain:,}/{len(fav_entries):,} ({pct_remain_high:.0f}%) have TENSION>=80 still at entry+{REF_HORIZON}d")

        # Conditional WR/mean/PF on those
        sub_remain = fav_entries.iloc[still_high]
        srem = stats(sub_remain[pnl_50])
        ratio_mean = srem['mean'] / s50['mean'] if s50['mean'] != 0 else np.nan
        print(f"    d050 conditional: WR={srem['WR']:.0f}%  mean={srem['mean']:+.0f}pts ({ratio_mean:.1f}x universo)  PF={srem['PF']:.0f} absolute")
        print(f"    [HTML]: PUT_SKEW (wait, TENSION) sigue >=80 al d{REF_HORIZON:03d} ({pct_remain_high:.0f}% de FAV entries):")
        print(f"            WR {srem['WR']:.0f}% d050, mean {ratio_mean:.1f}x universo, PF {srem['PF']:.0f} absoluto")

        # Of FAV entries, how many fall ≤20 some day in next 50?
        # Quick approximation: sample TENSION at entry+10, +20, +30, +40, +50
        any_drop = np.zeros(len(fav_entries), dtype=bool)
        for dt_offset in [10, 20, 30, 40, 50]:
            future_dates = fav_entries["trade_date"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(dt_offset, "D")
            ten_x = np.array([asof(pd.Timestamp(d)) for d in future_dates])
            any_drop |= (ten_x <= REGIME_ADV)
        pct_drop = 100.0 * np.mean(any_drop)
        n_drop = int(np.sum(any_drop))

        sub_drop = fav_entries.iloc[any_drop]
        sdrop = stats(sub_drop[pnl_50])
        ratio_drop_mean = sdrop['mean'] / srem['mean'] if (srem['mean'] != 0 and np.isfinite(srem['mean'])) else np.nan
        print(f"\n[B] Of FAV entries, {n_drop:,}/{len(fav_entries):,} ({pct_drop:.0f}%) see TENSION<=20 algun dia en proximos 50d")
        print(f"    d050 conditional sub-cohort: N={sdrop['N']:,} mean={sdrop['mean']:+.2f} ({ratio_drop_mean:.2f}x mean del cohort 'remain high')")

    # =============== Year stability ===============
    print("\n" + "=" * 100)
    print("YEAR STABILITY 2019-2025  (FAV cohort vs baseline, filtered)")
    print("=" * 100)
    df["year"] = df["trade_date"].dt.year
    fav_underperform = []
    fav_zero = []
    print(f"\n  {'Year':<6} {'N_fav':>7} {'mean_fav':>10} {'WR_fav':>8} {'baseline':>10} {'WR_base':>8}  {'verdict':<14}")
    for y, g in df.groupby("year"):
        fav_g = g[g["TENSION_3WAY_MIN"] >= REGIME_FAV]
        s_fav = stats(fav_g[pnl_ref])
        s_base = stats(g[pnl_ref])
        if s_fav['N'] == 0:
            verdict = "SIN FAV"
            fav_zero.append(y)
        elif s_fav['mean'] < s_base['mean']:
            verdict = "FAV PEOR"
            fav_underperform.append((y, s_fav['mean'], s_base['mean']))
        else:
            verdict = "FAV bate"
        print(f"  {y:<6} {s_fav['N']:>7} {s_fav['mean']:>+10.2f} {s_fav['WR']:>7.1f}% {s_base['mean']:>+10.2f} {s_base['WR']:>7.1f}%  {verdict}")

    # Count years FAV beats baseline
    n_years = df['year'].nunique()
    n_fav_beats = 0
    for y, g in df.groupby("year"):
        fav_g = g[g["TENSION_3WAY_MIN"] >= REGIME_FAV]
        s_fav = stats(fav_g[pnl_ref])
        s_base = stats(g[pnl_ref])
        if s_fav['N'] > 0 and s_fav['mean'] > s_base['mean']:
            n_fav_beats += 1
    print(f"\n[HTML caveat line]: FAV bate al universo en {n_fav_beats} de {n_years} anios.")
    if fav_underperform:
        details = ", ".join([f"{y} (FAV mean {m:+.0f})" for y, m, b in fav_underperform])
        print(f"  Fallos: {details}")
    if fav_zero:
        details = ", ".join([f"{y} (sin FAV)" for y in fav_zero])
        print(f"  Sin FAV: {details}")

    # =============== Spearman headline (filtered) ===============
    print("\n" + "=" * 100)
    print(f"HEADLINE SPEARMAN d{REF_HORIZON:03d} (filtered |SPX|<=3%)")
    print("=" * 100)
    sub = df[["TENSION_3WAY_MIN", pnl_ref]].dropna()
    r = spearmanr(sub["TENSION_3WAY_MIN"], sub[pnl_ref]).correlation
    # Bootstrap CI95 (rank-once)
    score_rank = pd.Series(sub["TENSION_3WAY_MIN"].values).rank().to_numpy(dtype=float)
    pnl_rank = pd.Series(sub[pnl_ref].values).rank().to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + REF_HORIZON)
    sp_vals = np.full(BOOTSTRAP_N, np.nan, dtype=float)
    n = len(sub)
    for b in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        sr = score_rank[idx]; pr = pnl_rank[idx]
        sx = sr.std(); sy = pr.std()
        if sx > 0 and sy > 0:
            sp_vals[b] = float(np.mean((sr - sr.mean()) * (pr - pr.mean())) / (sx * sy))
    ci_lo = float(np.nanpercentile(sp_vals, 2.5))
    ci_hi = float(np.nanpercentile(sp_vals, 97.5))
    print(f"  N={n:,}  Spearman r = {r:+.4f}  CI95% = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"\n[Cross-check vs generate_evidence.py]: should match within rounding")

    # =============== Section 7 spot-checks ===============
    print("\n" + "=" * 100)
    print("SECTION 7 SPOT-CHECKS (window-forward, unfiltered df)")
    print("=" * 100)
    # We want HIGH/LOW spreads at (t=0,x=20,sin filtro), (t=0,x=50,sin filtro), (t=20,x=50,sin filtro), (t=20,x=50,|SPX|<=2%)
    # Reuse df_full (unfiltered for window-forward). Apply window-SPX filters per cell.

    df_wf = df_full.copy()
    n_wf = len(df_wf)
    entry_dates = df_wf["trade_date"].to_numpy(dtype="datetime64[ns]")

    def wf_cell(t, x, spx_filter):
        if t == 0:
            ps_t = df_wf["TENSION_3WAY_MIN"].to_numpy(dtype=float)
            pnl_t = np.zeros(n_wf, dtype=float)
            spx_t = np.zeros(n_wf, dtype=float)
        else:
            obs_dates = entry_dates + np.timedelta64(t, "D")
            ps_t = np.array([asof(pd.Timestamp(d)) for d in obs_dates])
            pnl_t_col = f"PnL_d{t:03d}_mediana"
            spx_t_col = f"SPX_chg_pct_d{t:03d}"
            pnl_t = pd.to_numeric(df_wf.get(pnl_t_col, pd.Series(dtype=float)), errors="coerce").to_numpy() if pnl_t_col in df_wf.columns else np.full(n_wf, np.nan)
            spx_t = pd.to_numeric(df_wf.get(spx_t_col, pd.Series(dtype=float)), errors="coerce").to_numpy() if spx_t_col in df_wf.columns else np.zeros(n_wf)
        tx = t + x
        pnl_tx_col = f"PnL_d{tx:03d}_mediana"
        spx_tx_col = f"SPX_chg_pct_d{tx:03d}"
        if pnl_tx_col not in df_wf.columns:
            return None
        pnl_tx = pd.to_numeric(df_wf[pnl_tx_col], errors="coerce").to_numpy()
        spx_tx = pd.to_numeric(df_wf.get(spx_tx_col, pd.Series(dtype=float)), errors="coerce").to_numpy() if spx_tx_col in df_wf.columns else np.zeros(n_wf)
        delta_pnl = pnl_tx - pnl_t
        spx_window = spx_tx - spx_t
        if spx_filter == "sin filtro":
            sm = np.ones(n_wf, dtype=bool)
        elif spx_filter == "|SPX|<=3%":
            sm = np.abs(spx_window) <= 3.0
        elif spx_filter == "|SPX|<=2%":
            sm = np.abs(spx_window) <= 2.0
        else:
            return None
        ok = sm & ~np.isnan(ps_t) & ~np.isnan(delta_pnl)
        h = delta_pnl[ok & (ps_t >= REGIME_FAV)]
        l = delta_pnl[ok & (ps_t <= REGIME_ADV)]
        if len(h) < 5 or len(l) < 5:
            return None
        return dict(
            high_mean=float(np.mean(h)), high_N=len(h),
            low_mean=float(np.mean(l)), low_N=len(l),
            spread=float(np.mean(h) - np.mean(l)),
        )

    print(f"\n  (t, x, filter)        HIGH mean (N)        LOW mean (N)         spread")
    print(f"  {'-'*70}")
    for t, x, flt in [(0, 20, "sin filtro"), (0, 50, "sin filtro"),
                      (20, 50, "sin filtro"), (20, 50, "|SPX|<=2%"),
                      (40, 50, "sin filtro"), (40, 50, "|SPX|<=2%")]:
        c = wf_cell(t, x, flt)
        if c is None:
            print(f"  ({t}, +{x}d, {flt:<12}): no data")
            continue
        print(f"  ({t}, +{x}d, {flt:<12}): {c['high_mean']:>+8.2f} ({c['high_N']:>6,})    {c['low_mean']:>+8.2f} ({c['low_N']:>6,})    {c['spread']:>+8.2f}")

    # =============== KEY HEADLINES PRE-FORMATTED ===============
    print("\n" + "=" * 100)
    print("COPY-PASTE READY HTML INSERTS")
    print("=" * 100)
    print(f"\n[L342 subtitulo]:")
    print(f'  <div class="subtitle">Score TENSION_3WAY_MIN aplicado a Allantis MT - SPX 10:30 ET, DTE 60</div>')
    print(f"\n[L381 baseline strong line]:")
    print(f'  <strong>Baseline universo Allantis MT</strong> ({s30["N"]:,} trades, {y_min}-{y_max}, |SPX|<=3%):')
    print(f'  WR {s30["WR"]:.0f}% / {s50["WR"]:.0f}%  &nbsp;&middot;&nbsp;  mean +{s30["mean"]:.0f} / +{s50["mean"]:.0f} pts  &nbsp;&middot;&nbsp;  PF {s30["PF"]:.1f} / {s50["PF"]:.1f}')
    print(f'  <span class="h">(d030 / d050)</span>')
    print(f"\n[L484 evidencia regime split section text]:")
    print(f"  PnL d030 y d050 condicionado a FAVORABLE / NEUTRAL / ADVERSO.")
    print(f"\n[L516 deciles section title]:")
    print(f"  PnL d030 por decil del score")

    print("\n" + "=" * 100)
    print("DONE — copy numbers above into index.html, then run generate_evidence.py")
    print("=" * 100)


if __name__ == "__main__":
    main()
