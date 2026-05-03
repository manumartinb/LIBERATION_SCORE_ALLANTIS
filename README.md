# LIBERATION_SCORE Allantis Dashboard

Pagina web con grafico interactivo del score **TENSION_3WAY_MIN** (componente
regime/daily de la familia LIBERATION) validado sobre **Allantis MT** (SPX, DTE
60, snapshot 10:30 ET).

URL publica: https://manumartinb.github.io/LIBERATION_SCORE_ALLANTIS/

Hermano del dashboard `LIBERATION_SCORE_BATMAN_LT` (mismo score, validacion
contra Batman LT). El header chart es identico (mismo TENSION daily); la
seccion de evidencia es lo que cambia.

## Que muestra

- Linea principal: **TENSION_3WAY_MIN** (percentil 252d, minimo de los 3 subcomponentes)
- Bandas coloreadas: FAVORABLE (>=80), NEUTRAL (20-80), ADVERSO (<=20)
- 3 subcomponentes toggleables (legend click): curv 15-30-45, slope 10-40, skew 25-50
- Selector de rango: 30D / 90D / 1A / 3A / All

## Pipeline

Actualizacion automatica diaria via `V0.[PERMA] MASTER_DAILY_PIPELINE.py`,
**Step 6** (independiente de Steps 3-5 que sirven los otros 3 dashboards):

```
V18 (streaming) -> V8.0 (SKEW pipeline + Telegram) -> update_dashboard.py (este repo)
```

`update_dashboard.py` lee `SURFACE_SKEW_CONCAVITY_COMPONENTS_DAILY.csv`,
regenera `data.json` y hace `git push` a este repo. GitHub Pages sirve el HTML.

Solo `data.json` se actualiza diariamente. `evidence/*.png` y `evidence.json`
son estaticos &mdash; se regeneran manualmente con `generate_evidence.py`.

## Fuente de datos

- **Score daily**: `Skew/SURFACE_SKEW_CONCAVITY_COMPONENTS_DAILY.csv` (col `TENSION_3WAY_MIN`)
- **Trades validacion**: `Allantis/LIVE/[MAIN RANKEO MT]_combined_ALLANTIS_ALLDAYS.csv`
  (~145,870 trades, 2019-2025, BOM UTF-8, col `dia` &rarr; `trade_date`)

## Decisiones metodologicas Allantis

- **Horizonte de referencia**: d030 (canonico Allantis, NO d020 como Batman LT)
- **Filtro SPX headline**: `|SPX_chg_pct_d030| <= 3%` (canonico Allantis)
- **Bandas FAV/NEU/ADV**: 80/20 percentile (mismas que Batman LT)
- **Bootstrap CI95**: n=2000, seed=42 (mismo que Batman LT)
- **Section 7 window-forward**: 3 filtros internos (sin filtro / |SPX|<=3% / |SPX|<=2%)

## Seccion de evidencia estadistica

Bajo la grafica diaria, la pagina muestra 7 secciones de evidencia (Section 8
TRIPLE de Batman LT NO aplica a Allantis, BQI/TS son scores Batman-only):

1. Concepto: que mide TENSION_3WAY_MIN
2. Metodologia: dataset Allantis, filtro SPX, bootstrap params
3. Predictividad: Spearman r vs PnL por horizonte d001-d049 + bootstrap
4. Deciles D1-D10 + spread D10-D1 por horizonte
5. Year stability 2019-2025
6. Regime split FAV/NEU/ADV en d030 y d050
7. Window forward + curva continua HIGH vs LOW

Para regenerar evidencia (manual):

```
python generate_evidence.py            # local only
python generate_evidence.py --push     # local + git push
```

Auth: `GH_DASHBOARD_TOKEN` (User scope env var).

## Verificacion

Antes de publicar, correr `verify_recompute.py` (script standalone independiente
de `generate_evidence.py`) para sanity check de TODOS los numeros HTML hardcoded:

```
python verify_recompute.py
```

Cross-check: Spearman r d030 + decile spread + year stability deben matchear
entre `verify_recompute.py` (stdout) y `evidence.json` (output de `generate_evidence.py`)
dentro de rounding.

## Headline numbers (al fork inicial 2026-05-03)

- N filtrado: 53,492 trades / 456 dias (2019-06-25 a 2025-12-23)
- N unfiltered: 138,264 trades
- Spearman r d030 = +0.135 (CI95% [+0.126, +0.143])
- FAV (>=80): N=8,280 (15%), mean +13.91 d030, WR 88.0%, PF 15.89
- NEU (20-80): N=29,897 (56%), mean +6.88 d030, WR 67.1%, PF 3.50
- ADV (<=20): N=15,315 (29%), mean +6.42 d030, WR 66.8%, PF 2.47
- FAV bate al universo en 3/7 anios. Senal mas debil que en Batman LT.

## Troubleshooting

- **No correr entre 12:25 y 12:40 Madrid**: ventana del V0 daily push (riesgo conflict rebase).
- **`utf-8-sig` encoding**: la CSV Allantis lleva BOM. `pd.read_csv` con `encoding="utf-8-sig"`
  + `df.columns = [c.replace("﻿","") for c in df.columns]` defensive.
- **SPX filter sign**: `SPX_chg_pct_d030` esta en PERCENTAGE POINTS, NO en decimal.
  Filtrar `abs() <= 3.0`, NO `<= 0.03`.
