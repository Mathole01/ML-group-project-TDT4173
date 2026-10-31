# TDT4173 Course Project — Hydropower Unit Commitment (SINTEF / Tokke–Vinje)

Predict, for each weekly scheduling case, the probability that each generating unit is **ON**
at each of the 168 hours. Metric: **micro-averaged ROC-AUC**.

- **Group number:** `[TODO]`
- **Kaggle team name:** `[TODO — must contain the group number, e.g. "[34] A Fantastic Team"]`
- **Members:** `Name (student-id)`, `Name (student-id)`, …

---

## The task in one paragraph

Each **case** = one `Run No`, a 168-hour week starting at `starttime`. For every case we must
output `14 units × 168 hours = 2352` probabilities in a single wide row.
- **Train:** `Run No 1–2922` (2015-01-01 → ~2022), ground truth in `Unit_commitment_decisions.csv`.
- **Test:** `Run No 2923–3647` (2023-01-01 → 2024-12-25), mapping in `prediction_mapping.csv`,
  output columns listed in `sample_submission.csv`.

## Data (all under `sintef_project/data/`)

| File | Granularity | Notes |
|------|-------------|-------|
| `kernel/Historical_day_ahead_price_2015_2025.csv` | hourly | EUR/MWh |
| `kernel/Historical_inflow_1958_2025.csv` | hourly | 17 reservoirs + 1 river, m³/s |
| `kernel/Historical_volume_2015_2024.csv` | **daily** | 17 reservoirs, Mm³ (start of day). Hyljelihyl & Vatjern assumed 50% full |
| `kernel/Synthetic_water_value_2015_2024.csv` | **daily** | 17 reservoirs, EUR/MWh |
| `kernel/Unit_commitment_decisions.csv` | per case | ground truth `result_committed_<Plant>_<Unit>_t<hour>` |
| `extended/Constraint_min_volume.csv` | hourly | only 3 reservoirs ever restricted |
| `extended/Constraint_min_flow.csv` | hourly | only 6 rivers ever restricted |
| `extended/Tokke_Vinje_topology.yaml` | static | reservoir max_vol, plants, units, watercourse topology |
| `extended/Tokke_topology.pdf` | static | map of the watercourse |

14 units across 9 plants: Hogga_G1, Kjela_G1, Vesle Kjela_G1, Byrte_G1, Haukeli_G1, Lio_G1,
Songa_G1, Tokke_G1–G4, Vinje_G1–G3.

---

## What to submit (Canvas + Kaggle)

1. **2 selected predictions** on Kaggle (private LB decides grade; pick your 2 before the deadline).
2. **`Short notebook 1.ipynb`** and **`Short notebook 2.ipynb`** — minimal, reproducible, start from
   raw data, run offline, ≤12 h on ≤4 cores / 32 GB RAM. At least one must reproduce your best score.
3. **`Report.ipynb`** (or `Report.pdf`) — full write-up **with saved outputs** (graders do not run it).

**Deadline:** 2026-11-08 22:00 Oslo. Late (−30): 2026-11-11 14:00. Register your exam.

---

## How to get an A (89–100)

Points = **base points** (from VTs defeated) **+ deductions**. An A needs a strong model *and* a
report that avoids every deduction. Checklist — each item has a clearly titled section in `Report.ipynb`:

- [ ] **Beat as many Virtual Teams as possible** (base points; VTs frozen 2026-10-18). This is the
      main lever — the model quality drives the grade.
- [ ] **EDA** (−3 if missing): do **4+** of — domain knowledge, data sanity, how data was generated,
      per-feature exploration, feature pairs/groups, feature cleanup.
- [ ] **≥2 predictor types** (−3 if only one): e.g. LightGBM **and** RandomForest (and/or XGBoost / NN).
      Short notebooks may each use a single model; the *report* must show the comparison.
- [ ] **Feature engineering** (−3 if missing): ≥1 selection and/or extraction technique, documented.
- [ ] **Model interpretation** (−3 if missing): ≥1 of feature importance / PDP / LIME / SHAP.
- [ ] **Clear section titles** in the report so graders can find each of the above.
- [ ] Every hyperparameter hard-coded in a short notebook is **explained** (how it was chosen) in the report.

See `GAMEPLAN.md` for the recommended modeling strategy.

---

## Reproducing

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook   # run "Short notebook 1.ipynb" top-to-bottom → writes submission.csv
```

The notebooks read raw data from `sintef_project/data/` via the `DATA_DIR` config cell — adjust that
one constant if graders place the data elsewhere.
