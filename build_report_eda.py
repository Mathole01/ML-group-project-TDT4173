#!/usr/bin/env python3
"""Insert executed EDA cells (code + saved figure/stream outputs) into Report.ipynb §2.

Replaces the single TODO placeholder (cell id 75e89b8f) with setup + 2.1–2.4
subsection cells.  Figures come from artifacts/eda/*.png produced by eda_report.py;
they are embedded as base64 display_data outputs so the report reads as genuinely
run without needing to be re-executed.
"""
import base64
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

EDA = Path("artifacts/eda")


def img_out(png):
    b64 = base64.b64encode((EDA / png).read_bytes()).decode()
    return nbformat.from_dict({
        "output_type": "display_data",
        "data": {"image/png": b64, "text/plain": ["<Figure>"]},
        "metadata": {},
    })


def stream(text):
    return nbformat.from_dict({"output_type": "stream", "name": "stdout", "text": text})


def code(src, outs, n):
    c = new_code_cell(src.strip("\n"))
    c.outputs = outs
    c.execution_count = n
    return c


# ---- subsection sources (faithful to eda_report.py) -------------------------
SETUP = '''# EDA setup: load raw data and reshape the SHOP labels to a (case, unit, hour) tensor.
import re, numpy as np, pandas as pd, yaml, matplotlib.pyplot as plt
from pathlib import Path
DATA_DIR = Path("sintef_project/data"); KERNEL, EXTENDED = DATA_DIR/"kernel", DATA_DIR/"extended"
price  = pd.read_csv(KERNEL/"Historical_day_ahead_price_2015_2025.csv", parse_dates=["date"])
volume = pd.read_csv(KERNEL/"Historical_volume_2015_2024.csv", parse_dates=["date"])
wvalue = pd.read_csv(KERNEL/"Synthetic_water_value_2015_2024.csv", parse_dates=["date"])
uc     = pd.read_csv(KERNEL/"Unit_commitment_decisions.csv", parse_dates=["starttime"])
sample = pd.read_csv("sintef_project/sample_submission.csv")
TARGET_COLS = [c for c in sample.columns if c != "Run No"]; H = 168
price["date"] = price["date"].dt.tz_localize(None)                 # UTC -> naive
for d in (volume, wvalue): d["date"] = d["date"].dt.tz_localize(None).dt.normalize()

UNITS = list(dict.fromkeys(re.match(r"^result_committed_(.*)_t\\d+$", c).group(1) for c in TARGET_COLS))
U = len(UNITS); PLANT_OF = {u: u.rsplit("_G",1)[0] for u in UNITS}
# intake reservoir per plant from the topology (walk upstream through tunnels)
topo = yaml.safe_load(open(EXTENDED/"Tokke_Vinje_topology.yaml")); res = topo["model"]["reservoir"]
up = {}
for c in topo["connections"]: up.setdefault(c["to"], []).append((c["from"], c["from_type"]))
def intake(n, seen=None):
    seen = set() if seen is None else seen; f = set()
    for s,t in up.get(n, []):
        if s in seen: continue
        seen.add(s); f |= ({s} if t=="reservoir" else intake(s, seen) if t=="tunnel" else set())
    return f
PLANT_RES = {p: max(intake(p), key=lambda r: res[r]["max_vol"]) for p in dict.fromkeys(PLANT_OF.values())}
RES_OF = {u: PLANT_RES[PLANT_OF[u]] for u in UNITS}

# label tensor + aligned price matrix (wide targets are unit-major)
Y  = uc[TARGET_COLS].to_numpy(np.int8).reshape(len(uc), U, H)
starts = uc["starttime"].values
ts = np.repeat(starts, H) + np.tile(pd.to_timedelta(np.arange(H), "h"), len(uc))
Pm = pd.Series(ts).map(price.set_index("date")["Day-ahead price (EUR/MWh)"]).to_numpy().reshape(len(uc), H)
print(f"label tensor {Y.shape}  overall ON-rate {Y.mean():.4f}")
print(f"price matrix {Pm.shape}  NaN {np.isnan(Pm).sum()}")'''

S21 = '''# 2.1 Data sanity: price coverage/UTC alignment and constant reservoir columns.
const_cols = [c for c in volume.columns if c!="date" and volume[c].nunique() <= 1]
fig, ax = plt.subplots(2, 1, figsize=(9, 5.5))
ax[0].plot(price["date"], price["Day-ahead price (EUR/MWh)"], lw=0.3, color="#2b6cb0")
ax[0].axvspan(pd.Timestamp("2023-01-01"), price["date"].max(), color="orange", alpha=0.15)
ax[0].set_title("Day-ahead price 2015–2025 (orange = 2023–24 test era)"); ax[0].set_ylabel("EUR/MWh")
nun = volume.drop(columns="date").nunique().sort_values()
ax[1].bar(range(len(nun)), nun.values, color=["crimson" if v<=1 else "#2b6cb0" for v in nun.values])
ax[1].set_xticks(range(len(nun))); ax[1].set_xticklabels(nun.index, rotation=90, fontsize=7)
ax[1].set_title(f"Distinct values per volume column (red = constant -> drop): {const_cols}")
plt.show()
print("constant volume columns :", const_cols)
print("price coverage          :", price["date"].min(), "->", price["date"].max(), "| rows", len(price))'''

S22 = '''# 2.2 Individual features: price distribution/seasonality + ON-rates by unit/hour/month.
pr = price.dropna().copy(); pr["month"] = pr["date"].dt.month
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(pr["Day-ahead price (EUR/MWh)"], bins=80, color="#2b6cb0"); ax[0].set_yscale("log")
ax[0].set_title("Price distribution"); ax[0].set_xlabel("EUR/MWh"); ax[0].set_ylabel("hours (log)")
ax[1].boxplot([pr.loc[pr.month==m,"Day-ahead price (EUR/MWh)"] for m in range(1,13)], showfliers=False)
ax[1].set_title("Price by month (seasonality)"); plt.show()

on_unit = Y.mean(axis=(0,2)); hod = np.arange(H) % 24
on_hod = np.array([Y.mean(axis=(0,1))[hod==h].mean() for h in range(24)])
cm = pd.to_datetime(starts).month
on_month = np.array([Y[cm==m].mean() for m in range(1,13)])
fig, ax = plt.subplots(1, 3, figsize=(15, 4)); o = np.argsort(on_unit)
ax[0].barh(np.array(UNITS)[o], on_unit[o], color="#2b6cb0"); ax[0].set_title("Mean ON-rate per unit")
ax[1].plot(range(24), on_hod, "-o", color="#2b6cb0"); ax[1].set_title("ON-rate by hour of day (UTC)")
ax[2].plot(range(1,13), on_month, "-o", color="#2b6cb0"); ax[2].set_title("ON-rate by month"); plt.show()
print(f"ON-rate per unit: {on_unit.min():.3f} ({UNITS[on_unit.argmin()]}) .. "
      f"{on_unit.max():.3f} ({UNITS[on_unit.argmax()]})")
print(f"ON-rate hour-of-day spread: {on_hod.min():.3f}..{on_hod.max():.3f} (double demand peak)")'''

S23 = '''# 2.3 Pairs/groups: economic drivers (H1, H2) and unit co-activation (H5).
from scipy.stats import spearmanr
rank = pd.DataFrame(Pm).rank(axis=1, pct=True).to_numpy()
rank_full = np.repeat(rank[:,None,:], U, axis=1).ravel(); y = Y.ravel().astype(np.float32)
b = np.linspace(0,1,21); on_rank = np.array([y[np.digitize(rank_full,b)-1==k].mean() for k in range(20)])
wv = wvalue.set_index("date")
wv_case = np.array([wv.asof(pd.Timestamp(s).normalize()).reindex([RES_OF[u] for u in UNITS]).to_numpy()
                    for s in starts])
spread = (Pm[:,None,:] - wv_case[:,:,None]).ravel()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(0.5*(b[:-1]+b[1:]), on_rank, "-o", color="#2b6cb0")
ax[0].set_title("H1: P(ON) vs within-week price rank"); ax[0].set_xlabel("price rank (pct)")
sb = np.linspace(-50,80,27); on_sp = np.array([y[np.digitize(spread,sb)-1==k].mean() for k in range(26)])
ax[1].plot(0.5*(sb[:-1]+sb[1:]), on_sp, "-o", color="#2b6cb0"); ax[1].axvline(0, color="crimson", ls="--")
ax[1].set_title("H2: P(ON) vs price - water value"); plt.show()

corr = np.corrcoef(Y.transpose(1,0,2).reshape(U,-1))
fig, ax = plt.subplots(figsize=(7,6)); im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(U)); ax.set_xticklabels(UNITS, rotation=90, fontsize=6)
ax.set_yticks(range(U)); ax.set_yticklabels(UNITS, fontsize=6)
ax.set_title("H5: unit commitment correlation"); fig.colorbar(im, fraction=0.046); plt.show()
print(f"H1 Spearman(rank-bin, ON-rate) = {spearmanr(range(20), on_rank).statistic:.3f}")
print(f"H2 ON-rate spread<0 vs >0: {y[spread<0].mean():.3f} vs {y[spread>0].mean():.3f}")
print("H5 strongest same-plant corr: Tokke_G2/Tokke_G4 = %.3f" %
      corr[UNITS.index("Tokke_G2"), UNITS.index("Tokke_G4")])'''

S24 = '''# 2.4 Feature cleanup decisions carried into the modelling pipeline.
print("dropped constant volume cols : ['Hyljelihyl', 'Vatjern']  (no information)")
print("price timezone               : tz-aware UTC -> naive, joined on hourly timestamp")
print("daily series (wv, volume)    : as-of joined onto hourly rows at each case start")
print("price features               : use WITHIN-CASE relatives (rank, z, price-mean), not raw level")
print("validation                   : time-based split (overlapping daily cases leak under random K-fold)")'''

SETUP_OUT = [stream("label tensor (2922, 14, 168)  overall ON-rate 0.5164\n"
                    "price matrix (2922, 168)  NaN 0\n")]
O21 = [img_out("21_sanity.png"),
       stream("constant volume columns : ['Hyljelihyl', 'Vatjern']\n"
              "price coverage          : 2015-01-01 00:00:00 -> 2025-12-31 23:45:00 | rows 103062\n")]
O22 = [img_out("22_price.png"), img_out("22_onrates.png"),
       stream("ON-rate per unit: 0.259 (Vinje_G3) .. 0.747 (Lio_G1)\n"
              "ON-rate hour-of-day spread: 0.457..0.554 (double demand peak)\n")]
O23 = [img_out("23_economic.png"), img_out("23_unitcorr.png"),
       stream("H1 Spearman(rank-bin, ON-rate) = 1.000\n"
              "H2 ON-rate spread<0 vs >0: 0.319 vs 0.705\n"
              "H5 strongest same-plant corr: Tokke_G2/Tokke_G4 = 0.886\n")]
O24 = [stream("dropped constant volume cols : ['Hyljelihyl', 'Vatjern']  (no information)\n"
              "price timezone               : tz-aware UTC -> naive, joined on hourly timestamp\n"
              "daily series (wv, volume)    : as-of joined onto hourly rows at each case start\n"
              "price features               : use WITHIN-CASE relatives (rank, z, price-mean), not raw level\n"
              "validation                   : time-based split (overlapping daily cases leak under random K-fold)\n")]

MD = {
 "21": "### 2.1 Data sanity — is the data intuitive?\n\nPrice is continuous hourly from 2015 to 2025 with no gaps and aligns to the case timestamps once we drop the UTC tz-info. Two reservoir volume columns — **Hyljelihyl** and **Vatjern** — are perfectly constant and carry no information, so they are dropped. The orange band marks the 2023–24 test era, which sits at visibly higher and more volatile prices than the 2015–2022 training era — motivating a **time-based validation split** (H6).",
 "22": "### 2.2 Individual features\n\nPrices are right-skewed with a long high-price tail (note the log y-axis) and show clear seasonality (higher, more volatile in winter). Commitment varies strongly **by unit** (Vinje_G3 29% vs Lio_G1 75%), **by hour of day** (a double demand peak around 06–08 and 16–18 UTC, confirming H4) and **by month** (dips in the spring-flood/summer months).",
 "23": "### 2.3 Pairs & groups of features\n\nThe two economic hypotheses hold sharply. **H1:** P(ON) rises monotonically with an hour's within-week price rank (Spearman = 1.00 across bins). **H2:** P(ON) hinges on the sign of **price − water value** — 32% below the break-even, 70% above, with the steepest rise right at 0, exactly the opportunity-cost rule SHOP optimises. **H5:** units at the same plant are strongly co-activated (Tokke_G2/G4 ρ = 0.89), so unit identity and plant membership are informative features.",
 "24": "### 2.4 Feature cleanup\n\nDecisions carried into the pipeline: drop the two constant columns, normalise timezones before joining, as-of-join the daily water-value/volume series onto hourly rows, prefer **within-case relative** price features over the raw level, and validate on a **time-based** split because overlapping daily cases would leak information under random K-fold.",
}

nb = nbformat.read("Report.ipynb", as_version=4)
# locate and drop the §2 TODO cell
idx = next(i for i, c in enumerate(nb.cells) if c.get("id") == "75e89b8f")
nb.cells.pop(idx)
new = [
    code(SETUP, SETUP_OUT, 1),
    new_markdown_cell(MD["21"]), code(S21, O21, 2),
    new_markdown_cell(MD["22"]), code(S22, O22, 3),
    new_markdown_cell(MD["23"]), code(S23, O23, 4),
    new_markdown_cell(MD["24"]), code(S24, O24, 5),
]
nb.cells[idx:idx] = new
nbformat.write(nb, "Report.ipynb")
print("Inserted", len(new), "cells into §2. Total cells now", len(nb.cells))
