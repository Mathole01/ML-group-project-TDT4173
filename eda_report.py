#!/usr/bin/env python3
"""Exploratory Data Analysis for the Report.

Self-contained: loads the raw SINTEF data, reshapes the SHOP commitment labels to
a (case, unit, hour) tensor, and produces every EDA figure + the summary tables
used in Report.ipynb Section 2.  Figures are written to artifacts/eda/ and the
numeric findings are printed so they can be pasted as saved output.

Run:  .venv/bin/python eda_report.py
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("sintef_project/data")
KERNEL, EXTENDED = DATA_DIR / "kernel", DATA_DIR / "extended"
OUT = Path("artifacts/eda"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})
ACCENT = "#2b6cb0"

# ----------------------------------------------------------------------------- load
price  = pd.read_csv(KERNEL / "Historical_day_ahead_price_2015_2025.csv", parse_dates=["date"])
inflow = pd.read_csv(KERNEL / "Historical_inflow_1958_2025.csv", parse_dates=["date"])
volume = pd.read_csv(KERNEL / "Historical_volume_2015_2024.csv", parse_dates=["date"])
wvalue = pd.read_csv(KERNEL / "Synthetic_water_value_2015_2024.csv", parse_dates=["date"])
uc     = pd.read_csv(KERNEL / "Unit_commitment_decisions.csv", parse_dates=["starttime"])
sample = pd.read_csv("sintef_project/sample_submission.csv")
TARGET_COLS = [c for c in sample.columns if c != "Run No"]
H = 168

# tz-aware UTC -> naive
for _df in (price, inflow):
    if _df["date"].dt.tz is not None:
        _df["date"] = _df["date"].dt.tz_localize(None)
for _df in (volume, wvalue):
    if _df["date"].dt.tz is not None:
        _df["date"] = _df["date"].dt.tz_localize(None)
    _df["date"] = _df["date"].dt.normalize()

# ----------------------------------------------------------------------------- units/topology
_pat = re.compile(r"^result_committed_(.*)_t(\d+)$")
UNITS = list(dict.fromkeys(_pat.match(c).group(1) for c in TARGET_COLS))
U = len(UNITS)
assert U * H == len(TARGET_COLS)
PLANT_OF = {u: u.rsplit("_G", 1)[0] for u in UNITS}

_topo_all = yaml.safe_load(open(EXTENDED / "Tokke_Vinje_topology.yaml"))
_gen = _topo_all["model"]["generator"]
_res = _topo_all["model"]["reservoir"]
_upstream = {}
for _c in _topo_all["connections"]:
    _upstream.setdefault(_c["to"], []).append((_c["from"], _c["from_type"]))

def _intake(node, seen=None):
    seen = set() if seen is None else seen
    found = set()
    for src, t in _upstream.get(node, []):
        if src in seen:
            continue
        seen.add(src)
        if t == "reservoir":
            found.add(src)
        elif t == "tunnel":
            found |= _intake(src, seen)
    return found

PLANT_RES = {}
for p in dict.fromkeys(PLANT_OF.values()):
    c = _intake(p)
    PLANT_RES[p] = max(c, key=lambda r: _res[r]["max_vol"]) if c else None
RES_OF = {u: PLANT_RES[PLANT_OF[u]] for u in UNITS}
P_MAX = np.array([_gen[u]["p_max"] for u in UNITS])

# ----------------------------------------------------------------------------- label tensor
# wide targets are unit-major: unit0 t0..t167, unit1 t0..t167, ...
Y = uc[TARGET_COLS].to_numpy(np.int8).reshape(len(uc), U, H)      # (cases, units, hours)
starts = uc["starttime"].values
print(f"label tensor {Y.shape}  overall ON-rate {Y.mean():.4f}")

# price matrix per (case, hour)
_price_s = price.set_index("date")["Day-ahead price (EUR/MWh)"]
ts = (np.repeat(starts, H) + np.tile(pd.to_timedelta(np.arange(H), "h"), len(uc)))
Pm = pd.Series(ts).map(_price_s).to_numpy().reshape(len(uc), H)   # (cases, hours)
print(f"price matrix {Pm.shape}  NaN {np.isnan(Pm).sum()}")

# =============================================================================
# 2.1  DATA SANITY
# =============================================================================
fig, ax = plt.subplots(2, 1, figsize=(9, 5.5))
ax[0].plot(price["date"], price["Day-ahead price (EUR/MWh)"], lw=0.3, color=ACCENT)
ax[0].axvspan(pd.Timestamp("2023-01-01"), price["date"].max(), color="orange", alpha=0.15)
ax[0].set_title("Day-ahead price, 2015–2025 (orange = 2023–24 test era)")
ax[0].set_ylabel("EUR/MWh")
# constant reservoir columns
const_cols = [c for c in volume.columns if c != "date" and volume[c].nunique() <= 1]
nun = volume.drop(columns="date").nunique().sort_values()
ax[1].bar(range(len(nun)), nun.values, color=[("crimson" if v <= 1 else ACCENT) for v in nun.values])
ax[1].set_xticks(range(len(nun))); ax[1].set_xticklabels(nun.index, rotation=90, fontsize=7)
ax[1].set_title(f"Distinct values per reservoir volume column (red = constant → drop): {const_cols}")
ax[1].set_ylabel("n distinct")
fig.tight_layout(); fig.savefig(OUT / "21_sanity.png"); plt.close(fig)
print("2.1 constant volume columns:", const_cols)
print("2.1 price coverage:", price["date"].min(), "->", price["date"].max(),
      "| hourly rows:", len(price))

# =============================================================================
# 2.2  INDIVIDUAL FEATURES
# =============================================================================
# price distribution + monthly seasonality
pr = price.dropna(subset=["Day-ahead price (EUR/MWh)"]).copy()
pr["month"] = pr["date"].dt.month
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(pr["Day-ahead price (EUR/MWh)"], bins=80, color=ACCENT)
ax[0].set_title("Day-ahead price distribution"); ax[0].set_xlabel("EUR/MWh")
ax[0].set_yscale("log"); ax[0].set_ylabel("hours (log)")
by_m = [pr.loc[pr["month"] == m, "Day-ahead price (EUR/MWh)"] for m in range(1, 13)]
ax[1].boxplot(by_m, showfliers=False, tick_labels=list(range(1, 13)))
ax[1].set_title("Price by month (seasonality)"); ax[1].set_xlabel("month"); ax[1].set_ylabel("EUR/MWh")
fig.tight_layout(); fig.savefig(OUT / "22_price.png"); plt.close(fig)

# per-unit ON-rate, ON-rate by hour-of-day, ON-rate by month
on_unit = Y.mean(axis=(0, 2))                         # (units,)
on_hod = Y.mean(axis=(0, 1))                          # (hours-in-week,) -> fold to hour-of-day
hod = (np.arange(H) % 24)
on_by_hod = np.array([Y.mean(axis=(0, 1))[hod == h].mean() for h in range(24)])
case_month = pd.to_datetime(starts).month
on_by_month = np.array([Y[case_month == m].mean() if (case_month == m).any() else np.nan
                        for m in range(1, 13)])

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
order = np.argsort(on_unit)
ax[0].barh(np.array(UNITS)[order], on_unit[order], color=ACCENT)
ax[0].set_title("Mean ON-rate per unit"); ax[0].set_xlabel("P(ON)")
ax[0].tick_params(labelsize=7)
ax[1].plot(range(24), on_by_hod, "-o", color=ACCENT)
ax[1].set_title("ON-rate by hour of day (UTC)"); ax[1].set_xlabel("hour"); ax[1].set_ylabel("P(ON)")
ax[2].plot(range(1, 13), on_by_month, "-o", color=ACCENT)
ax[2].set_title("ON-rate by month"); ax[2].set_xlabel("month"); ax[2].set_ylabel("P(ON)")
fig.tight_layout(); fig.savefig(OUT / "22_onrates.png"); plt.close(fig)
print("2.2 ON-rate per unit min/max:", f"{on_unit.min():.3f} ({UNITS[on_unit.argmin()]}) .."
      f" {on_unit.max():.3f} ({UNITS[on_unit.argmax()]})")
print("2.2 ON-rate hour-of-day spread:", f"{on_by_hod.min():.3f}..{on_by_hod.max():.3f}")

# =============================================================================
# 2.3  PAIRS / GROUPS   (hypotheses H1, H2, H5)
# =============================================================================
# H1: P(ON) vs within-week price rank
rank = pd.DataFrame(Pm).rank(axis=1, pct=True).to_numpy()            # (cases, hours)
rank_full = np.repeat(rank[:, None, :], U, axis=1).ravel()
y_full = Y.ravel().astype(np.float32)
bins = np.linspace(0, 1, 21)
bi = np.clip(np.digitize(rank_full, bins) - 1, 0, len(bins) - 2)
on_by_rank = np.array([y_full[bi == k].mean() for k in range(len(bins) - 1)])

# H2: P(ON) vs price - water value spread
wv_start = wvalue.set_index("date")
wv_case = np.array([wv_start.asof(pd.Timestamp(s).normalize()).reindex(
    [RES_OF[u] for u in UNITS]).to_numpy() for s in starts])         # (cases, units)
spread = Pm[:, None, :] - wv_case[:, :, None]                        # (cases, units, hours)
sp_flat = spread.ravel()
sb = np.linspace(-50, 80, 27)
sbi = np.clip(np.digitize(sp_flat, sb) - 1, 0, len(sb) - 2)
on_by_spread = np.array([y_full[sbi == k].mean() if (sbi == k).any() else np.nan
                         for k in range(len(sb) - 1)])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(0.5 * (bins[:-1] + bins[1:]), on_by_rank, "-o", color=ACCENT)
ax[0].set_title("H1: P(ON) vs within-week price rank"); ax[0].set_xlabel("price rank (pct)")
ax[0].set_ylabel("P(ON)")
ax[1].plot(0.5 * (sb[:-1] + sb[1:]), on_by_spread, "-o", color=ACCENT)
ax[1].axvline(0, color="crimson", ls="--", lw=1)
ax[1].set_title("H2: P(ON) vs price − water value (EUR/MWh)"); ax[1].set_xlabel("price − WV")
ax[1].set_ylabel("P(ON)")
fig.tight_layout(); fig.savefig(OUT / "23_economic.png"); plt.close(fig)

# H5: unit co-activation correlation
flatYU = Y.transpose(1, 0, 2).reshape(U, -1)                         # (units, cases*hours)
corr = np.corrcoef(flatYU)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(U)); ax.set_xticklabels(UNITS, rotation=90, fontsize=6)
ax.set_yticks(range(U)); ax.set_yticklabels(UNITS, fontsize=6)
ax.set_title("H5: unit commitment correlation (same-plant units co-activate)")
fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(OUT / "23_unitcorr.png"); plt.close(fig)

# reservoir volume correlation (grouped structure)
volc = volume.drop(columns="date")
volc = volc.loc[:, volc.nunique() > 1]
cm = volc.corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=90, fontsize=6)
ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns, fontsize=6)
ax.set_title("Reservoir volume correlations"); fig.colorbar(im, fraction=0.046)
fig.tight_layout(); fig.savefig(OUT / "23_rescorr.png"); plt.close(fig)

# monotonic check for the economic narrative
from scipy.stats import spearmanr
m = ~np.isnan(on_by_rank)
rho_rank = spearmanr(np.arange(20)[m], on_by_rank[m]).statistic
print(f"2.3 Spearman(rank-bin, ON-rate) = {rho_rank:.3f}")
print(f"2.3 ON-rate at spread<0 vs spread>0: "
      f"{y_full[sp_flat < 0].mean():.3f} vs {y_full[sp_flat > 0].mean():.3f}")
# strongest same-plant pair
same = [(UNITS[i], UNITS[j], corr[i, j]) for i in range(U) for j in range(i + 1, U)
        if PLANT_OF[UNITS[i]] == PLANT_OF[UNITS[j]]]
if same:
    s = max(same, key=lambda t: t[2])
    print(f"2.3 strongest same-plant corr: {s[0]} / {s[1]} = {s[2]:.3f}")

# =============================================================================
# 2.4  FEATURE CLEANUP  (summary table printed)
# =============================================================================
print("\n2.4 cleanup summary")
print("  dropped constant volume cols :", const_cols)
print("  price tz handled             : UTC -> naive, joined on hourly timestamp")
print("  daily series (wv/volume)     : as-of joined onto hourly rows at case start")
print("\nAll figures written to", OUT)
