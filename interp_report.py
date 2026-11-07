#!/usr/bin/env python3
"""Model interpretation for Report.ipynb Section 6.

Reuses the EXACT feature engineering of Short notebook 1 (its load + feature cells
are executed verbatim, skipping the slow LP and the final training), trains a
LightGBM on a time-based split, and produces:
  * gain-based feature importance (top 20)
  * SHAP beeswarm summary on a sample
  * partial dependence of P(ON) on price_rank and pwv_spread_end
Figures -> artifacts/interp/.  Validation ROC-AUC is printed for the report.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("artifacts/interp"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})

# --- run NB1's load + feature-engineering cells verbatim (cells 2, 4, 6) ------
import nbformat
nb = nbformat.read("Short notebook 1.ipynb", as_version=4)
code_cells = [c.source for c in nb.cells if c.cell_type == "code"]
g = {}
for src in code_cells[:3]:                 # config, load raw, feature engineering (no LP, no train)
    exec(compile(src, "<nb1>", "exec"), g)
train_long = g["train_long"]
print("reused NB1 features:", train_long.shape)

# --- feature matrix (drop identifiers/target, keep engineered features) -------
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

DROP = {"Run No", "target", "unit", "reservoir", "outlet_res"}
feat_cols = [c for c in train_long.columns if c not in DROP
             and train_long[c].dtype.kind in "fiu"]
# time-based split: earliest 80% of cases -> train, latest 20% -> validation
runs = np.sort(train_long["Run No"].unique())
cut = runs[int(len(runs) * 0.8)]
tr = train_long[train_long["Run No"] < cut]
va = train_long[train_long["Run No"] >= cut]
# subsample training rows for a fast interpretation model (every other hour)
tr = tr[tr["hour"] % 2 == 0]
print(f"train rows {len(tr):,}  val rows {len(va):,}  features {len(feat_cols)}")

dtr = lgb.Dataset(tr[feat_cols], tr["target"])
dva = lgb.Dataset(va[feat_cols], va["target"], reference=dtr)
params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=127,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              min_child_samples=200, seed=42, verbose=-1)
model = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
val_auc = roc_auc_score(va["target"], model.predict(va[feat_cols]))
print(f"interpretation model: best_iter={model.best_iteration}  VAL ROC-AUC={val_auc:.4f}")

# --- 1) gain-based feature importance ----------------------------------------
imp = (pd.Series(model.feature_importance("gain"), index=feat_cols)
       .sort_values(ascending=False))
top = imp.head(20)[::-1]
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top.index, top.values / top.values.max(), color="#2b6cb0")
ax.set_title("LightGBM gain importance (top 20, normalised)"); ax.set_xlabel("relative gain")
ax.tick_params(labelsize=8); fig.savefig(OUT / "61_importance.png"); plt.close(fig)
print("\ntop 10 features by gain:")
for k, v in imp.head(10).items():
    print(f"  {k:24s} {v/imp.max():.3f}")

# --- 2) SHAP beeswarm summary ------------------------------------------------
import shap
samp = va[feat_cols].sample(n=min(8000, len(va)), random_state=42)
expl = shap.TreeExplainer(model)
sv = expl.shap_values(samp)
sv = sv[1] if isinstance(sv, list) else sv
plt.figure()
shap.summary_plot(sv, samp, max_display=15, show=False)
plt.title("SHAP summary (validation sample)")
plt.savefig(OUT / "62_shap.png", bbox_inches="tight"); plt.close()
mean_abs = pd.Series(np.abs(sv).mean(0), index=feat_cols).sort_values(ascending=False)
print("\ntop 8 features by mean|SHAP|:")
for k, v in mean_abs.head(8).items():
    print(f"  {k:24s} {v:.4f}")

# --- 3) partial dependence for the two economic features ---------------------
def pdp(col, lo, hi, n=40):
    base = va[feat_cols].sample(n=min(4000, len(va)), random_state=1).copy()
    xs = np.linspace(lo, hi, n); ys = []
    for x in xs:
        base[col] = x
        ys.append(model.predict(base).mean())
    return xs, np.array(ys)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, (col, lo, hi, lab) in zip(ax, [
        ("price_rank", 0, 1, "within-week price rank"),
        ("pwv_spread_end", -40, 80, "price - end-of-week water value")]):
    xs, ys = pdp(col, lo, hi)
    a.plot(xs, ys, "-", color="#2b6cb0", lw=2)
    if col == "pwv_spread_end":
        a.axvline(0, color="crimson", ls="--", lw=1)
    a.set_title(f"PDP: P(ON) vs {lab}"); a.set_xlabel(col); a.set_ylabel("predicted P(ON)")
fig.tight_layout(); fig.savefig(OUT / "63_pdp.png"); plt.close(fig)

# --- persist numbers for the report ------------------------------------------
summary = {
    "val_auc": round(float(val_auc), 4),
    "best_iter": int(model.best_iteration),
    "n_features": len(feat_cols),
    "top_gain": [[k, round(float(v / imp.max()), 3)] for k, v in imp.head(10).items()],
    "top_shap": [[k, round(float(v), 4)] for k, v in mean_abs.head(8).items()],
}
json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
print("\nfigures + summary written to", OUT)
