# Modeling game plan

A recommended path to a strong score. This is guidance — the notebooks contain `TODO` stubs, not a
finished solution.

## 1. How to frame the problem

The submission is **wide** (one row per `Run No`, 2352 target columns), but you should *model* it in a
**long / stacked** table and pivot back to wide at the end:

```
one training row  =  (Run No, plant, unit, hour)   →   target = committed ∈ {0,1}
```

That turns 2922 cases into ~2922 × 14 × 168 ≈ 6.9M rows — a single binary classifier predicts
`P(ON)` for every (unit, hour), which is exactly the micro-averaged ROC-AUC target. Tree models
(LightGBM/XGBoost) handle this volume well; you can subsample hours/cases if training is slow.

Alternative framings worth a sentence in the report (even if rejected): one model **per unit** (14
models), or a sequence/CNN model over the 168-hour week. Trying/mentioning these helps the writeup.

## 2. Features to build (feature engineering section)

Join the time-series onto each `(Run No, hour)` using `starttime + hour`:

- **Price:** value at (case, hour); rolling mean/std/min/max over the week; **rank of this hour's price
  within the 168-hour week** (units run when price is high → very predictive); price − week-mean;
  hour-of-day and day-of-week from the absolute timestamp; lagged price (t-1, t-24).
- **Water value:** per-reservoir water value at case start and at case end (date+7); water-value trend.
  Price-to-water-value spread is the core economic signal for whether it pays to generate.
- **Volume / storage:** reservoir fill fraction = `volume / max_vol` (max_vol from the topology YAML);
  storage relative to `Constraint_min_volume`.
- **Inflow:** inflow at (case, hour) and weekly aggregates per reservoir; upstream inflow sums using
  the watercourse topology.
- **Constraints:** min-flow / min-volume active flags (mostly 0 — cheap but occasionally decisive).
- **Unit / plant identity:** categorical unit id, plant id, installed capacity, number of units at the
  plant, position in the cascade (from topology).
- **Static ON-rate priors:** historical ON-frequency of each unit by hour-of-day / season (strong
  baseline feature; compute from train only to avoid leakage).

**Feature selection** (needed to avoid the −3): drop near-constant columns, use gain/permutation
importance or a correlation filter, and report what you removed and why.

## 3. Models (need ≥2 types in the report)

1. **LightGBM** (primary) — fast on millions of rows, strong default.
2. **XGBoost** and/or **RandomForest** — second type for the required comparison.
3. Optional: logistic regression (baseline) and/or a small MLP/1D-CNN over the weekly sequence.

Report a table: model × CV ROC-AUC × train time. Your two short notebooks can be e.g.
**NB1 = best single LightGBM** (your best score) and **NB2 = LightGBM+XGBoost blend** (or XGBoost
alone), so the two selected Kaggle predictions are genuinely different.

## 4. Validation (avoid overfitting the public LB)

- Use a **time-based split**: train on early `Run No`, validate on the latest cases before test —
  mirrors the train→test (2015–2022 → 2023–2024) shift. Random K-fold leaks across adjacent weeks.
- Watch for the public/private LB gap the brief warns about; trust CV + a time-honest holdout, not the
  public LB. Select your 2 final submissions on CV, not public rank.

## 5. Interpretation (avoid the −3)

- Global: LightGBM gain importance + **SHAP** summary plot.
- Local/behavioral: **partial dependence** of `P(ON)` on price rank and on water-value spread — these
  should match the economic intuition (higher price ⇒ more likely ON). That narrative is exactly what
  the "model interpretation" deduction checks for.

## 6. Reproducibility guardrails for the short notebooks

- Fix all random seeds; no external data; no internet.
- Read raw data via the single `DATA_DIR` config cell.
- Keep total runtime < 12 h on 4 cores / 32 GB (subsample rows, cap `n_estimators`, or persist an
  intermediate feature table to disk).
- Hard-code final hyperparameters, but **document in the report how you found them** (e.g. Optuna /
  grid search run offline).
- End by writing `submission.csv` whose columns exactly match `sample_submission.csv`.
