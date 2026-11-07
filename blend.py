#!/usr/bin/env python3
"""Rank-average blend of the two selected submissions (improvement #2).

The Kaggle metric is a *micro-average* ROC-AUC (see sintef_project/kaggle_metric.py):
all (Run No, unit, hour) cells are pooled into one vector and a single AUC is
computed over the global ordering. AUC depends only on that ordering, so the
right way to combine two models is to average their *global ranks*, not their
raw probabilities -- this is invariant to any monotonic miscalibration between
LightGBM (NB1) and XGBoost (NB2) and cannot be dominated by whichever model
happens to output more extreme values.

Reads  submission_nb1.csv, submission_nb2.csv  (wide: 'Run No' + 2352 targets)
Writes submission_blend.csv  in the same layout as sample_submission.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
SAMPLE = pd.read_csv(HERE / "sample_submission.csv")
TARGET_COLS = [c for c in SAMPLE.columns if c != "Run No"]


def global_rank01(df: pd.DataFrame) -> np.ndarray:
    """Flatten the target matrix and map every cell to its global percentile in
    [0, 1] (ties share the average rank)."""
    flat = df[TARGET_COLS].to_numpy(dtype="float64").ravel()
    r = rankdata(flat, method="average")           # 1 .. N, ties averaged
    return (r - 1) / (len(r) - 1)                   # -> [0, 1]


def load_aligned(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # align rows to the sample's Run No order and columns to the target layout
    df = df.set_index("Run No").reindex(SAMPLE["Run No"])[TARGET_COLS].reset_index()
    assert df[TARGET_COLS].isna().sum().sum() == 0, f"missing predictions in {path.name}"
    return df


def blend(paths, weights=None):
    subs = [load_aligned(HERE / p) for p in paths]
    w = np.ones(len(subs)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    shape = (len(SAMPLE), len(TARGET_COLS))
    acc = np.zeros(shape[0] * shape[1])
    for wi, s in zip(w, subs):
        acc += wi * global_rank01(s)
    out = SAMPLE.copy()
    out[TARGET_COLS] = acc.reshape(shape)
    return out


if __name__ == "__main__":
    out = blend(["submission_nb1.csv", "submission_nb2.csv"])   # equal weight
    assert list(out.columns) == ["Run No"] + TARGET_COLS
    assert out["Run No"].tolist() == SAMPLE["Run No"].tolist()
    assert out[TARGET_COLS].isna().sum().sum() == 0
    out.to_csv(HERE / "submission_blend.csv", index=False)
    print("Wrote submission_blend.csv", out.shape)
