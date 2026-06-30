"""
group_cv_eval.py
=================
Re-scores the probe at its already-selected best (layer, mode) under
base-problem-grouped 5-fold CV, to check for leakage from shared
base_id / template structure in the plain stratified CV used in Stage 2.

Train/test split: StratifiedGroupKFold(n_splits=5) with
    X      = act_{mode}.npy[:, layer, :]
    y      = act_labels.npy
    groups = stage1_records.csv["base_id"]
so every variant (clean / error_standard / error_verbalized) of a given
base problem is confined to a single fold.

Reads (per output dir):
    act_last.npy, act_mean.npy, act_labels.npy, stage1_records.csv,
    probe_threshold.json   (for the best layer/mode to re-evaluate)

Usage:
    python group_cv_eval.py --output_dir results_qwen25_3b_instruct/ --model_name Qwen2.5-3B

Output:
    <output_dir>/grouped_cv_results.json
    appends a row to final_exps/tab_grouped_cv.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

N_SPLITS = 5
RANDOM_STATE = 42
TABLE_PATH = Path(__file__).parent / "tab_grouped_cv.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", required=True,
                    help="Display name for the results table, e.g. 'Qwen2.5-3B'")
    return p.parse_args()


def pick_best_aggregation(thresh: dict):
    """Best (layer, mode) is whichever mode the original Stage 2 run reported
    as having the higher final cv_auroc (tab:probe_summary uses this figure)."""
    last_auroc = thresh["last"]["cv_auroc"]
    mean_auroc = thresh["mean"]["cv_auroc"]
    mode = "mean" if mean_auroc >= last_auroc else "last"
    return thresh[mode]["layer"], mode, thresh[mode]["cv_auroc"]


def grouped_cv_auroc(X, y, groups):
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, test_idx in skf.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[test_idx])[:, 1]
        scores.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    args = parse_args()
    out = Path(args.output_dir)

    with open(out / "probe_threshold.json") as f:
        thresh = json.load(f)
    layer, mode, stratified_auroc = pick_best_aggregation(thresh)

    act = np.load(out / f"act_{mode}.npy")          # (N, n_layers+1, d_model)
    y   = np.load(out / "act_labels.npy")            # (N,)
    records = pd.read_csv(out / "stage1_records.csv")
    groups = records["base_id"].values

    assert len(y) == len(groups) == act.shape[0], "row count mismatch between act/labels/records"

    X = act[:, layer, :]
    grouped_auroc, grouped_std = grouped_cv_auroc(X, y, groups)
    delta = grouped_auroc - stratified_auroc

    print(f"Model:      {args.model_name}")
    print(f"Best layer: {layer}  mode: {mode}")
    print(f"Stratified AUROC: {stratified_auroc:.4f}")
    print(f"Grouped    AUROC: {grouped_auroc:.4f} (+/- {grouped_std:.4f})")
    print(f"Delta:            {delta:+.4f}")

    result = {
        "model":            args.model_name,
        "layer":            layer,
        "mode":             mode,
        "stratified_auroc": stratified_auroc,
        "grouped_auroc":    grouped_auroc,
        "grouped_auroc_std": grouped_std,
        "delta":            delta,
    }
    with open(out / "grouped_cv_results.json", "w") as f:
        json.dump(result, f, indent=2)

    row = pd.DataFrame([result])
    if TABLE_PATH.exists():
        existing = pd.read_csv(TABLE_PATH)
        existing = existing[existing["model"] != args.model_name]
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(TABLE_PATH, index=False)
    print(f"\nSaved -> {out / 'grouped_cv_results.json'}")
    print(f"Saved -> {TABLE_PATH}")


if __name__ == "__main__":
    main()
