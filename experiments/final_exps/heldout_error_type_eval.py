"""
heldout_error_type_eval.py
===========================
Leave-one-error-type-out generalisation: for each of the 5 injected error
types, train the probe on clean + the OTHER 4 types, and evaluate detection
AUROC on clean + the held-out type. Clean records are reused (pooled) as
negatives in every split; only the error-type composition of the positive
class changes.

Reads (per output dir):
    act_last.npy, act_mean.npy, act_labels.npy, stage1_records.csv,
    probe_threshold.json   (for the best layer/mode to use)

Usage:
    python heldout_error_type_eval.py --output_dir results_qwen25_3b_instruct/ --model_name Qwen2.5-3B

Output:
    <output_dir>/heldout_error_type_results.csv
    final_exps/tab_heldout_error_type.csv  (one column per model)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ERROR_TYPES = [
    "off_by_one",
    "wrong_operator",
    "wrong_unit",
    "magnitude_error",
    "wrong_percentage_base",
]
TABLE_PATH = Path(__file__).parent / "tab_heldout_error_type.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", required=True,
                    help="Display name for the results table, e.g. 'Qwen2.5-3B'")
    return p.parse_args()


def pick_best_aggregation(thresh: dict):
    last_auroc = thresh["last"]["cv_auroc"]
    mean_auroc = thresh["mean"]["cv_auroc"]
    mode = "mean" if mean_auroc >= last_auroc else "last"
    return thresh[mode]["layer"], mode


def main():
    args = parse_args()
    out = Path(args.output_dir)

    with open(out / "probe_threshold.json") as f:
        thresh = json.load(f)
    layer, mode = pick_best_aggregation(thresh)

    act = np.load(out / f"act_{mode}.npy")
    y_all = np.load(out / "act_labels.npy")
    records = pd.read_csv(out / "stage1_records.csv")
    assert len(records) == act.shape[0] == len(y_all)

    X_all = act[:, layer, :]
    is_clean = records["condition"].values == "clean"
    error_type = records["error_type"].values

    missing = set(records.loc[~is_clean, "error_type"].unique()) - set(ERROR_TYPES)
    if missing:
        raise ValueError(f"Unexpected error types in data not in ERROR_TYPES: {missing}")

    rows = []
    for held_type in ERROR_TYPES:
        is_held = (error_type == held_type)
        is_other_error = (~is_clean) & (~is_held)

        train_mask = is_clean | is_other_error
        test_mask  = is_clean | is_held

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ])
        pipe.fit(X_all[train_mask], y_all[train_mask])
        proba = pipe.predict_proba(X_all[test_mask])[:, 1]
        auroc = roc_auc_score(y_all[test_mask], proba)

        rows.append({
            "held_out_type": held_type,
            "n_train": int(train_mask.sum()),
            "n_test":  int(test_mask.sum()),
            "auroc":   auroc,
        })
        print(f"  held-out={held_type:<24s} AUROC={auroc:.4f}  "
              f"(train={train_mask.sum()}, test={test_mask.sum()})")

    df = pd.DataFrame(rows)
    mean_auroc = df["auroc"].mean()
    print(f"\nMean held-out AUROC: {mean_auroc:.4f}")

    df.to_csv(out / "heldout_error_type_results.csv", index=False)
    print(f"Saved -> {out / 'heldout_error_type_results.csv'}")

    col = df.set_index("held_out_type")["auroc"]
    col.loc["Mean"] = mean_auroc
    col.name = args.model_name

    if TABLE_PATH.exists():
        wide = pd.read_csv(TABLE_PATH, index_col=0)
        wide[args.model_name] = col
    else:
        wide = col.to_frame()
    wide.to_csv(TABLE_PATH)
    print(f"Saved -> {TABLE_PATH}")


if __name__ == "__main__":
    main()
