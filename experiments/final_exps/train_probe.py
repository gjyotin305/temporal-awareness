"""
train_probe.py
==============
Minimal script to train layer-wise logistic probes on Stage 1 activation data.

Reads:
    <output_dir>/act_last.npy    (N, n_layers+1, d_model)
    <output_dir>/act_mean.npy    (N, n_layers+1, d_model)
    <output_dir>/act_labels.npy  (N,)  binary: 0=clean, 1=error

Writes:
    <output_dir>/layer_probe_results.csv
    <output_dir>/probe_weights.npz
    <output_dir>/probe_threshold.json

Usage:
    python train_probe.py results_qwen25_3b_instruct/
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CV_FOLDS = 5
SEED = 42


def train(output_dir: str):
    out = Path(output_dir)

    act_last = np.load(out / "act_last.npy")   # (N, L+1, d)
    act_mean = np.load(out / "act_mean.npy")   # (N, L+1, d)
    labels   = np.load(out / "act_labels.npy") # (N,)

    n_layers_p1 = act_last.shape[1]
    print(f"Loaded: {len(labels)} samples, {n_layers_p1} layers (incl. embedding), d={act_last.shape[2]}")
    print(f"Label balance: {labels.sum()} error / {(labels==0).sum()} clean\n")

    kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    rows = []
    best_per_mode = {
        "last": {"auroc": 0.0, "layer": 0},
        "mean": {"auroc": 0.0, "layer": 0},
    }

    for mode, arr in [("last", act_last), ("mean", act_mean)]:
        print(f"--- {mode}-token aggregation ---")
        for li in range(n_layers_p1):
            X = arr[:, li, :]
            pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, C=1.0))])
            aurocs = cross_val_score(pipe, X, labels, cv=kf, scoring="roc_auc")
            accs   = cross_val_score(pipe, X, labels, cv=kf, scoring="accuracy")

            rows.append({
                "layer": li, "mode": mode,
                "auroc_mean": aurocs.mean(), "auroc_std": aurocs.std(),
                "acc_mean":   accs.mean(),   "acc_std":   accs.std(),
            })

            if li % 5 == 0 or li == n_layers_p1 - 1:
                print(f"  L{li:3d}: AUROC={aurocs.mean():.4f} ± {aurocs.std():.4f}")

            if aurocs.mean() > best_per_mode[mode]["auroc"]:
                best_per_mode[mode]["auroc"] = aurocs.mean()
                best_per_mode[mode]["layer"] = li

    probe_df = pd.DataFrame(rows)
    probe_df.to_csv(out / "layer_probe_results.csv", index=False)

    all_thresholds = {}
    for mode, arr in [("last", act_last), ("mean", act_mean)]:
        best_layer = best_per_mode[mode]["layer"]
        best_auroc = best_per_mode[mode]["auroc"]
        print(f"\nBest ({mode}): L{best_layer}  AUROC={best_auroc:.4f}")

        X_best = arr[:, best_layer, :]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_best)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf.fit(X_scaled, labels)

        pipe_final = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED))])
        oof = cross_val_predict(pipe_final, X_best, labels, cv=kf, method="predict_proba")[:, 1]
        fpr, tpr, thresholds = roc_curve(labels, oof)
        youden_j   = tpr - fpr
        youden_idx = int(np.argmax(youden_j))
        youden_tau = float(thresholds[youden_idx])
        cons_tau   = float(np.percentile(oof[labels == 0], 99))

        assert scaler.mean_ is not None and scaler.scale_ is not None
        np.savez(
            out / f"probe_weights_{mode}.npz",
            W=clf.coef_[0], b=clf.intercept_[0],
            scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
            layer=best_layer, mode=mode,
        )

        all_thresholds[mode] = {
            "threshold":              youden_tau,
            "youden_J":               float(youden_j[youden_idx]),
            "tpr_at_threshold":       float(tpr[youden_idx]),
            "fpr_at_threshold":       float(fpr[youden_idx]),
            "conservative_threshold": cons_tau,
            "cv_auroc":               best_auroc,
            "layer":                  best_layer,
            "mode":                   mode,
        }
        print(f"  Youden threshold: {youden_tau:.4f}  TPR={tpr[youden_idx]:.3f}  FPR={fpr[youden_idx]:.3f}")
        print(f"  Conservative threshold (99th pctile clean): {cons_tau:.4f}")

    with open(out / "probe_threshold.json", "w") as f:
        json.dump(all_thresholds, f, indent=2)

    print(f"\nSaved:")
    print(f"  {out}/layer_probe_results.csv")
    print(f"  {out}/probe_weights_last.npz")
    print(f"  {out}/probe_weights_mean.npz")
    print(f"  {out}/probe_threshold.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_probe.py <output_dir>")
        sys.exit(1)
    train(sys.argv[1])
