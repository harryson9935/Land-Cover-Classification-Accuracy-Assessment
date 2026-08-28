"""
experiment.py
--------------
Runs the full pipeline (feature extraction -> sampling -> RF train ->
full-scene predict -> accuracy assessment) across a grid of window sizes
and sampling strategies, and reports the comparison table plus the best
configuration's full diagnostics.
"""

import time
import numpy as np
import pandas as pd

from data_generator import generate_scene, CLASS_NAMES, BAND_NAMES, scene_to_rgb
from features import extract_features
from sampling import get_samples, train_test_split_coords
from classifier import train_rf, predict_full_scene, feature_importances
from accuracy import assess


def run_single_experiment(image, labels, window_size, strategy, n_samples=7200,
                           test_size=0.3, seed=0, n_estimators=300):
    t0 = time.time()
    feats, feature_names = extract_features(image, window_size=window_size, band_names=BAND_NAMES)

    rows, cols = get_samples(labels, strategy=strategy, n_samples=n_samples, seed=seed)
    (tr_rows, tr_cols), (te_rows, te_cols) = train_test_split_coords(
        rows, cols, labels, test_size=test_size, seed=seed
    )

    X_train = feats[tr_rows, tr_cols]
    y_train = labels[tr_rows, tr_cols]
    X_test = feats[te_rows, te_cols]
    y_test = labels[te_rows, te_cols]

    clf = train_rf(X_train, y_train, n_estimators=n_estimators, seed=seed)
    y_pred = clf.predict(X_test)

    result = assess(y_test, y_pred, CLASS_NAMES)
    elapsed = time.time() - t0

    return {
        "window_size": window_size,
        "sampling_strategy": strategy,
        "n_train": len(tr_rows),
        "n_test": len(te_rows),
        "n_features": feats.shape[-1],
        "overall_accuracy": result["overall_accuracy"],
        "kappa": result["kappa"],
        "macro_f1": float(np.mean([v["f1_score"] for v in result["per_class"].values()])),
        "train_time_sec": round(elapsed, 2),
        "_clf": clf,
        "_feats": feats,
        "_feature_names": feature_names,
        "_full_result": result,
        "_test_coords": (te_rows, te_cols),
        "_y_test": y_test,
        "_y_pred": y_pred,
    }


def run_grid(image, labels, window_sizes=(1, 3, 5, 7), strategies=("random", "stratified"),
             n_samples=7200, seed=0):
    rows = []
    detailed = {}
    for ws in window_sizes:
        for strat in strategies:
            r = run_single_experiment(image, labels, ws, strat, n_samples=n_samples, seed=seed)
            key = (ws, strat)
            detailed[key] = r
            rows.append({
                "window_size": ws,
                "sampling_strategy": strat,
                "n_features": r["n_features"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
                "overall_accuracy_%": round(r["overall_accuracy"] * 100, 2),
                "kappa": round(r["kappa"], 3),
                "macro_f1": round(r["macro_f1"], 3),
                "train_time_sec": r["train_time_sec"],
            })
            print(f"window={ws:>2} strategy={strat:<11} "
                  f"OA={r['overall_accuracy']*100:6.2f}%  kappa={r['kappa']:.3f}  "
                  f"macro_f1={r['macro_f1']:.3f}  ({r['train_time_sec']}s)")
    df = pd.DataFrame(rows).sort_values(
        "overall_accuracy_%", ascending=False
    ).reset_index(drop=True)
    return df, detailed


if __name__ == "__main__":
    image, labels = generate_scene(height=300, width=300, seed=42)
    df, detailed = run_grid(image, labels)
    print("\n=== Comparison table (sorted by OA) ===")
    print(df.to_string(index=False))
    best_key = (df.iloc[0]["window_size"], df.iloc[0]["sampling_strategy"])
    print("\nBest config:", best_key)
