"""
app.py
-------
Interactive Gradio interface for the Land Cover Classification &
Accuracy Assessment pipeline.

Lets the user:
  - generate a (synthetic) Sentinel-2 / Dynamic-World scene from a seed
  - pick a spatial window size (1/3/5/7) for spectral-spatial feature
    extraction
  - pick a sampling strategy (random / stratified) and sample budget
  - train a Random Forest classifier and classify the full scene
  - view the RGB composite, ground truth map, and classified map
  - view the confusion matrix and per-class accuracy table
  - download the accuracy assessment as an Excel workbook and an HTML report

Run with:
    pip install gradio   (not installed in this sandbox — no internet
                           access here — but this file is a complete,
                           ready-to-run app for any machine with internet)
    python3 app.py
Then open the local URL Gradio prints (typically http://127.0.0.1:7860).

NOTE: gradio could not be installed or executed in this sandboxed
environment (no internet access), so this file has been written and
reviewed but not run end-to-end here. All of its building blocks
(data_generator, features, sampling, classifier, accuracy, report) ARE
tested and used by main.py, which produced the accuracy numbers and
images already generated in outputs/.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr

from data_generator import generate_scene, CLASS_NAMES, BAND_NAMES, scene_to_rgb
from features import extract_features
from sampling import get_samples, train_test_split_coords
from classifier import train_rf, predict_full_scene, feature_importances
from accuracy import assess
from report import (
    CLASS_COLORS, save_scene_images, save_classified_map,
    save_confusion_matrix_plot, save_feature_importance_plot, export_excel,
    export_html,
)
from matplotlib.colors import ListedColormap

CMAP = ListedColormap(CLASS_COLORS)


# ----------------------------------------------------------------------
# Core pipeline call used by every Gradio button
# ----------------------------------------------------------------------
def run_pipeline(seed, scene_size, window_size, strategy, n_samples, test_size, n_estimators):
    seed = int(seed)
    scene_size = int(scene_size)
    window_size = int(window_size)
    n_samples = int(n_samples)
    n_estimators = int(n_estimators)

    image, labels = generate_scene(height=scene_size, width=scene_size, seed=seed)

    feats, feature_names = extract_features(image, window_size=window_size, band_names=BAND_NAMES)

    rows, cols = get_samples(labels, strategy=strategy, n_samples=n_samples, seed=seed)
    (tr_rows, tr_cols), (te_rows, te_cols) = train_test_split_coords(
        rows, cols, labels, test_size=test_size, seed=seed
    )

    X_train, y_train = feats[tr_rows, tr_cols], labels[tr_rows, tr_cols]
    X_test, y_test = feats[te_rows, te_cols], labels[te_rows, te_cols]

    clf = train_rf(X_train, y_train, n_estimators=n_estimators, seed=seed)
    y_pred = clf.predict(X_test)
    result = assess(y_test, y_pred, CLASS_NAMES)

    full_pred = predict_full_scene(clf, feats)

    # ---- figures ----
    rgb = scene_to_rgb(image)
    fig_rgb, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb); ax.axis("off"); ax.set_title("RGB composite")
    plt.tight_layout()

    fig_gt, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(labels, cmap=CMAP, vmin=0, vmax=len(CLASS_NAMES) - 1)
    ax.axis("off"); ax.set_title("Ground truth")
    cbar = plt.colorbar(im, ticks=range(len(CLASS_NAMES)), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    plt.tight_layout()

    fig_pred, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(full_pred, cmap=CMAP, vmin=0, vmax=len(CLASS_NAMES) - 1)
    ax.axis("off"); ax.set_title(f"Classified (w={window_size}, {strategy})")
    cbar = plt.colorbar(im, ticks=range(len(CLASS_NAMES)), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    plt.tight_layout()

    cm = result["confusion_matrix"]
    fig_cm, ax = plt.subplots(figsize=(6.5, 6))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Reference")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=7)
    ax.set_title("Confusion matrix")
    plt.tight_layout()

    fig_fi, ax = plt.subplots(figsize=(6, 5))
    top = feature_importances(clf, feature_names, top_n=12)[::-1]
    ax.barh([t[0] for t in top], [t[1] for t in top], color="#397D49")
    ax.set_title("Top feature importances")
    plt.tight_layout()

    # ---- metrics table ----
    per_class = result["per_class"]
    table = [
        [c, per_class[c]["support"],
         f"{per_class[c]['producers_accuracy']*100:.2f}%",
         f"{per_class[c]['users_accuracy']*100:.2f}%",
         f"{per_class[c]['f1_score']:.3f}"]
        for c in CLASS_NAMES
    ]

    summary_md = (
        f"### Results — window={window_size}, sampling={strategy}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Overall Accuracy | **{result['overall_accuracy']*100:.2f}%** |\n"
        f"| Kappa coefficient | **{result['kappa']:.3f}** |\n"
        f"| Train / Test pixels | {len(tr_rows)} / {len(te_rows)} |\n"
        f"| Number of features | {feats.shape[-1]} |\n"
    )

    # ---- exportable files ----
    tmp_dir = tempfile.mkdtemp(prefix="landcover_gradio_")
    img_dir = os.path.join(tmp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    save_scene_images(image, labels, img_dir)
    save_classified_map(full_pred, CMAP, img_dir,
                         title=f"Classified (w={window_size}, {strategy})",
                         filename="03_classified_map.png")
    save_confusion_matrix_plot(cm, CLASS_NAMES, os.path.join(img_dir, "04_confusion_matrix.png"))
    save_feature_importance_plot(clf, feature_names, os.path.join(img_dir, "05_feature_importance.png"))

    import pandas as pd
    df_single = pd.DataFrame([{
        "window_size": window_size, "sampling_strategy": strategy,
        "n_features": feats.shape[-1], "n_train": len(tr_rows), "n_test": len(te_rows),
        "overall_accuracy_%": round(result["overall_accuracy"] * 100, 2),
        "kappa": round(result["kappa"], 3),
        "macro_f1": round(np.mean([v["f1_score"] for v in per_class.values()]), 3),
        "train_time_sec": 0.0,
    }])
    best_like = {
        "window_size": window_size, "sampling_strategy": strategy,
        "overall_accuracy": result["overall_accuracy"], "kappa": result["kappa"],
        "macro_f1": float(np.mean([v["f1_score"] for v in per_class.values()])),
        "n_train": len(tr_rows), "n_test": len(te_rows), "n_features": feats.shape[-1],
        "_full_result": result,
    }
    xlsx_path = os.path.join(tmp_dir, "accuracy_assessment.xlsx")
    html_path = os.path.join(tmp_dir, "accuracy_report.html")
    export_excel(df_single, best_like, CLASS_NAMES, xlsx_path)
    export_html(df_single, best_like, CLASS_NAMES, img_dir, html_path)

    return (fig_rgb, fig_gt, fig_pred, fig_cm, fig_fi, table, summary_md,
            xlsx_path, html_path)


# ----------------------------------------------------------------------
# Gradio layout
# ----------------------------------------------------------------------
with gr.Blocks(title="Land Cover Classification & Accuracy Assessment") as demo:
    gr.Markdown(
        "# 🌍 Land Cover Classification — Accuracy Assessment\n"
        "Sentinel-2 multispectral imagery &middot; Dynamic World 9-class schema &middot; "
        "Random Forest with configurable window-based spectral-spatial features.\n\n"
        "> **Data note:** scenes are synthesized (`src/data_generator.py`) to mimic Sentinel-2 "
        "reflectance + Dynamic World labels, since this pipeline is designed to also accept real "
        "GeoTIFFs of the same shape — swap in real rasters there for production use."
    )

    with gr.Row():
        with gr.Column(scale=1):
            seed = gr.Slider(0, 999, value=42, step=1, label="Scene random seed")
            scene_size = gr.Slider(100, 400, value=300, step=50, label="Scene size (pixels, square)")
            window_size = gr.Radio([1, 3, 5, 7], value=5, label="Spectral-spatial window size")
            strategy = gr.Radio(["random", "stratified"], value="stratified", label="Sampling strategy")
            n_samples = gr.Slider(900, 14400, value=7200, step=900, label="Total sample budget")
            test_size = gr.Slider(0.1, 0.5, value=0.3, step=0.05, label="Test set fraction")
            n_estimators = gr.Slider(50, 500, value=300, step=50, label="RF n_estimators")
            run_btn = gr.Button("🚀 Run classification", variant="primary")
            summary_md = gr.Markdown()
            xlsx_out = gr.File(label="Download accuracy_assessment.xlsx")
            html_out = gr.File(label="Download accuracy_report.html")

        with gr.Column(scale=2):
            with gr.Row():
                plot_rgb = gr.Plot(label="RGB composite")
                plot_gt = gr.Plot(label="Ground truth")
                plot_pred = gr.Plot(label="Classified map")
            with gr.Row():
                plot_cm = gr.Plot(label="Confusion matrix")
                plot_fi = gr.Plot(label="Feature importance")
            table_out = gr.Dataframe(
                headers=["Class", "Support", "Producer's Acc.", "User's Acc.", "F1"],
                label="Per-class accuracy",
            )

    run_btn.click(
        run_pipeline,
        inputs=[seed, scene_size, window_size, strategy, n_samples, test_size, n_estimators],
        outputs=[plot_rgb, plot_gt, plot_pred, plot_cm, plot_fi, table_out, summary_md,
                 xlsx_out, html_out],
    )

    demo.load(
        run_pipeline,
        inputs=[seed, scene_size, window_size, strategy, n_samples, test_size, n_estimators],
        outputs=[plot_rgb, plot_gt, plot_pred, plot_cm, plot_fi, table_out, summary_md,
                 xlsx_out, html_out],
    )


if __name__ == "__main__":
    demo.launch()
