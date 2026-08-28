"""
main.py
--------
End-to-end pipeline runner. Produces every deliverable in outputs/:
  images/01_rgb_composite.png
  images/02_ground_truth_map.png
  images/03_classified_map.png
  images/04_confusion_matrix.png
  images/05_feature_importance.png
  images/06_comparison_chart.png
  comparison_results.xlsx
  accuracy_report.html

Run with:  python3 main.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_generator import generate_scene, CLASS_NAMES
from experiment import run_grid
from classifier import predict_full_scene
from report import (
    save_scene_images, save_classified_map, save_confusion_matrix_plot,
    save_feature_importance_plot, save_comparison_chart, export_excel, export_html
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
IMG_DIR = os.path.join(OUT_DIR, "images")


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    print("1/5  Generating synthetic Sentinel-2 / Dynamic-World scene ...")
    image, labels = generate_scene(height=300, width=300, seed=42)

    print("2/5  Running experiment grid (window sizes x sampling strategies) ...")
    df, detailed = run_grid(
        image, labels,
        window_sizes=(1, 3, 5),
        strategies=("random", "stratified"),
        n_samples=7200,
        seed=0,
    )

    best_row = df.iloc[0]
    best_key = (int(best_row["window_size"]), best_row["sampling_strategy"])
    best = detailed[best_key]
    print(f"\nBest configuration: window={best_key[0]}, strategy={best_key[1]} "
          f"-> OA={best['overall_accuracy']*100:.2f}%, kappa={best['kappa']:.3f}")

    print("3/5  Rendering scene, classification map & confusion matrix images ...")
    cmap = save_scene_images(image, labels, IMG_DIR)

    full_pred = predict_full_scene(best["_clf"], best["_feats"])
    save_classified_map(
        full_pred, cmap, IMG_DIR,
        title=f"RF classified map (window={best_key[0]}, {best_key[1]} sampling)",
        filename="03_classified_map.png",
    )
    save_confusion_matrix_plot(
        best["_full_result"]["confusion_matrix"], CLASS_NAMES,
        os.path.join(IMG_DIR, "04_confusion_matrix.png"),
    )
    save_feature_importance_plot(
        best["_clf"], best["_feature_names"],
        os.path.join(IMG_DIR, "05_feature_importance.png"),
    )
    save_comparison_chart(df, os.path.join(IMG_DIR, "06_comparison_chart.png"))

    print("4/5  Exporting Excel accuracy-assessment workbook ...")
    export_excel(df, best, CLASS_NAMES, os.path.join(OUT_DIR, "comparison_results.xlsx"))

    print("5/5  Exporting HTML accuracy assessment report ...")
    export_html(df, best, CLASS_NAMES, IMG_DIR, os.path.join(OUT_DIR, "accuracy_report.html"))

    print("\nAll outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
