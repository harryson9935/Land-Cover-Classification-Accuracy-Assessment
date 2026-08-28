# Land Cover Classification & Accuracy Assessment

A machine-learning pipeline for land cover classification using Sentinel-2
multispectral imagery and Dynamic World-style ground-truth labels across
9 land-cover classes, with a Random Forest classifier, configurable
window-based spectral-spatial feature extraction, random/stratified
sampling, an interactive Gradio interface, confusion-matrix visualization,
and automated accuracy reporting (Excel + HTML).

## Results (this run)

| Metric | Value |
|---|---|
| Best configuration | window size = 5, stratified sampling |
| Overall Accuracy | **91.52%** |
| Kappa coefficient | **0.905** |
| Macro F1 | 0.916 |

Full comparison across window sizes (1/3/5) x sampling strategies
(random/stratified) is in `outputs/comparison_results.xlsx` and
`outputs/accuracy_report.html`.

**Data note:** this environment has no internet access, so the pipeline
runs on a *synthetically generated* Sentinel-2-style scene (see
"Data source" below) instead of pulling real imagery from Google Earth
Engine. The pipeline itself — feature extraction, sampling, RF training,
accuracy assessment, reporting, Gradio UI — is written to be data-source
agnostic; point it at real rasters and nothing else changes.

## Project structure

```
landcover_project/
├── main.py                     # run the full pipeline end-to-end
├── app.py                      # Gradio interactive interface
├── requirements.txt
├── src/
│   ├── data_generator.py       # synthetic Sentinel-2 + Dynamic World scene generator
│   ├── features.py             # window-based spectral-spatial feature extraction
│   ├── sampling.py             # random / stratified pixel sampling
│   ├── classifier.py           # Random Forest train / predict wrapper
│   ├── accuracy.py             # OA, Kappa, confusion matrix, producer's/user's accuracy
│   ├── experiment.py           # grid search over window sizes x sampling strategies
│   └── report.py               # Excel / HTML export + all chart generation
└── outputs/
    ├── comparison_results.xlsx
    ├── accuracy_report.html
    └── images/
        ├── 01_rgb_composite.png
        ├── 02_ground_truth_map.png
        ├── 03_classified_map.png
        ├── 04_confusion_matrix.png
        ├── 05_feature_importance.png
        └── 06_comparison_chart.png
```

## Approach

1. **Data**: 6-band Sentinel-2-style reflectance (Blue, Green, Red, NIR,
   SWIR1, SWIR2) + a 9-class Dynamic World-style label map
   (`water, trees, grass, flooded_vegetation, crops, shrub_and_scrub,
   built, bare, snow_and_ice`), spatially clustered into realistic patches
   with class-specific spectral signatures, sensor noise, and mixed-pixel
   boundary blending.
2. **Feature extraction**: for each pixel, a square window (size 1/3/5/7)
   is used to compute per-band mean / std / min / max plus NDVI and NDWI,
   in addition to the raw center-pixel spectral values. Window size 1
   reduces to pixel-only (no spatial context) classification, used as a
   baseline.
3. **Sampling**: `random` draws pixels uniformly regardless of class;
   `stratified` draws a balanced number of pixels per class so rare
   classes (e.g. snow_and_ice, flooded_vegetation) aren't starved during
   training.
4. **Classification**: `RandomForestClassifier` (scikit-learn),
   class-balanced, 300 trees by default.
5. **Accuracy assessment**: overall accuracy, Cohen's Kappa, confusion
   matrix, and per-class producer's accuracy / user's accuracy / F1 — the
   standard remote-sensing accuracy-assessment trio.
6. **Reporting**: results are exported to a styled `.xlsx` workbook
   (Summary, Experiment_Comparison, Per_Class_Accuracy, Confusion_Matrix
   sheets) and a self-contained `.html` report with embedded charts.
7. **Interactive app**: `app.py` is a Gradio interface for picking window
   size / sampling strategy / sample budget interactively, viewing the
   RGB composite, ground truth, classified map, confusion matrix and
   feature importances, and downloading the Excel/HTML report.

## Running it

```bash
pip install -r requirements.txt

# 1. Run the full pipeline (data generation -> grid search -> reports)
python3 main.py

# 2. Launch the interactive Gradio app
python3 app.py
```

`main.py` regenerates everything in `outputs/`. `app.py` opens a local
web UI (default http://127.0.0.1:7860) for interactive experimentation.

> **Sandbox note:** `gradio` could not be installed or executed in the
> environment that produced this project (no internet access), so
> `app.py` was verified by unit-testing its underlying `run_pipeline()`
> function directly (confirmed to produce correct plots, tables, and
> Excel/HTML exports) but the Gradio UI itself was not launched live.
> It will run normally on any machine with internet access to `pip
> install gradio`.

## Using real Sentinel-2 + Dynamic World data instead of synthetic data

Replace the body of `generate_scene()` in `src/data_generator.py` with a
loader for your own rasters, keeping the same return signature:

```python
def generate_scene(...):
    image  = <read Sentinel-2 GeoTIFF as (H, W, 6) float32 array, e.g. via rasterio>
    labels = <read Dynamic World GeoTIFF as (H, W) uint8 array, values 0-8>
    return image, labels
```

A typical real-data workflow with Google Earth Engine would export a
Sentinel-2 L2A median composite (`COPERNICUS/S2_SR_HARMONIZED`, cloud
masked) and the matching Dynamic World mode composite
(`GOOGLE/DYNAMICWORLD/V1`) for the same AOI and date range, clipped and
reprojected to the same grid, then export both as GeoTIFFs. Everything
downstream (feature extraction, sampling, RF training, accuracy
assessment, Gradio app) works unchanged.

## Key dependencies

- `numpy`, `scipy` — array ops, spatial filtering for feature windows
- `scikit-learn` — Random Forest, train/test split, metrics
- `pandas`, `openpyxl` — Excel export
- `matplotlib` — all chart/map rendering
- `gradio` — interactive interface (see sandbox note above)
- (for real imagery) `rasterio` or `earthengine-api` to source Sentinel-2 / Dynamic World rasters
