"""
data_generator.py
------------------
Generates a synthetic Sentinel-2-like multispectral raster together with a
Dynamic-World-style 9-class land cover ground-truth map.

NOTE ON DATA SOURCE
====================
This pipeline was originally designed to run on real Sentinel-2 L2A surface
reflectance composites paired with Google's Dynamic World V1 label product
(both pulled from Google Earth Engine). Because this execution environment
has no internet access, real imagery cannot be downloaded here. This module
instead synthesizes a spatially-realistic scene with class-specific spectral
signatures drawn from published Sentinel-2 reflectance ranges for each
Dynamic World class, plus spatially-correlated noise and mixed-pixel effects,
so that the rest of the pipeline (feature extraction, sampling, RF
classification, accuracy assessment, Gradio app) is exercised exactly as it
would be on real imagery.

To switch to real data: replace `generate_scene()` with a rasterio read of
your Sentinel-2 GeoTIFF (bands stacked as [H, W, C]) and your Dynamic World
label GeoTIFF (as [H, W] with values 0-8), keeping the same return shapes.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

# ----------------------------------------------------------------------
# Dynamic World classes (official 9-class schema)
# ----------------------------------------------------------------------
CLASS_NAMES = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

BAND_NAMES = ["B2_Blue", "B3_Green", "B4_Red", "B8_NIR", "B11_SWIR1", "B12_SWIR2"]

# Approximate class-mean surface reflectance signatures (scaled 0-1) per
# band [Blue, Green, Red, NIR, SWIR1, SWIR2], loosely based on typical
# Sentinel-2 spectral behaviour reported in remote-sensing literature.
CLASS_SIGNATURES = {
    "water":               [0.045, 0.055, 0.035, 0.020, 0.010, 0.006],
    "trees":               [0.032, 0.058, 0.040, 0.360, 0.165, 0.085],
    "grass":               [0.048, 0.078, 0.065, 0.310, 0.230, 0.135],
    "flooded_vegetation":  [0.042, 0.068, 0.050, 0.280, 0.150, 0.090],
    "crops":               [0.050, 0.086, 0.072, 0.330, 0.245, 0.150],
    "shrub_and_scrub":     [0.058, 0.084, 0.088, 0.250, 0.255, 0.175],
    "built":               [0.090, 0.108, 0.128, 0.195, 0.225, 0.190],
    "bare":                [0.120, 0.142, 0.168, 0.225, 0.280, 0.235],
    "snow_and_ice":        [0.500, 0.510, 0.495, 0.460, 0.130, 0.075],
}

# Deliberately overlapping spectrally-similar classes (grass/crops/shrub,
# trees/flooded_vegetation, built/bare) so the classifier faces realistic
# confusion, similar to real Dynamic World / Sentinel-2 class overlap.
CLASS_SPECTRAL_STD = 0.068   # per-band random reflectance noise (sensor + within-class variability)
SPATIAL_NOISE_SIGMA = 1.2     # smoothing sigma controlling texture correlation length


def _make_label_map(height, width, n_classes, smooth_sigma=18.0, seed=0):
    """Create a spatially-clustered land-cover label map. Each class gets a
    smooth random 'suitability' field (white noise heavily low-pass
    filtered), and every pixel is assigned to the class with the highest
    local suitability. This produces large, irregular, Voronoi-like patches
    that resemble real land-cover parcels, instead of salt-and-pepper noise."""
    rng = np.random.default_rng(seed)
    score = np.zeros((n_classes, height, width), dtype=np.float32)

    for c in range(n_classes):
        field = rng.normal(0, 1, size=(height, width)).astype(np.float32)
        field = gaussian_filter(field, sigma=smooth_sigma)
        field = (field - field.mean()) / (field.std() + 1e-6)  # normalize to unit scale
        # small amount of finer-grained noise on top for irregular boundaries
        fine = gaussian_filter(rng.normal(0, 1, size=(height, width)).astype(np.float32), sigma=smooth_sigma / 4)
        fine = (fine - fine.mean()) / (fine.std() + 1e-6)
        score[c] = 0.8 * field + 0.2 * fine

    # Give a relative-area bias so classes aren't perfectly balanced
    # (more realistic: crops/trees/grass larger, snow/flooded smaller)
    area_bias = np.array([0.35, 0.55, 0.45, -0.35, 0.40, 0.15, 0.05, -0.05, -0.55])[:n_classes]
    score += area_bias[:, None, None]

    labels = np.argmax(score, axis=0).astype(np.uint8)
    return labels


def generate_scene(height=300, width=300, seed=42):
    """
    Returns
    -------
    image : np.ndarray, shape (H, W, 6), float32, reflectance 0-1
    labels: np.ndarray, shape (H, W), uint8, values 0-8 (Dynamic World classes)
    """
    rng = np.random.default_rng(seed)
    n_classes = len(CLASS_NAMES)

    labels = _make_label_map(height, width, n_classes, seed=seed)

    sig_matrix = np.array([CLASS_SIGNATURES[c] for c in CLASS_NAMES], dtype=np.float32)  # (9,6)
    n_bands = sig_matrix.shape[1]

    image = sig_matrix[labels]  # (H, W, 6) — base reflectance per pixel from its class

    # Per-pixel sensor/spectral noise
    image = image + rng.normal(0, CLASS_SPECTRAL_STD, size=image.shape).astype(np.float32)

    # Spatially-correlated illumination / texture noise (adds realistic texture per band)
    for b in range(n_bands):
        tex = rng.normal(0, 1, size=(height, width)).astype(np.float32)
        tex = gaussian_filter(tex, sigma=SPATIAL_NOISE_SIGMA)
        tex = tex / (tex.std() + 1e-6) * 0.012
        image[:, :, b] += tex

    # Mixed-pixel effect at class boundaries: blend each pixel a little with
    # the local neighbourhood mean spectrum to simulate the ~10-20m mixing
    # that real Sentinel-2 boundary pixels exhibit.
    blended = np.empty_like(image)
    for b in range(n_bands):
        blended[:, :, b] = gaussian_filter(image[:, :, b], sigma=0.9)
    image = 0.74 * image + 0.26 * blended

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image, labels


def scene_to_rgb(image):
    """Simple true-colour-ish stretch (Red=B4, Green=B3, Blue=B2) for display."""
    rgb = image[:, :, [2, 1, 0]]
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return rgb


if __name__ == "__main__":
    img, lab = generate_scene()
    print("image shape:", img.shape, "labels shape:", lab.shape)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:20s} pixels={np.sum(lab == i):6d}  ({100*np.mean(lab==i):.1f}%)")
