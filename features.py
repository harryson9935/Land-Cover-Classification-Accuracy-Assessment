"""
features.py
------------
Window-based spectral-spatial feature extraction for pixel-wise land cover
classification.

For every pixel, a square window of size `window_size` (1, 3, 5, 7, ...) is
centred on it. From that window, per band we compute:
    - mean            (spectral-spatial context)
    - std             (local texture / homogeneity)
    - min, max        (range within neighbourhood)
    - center pixel value (raw spectral value, unaffected by smoothing)

window_size = 1 reduces to plain per-pixel spectral classification (no
spatial context) — this is the baseline used for comparison against the
spectral-spatial windows.

Two spectral indices are also appended (computed per-pixel then windowed):
    - NDVI = (NIR - Red) / (NIR + Red)
    - NDWI = (Green - NIR) / (Green + NIR)
"""

import numpy as np
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter


def _ndvi(image):
    red, nir = image[:, :, 2], image[:, :, 3]
    return (nir - red) / (nir + red + 1e-6)


def _ndwi(image):
    green, nir = image[:, :, 1], image[:, :, 3]
    return (green - nir) / (green + nir + 1e-6)


def extract_features(image, window_size=3, band_names=None):
    """
    Parameters
    ----------
    image : (H, W, C) float32 array of reflectance values
    window_size : odd int, size of the spatial window (1 = pixel-only)
    band_names : optional list of C band display names (defaults to B0..Bn-1)

    Returns
    -------
    feats : (H, W, F) float32 array of stacked features
    feature_names : list[str]
    """
    h, w, c = image.shape
    ndvi = _ndvi(image)[:, :, None]
    ndwi = _ndwi(image)[:, :, None]
    full = np.concatenate([image, ndvi, ndwi], axis=2)
    n_bands = full.shape[2]
    if band_names is None:
        band_names = [f"B{i}" for i in range(c)]
    band_labels = list(band_names) + ["NDVI", "NDWI"]

    feature_layers = []
    feature_names = []

    # Always include the raw (center-pixel) spectral values
    for i in range(n_bands):
        feature_layers.append(full[:, :, i])
        feature_names.append(f"{band_labels[i]}_center")

    if window_size > 1:
        for i in range(n_bands):
            band = full[:, :, i]
            mean = uniform_filter(band, size=window_size, mode="reflect")
            sq_mean = uniform_filter(band ** 2, size=window_size, mode="reflect")
            var = np.clip(sq_mean - mean ** 2, 0, None)
            std = np.sqrt(var)
            bmin = minimum_filter(band, size=window_size, mode="reflect")
            bmax = maximum_filter(band, size=window_size, mode="reflect")

            feature_layers += [mean, std, bmin, bmax]
            feature_names += [
                f"{band_labels[i]}_mean_w{window_size}",
                f"{band_labels[i]}_std_w{window_size}",
                f"{band_labels[i]}_min_w{window_size}",
                f"{band_labels[i]}_max_w{window_size}",
            ]

    feats = np.stack(feature_layers, axis=-1).astype(np.float32)
    return feats, feature_names


if __name__ == "__main__":
    from data_generator import generate_scene
    img, lab = generate_scene(height=64, width=64)
    for ws in [1, 3, 5, 7]:
        feats, names = extract_features(img, window_size=ws)
        print(f"window_size={ws}: feature array shape={feats.shape}, n_features={len(names)}")
