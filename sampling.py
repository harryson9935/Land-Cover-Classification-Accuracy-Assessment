"""
sampling.py
------------
Configurable pixel sampling strategies for building train/test sets from a
labeled raster scene.

Strategies
----------
random      : pixels are drawn uniformly at random across the whole scene,
              irrespective of class. Fast, simple, but under-represents rare
              classes (e.g. snow_and_ice, flooded_vegetation).
stratified  : a fixed number (or proportion) of pixels is drawn per class,
              guaranteeing minority classes are adequately represented in
              training.
"""

import numpy as np


def random_sample(labels, n_samples, seed=0):
    """Uniform random sampling of pixel coordinates, independent of class."""
    rng = np.random.default_rng(seed)
    h, w = labels.shape
    flat_idx = rng.choice(h * w, size=min(n_samples, h * w), replace=False)
    rows, cols = np.unravel_index(flat_idx, (h, w))
    return rows, cols


def stratified_sample(labels, n_per_class, seed=0):
    """Draw up to `n_per_class` pixels from every class (fewer if a class
    doesn't have enough pixels)."""
    rng = np.random.default_rng(seed)
    rows_all, cols_all = [], []
    for c in np.unique(labels):
        rr, cc = np.where(labels == c)
        n = min(n_per_class, len(rr))
        idx = rng.choice(len(rr), size=n, replace=False)
        rows_all.append(rr[idx])
        cols_all.append(cc[idx])
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    return rows, cols


def train_test_split_coords(rows, cols, labels, test_size=0.3, seed=0, stratify=True):
    """Split sampled pixel coordinates into train/test sets."""
    from sklearn.model_selection import train_test_split
    y = labels[rows, cols]
    idx = np.arange(len(rows))
    strat = y if stratify else None
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=strat
    )
    return (rows[idx_train], cols[idx_train]), (rows[idx_test], cols[idx_test])


def get_samples(labels, strategy="stratified", n_samples=6000, seed=0):
    """
    Unified entry point.

    strategy   : 'random' or 'stratified'
    n_samples  : total budget for 'random'; per-class budget for 'stratified'
                 (per-class budget is derived so total is comparable across
                 strategies: n_samples // n_classes)
    """
    n_classes = len(np.unique(labels))
    if strategy == "random":
        rows, cols = random_sample(labels, n_samples, seed=seed)
    elif strategy == "stratified":
        per_class = max(1, n_samples // n_classes)
        rows, cols = stratified_sample(labels, per_class, seed=seed)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    return rows, cols


if __name__ == "__main__":
    from data_generator import generate_scene
    img, lab = generate_scene(height=200, width=200)
    for strat in ["random", "stratified"]:
        rows, cols = get_samples(lab, strategy=strat, n_samples=3600)
        y = lab[rows, cols]
        print(strat, "total samples:", len(rows))
        vals, counts = np.unique(y, return_counts=True)
        print(" per-class counts:", dict(zip(vals.tolist(), counts.tolist())))
