"""
classifier.py
--------------
Random Forest training / full-scene prediction wrapper.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_rf(X_train, y_train, n_estimators=300, max_depth=None, seed=0):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    return clf


def predict_full_scene(clf, feats):
    """feats: (H, W, F) -> returns (H, W) predicted label map."""
    h, w, f = feats.shape
    flat = feats.reshape(-1, f)
    preds = clf.predict(flat)
    return preds.reshape(h, w)


def feature_importances(clf, feature_names, top_n=15):
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], float(importances[i])) for i in order]
