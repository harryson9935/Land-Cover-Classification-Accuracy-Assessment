"""
accuracy.py
------------
Accuracy assessment utilities: overall accuracy, Cohen's Kappa, confusion
matrix, and per-class producer's / user's accuracy (the standard remote
sensing accuracy-assessment trio).
"""

import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score


def assess(y_true, y_pred, class_names):
    n_classes = len(class_names)
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Producer's accuracy (recall per class) = diag / row sum (reference totals)
    row_sums = cm.sum(axis=1)
    producers = np.divide(np.diag(cm), row_sums, out=np.zeros(n_classes), where=row_sums != 0)

    # User's accuracy (precision per class) = diag / col sum (predicted totals)
    col_sums = cm.sum(axis=0)
    users = np.divide(np.diag(cm), col_sums, out=np.zeros(n_classes), where=col_sums != 0)

    f1 = np.divide(
        2 * producers * users, producers + users,
        out=np.zeros(n_classes), where=(producers + users) != 0
    )

    per_class = {
        class_names[i]: {
            "producers_accuracy": float(producers[i]),
            "users_accuracy": float(users[i]),
            "f1_score": float(f1[i]),
            "support": int(row_sums[i]),
        }
        for i in range(n_classes)
    }

    return {
        "overall_accuracy": float(oa),
        "kappa": float(kappa),
        "confusion_matrix": cm,
        "per_class": per_class,
    }
