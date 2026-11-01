import math
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import FEATURES
from .metrics import latency


def rules(x):
    """Return the fixed numeric toxicity-proxy probability."""
    return float(min(0.95, max(0.05, 0.15 + 0.25 * abs(x[1]) + 0.2 * abs(x[2]) + 0.1 * abs(x[3]))))


def fit_classifier(rows, indices):
    """Fit chronological scaling, logistic regression and purged isotonic calibration."""
    cut = rows[indices[len(indices) * 2 // 3]]["ns"]
    train = [i for i in indices if rows[i]["label_end_ns"] < cut]
    calibrate = [i for i in indices if rows[i]["ns"] >= cut]
    x = np.array([r["x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    if len(train) < 10 or len(calibrate) < 10 or len(set(y[train])) < 2:
        raise ValueError("insufficient training/calibration observations or label classes")
    scaler = StandardScaler().fit(x[train])
    model = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(
        scaler.transform(x[train]), y[train]
    )
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        model.predict_proba(scaler.transform(x[calibrate]))[:, 1], y[calibrate]
    )
    return {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weights": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "fit_n": len(train),
        "calibration_n": len(calibrate),
        "fit_end_ns": cut,
        "calibration_end_ns": max(rows[i]["label_end_ns"] for i in calibrate),
    }


def classifier(x, model):
    """Evaluate stored logistic coefficients and isotonic probability levels."""
    z = model["intercept"] + sum(
        w * (v - m) / s
        for w, v, m, s in zip(model["weights"], x, model["mean"], model["scale"], strict=True)
    )
    raw = 1 / (1 + math.exp(-max(-700, min(700, z))))
    return float(np.interp(raw, model["isotonic_x"], model["isotonic_y"]))


def benchmark_local(rows, model, repetitions=2000):
    """Retain every local inference timing rather than select a favourable burst."""
    result = {}
    for name, predict in (("rules", rules), ("classifier", lambda x: classifier(x, model))):
        samples = []
        for i in range(repetitions + 100):
            start = time.perf_counter_ns()
            p = predict(rows[i % len(rows)]["x"])
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if not 0 <= p <= 1:
                raise ValueError("invalid probability")
            if i >= 100:
                samples.append(elapsed)
        result[name] = latency(samples)
        result[name]["cost_per_million_usd"] = None
        result[name]["cost_note"] = "no API charge; hardware/energy cost not measured"
    return result


def select_abstention(y, p):
    # Proper-score objective with an explicit fixed rejection cost; never test P&L.
    """Choose a confidence threshold using validation loss and rejection cost."""
    candidates = []
    for threshold in (0.0, 0.6, 0.7, 0.8, 0.9):
        accept = np.maximum(p, 1 - p) >= threshold
        loss = np.where(accept, (p - y) ** 2, 0.20).mean()
        candidates.append(
            {"threshold": threshold, "loss": float(loss), "coverage": float(accept.mean())}
        )
    return min(candidates, key=lambda r: r["loss"])["threshold"], candidates


def fit_stack(y, first, second):
    """Select the classifier weight using validation Brier loss only."""
    candidates = [
        {"weight": float(w), "brier": float(np.mean((w * first + (1 - w) * second - y) ** 2))}
        for w in np.linspace(0, 1, 11)
    ]
    return min(candidates, key=lambda r: r["brier"])["weight"], candidates
