"""Evaluation metrics for probabilistic football forecasting.

All metrics focus on CALIBRATION and PROBABILISTIC ACCURACY,
not just classification accuracy. In sports betting, well-calibrated
probabilities are far more valuable than raw accuracy.

Key metrics:
- Log Loss: Primary metric — penalizes confident wrong predictions
- Brier Score: MSE of probability estimates
- Expected Calibration Error (ECE): Reliability diagram summary
- Ranked Probability Score (RPS): Ordinal-aware metric
- ROI: Simulated betting return on investment
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute multi-class log loss (cross-entropy).

    The primary evaluation metric. Lower is better.
    Random baseline ≈ 1.099 (log(3)). Good models achieve < 1.0.
    Elite models achieve < 0.95 on international football.

    Args:
        y_true: True labels (0, 1, 2).
        y_prob: Predicted probabilities, shape (n, 3).

    Returns:
        Log loss value.
    """
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return float(log_loss(y_true, y_prob))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute multi-class Brier score.

    Mean squared error of probability predictions.
    Lower is better. Random baseline ≈ 0.667.

    Brier = (1/N) × Σ_i Σ_c (p_ic - y_ic)²

    Args:
        y_true: True labels (0, 1, 2).
        y_prob: Predicted probabilities, shape (n, 3).

    Returns:
        Brier score value.
    """
    n = len(y_true)
    y_one_hot = np.zeros_like(y_prob)
    y_one_hot[np.arange(n), y_true.astype(int)] = 1
    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    """Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match observed frequencies.
    Perfect calibration: ECE = 0. Target: ECE < 0.03.

    For each predicted probability bin:
    - bin_confidence = mean predicted probability
    - bin_accuracy = fraction of correct predictions
    - ECE = weighted average of |confidence - accuracy| across bins

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities, shape (n, 3).
        n_bins: Number of probability bins.

    Returns:
        Tuple of (ECE value, list of per-bin calibration data).
    """
    # Use the predicted probability for the winning class
    predicted_class = np.argmax(y_prob, axis=1)
    max_probs = np.max(y_prob, axis=1)
    correct = (predicted_class == y_true.astype(int))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data: list[dict[str, float]] = []
    ece = 0.0

    for i in range(n_bins):
        mask = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i + 1])
        n_in_bin = mask.sum()

        if n_in_bin == 0:
            continue

        bin_accuracy = correct[mask].mean()
        bin_confidence = max_probs[mask].mean()

        ece += (n_in_bin / len(y_true)) * abs(bin_accuracy - bin_confidence)
        bin_data.append({
            "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
            "confidence": float(bin_confidence),
            "accuracy": float(bin_accuracy),
            "count": int(n_in_bin),
        })

    return float(ece), bin_data


def compute_rps(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Ranked Probability Score (RPS).

    Unlike log loss and Brier score, RPS respects the ordinal nature
    of the outcome space (home_win < draw < away_win). It penalizes
    predictions that are "further away" in the outcome ordering.

    RPS = (1/N) × (1/(K-1)) × Σ_i Σ_k (CDF_pred_ik - CDF_true_ik)²

    Lower is better. Random baseline ≈ 0.222. Good models < 0.19.

    Args:
        y_true: True labels (0, 1, 2).
        y_prob: Predicted probabilities, shape (n, 3).

    Returns:
        RPS value.
    """
    n = len(y_true)
    k = y_prob.shape[1]

    # Cumulative probabilities
    cdf_pred = np.cumsum(y_prob, axis=1)

    # True CDF
    cdf_true = np.zeros_like(y_prob)
    for i in range(n):
        for j in range(k):
            cdf_true[i, j] = 1.0 if j >= int(y_true[i]) else 0.0

    rps = np.mean(np.sum((cdf_pred - cdf_true) ** 2, axis=1) / (k - 1))
    return float(rps)


def compute_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute classification accuracy (predicted class = argmax of probs).

    This is a secondary metric. NOT the primary optimization target.
    Random baseline ≈ 33-40% (depending on class balance).

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.

    Returns:
        Accuracy (0.0 to 1.0).
    """
    y_pred = np.argmax(y_prob, axis=1)
    return float(accuracy_score(y_true, y_pred))


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
) -> dict[str, Any]:
    """Compute all evaluation metrics for a model.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities, shape (n, 3).
        model_name: Name for logging.

    Returns:
        Dictionary of all metrics.
    """
    ll = compute_log_loss(y_true, y_prob)
    brier = compute_brier_score(y_true, y_prob)
    ece, cal_bins = compute_ece(y_true, y_prob)
    rps = compute_rps(y_true, y_prob)
    acc = compute_accuracy(y_true, y_prob)

    # Class distribution of predictions
    pred_classes = np.argmax(y_prob, axis=1)
    pred_dist = {
        "pred_home_pct": float((pred_classes == 0).mean()),
        "pred_draw_pct": float((pred_classes == 1).mean()),
        "pred_away_pct": float((pred_classes == 2).mean()),
    }

    metrics = {
        "log_loss": round(ll, 6),
        "brier_score": round(brier, 6),
        "ece": round(ece, 6),
        "rps": round(rps, 6),
        "accuracy": round(acc, 4),
        "n_samples": len(y_true),
        **pred_dist,
    }

    logger.info(f"evaluation_{model_name}", **metrics)
    return metrics


def compare_to_baseline(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    baseline_name: str = "uniform",
) -> dict[str, float]:
    """Compare model performance against baselines.

    Baselines:
    - Uniform: P = [1/3, 1/3, 1/3] for all matches
    - Historical: P = [observed home_win%, draw%, away%]

    Args:
        y_true: True labels.
        y_prob: Model predictions.
        baseline_name: Which baseline to compare against.

    Returns:
        Improvement metrics over baseline.
    """
    n = len(y_true)

    if baseline_name == "uniform":
        baseline_prob = np.full((n, 3), 1 / 3)
    elif baseline_name == "historical":
        # Use observed class frequencies as baseline
        hist = np.bincount(y_true.astype(int), minlength=3) / n
        baseline_prob = np.tile(hist, (n, 1))
    else:
        baseline_prob = np.full((n, 3), 1 / 3)

    model_ll = compute_log_loss(y_true, y_prob)
    baseline_ll = compute_log_loss(y_true, baseline_prob)

    model_brier = compute_brier_score(y_true, y_prob)
    baseline_brier = compute_brier_score(y_true, baseline_prob)

    return {
        f"log_loss_improvement_vs_{baseline_name}": round(baseline_ll - model_ll, 6),
        f"log_loss_pct_improvement_vs_{baseline_name}": round(
            (baseline_ll - model_ll) / baseline_ll * 100, 2
        ),
        f"brier_improvement_vs_{baseline_name}": round(baseline_brier - model_brier, 6),
    }
