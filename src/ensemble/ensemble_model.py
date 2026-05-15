"""Ensemble model: combines base model predictions via stacking and calibration.

The ensemble pipeline:
1. Collect out-of-fold predictions from all base models
2. Optimize blend weights (minimize log loss)
3. Train stacking meta-learner on base model predictions
4. Calibrate final probabilities (isotonic regression)

This produces the final calibrated probabilities used for prediction
and downstream simulation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.models.base_model import BasePredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel:
    """Ensemble combiner with weighted averaging, stacking, and calibration.

    Three combination methods (used in sequence):
    1. **Weighted average**: Optimize weights to minimize log loss
    2. **Stacking**: Train logistic regression on base model predictions
    3. **Calibration**: Isotonic regression on stacking output

    The ensemble is trained on validation-set predictions (out-of-fold)
    to prevent overfitting to the training data.

    Attributes:
        models: Dictionary of trained base models.
        blend_weights: Optimized weights for weighted averaging.
        meta_model: Stacking meta-learner (logistic regression).
        calibrators: Per-class isotonic regression calibrators.
    """

    def __init__(self, models: dict[str, BasePredictor] | None = None) -> None:
        self.models: dict[str, BasePredictor] = models or {}
        self.blend_weights: dict[str, float] = {}
        self.meta_model: LogisticRegression | None = None
        self.calibrators: list[IsotonicRegression] = []
        self._fitted = False

    def add_model(self, name: str, model: BasePredictor) -> None:
        """Add a base model to the ensemble.

        Args:
            name: Model identifier.
            model: Trained BasePredictor instance.
        """
        self.models[name] = model

    def _collect_predictions(
        self, X: pd.DataFrame | np.ndarray, model_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Collect predictions from all base models.

        Args:
            X: Feature matrix.
            model_names: Optional subset of models to use.

        Returns:
            Dictionary mapping model name to prediction array (n, 3).
        """
        names = model_names or list(self.models.keys())
        predictions = {}
        for name in names:
            if name in self.models:
                try:
                    pred = self.models[name].predict_proba(X)
                    predictions[name] = pred
                except Exception as e:
                    logger.warning(f"model_{name}_prediction_failed", error=str(e))
        return predictions

    def optimize_weights(
        self,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> dict[str, float]:
        """Optimize blend weights to minimize log loss.

        Uses constrained optimization: weights sum to 1, all non-negative.

        Args:
            predictions: Per-model prediction arrays.
            y_true: True labels.

        Returns:
            Optimized weights per model.
        """
        model_names = list(predictions.keys())
        n_models = len(model_names)
        pred_stack = np.stack([predictions[name] for name in model_names])

        def objective(weights: np.ndarray) -> float:
            # Weighted average across models
            blended = np.tensordot(weights, pred_stack, axes=(0, 0))
            blended = np.clip(blended, 1e-15, 1 - 1e-15)
            # Renormalize
            blended /= blended.sum(axis=1, keepdims=True)
            return log_loss(y_true, blended)

        # Initial weights: uniform
        x0 = np.ones(n_models) / n_models

        # Constraints: sum to 1
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0.01, 1.0)] * n_models  # Minimum 1% weight

        result = minimize(
            objective, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200},
        )

        weights = dict(zip(model_names, result.x.tolist()))
        self.blend_weights = weights

        logger.info(
            "blend_weights_optimized",
            weights={k: round(v, 4) for k, v in weights.items()},
            log_loss=round(result.fun, 6),
        )
        return weights

    def weighted_average(
        self, predictions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute weighted average of model predictions.

        Args:
            predictions: Per-model prediction arrays.

        Returns:
            Blended prediction array of shape (n, 3).
        """
        result = np.zeros_like(list(predictions.values())[0])
        total_weight = 0.0

        for name, pred in predictions.items():
            weight = self.blend_weights.get(name, 1.0 / len(predictions))
            result += weight * pred
            total_weight += weight

        result /= total_weight
        # Renormalize rows
        result /= result.sum(axis=1, keepdims=True)
        return result

    def fit_stacking(
        self,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Train the stacking meta-learner.

        The meta-learner takes concatenated base model predictions as input
        and learns optimal combination weights with interaction effects.

        Uses logistic regression with L2 regularization — simple enough
        to avoid overfitting on the relatively small validation set.

        Args:
            predictions: Per-model out-of-fold predictions.
            y_true: True labels.
        """
        # Stack all predictions into meta-features
        meta_features = np.hstack([predictions[name] for name in sorted(predictions.keys())])

        self.meta_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )
        self.meta_model.fit(meta_features, y_true)

        meta_pred = self.meta_model.predict_proba(meta_features)
        stacking_loss = log_loss(y_true, meta_pred)

        logger.info("stacking_meta_learner_fitted", log_loss=round(stacking_loss, 6))

    def fit_calibration(
        self,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Fit per-class isotonic regression calibrators.

        Isotonic regression maps raw probabilities to calibrated
        probabilities using a monotonic step function. It's non-parametric
        and works well with enough data (>500 samples).

        Args:
            predictions: Base model predictions.
            y_true: True labels.
        """
        # Get stacking output
        if self.meta_model is not None:
            meta_features = np.hstack(
                [predictions[name] for name in sorted(predictions.keys())]
            )
            raw_probs = self.meta_model.predict_proba(meta_features)
        else:
            raw_probs = self.weighted_average(predictions)

        self.calibrators = []
        for cls in range(3):
            iso = IsotonicRegression(
                y_min=0.01, y_max=0.99, out_of_bounds="clip",
            )
            binary_target = (y_true == cls).astype(float)
            iso.fit(raw_probs[:, cls], binary_target)
            self.calibrators.append(iso)

        # Verify calibration
        calibrated = self._apply_calibration(raw_probs)
        cal_loss = log_loss(y_true, calibrated)
        logger.info("calibration_fitted", log_loss=round(cal_loss, 6))

    def _apply_calibration(self, probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration to raw probabilities.

        Args:
            probs: Raw probability array (n, 3).

        Returns:
            Calibrated probability array (n, 3).
        """
        if not self.calibrators:
            return probs

        calibrated = np.column_stack([
            self.calibrators[cls].predict(probs[:, cls])
            for cls in range(3)
        ])

        # Renormalize to sum to 1
        calibrated = np.clip(calibrated, 1e-10, None)
        calibrated /= calibrated.sum(axis=1, keepdims=True)

        return calibrated

    def fit(
        self,
        X_val: pd.DataFrame | np.ndarray,
        y_val: np.ndarray,
        X_cal: pd.DataFrame | np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
        precomputed_predictions: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Fit the complete ensemble pipeline.

        Args:
            X_val: Validation features (for weight optimization + stacking).
            y_val: Validation labels.
            X_cal: Calibration features (separate from validation).
            y_cal: Calibration labels.
            precomputed_predictions: Optional pre-computed predictions.

        Returns:
            Ensemble training metadata.
        """
        # Step 1: Collect base model predictions
        val_preds = precomputed_predictions or self._collect_predictions(X_val)

        if not val_preds:
            raise ValueError("No model predictions available")

        # Step 2: Optimize blend weights
        self.optimize_weights(val_preds, y_val)

        # Step 3: Fit stacking meta-learner
        self.fit_stacking(val_preds, y_val)

        # Step 4: Fit calibration (use calibration set if available)
        if X_cal is not None and y_cal is not None:
            cal_preds = self._collect_predictions(X_cal)
            self.fit_calibration(cal_preds, y_cal)
        else:
            self.fit_calibration(val_preds, y_val)

        self._fitted = True

        # Compute final ensemble performance
        final_pred = self.predict(X_val, precomputed=val_preds)
        ensemble_loss = log_loss(y_val, final_pred)

        metadata = {
            "ensemble_log_loss": round(ensemble_loss, 6),
            "blend_weights": {k: round(v, 4) for k, v in self.blend_weights.items()},
            "n_models": len(self.models),
        }

        logger.info("ensemble_fitted", **metadata)
        return metadata

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        precomputed: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Generate calibrated ensemble predictions.

        Args:
            X: Feature matrix.
            precomputed: Optional pre-computed base model predictions.

        Returns:
            Calibrated probability array of shape (n, 3).
        """
        predictions = precomputed or self._collect_predictions(X)

        if self.meta_model is not None:
            # Use stacking meta-learner
            meta_features = np.hstack(
                [predictions[name] for name in sorted(predictions.keys())]
            )
            raw_probs = self.meta_model.predict_proba(meta_features)
        else:
            # Fall back to weighted average
            raw_probs = self.weighted_average(predictions)

        # Apply calibration
        calibrated = self._apply_calibration(raw_probs)

        return calibrated

    def save(self, path: Path) -> None:
        """Save ensemble metadata (weights, calibrators). Base models saved separately."""
        import pickle

        path.mkdir(parents=True, exist_ok=True)

        with open(path / "ensemble_weights.json", "w") as f:
            json.dump(self.blend_weights, f, indent=2)

        if self.meta_model is not None:
            with open(path / "meta_model.pkl", "wb") as f:
                pickle.dump(self.meta_model, f)

        if self.calibrators:
            with open(path / "calibrators.pkl", "wb") as f:
                pickle.dump(self.calibrators, f)

        logger.info("ensemble_saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load ensemble metadata."""
        import pickle

        with open(path / "ensemble_weights.json") as f:
            self.blend_weights = json.load(f)

        meta_path = path / "meta_model.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                self.meta_model = pickle.load(f)

        cal_path = path / "calibrators.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                self.calibrators = pickle.load(f)

        self._fitted = True
        logger.info("ensemble_loaded", path=str(path))
