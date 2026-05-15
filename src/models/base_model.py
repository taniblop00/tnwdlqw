"""Base model interface for all predictive models.

Defines the contract that every model (XGBoost, LightGBM, CatBoost,
PyTorch NN, Poisson, Bayesian) must implement. This enables uniform
training, evaluation, and ensembling across heterogeneous model types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Abstract base class for match outcome predictors.

    All models must implement:
    - fit(): Train the model on feature matrix + targets.
    - predict_proba(): Return calibrated 3-class probabilities.
    - save() / load(): Persist and restore model artifacts.

    Optional:
    - feature_importance(): Return feature importance scores.
    - get_params(): Return current hyperparameters.
    """

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the model.

        Args:
            X_train: Training features.
            y_train: Training targets (0=home_win, 1=draw, 2=away_win).
            X_val: Validation features (for early stopping).
            y_val: Validation targets.
            **kwargs: Model-specific training arguments.

        Returns:
            Training metadata (e.g., best iteration, training loss).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict match outcome probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 3) with columns:
            [P(home_win), P(draw), P(away_win)].
            Each row sums to 1.0.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Directory to save model artifacts.
        """
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk.

        Args:
            path: Directory containing model artifacts.
        """
        ...

    def feature_importance(self) -> dict[str, float] | None:
        """Return feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not supported.
        """
        return None

    def get_params(self) -> dict[str, Any]:
        """Return current hyperparameters.

        Returns:
            Dictionary of hyperparameter names and values.
        """
        return {}

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict the most likely outcome.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted classes (0, 1, or 2).
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
