"""Gradient Boosted Decision Tree models: XGBoost, LightGBM, CatBoost.

These three GBDT variants form the core of the ensemble. Each has
different strengths:
- XGBoost: robust, widely validated, good with regularization
- LightGBM: fastest training, leaf-wise growth, handles sparse data
- CatBoost: best with categoricals, ordered boosting reduces overfitting

All three support GPU training and handle missing values natively.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.models.base_model import BasePredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


# -- XGBoost Model ---------------------------------------------------------


class XGBoostPredictor(BasePredictor):
    """XGBoost multi-class classifier for match outcome prediction.

    Uses multi:softprob objective with GPU acceleration when available.
    Handles missing values natively (learns optimal split direction).
    """

    name = "xgboost"

    def __init__(self, params: dict[str, Any] | None = None, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu
        self.default_params: dict[str, Any] = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "colsample_bylevel": 0.7,
            "gamma": 0.1,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if use_gpu:
            self.default_params.update({
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "gpu_id": 0,
                "n_jobs": 1,
            })
        else:
            self.default_params["tree_method"] = "hist"

        if params:
            self.default_params.update(params)

        self.model: Any = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        early_stopping_rounds: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import xgboost as xgb

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        params = {**self.default_params}
        n_estimators = params.pop("n_estimators", 1000)

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            **params,
        )

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set if eval_set else None,
            verbose=False,
        )

        metadata: dict[str, Any] = {
            "best_iteration": getattr(self.model, "best_iteration", n_estimators),
            "n_features": X_train.shape[1],
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict_proba(X_val)
            metadata["val_log_loss"] = log_loss(y_val, val_pred)

        logger.info(
            "xgboost_trained",
            **metadata,
        )
        return metadata

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path / "xgboost_model.ubj"))
        with open(path / "xgboost_meta.json", "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "params": self.default_params,
            }, f, indent=2)
        logger.info("xgboost_saved", path=str(path))

    def load(self, path: Path) -> None:
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path / "xgboost_model.ubj"))
        meta_path = path / "xgboost_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
        logger.info("xgboost_loaded", path=str(path))

    def feature_importance(self) -> dict[str, float] | None:
        if self.model is None or not self.feature_names:
            return None
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))

    def get_params(self) -> dict[str, Any]:
        return {**self.default_params}


# -- LightGBM Model --------------------------------------------------------


class LightGBMPredictor(BasePredictor):
    """LightGBM multi-class classifier.

    Leaf-wise growth typically converges faster than level-wise (XGBoost).
    Better handling of sparse features and native categorical support.
    """

    name = "lightgbm"

    def __init__(self, params: dict[str, Any] | None = None, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu
        self.default_params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "n_estimators": 1000,
            "num_leaves": 63,
            "max_depth": -1,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "min_split_gain": 0.01,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if use_gpu:
            self.default_params.update({
                "device": "gpu",
                "gpu_use_dp": True,
            })

        if params:
            self.default_params.update(params)

        self.model: Any = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        early_stopping_rounds: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import lightgbm as lgb

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        params = {**self.default_params}
        n_estimators = params.pop("n_estimators", 1000)

        callbacks = [lgb.log_evaluation(period=0)]
        if early_stopping_rounds > 0 and X_val is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

        self.model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set if eval_set else None,
            callbacks=callbacks,
        )

        metadata: dict[str, Any] = {
            "best_iteration": getattr(self.model, "best_iteration_", n_estimators),
            "n_features": X_train.shape[1],
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict_proba(X_val)
            metadata["val_log_loss"] = log_loss(y_val, val_pred)

        logger.info("lightgbm_trained", **metadata)
        return metadata

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path / "lightgbm_model.txt"))
        with open(path / "lightgbm_meta.json", "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "params": self.default_params,
            }, f, indent=2)
        logger.info("lightgbm_saved", path=str(path))

    def load(self, path: Path) -> None:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(path / "lightgbm_model.txt"))
        self.model = lgb.LGBMClassifier()
        self.model._Booster = booster
        self.model.fitted_ = True
        self.model._n_classes = 3
        meta_path = path / "lightgbm_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
        logger.info("lightgbm_loaded", path=str(path))

    def feature_importance(self) -> dict[str, float] | None:
        if self.model is None or not self.feature_names:
            return None
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))

    def get_params(self) -> dict[str, Any]:
        return {**self.default_params}


# -- CatBoost Model --------------------------------------------------------


class CatBoostPredictor(BasePredictor):
    """CatBoost multi-class classifier.

    Best native categorical feature handling via target statistics
    with ordered boosting. Most robust to overfitting among GBDT models.
    """

    name = "catboost"

    def __init__(self, params: dict[str, Any] | None = None, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu
        self.default_params: dict[str, Any] = {
            "loss_function": "MultiClass",
            "classes_count": 3,
            "iterations": 1000,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "bagging_temperature": 0.8,
            "random_strength": 1.0,
            "border_count": 128,
            "random_seed": 42,
            "verbose": 0,
            "allow_writing_files": False,
        }
        if use_gpu:
            self.default_params.update({
                "task_type": "GPU",
                "devices": "0",
                "bootstrap_type": "Poisson",
            })

        if params:
            self.default_params.update(params)

        self.model: Any = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        early_stopping_rounds: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from catboost import CatBoostClassifier, Pool

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        self.model = CatBoostClassifier(**self.default_params)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(X_val, y_val)

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose=False,
        )

        metadata: dict[str, Any] = {
            "best_iteration": self.model.get_best_iteration() or self.default_params.get("iterations", 1000),
            "n_features": X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0]),
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict_proba(X_val)
            metadata["val_log_loss"] = log_loss(y_val, val_pred)

        logger.info("catboost_trained", **metadata)
        return metadata

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path / "catboost_model.cbm"))
        with open(path / "catboost_meta.json", "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "params": self.default_params,
            }, f, indent=2, default=str)
        logger.info("catboost_saved", path=str(path))

    def load(self, path: Path) -> None:
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier()
        self.model.load_model(str(path / "catboost_model.cbm"))
        meta_path = path / "catboost_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
        logger.info("catboost_loaded", path=str(path))

    def feature_importance(self) -> dict[str, float] | None:
        if self.model is None or not self.feature_names:
            return None
        importance = self.model.get_feature_importance()
        return dict(zip(self.feature_names, importance.tolist()))

    def get_params(self) -> dict[str, Any]:
        return {**self.default_params}
