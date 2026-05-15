"""Walk-forward temporal cross-validation for time-series data.

Standard k-fold CV is INVALID for time-series because it leaks future
information into training. Walk-forward validation ensures that:
- Training data is always BEFORE validation data
- Each fold expands the training set (growing window)
- No match result from the future is ever seen during training

This is the ONLY acceptable validation strategy for match prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TemporalFold:
    """A single fold in walk-forward validation."""

    fold_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    n_train: int
    n_val: int


def create_temporal_folds(
    dates: pd.Series,
    n_folds: int = 5,
    min_train_size: int = 100,
    val_months: int = 12,
) -> list[TemporalFold]:
    """Create walk-forward temporal CV folds.

    Strategy:
    - Sort all matches by date
    - Reserve the last `val_months` of data per fold as validation
    - Each fold's training window starts from the beginning
    - Each fold's training end advances forward

    Example with 5 folds over 10 years of data:
        Fold 1: Train [2010-2016], Val [2016-2018]
        Fold 2: Train [2010-2018], Val [2018-2020]
        Fold 3: Train [2010-2020], Val [2020-2021]
        Fold 4: Train [2010-2021], Val [2021-2022]
        Fold 5: Train [2010-2022], Val [2022-2023]

    Args:
        dates: Series of match dates (pd.Timestamp).
        n_folds: Number of validation folds.
        min_train_size: Minimum number of training samples required.
        val_months: Number of months per validation window.

    Returns:
        List of TemporalFold objects.
    """
    dates = pd.to_datetime(dates)
    sorted_idx = dates.argsort()
    sorted_dates = dates.iloc[sorted_idx]

    min_date = sorted_dates.min()
    max_date = sorted_dates.max()
    total_range = (max_date - min_date).days

    # Compute fold boundaries
    # Reserve space for n_folds validation windows at the end
    val_days = val_months * 30
    available_train_end = max_date - pd.Timedelta(days=val_days)
    train_range = (available_train_end - min_date).days

    # Space fold boundaries evenly across the trainable range
    fold_step = train_range // n_folds

    folds: list[TemporalFold] = []

    for i in range(n_folds):
        train_end_date = min_date + pd.Timedelta(
            days=min(train_range, (i + 1) * fold_step)
        )
        val_start_date = train_end_date
        val_end_date = train_end_date + pd.Timedelta(days=val_days)

        train_mask = dates < train_end_date
        val_mask = (dates >= val_start_date) & (dates < val_end_date)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) < min_train_size or len(val_idx) < 10:
            continue

        fold = TemporalFold(
            fold_id=i,
            train_start=str(dates[train_idx[0]].date()),
            train_end=str(train_end_date.date()),
            val_start=str(val_start_date.date()),
            val_end=str(val_end_date.date()),
            train_idx=train_idx,
            val_idx=val_idx,
            n_train=len(train_idx),
            n_val=len(val_idx),
        )
        folds.append(fold)

        logger.info(
            "temporal_fold_created",
            fold_id=i,
            train_period=f"{fold.train_start} -> {fold.train_end}",
            val_period=f"{fold.val_start} -> {fold.val_end}",
            n_train=fold.n_train,
            n_val=fold.n_val,
        )

    logger.info("temporal_folds_created", n_folds=len(folds))
    return folds


def create_train_val_test_split(
    df: pd.DataFrame,
    test_months: int = 6,
    cal_months: int = 6,
    val_months: int = 12,
) -> dict[str, pd.DataFrame]:
    """Create a single train/val/cal/test split for final model training.

    Split layout (going backward from latest date):
    |--- Train ---|--- Validation ---|--- Calibration ---|--- Test ---|
                  ^                  ^                   ^           ^
             T-test-cal-val     T-test-cal            T-test       T(now)

    Args:
        df: Feature matrix with 'match_date' column.
        test_months: Months reserved for final test set.
        cal_months: Months reserved for calibration set.
        val_months: Months reserved for validation set.

    Returns:
        Dictionary with 'train', 'val', 'cal', 'test' DataFrames.
    """
    dates = pd.to_datetime(df["match_date"])
    max_date = dates.max()

    test_cutoff = max_date - pd.DateOffset(months=test_months)
    cal_cutoff = test_cutoff - pd.DateOffset(months=cal_months)
    val_cutoff = cal_cutoff - pd.DateOffset(months=val_months)

    splits = {
        "train": df[dates < val_cutoff].copy(),
        "val": df[(dates >= val_cutoff) & (dates < cal_cutoff)].copy(),
        "cal": df[(dates >= cal_cutoff) & (dates < test_cutoff)].copy(),
        "test": df[dates >= test_cutoff].copy(),
    }

    for name, split_df in splits.items():
        date_range = ""
        if len(split_df) > 0:
            d = pd.to_datetime(split_df["match_date"])
            date_range = f"{d.min().date()} -> {d.max().date()}"
        logger.info(
            f"split_{name}",
            n_matches=len(split_df),
            date_range=date_range,
        )

    return splits


def walk_forward_evaluate(
    model_class: type,
    model_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    n_folds: int = 5,
    val_months: int = 12,
) -> dict[str, Any]:
    """Run walk-forward validation for a model.

    For each temporal fold:
    1. Train model on training portion
    2. Predict on validation portion
    3. Compute evaluation metrics

    Returns aggregated metrics across all folds.

    Args:
        model_class: Model class to instantiate.
        model_params: Hyperparameters for the model.
        X: Feature matrix.
        y: Target labels.
        dates: Match dates (for temporal splitting).
        n_folds: Number of folds.
        val_months: Months per validation window.

    Returns:
        Aggregated metrics and per-fold breakdown.
    """
    folds = create_temporal_folds(dates, n_folds=n_folds, val_months=val_months)

    fold_metrics: list[dict[str, Any]] = []
    all_val_preds: list[np.ndarray] = []
    all_val_true: list[np.ndarray] = []

    for fold in folds:
        X_train = X.iloc[fold.train_idx]
        y_train = y.iloc[fold.train_idx]
        X_val = X.iloc[fold.val_idx]
        y_val = y.iloc[fold.val_idx]

        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # Predict
        val_pred = model.predict_proba(X_val)

        # Evaluate
        metrics = evaluate_model(y_val.values, val_pred, model_name=f"fold_{fold.fold_id}")
        metrics["fold_id"] = fold.fold_id
        metrics["train_period"] = f"{fold.train_start} -> {fold.train_end}"
        metrics["val_period"] = f"{fold.val_start} -> {fold.val_end}"
        fold_metrics.append(metrics)

        all_val_preds.append(val_pred)
        all_val_true.append(y_val.values)

    # Aggregate metrics
    combined_preds = np.concatenate(all_val_preds)
    combined_true = np.concatenate(all_val_true)
    aggregate = evaluate_model(combined_true, combined_preds, model_name="walk_forward_aggregate")

    result = {
        "aggregate_metrics": aggregate,
        "per_fold_metrics": fold_metrics,
        "n_folds": len(folds),
        "total_val_samples": len(combined_true),
    }

    logger.info(
        "walk_forward_complete",
        aggregate_log_loss=aggregate["log_loss"],
        aggregate_brier=aggregate["brier_score"],
        n_folds=len(folds),
    )

    return result
