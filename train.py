"""train.py - Main entry point for the World Cup AI prediction engine.

Single command to run the entire pipeline end-to-end:
    python train.py

Pipeline steps:
1. Download international results dataset (45,000+ matches)
2. Download StatsBomb open data (300+ matches with xG)
3. Merge datasets with deduplication
4. Build 200+ temporal features (Elo, Glicko-2, xG, form, matchup)
5. Train XGBoost, LightGBM, CatBoost on walk-forward CV
6. Fit Dixon-Coles Poisson score model
7. Build calibrated ensemble (stacking + isotonic regression)
8. Evaluate against baselines (uniform, historical, Elo-only)
9. Save all models and evaluation artifacts

Usage:
    python train.py                     # Full pipeline
    python train.py --skip-ingestion    # Skip data download (use cached)
    python train.py --skip-features     # Skip feature engineering (use cached)
    python train.py --models xgboost    # Train single model
    python train.py --gpu               # Enable GPU training
    python train.py --no-statsbomb      # Skip StatsBomb (faster, no xG)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import settings
from src.ensemble.ensemble_model import EnsembleModel
from src.evaluation.metrics import compare_to_baseline, evaluate_model
from src.evaluation.walk_forward import create_train_val_test_split
from src.feature_engineering.builder import FeatureBuilder
from src.ingestion.data_merger import merge_datasets, export_merged
from src.ingestion.international_results_ingestor import InternationalResultsIngestor
from src.ingestion.statsbomb_ingestor import StatsBombIngestor
from src.models.base_model import BasePredictor
from src.models.gbdt_models import CatBoostPredictor, LightGBMPredictor, XGBoostPredictor
from src.models.poisson_model import PoissonGoalModel
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="World Cup AI - Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-ingestion", action="store_true",
        help="Skip data download (use cached data).",
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip feature engineering (use cached features).",
    )
    parser.add_argument(
        "--no-statsbomb", action="store_true",
        help="Skip StatsBomb ingestion (faster, no xG features).",
    )
    parser.add_argument(
        "--models", type=str, default="xgboost,lightgbm,catboost,poisson",
        help="Comma-separated list of models to train.",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Enable GPU training for supported models.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--min-year", type=int, default=2000,
        help="Earliest year to include from international results.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/models",
        help="Directory to save trained models.",
    )
    return parser.parse_args()


# ======================================================================
# STEP 1: DATA INGESTION
# ======================================================================

def step_1_ingest(
    skip: bool = False,
    skip_statsbomb: bool = False,
    min_year: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and merge all data sources.

    Returns:
        Tuple of (matches_df, events_df).
    """
    print("\n" + "=" * 60)
    print("  STEP 1: DATA INGESTION")
    print("=" * 60)

    processed_dir = Path("data/processed")
    merged_path = processed_dir / "merged_matches.parquet"
    events_path = processed_dir / "merged_events.parquet"

    if skip and merged_path.exists():
        print("  -> Loading cached data...")
        matches_df = pd.read_parquet(merged_path)
        events_df = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()
        print(f"  -> Loaded {len(matches_df):,} matches, {len(events_df):,} events")
        return matches_df, events_df

    # -- 1a: International Results (45,000+ matches) ---------------
    print("\n  [1a] Downloading international results dataset...")
    intl_ingestor = InternationalResultsIngestor(
        cache_dir=Path("data/raw/international_results")
    )
    intl_df = intl_ingestor.fetch_results(min_year=min_year)
    print(f"  -> {len(intl_df):,} matches from {intl_df['match_date'].min().date()} to {intl_df['match_date'].max().date()}")
    print(f"  -> {intl_df['home_team'].nunique()} teams, {intl_df['competition'].nunique()} competitions")

    # -- 1b: StatsBomb (300+ matches with xG) ---------------------
    sb_matches = None
    sb_events = pd.DataFrame()

    if not skip_statsbomb:
        print("\n  [1b] Downloading StatsBomb open data (xG events)...")
        sb_ingestor = StatsBombIngestor(
            raw_dir=Path("data/raw/statsbomb"),
            rate_limit_seconds=0.2,
        )
        sb_data = sb_ingestor.ingest_all(include_events=True, include_lineups=False)
        sb_dfs = sb_ingestor.to_dataframes(sb_data)
        sb_matches = sb_dfs["matches"]
        sb_events = sb_dfs["events"]
        print(f"  -> {len(sb_matches):,} StatsBomb matches, {len(sb_events):,} shot events")
    else:
        print("\n  [1b] Skipping StatsBomb (--no-statsbomb)")

    # -- 1c: Merge datasets ----------------------------------------
    print("\n  [1c] Merging datasets...")
    merged_df, events_df = merge_datasets(intl_df, sb_matches, sb_events)
    export_merged(merged_df, events_df, processed_dir)

    xg_count = merged_df["has_xg"].sum() if "has_xg" in merged_df.columns else 0
    print(f"  -> Merged: {len(merged_df):,} total matches ({xg_count:,} with xG)")
    print(f"  -> Date range: {merged_df['match_date'].min().date()} -> {merged_df['match_date'].max().date()}")

    return merged_df, events_df


# ======================================================================
# STEP 2: FEATURE ENGINEERING
# ======================================================================

def step_2_features(
    matches_df: pd.DataFrame,
    events_df: pd.DataFrame,
    skip: bool = False,
) -> pd.DataFrame:
    """Build the complete feature matrix.

    Returns:
        Feature matrix DataFrame with ~200 features per match.
    """
    print("\n" + "=" * 60)
    print("  STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    features_path = Path("data/features/feature_matrix.parquet")

    if skip and features_path.exists():
        print("  -> Loading cached features...")
        feature_matrix = pd.read_parquet(features_path)
        print(f"  -> Loaded {len(feature_matrix):,} matches × {len(feature_matrix.columns)} columns")
        return feature_matrix

    builder = FeatureBuilder(
        elo_config={
            "k_base": settings.elo_k_factor_competitive,
            "home_advantage": settings.elo_home_advantage,
            "initial_rating": settings.elo_initial_rating,
        },
        form_windows=settings.form_windows,
        xg_windows=settings.xg_rolling_windows,
    )

    feature_matrix = builder.build(
        matches_df=matches_df,
        events_df=events_df if not events_df.empty else None,
        output_path=features_path,
    )

    feature_cols = builder.get_feature_columns(feature_matrix)
    print(f"  -> Built {len(feature_matrix):,} matches × {len(feature_cols)} features")

    # Feature completeness report
    completeness = feature_matrix[feature_cols].notna().mean()
    good_features = (completeness > 0.5).sum()
    print(f"  -> Features with >50% coverage: {good_features}/{len(feature_cols)}")

    return feature_matrix


# ======================================================================
# STEP 3: MODEL TRAINING
# ======================================================================

def step_3_train_models(
    feature_matrix: pd.DataFrame,
    model_names: list[str],
    use_gpu: bool = False,
    output_dir: Path = Path("data/models"),
) -> dict[str, BasePredictor]:
    """Train all specified models.

    Returns:
        Dictionary of trained model instances.
    """
    print("\n" + "=" * 60)
    print("  STEP 3: MODEL TRAINING")
    print("=" * 60)

    # Create temporal splits
    splits = create_train_val_test_split(
        feature_matrix,
        test_months=settings.test_months,
        cal_months=settings.calibration_months,
        val_months=settings.validation_months,
    )

    # Extract numeric feature columns only
    feature_cols = FeatureBuilder.get_feature_columns(feature_matrix)
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(feature_matrix[c])
    ]

    print(f"\n  Using {len(numeric_cols)} numeric features")
    for name, split_df in splits.items():
        print(f"  {name:>8s}: {len(split_df):>6,} matches", end="")
        if len(split_df) > 0:
            dates = pd.to_datetime(split_df["match_date"])
            print(f"  ({dates.min().date()} -> {dates.max().date()})")
        else:
            print()

    X_train = splits["train"][numeric_cols]
    y_train = splits["train"]["target"].astype(int)
    X_val = splits["val"][numeric_cols]
    y_val = splits["val"]["target"].astype(int)
    X_test = splits["test"][numeric_cols]
    y_test = splits["test"]["target"].astype(int)

    trained_models: dict[str, BasePredictor] = {}

    # -- Train GBDT Models -----------------------------------------
    gbdt_models = {
        "xgboost": XGBoostPredictor,
        "lightgbm": LightGBMPredictor,
        "catboost": CatBoostPredictor,
    }

    for name in model_names:
        if name in gbdt_models:
            print(f"\n  Training {name}...")
            start = time.time()

            model = gbdt_models[name](use_gpu=use_gpu)
            metadata = model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                early_stopping_rounds=settings.early_stopping_rounds,
            )

            elapsed = time.time() - start

            # Evaluate on test set
            test_pred = model.predict_proba(X_test)
            test_metrics = evaluate_model(y_test.values, test_pred, model_name=f"{name}_test")

            print(f"  -> {name} trained in {elapsed:.1f}s")
            print(f"    Log Loss: {test_metrics['log_loss']:.6f}")
            print(f"    Brier:    {test_metrics['brier_score']:.6f}")
            print(f"    Accuracy: {test_metrics['accuracy']:.4f}")

            # Save model
            model_path = output_dir / name
            model.save(model_path)
            trained_models[name] = model

    # -- Train Poisson Model ---------------------------------------
    if "poisson" in model_names:
        print(f"\n  Training Poisson score model...")
        start = time.time()

        poisson = PoissonGoalModel(time_decay=0.003)
        poisson_train = splits["train"][
            ["home_team", "away_team", "home_score", "away_score", "match_date"]
        ].copy()
        poisson.fit(poisson_train, y_train)

        elapsed = time.time() - start

        # Evaluate
        poisson_test_data = splits["test"][["home_team", "away_team"]].copy()
        poisson_test_pred = poisson.predict_proba(poisson_test_data)
        poisson_metrics = evaluate_model(
            y_test.values, poisson_test_pred, model_name="poisson_test"
        )

        print(f"  -> Poisson trained in {elapsed:.1f}s")
        print(f"    Log Loss: {poisson_metrics['log_loss']:.6f}")
        print(f"    Brier:    {poisson_metrics['brier_score']:.6f}")
        print(f"    Teams:    {len(poisson.teams)}")

        poisson.save(output_dir / "poisson")
        trained_models["poisson"] = poisson

        # Print top teams
        strengths = poisson.get_team_strengths().head(15)
        print(f"\n  Top 15 teams by attack strength:")
        for _, row in strengths.iterrows():
            print(f"    {row['team']:<25s} ATK={row['attack']:+.4f}  DEF={row['defense']:+.4f}  E[G]={row['expected_goals_per_match']:.2f}")

    return trained_models


# ======================================================================
# STEP 4: ENSEMBLE & CALIBRATION
# ======================================================================

def step_4_ensemble(
    trained_models: dict[str, BasePredictor],
    feature_matrix: pd.DataFrame,
    output_dir: Path = Path("data/models"),
) -> EnsembleModel:
    """Build the calibrated ensemble."""
    print("\n" + "=" * 60)
    print("  STEP 4: ENSEMBLE & CALIBRATION")
    print("=" * 60)

    splits = create_train_val_test_split(
        feature_matrix,
        test_months=settings.test_months,
        cal_months=settings.calibration_months,
        val_months=settings.validation_months,
    )

    numeric_cols = [
        c for c in FeatureBuilder.get_feature_columns(feature_matrix)
        if pd.api.types.is_numeric_dtype(feature_matrix[c])
    ]

    X_val = splits["val"][numeric_cols]
    y_val = splits["val"]["target"].astype(int).values
    X_cal = splits["cal"][numeric_cols]
    y_cal = splits["cal"]["target"].astype(int).values

    # Collect validation predictions
    val_predictions: dict[str, np.ndarray] = {}
    for name, model in trained_models.items():
        try:
            if name == "poisson":
                val_data = splits["val"][["home_team", "away_team"]].copy()
                val_predictions[name] = model.predict_proba(val_data)
            else:
                val_predictions[name] = model.predict_proba(X_val)
            print(f"  -> {name} validation predictions collected")
        except Exception as e:
            print(f"  [X] {name} predictions failed: {e}")

    if len(val_predictions) < 2:
        print("  [!] Less than 2 models - ensemble will be limited")

    # Build ensemble
    ensemble = EnsembleModel(models=trained_models)
    ensemble_metadata = ensemble.fit(
        X_val=X_val, y_val=y_val,
        precomputed_predictions=val_predictions,
    )

    print(f"\n  Ensemble weights:")
    for name, weight in sorted(ensemble.blend_weights.items(), key=lambda x: -x[1]):
        bar = "#" * int(weight * 40)
        print(f"    {name:<12s} {weight:.4f} {bar}")

    print(f"  Ensemble Log Loss: {ensemble_metadata.get('ensemble_log_loss', 'N/A')}")

    ensemble.save(output_dir / "ensemble")
    return ensemble


# ======================================================================
# STEP 5: EVALUATION & REPORTING
# ======================================================================

def step_5_evaluate(
    trained_models: dict[str, BasePredictor],
    ensemble: EnsembleModel,
    feature_matrix: pd.DataFrame,
    output_dir: Path = Path("data/models"),
) -> dict[str, dict]:
    """Comprehensive evaluation and reporting."""
    print("\n" + "=" * 60)
    print("  STEP 5: EVALUATION & REPORTING")
    print("=" * 60)

    splits = create_train_val_test_split(
        feature_matrix,
        test_months=settings.test_months,
        cal_months=settings.calibration_months,
        val_months=settings.validation_months,
    )

    numeric_cols = [
        c for c in FeatureBuilder.get_feature_columns(feature_matrix)
        if pd.api.types.is_numeric_dtype(feature_matrix[c])
    ]

    X_test = splits["test"][numeric_cols]
    y_test = splits["test"]["target"].astype(int).values

    results: dict[str, dict] = {}

    # Per-model evaluation
    test_preds_dict: dict[str, np.ndarray] = {}
    for name, model in trained_models.items():
        try:
            if name == "poisson":
                test_data = splits["test"][["home_team", "away_team"]].copy()
                pred = model.predict_proba(test_data)
            else:
                pred = model.predict_proba(X_test)

            test_preds_dict[name] = pred
            metrics = evaluate_model(y_test, pred, model_name=name)
            uniform_comp = compare_to_baseline(y_test, pred, "uniform")
            hist_comp = compare_to_baseline(y_test, pred, "historical")
            results[name] = {**metrics, **uniform_comp, **hist_comp}
        except Exception as e:
            print(f"  [X] {name} evaluation failed: {e}")

    # Ensemble evaluation
    if test_preds_dict:
        ensemble_pred = ensemble.predict(X_test, precomputed=test_preds_dict)
        ens_metrics = evaluate_model(y_test, ensemble_pred, model_name="ensemble")
        uniform_comp = compare_to_baseline(y_test, ensemble_pred, "uniform")
        hist_comp = compare_to_baseline(y_test, ensemble_pred, "historical")
        results["ensemble"] = {**ens_metrics, **uniform_comp, **hist_comp}

    # Print summary table
    print(f"\n  {'-' * 78}")
    print(f"  {'Model':<15} {'Log Loss':>10} {'Brier':>10} {'ECE':>10} {'RPS':>10} {'Acc':>8}")
    print(f"  {'-' * 78}")

    baseline_ll = np.log(3)  # ~1.0986 (uniform baseline)
    for name, metrics in sorted(results.items(), key=lambda x: x[1].get("log_loss", 99)):
        ll = metrics.get("log_loss", -1)
        marker = " *" if name == "ensemble" else ""
        print(
            f"  {name:<15} "
            f"{ll:>10.6f} "
            f"{metrics.get('brier_score', -1):>10.6f} "
            f"{metrics.get('ece', -1):>10.6f} "
            f"{metrics.get('rps', -1):>10.6f} "
            f"{metrics.get('accuracy', -1):>8.4f}{marker}"
        )
    print(f"  {'-' * 78}")
    print(f"  Uniform baseline:    {baseline_ll:.6f}")

    if "ensemble" in results:
        ens = results["ensemble"]
        print(f"\n  Ensemble vs Uniform:    {ens.get('log_loss_pct_improvement_vs_uniform', 0):+.1f}% log loss")
        print(f"  Ensemble vs Historical: {ens.get('log_loss_pct_improvement_vs_historical', 0):+.1f}% log loss")

    # Save report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    return results


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    """Execute the full training pipeline."""
    args = parse_args()

    setup_logging(level=settings.log_level, log_file=settings.logs_dir / "train.log")
    settings.ensure_dirs()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    print("\n" + "=" * 60)
    print("  WORLD CUP AI - TRAINING PIPELINE")
    print("=" * 60)

    # Step 1
    matches_df, events_df = step_1_ingest(
        skip=args.skip_ingestion,
        skip_statsbomb=args.no_statsbomb,
        min_year=args.min_year,
    )
    if len(matches_df) == 0:
        print("ERROR: No matches loaded. Aborting.")
        sys.exit(1)

    # Step 2
    feature_matrix = step_2_features(matches_df, events_df, skip=args.skip_features)
    if len(feature_matrix) == 0:
        print("ERROR: Empty feature matrix. Aborting.")
        sys.exit(1)

    # Step 3
    model_names = [m.strip() for m in args.models.split(",")]
    trained_models = step_3_train_models(
        feature_matrix,
        model_names=model_names,
        use_gpu=args.gpu,
        output_dir=output_dir,
    )
    if not trained_models:
        print("ERROR: No models trained. Aborting.")
        sys.exit(1)

    # Step 4
    ensemble = step_4_ensemble(trained_models, feature_matrix, output_dir)

    # Step 5
    results = step_5_evaluate(trained_models, ensemble, feature_matrix, output_dir)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print(f"  [OK] Pipeline complete in {total_elapsed/60:.1f} minutes")
    print(f"  [OK] Models saved to: {output_dir}")
    print(f"  [OK] Features: data/features/feature_matrix.parquet")
    print(f"  [OK] Data: data/processed/merged_matches.parquet")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
