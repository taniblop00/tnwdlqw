"""predict.py - CLI tool for match prediction.

Usage:
    python predict.py --home Brazil --away Germany
    python predict.py --home France --away Argentina --neutral
    python predict.py --home England --away Spain --scores

Loads the trained ensemble and produces:
- Win/draw/loss probabilities
- Predicted score distribution (with --scores)
- Team strength comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ensemble.ensemble_model import EnsembleModel
from src.feature_engineering.builder import FeatureBuilder
from src.models.gbdt_models import CatBoostPredictor, LightGBMPredictor, XGBoostPredictor
from src.models.poisson_model import PoissonGoalModel
from src.utils.constants import normalize_team_name
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def load_models(models_dir: Path) -> dict:
    """Load all trained models from disk."""
    models = {}

    for name, cls in [("xgboost", XGBoostPredictor), ("lightgbm", LightGBMPredictor),
                       ("catboost", CatBoostPredictor)]:
        path = models_dir / name
        if path.exists():
            model = cls()
            model.load(path)
            models[name] = model
            logger.info(f"loaded_{name}")

    poisson_path = models_dir / "poisson"
    if poisson_path.exists():
        poisson = PoissonGoalModel()
        poisson.load(poisson_path)
        models["poisson"] = poisson
        logger.info("loaded_poisson")

    return models


def predict_match(
    home_team: str,
    away_team: str,
    models_dir: Path = Path("data/models"),
    show_scores: bool = False,
    is_neutral: bool = False,
) -> dict:
    """Predict a single match outcome.

    Args:
        home_team: Home team name.
        away_team: Away team name.
        models_dir: Directory containing trained models.
        show_scores: Whether to compute score distribution.
        is_neutral: Whether venue is neutral.

    Returns:
        Prediction dictionary.
    """
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)

    models = load_models(models_dir)

    if not models:
        logger.error("no_trained_models_found")
        return {}

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "neutral_venue": is_neutral,
    }

    # Poisson model predictions (always available if trained)
    if "poisson" in models:
        poisson: PoissonGoalModel = models["poisson"]
        probs = poisson.predict_proba(pd.DataFrame([{
            "home_team": home_team,
            "away_team": away_team,
        }]))

        result["probabilities"] = {
            "home_win": round(float(probs[0, 0]), 4),
            "draw": round(float(probs[0, 1]), 4),
            "away_win": round(float(probs[0, 2]), 4),
        }

        # Team strengths
        strengths = poisson.get_team_strengths()
        home_str = strengths[strengths["team"] == home_team]
        away_str = strengths[strengths["team"] == away_team]

        if not home_str.empty and not away_str.empty:
            result["team_strengths"] = {
                home_team: {
                    "attack": float(home_str["attack"].iloc[0]),
                    "defense": float(home_str["defense"].iloc[0]),
                    "expected_goals": float(home_str["expected_goals_per_match"].iloc[0]),
                },
                away_team: {
                    "attack": float(away_str["attack"].iloc[0]),
                    "defense": float(away_str["defense"].iloc[0]),
                    "expected_goals": float(away_str["expected_goals_per_match"].iloc[0]),
                },
            }

        # Score distribution
        if show_scores:
            matrix = poisson.predict_score_matrix(home_team, away_team, is_neutral)
            top_scores = []
            for i in range(min(6, matrix.shape[0])):
                for j in range(min(6, matrix.shape[1])):
                    if matrix[i, j] > 0.01:
                        top_scores.append({
                            "score": f"{i}-{j}",
                            "probability": round(float(matrix[i, j]), 4),
                        })
            top_scores.sort(key=lambda x: x["probability"], reverse=True)
            result["score_distribution"] = top_scores[:15]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict match outcome")
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument("--models-dir", default="data/models", help="Models directory")
    parser.add_argument("--scores", action="store_true", help="Show score distribution")
    parser.add_argument("--neutral", action="store_true", help="Neutral venue")
    args = parser.parse_args()

    setup_logging(level="WARNING")

    prediction = predict_match(
        home_team=args.home,
        away_team=args.away,
        models_dir=Path(args.models_dir),
        show_scores=args.scores,
        is_neutral=args.neutral,
    )

    if not prediction:
        print("Error: No trained models found. Run `python train.py` first.")
        sys.exit(1)

    # Pretty print
    print(f"\n{'='*50}")
    print(f"  {prediction['home_team']}  vs  {prediction['away_team']}")
    if prediction.get("neutral_venue"):
        print(f"  (neutral venue)")
    print(f"{'='*50}")

    probs = prediction.get("probabilities", {})
    if probs:
        print(f"\n  Probabilities:")
        print(f"    Home Win:  {probs.get('home_win', 0)*100:5.1f}%")
        print(f"    Draw:      {probs.get('draw', 0)*100:5.1f}%")
        print(f"    Away Win:  {probs.get('away_win', 0)*100:5.1f}%")

    strengths = prediction.get("team_strengths", {})
    if strengths:
        print(f"\n  Team Strengths:")
        for team, s in strengths.items():
            print(f"    {team}: ATK={s['attack']:+.3f}  DEF={s['defense']:+.3f}  E[G]={s['expected_goals']:.2f}")

    scores = prediction.get("score_distribution", [])
    if scores:
        print(f"\n  Most Likely Scores:")
        for s in scores[:10]:
            bar = "#" * int(s["probability"] * 100)
            print(f"    {s['score']:>5s}  {s['probability']*100:5.1f}%  {bar}")

    print()


if __name__ == "__main__":
    main()
