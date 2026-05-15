"""Feature builder - orchestrates all feature engineering modules.

This is the central coordinator that:
1. Takes raw match and event data
2. Runs all feature computation modules in dependency order
3. Produces the final feature matrix for model training

The builder enforces temporal safety: all features for match M
are computed using only data from matches that occurred before M.

Output: A single Parquet file containing one row per match with
all ~200 features plus the target variable (match result).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering.expected_goals import XGFeatureComputer
from src.feature_engineering.ratings import (
    AttackDefenseElo,
    EloRatingSystem,
    Glicko2RatingSystem,
)
from src.feature_engineering.team_form import TeamFormComputer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureBuilder:
    """Orchestrates all feature engineering and produces the feature matrix.

    Processing order (dependencies flow downward):
    1. Elo ratings (depends on: matches only)
    2. Glicko-2 ratings (depends on: matches only)
    3. Attack/defense ratings (depends on: matches only)
    4. xG aggregation (depends on: events + matches)
    5. xG features (depends on: xG aggregation)
    6. Team form features (depends on: matches)
    7. Derived/interaction features (depends on: all of the above)

    Each module processes matches chronologically and records results
    AFTER computing features, preventing any temporal leakage.
    """

    def __init__(
        self,
        elo_config: dict[str, Any] | None = None,
        form_windows: list[int] | None = None,
        xg_windows: list[int] | None = None,
    ) -> None:
        """Initialize feature builder with configuration.

        Args:
            elo_config: Elo system configuration overrides.
            form_windows: Rolling window sizes for form features.
            xg_windows: Rolling window sizes for xG features.
        """
        elo_config = elo_config or {}
        self.elo_system = EloRatingSystem(**elo_config)
        self.glicko_system = Glicko2RatingSystem()
        self.ad_elo_system = AttackDefenseElo()
        self.form_computer = TeamFormComputer(windows=form_windows or [3, 5, 10, 20])
        self.xg_computer = XGFeatureComputer(windows=xg_windows or [5, 10, 20])

    def build(
        self,
        matches_df: pd.DataFrame,
        events_df: pd.DataFrame | None = None,
        output_path: Path | None = None,
    ) -> pd.DataFrame:
        """Build the complete feature matrix.

        Args:
            matches_df: Raw matches DataFrame with columns:
                        match_id, match_date, home_team, away_team,
                        home_score, away_score, competition.
            events_df: Optional events DataFrame for xG features.
            output_path: Optional path to save feature matrix as Parquet.

        Returns:
            Feature matrix DataFrame with one row per match.
        """
        logger.info("starting_feature_build", matches=len(matches_df))

        # Ensure chronological order
        matches_df = matches_df.sort_values("match_date").reset_index(drop=True)

        # -- Step 1: Elo ratings ----------------------------------------
        logger.info("computing_elo_ratings")
        matches_df = self.elo_system.process_match_history(matches_df.copy())

        # -- Step 2: Glicko-2 ratings -----------------------------------
        logger.info("computing_glicko_ratings")
        matches_df = self.glicko_system.process_match_history(matches_df.copy())

        # -- Step 3: Attack/defense ratings -----------------------------
        logger.info("computing_attack_defense_ratings")
        matches_df = self.ad_elo_system.process_match_history(matches_df.copy())

        # -- Step 4: xG aggregation from events ------------------------
        if events_df is not None and not events_df.empty:
            logger.info("aggregating_xg_from_events")
            matches_df = self.xg_computer.aggregate_match_xg(events_df, matches_df)
        else:
            logger.info("no_events_data_skipping_xg_aggregation")
            matches_df["home_xg"] = np.nan
            matches_df["away_xg"] = np.nan

        # -- Step 5: xG rolling features -------------------------------
        logger.info("computing_xg_features")
        matches_df = self.xg_computer.process_matches(matches_df)

        # -- Step 6: Team form features --------------------------------
        logger.info("computing_form_features")
        matches_df = self.form_computer.process_matches(matches_df)

        # -- Step 7: Derived/interaction features ----------------------
        logger.info("computing_derived_features")
        matches_df = self._add_derived_features(matches_df)

        # -- Step 8: Target variable -----------------------------------
        matches_df["target"] = matches_df["result"].map({"H": 0, "D": 1, "A": 2})

        # -- Step 9: Clean up ------------------------------------------
        feature_cols = self.get_feature_columns(matches_df)
        n_features = len(feature_cols)
        n_complete = matches_df[feature_cols].dropna(how="all", axis=0).shape[0]

        logger.info(
            "feature_build_complete",
            total_matches=len(matches_df),
            total_features=n_features,
            matches_with_features=n_complete,
            feature_completeness=round(
                matches_df[feature_cols].notna().mean().mean() * 100, 1
            ),
        )

        # -- Step 10: Export -------------------------------------------
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            matches_df.to_parquet(output_path, index=False, engine="pyarrow")
            logger.info("feature_matrix_saved", path=str(output_path))

        return matches_df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived and interaction features.

        These combine signals from multiple feature modules:
        - Rating interactions (Elo × form)
        - Matchup advantages (attack vs defense)
        - Quality differentials
        - Historical context

        Args:
            df: DataFrame with all base features computed.

        Returns:
            DataFrame with derived features added.
        """
        # -- Rating-based differentials --------------------------------
        # Already computed by individual modules, but add combined signals
        if "home_elo" in df.columns and "away_elo" in df.columns:
            df["elo_ratio"] = df["home_elo"] / df["away_elo"].clip(lower=1)
            df["elo_abs_diff"] = (df["home_elo"] - df["away_elo"]).abs()

        if "home_glicko" in df.columns and "away_glicko" in df.columns:
            df["glicko_ratio"] = df["home_glicko"] / df["away_glicko"].clip(lower=1)
            # Combined uncertainty: higher = less predictable match
            if "home_glicko_rd" in df.columns:
                df["combined_uncertainty"] = (
                    df["home_glicko_rd"] + df["away_glicko_rd"]
                )

        # -- Attack vs Defense matchup ---------------------------------
        if "home_attack_rating" in df.columns and "away_defense_rating" in df.columns:
            # Home attack vs away defense (positive = home advantage)
            df["attack_vs_defense_home"] = (
                df["home_attack_rating"] - df["away_defense_rating"]
            )
            df["attack_vs_defense_away"] = (
                df["away_attack_rating"] - df["home_defense_rating"]
            )
            df["attack_defense_matchup_diff"] = (
                df["attack_vs_defense_home"] - df["attack_vs_defense_away"]
            )

        # -- Form × Rating interaction ---------------------------------
        if "home_ppg_5" in df.columns and "home_elo" in df.columns:
            # High-rated teams in good form vs low-rated teams in bad form
            df["home_form_x_rating"] = df["home_ppg_5"] * df["home_elo"] / 1500
            df["away_form_x_rating"] = df["away_ppg_5"] * df["away_elo"] / 1500

        # -- Quality differential summary ------------------------------
        for window in [5, 10]:
            h_wr = f"home_win_rate_{window}"
            a_wr = f"away_win_rate_{window}"
            if h_wr in df.columns and a_wr in df.columns:
                df[f"form_quality_diff_{window}"] = df[h_wr] - df[a_wr]

        # -- xG matchup features ---------------------------------------
        for window in [5, 10]:
            h_xg = f"home_xg_avg_{window}"
            a_xga = f"away_xga_avg_{window}"
            if h_xg in df.columns and a_xga in df.columns:
                # Home xG vs Away xGA: positive means home attacks well vs away defense
                df[f"xg_attack_vs_defense_{window}"] = df[h_xg] - df[a_xga]

        return df

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list[str]:
        """Get the list of feature columns (exclude metadata and targets).

        Args:
            df: Feature matrix DataFrame.

        Returns:
            List of feature column names.
        """
        exclude_cols = {
            "match_id", "match_date", "home_team", "away_team",
            "home_score", "away_score", "result", "target",
            "competition", "season", "stadium", "referee",
            "home_manager", "away_manager", "goal_diff",
            "competition_id", "season_id",
            # Post-match values (not features, they'd leak)
            "home_elo_after", "away_elo_after",
        }
        return [c for c in df.columns if c not in exclude_cols]

    @staticmethod
    def get_trainable_data(
        df: pd.DataFrame,
        min_matches_per_team: int = 5,
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Extract trainable subset of the feature matrix.

        Filters to matches where both teams have enough history
        for meaningful features, and drops rows with missing targets.

        Args:
            df: Full feature matrix.
            min_matches_per_team: Minimum prior matches needed.

        Returns:
            Tuple of (feature_matrix, target_vector, feature_names).
        """
        feature_cols = FeatureBuilder.get_feature_columns(df)

        # Filter to matches with valid targets
        valid = df["target"].notna()

        # Filter to matches where teams have history
        if "home_matches_played" in df.columns:
            valid &= df["home_matches_played"] >= min_matches_per_team
        if "away_matches_played" in df.columns:
            valid &= df["away_matches_played"] >= min_matches_per_team

        df_valid = df[valid].copy()

        X = df_valid[feature_cols]
        y = df_valid["target"].astype(int)

        logger.info(
            "trainable_data_extracted",
            total_matches=len(df),
            trainable_matches=len(df_valid),
            features=len(feature_cols),
            class_distribution=y.value_counts().to_dict(),
        )

        return X, y, feature_cols
