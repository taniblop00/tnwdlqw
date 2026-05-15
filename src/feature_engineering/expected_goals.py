"""Expected goals (xG) based feature engineering.

Computes rolling xG features from StatsBomb event data. These features
capture the QUALITY of chances created/conceded, which is a better
predictor than raw goals (less noisy, more process-driven).

xG features explain attacking/defensive quality independent of finishing
luck, making them critical for predictive models.

All features use strict temporal filtering — no future data leakage.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MatchXGRecord:
    """xG summary for a single match from one team's perspective."""

    date: str
    opponent: str
    xg_for: float
    xg_against: float
    shots: int
    shots_on_target: int
    goals: int
    goals_against: int
    xg_open_play: float = 0.0
    xg_set_piece: float = 0.0
    big_chances: int = 0  # Shots with xG > 0.3


class XGFeatureComputer:
    """Computes rolling xG-based features for all teams.

    Processes event-level data to aggregate per-match xG statistics,
    then computes rolling averages, trends, and differentials.

    Features produced per team (×2 for home/away):
    - Rolling xG per 90 (windows: 5, 10, 20)
    - Rolling xGA per 90
    - xG difference (attack - defense quality)
    - xG overperformance (goals - xG, measures finishing quality)
    - xGA overperformance (goals conceded - xGA, measures GK quality)
    - Shot quality (xG per shot)
    - Shot volume (shots per match)
    - Shot accuracy (SOT / shots)
    - Conversion rate (goals / shots)
    - Big chance creation (xG > 0.3 per match)
    - Open play vs set piece xG breakdown
    - xG trend (slope of recent xG values)
    - xG variance (consistency measure)
    """

    def __init__(self, max_history: int = 30, windows: list[int] | None = None) -> None:
        self.max_history = max_history
        self.windows = windows or [5, 10, 20]
        self.team_xg_history: dict[str, deque[MatchXGRecord]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    def aggregate_match_xg(self, events_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate event-level data into per-match xG summaries.

        Joins shot events with match records to compute:
        - Total xG for each team in each match
        - Shot counts and types
        - Open play vs set piece xG breakdown

        Args:
            events_df: StatsBomb events DataFrame with columns:
                       match_id, event_type, team, xg, outcome, x, y.
            matches_df: Matches DataFrame with match_id, home_team, away_team.

        Returns:
            Matches DataFrame with added xG columns per team.
        """
        if events_df.empty or "xg" not in events_df.columns:
            logger.warning("no_xg_data_in_events")
            matches_df["home_xg"] = np.nan
            matches_df["away_xg"] = np.nan
            return matches_df

        # Filter to shots only
        shots = events_df[events_df["event_type"] == "Shot"].copy()

        if shots.empty:
            logger.warning("no_shot_events_found")
            matches_df["home_xg"] = np.nan
            matches_df["away_xg"] = np.nan
            return matches_df

        # Aggregate xG per team per match
        match_xg = (
            shots.groupby(["match_id", "team"])
            .agg(
                xg_total=("xg", "sum"),
                shot_count=("xg", "count"),
                shots_on_target=("outcome", lambda x: (x == "Goal").sum() + (x == "Saved").sum()),
                goals=("outcome", lambda x: (x == "Goal").sum()),
                big_chances=("xg", lambda x: (x > 0.3).sum()),
            )
            .reset_index()
        )

        # Merge with matches to identify home/away xG
        home_xg = matches_df[["match_id", "home_team"]].merge(
            match_xg,
            left_on=["match_id", "home_team"],
            right_on=["match_id", "team"],
            how="left",
        ).drop(columns=["team"])

        away_xg = matches_df[["match_id", "away_team"]].merge(
            match_xg,
            left_on=["match_id", "away_team"],
            right_on=["match_id", "team"],
            how="left",
        ).drop(columns=["team"])

        matches_df["home_xg"] = home_xg["xg_total"].fillna(0.0).values
        matches_df["away_xg"] = away_xg["xg_total"].fillna(0.0).values
        matches_df["home_shots"] = home_xg["shot_count"].fillna(0).astype(int).values
        matches_df["away_shots"] = away_xg["shot_count"].fillna(0).astype(int).values
        matches_df["home_sot"] = home_xg["shots_on_target"].fillna(0).astype(int).values
        matches_df["away_sot"] = away_xg["shots_on_target"].fillna(0).astype(int).values
        matches_df["home_big_chances"] = home_xg["big_chances"].fillna(0).astype(int).values
        matches_df["away_big_chances"] = away_xg["big_chances"].fillna(0).astype(int).values

        logger.info(
            "match_xg_aggregated",
            matches_with_xg=(matches_df["home_xg"] > 0).sum(),
            total_matches=len(matches_df),
        )
        return matches_df

    def _compute_rolling_xg(
        self, records: list[MatchXGRecord], window: int, prefix: str,
    ) -> dict[str, float]:
        """Compute rolling xG statistics over a window.

        Args:
            records: Team's match xG records (most recent last).
            window: Number of recent matches.
            prefix: Feature name prefix (home/away).

        Returns:
            Dictionary of xG features.
        """
        recent = records[-window:] if len(records) >= window else records
        n = len(recent)

        if n == 0:
            return {
                f"{prefix}_xg_avg_{window}": np.nan,
                f"{prefix}_xga_avg_{window}": np.nan,
                f"{prefix}_xg_diff_{window}": np.nan,
                f"{prefix}_xg_overperf_{window}": np.nan,
                f"{prefix}_xga_overperf_{window}": np.nan,
                f"{prefix}_shot_quality_{window}": np.nan,
                f"{prefix}_shots_avg_{window}": np.nan,
                f"{prefix}_sot_rate_{window}": np.nan,
                f"{prefix}_conversion_{window}": np.nan,
                f"{prefix}_big_chances_avg_{window}": np.nan,
            }

        xg_vals = [r.xg_for for r in recent]
        xga_vals = [r.xg_against for r in recent]
        goals_vals = [r.goals for r in recent]
        ga_vals = [r.goals_against for r in recent]
        shots_vals = [r.shots for r in recent]
        sot_vals = [r.shots_on_target for r in recent]
        bc_vals = [r.big_chances for r in recent]

        total_shots = sum(shots_vals)
        total_goals = sum(goals_vals)

        return {
            f"{prefix}_xg_avg_{window}": np.mean(xg_vals),
            f"{prefix}_xga_avg_{window}": np.mean(xga_vals),
            f"{prefix}_xg_diff_{window}": np.mean(xg_vals) - np.mean(xga_vals),
            f"{prefix}_xg_overperf_{window}": np.mean(goals_vals) - np.mean(xg_vals),
            f"{prefix}_xga_overperf_{window}": np.mean(ga_vals) - np.mean(xga_vals),
            f"{prefix}_shot_quality_{window}": (
                sum(xg_vals) / total_shots if total_shots > 0 else np.nan
            ),
            f"{prefix}_shots_avg_{window}": np.mean(shots_vals),
            f"{prefix}_sot_rate_{window}": (
                sum(sot_vals) / total_shots if total_shots > 0 else np.nan
            ),
            f"{prefix}_conversion_{window}": (
                total_goals / total_shots if total_shots > 0 else np.nan
            ),
            f"{prefix}_big_chances_avg_{window}": np.mean(bc_vals),
        }

    def _compute_xg_trend(
        self, records: list[MatchXGRecord], prefix: str,
    ) -> dict[str, float]:
        """Compute the trend (slope) of xG over recent matches.

        Positive slope = improving attacking quality.

        Args:
            records: Match xG records.
            prefix: Feature name prefix.

        Returns:
            Trend features.
        """
        if len(records) < 5:
            return {
                f"{prefix}_xg_trend": np.nan,
                f"{prefix}_xg_variance": np.nan,
            }

        recent = records[-10:]
        xg_vals = np.array([r.xg_for for r in recent])

        # Linear regression slope
        x = np.arange(len(xg_vals))
        if len(x) > 1:
            slope = np.polyfit(x, xg_vals, 1)[0]
        else:
            slope = 0.0

        return {
            f"{prefix}_xg_trend": slope,
            f"{prefix}_xg_variance": float(np.var(xg_vals)),
        }

    def compute_features_for_team(
        self, team: str, prefix: str,
    ) -> dict[str, float]:
        """Compute all xG features for a team.

        Args:
            team: Team name.
            prefix: "home" or "away".

        Returns:
            Dictionary of xG features.
        """
        records = list(self.team_xg_history[team])
        features: dict[str, float] = {}

        for window in self.windows:
            xg_stats = self._compute_rolling_xg(records, window, prefix)
            features.update(xg_stats)

        trend = self._compute_xg_trend(records, prefix)
        features.update(trend)

        return features

    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Process matches chronologically and compute xG features.

        Requires matches_df to have home_xg, away_xg columns
        (from aggregate_match_xg).

        Args:
            matches_df: DataFrame with xG data, sorted by match_date.

        Returns:
            DataFrame with xG features added.
        """
        has_xg = "home_xg" in matches_df.columns and not matches_df["home_xg"].isna().all()
        if not has_xg:
            logger.warning("no_xg_data_available_skipping_xg_features")
            return matches_df

        matches_df = matches_df.sort_values("match_date").reset_index(drop=True)
        all_features: list[dict[str, Any]] = []

        for _, row in matches_df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            date_str = (
                str(row["match_date"].date())
                if hasattr(row["match_date"], "date")
                else str(row["match_date"])
            )

            # Compute features BEFORE this match
            home_feats = self.compute_features_for_team(home, "home")
            away_feats = self.compute_features_for_team(away, "away")
            match_feats = {**home_feats, **away_feats}

            # Matchup differentials
            for window in self.windows:
                h_xg = home_feats.get(f"home_xg_avg_{window}", np.nan)
                a_xga = away_feats.get(f"away_xga_avg_{window}", np.nan)
                match_feats[f"xg_matchup_diff_{window}"] = h_xg - a_xga

            all_features.append(match_feats)

            # Record this match for future features
            home_xg = row.get("home_xg", 0.0)
            away_xg = row.get("away_xg", 0.0)

            if pd.notna(home_xg):
                self.team_xg_history[home].append(MatchXGRecord(
                    date=date_str, opponent=away,
                    xg_for=float(home_xg), xg_against=float(away_xg),
                    shots=int(row.get("home_shots", 0)),
                    shots_on_target=int(row.get("home_sot", 0)),
                    goals=int(row["home_score"]),
                    goals_against=int(row["away_score"]),
                    big_chances=int(row.get("home_big_chances", 0)),
                ))
                self.team_xg_history[away].append(MatchXGRecord(
                    date=date_str, opponent=home,
                    xg_for=float(away_xg), xg_against=float(home_xg),
                    shots=int(row.get("away_shots", 0)),
                    shots_on_target=int(row.get("away_sot", 0)),
                    goals=int(row["away_score"]),
                    goals_against=int(row["home_score"]),
                    big_chances=int(row.get("away_big_chances", 0)),
                ))

        features_df = pd.DataFrame(all_features)
        result = pd.concat([matches_df.reset_index(drop=True), features_df], axis=1)

        logger.info(
            "xg_features_computed",
            matches=len(result),
            features=len(features_df.columns),
        )
        return result
