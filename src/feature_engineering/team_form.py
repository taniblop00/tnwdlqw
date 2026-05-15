"""Rolling team form features.

Computes temporal rolling statistics for each team: win rates, goal averages,
streaks, weighted form, head-to-head records, and momentum indicators.

All features are computed using ONLY data available before each match date.
This is enforced by processing matches chronologically and maintaining
running state per team.

This module produces ~60 features per match (30 per team × 2 teams).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.constants import get_competition_weight
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MatchRecord:
    """A single match record from a team's perspective."""

    date: str
    opponent: str
    goals_for: int
    goals_against: int
    result: str  # "W", "D", "L"
    competition: str
    is_home: bool
    points: int  # 3/1/0
    xg_for: float = 0.0
    xg_against: float = 0.0


class TeamFormComputer:
    """Computes rolling form features for all teams.

    Maintains a sliding window of recent match records for each team.
    As matches are processed chronologically, features are computed
    from the team's history UP TO (but not including) the current match.

    Features computed per team (for both home and away):
    - Win/draw/loss rates (windows: 3, 5, 10, 20)
    - Points per game (windows: 3, 5, 10, 20)
    - Goals scored/conceded averages (windows: 5, 10)
    - Goal difference average
    - Clean sheet rate
    - BTTS rate
    - Over 2.5 rate
    - Exponentially weighted form
    - Streak features (unbeaten, wins, losses, scoring, clean sheets)
    - Head-to-head record
    - Momentum indicators
    - Venue-specific form
    - Competition-specific form
    """

    def __init__(self, max_history: int = 50, windows: list[int] | None = None) -> None:
        """Initialize the form computer.

        Args:
            max_history: Maximum number of recent matches to retain per team.
            windows: Rolling window sizes. Default: [3, 5, 10, 20].
        """
        self.max_history = max_history
        self.windows = windows or [3, 5, 10, 20]
        self.team_history: dict[str, deque[MatchRecord]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        # Head-to-head tracking: (team, opponent) -> list of results
        self.h2h: dict[tuple[str, str], list[MatchRecord]] = defaultdict(list)

    def _compute_rolling_stats(
        self, records: list[MatchRecord], window: int,
    ) -> dict[str, float]:
        """Compute rolling statistics over a window of recent matches.

        Args:
            records: List of match records (most recent last).
            window: Number of recent matches to consider.

        Returns:
            Dictionary of computed statistics.
        """
        recent = records[-window:] if len(records) >= window else records
        n = len(recent)

        if n == 0:
            return {
                f"win_rate_{window}": np.nan,
                f"draw_rate_{window}": np.nan,
                f"loss_rate_{window}": np.nan,
                f"ppg_{window}": np.nan,
                f"goals_scored_avg_{window}": np.nan,
                f"goals_conceded_avg_{window}": np.nan,
                f"goal_diff_avg_{window}": np.nan,
                f"clean_sheet_rate_{window}": np.nan,
                f"btts_rate_{window}": np.nan,
                f"over25_rate_{window}": np.nan,
            }

        wins = sum(1 for r in recent if r.result == "W")
        draws = sum(1 for r in recent if r.result == "D")
        losses = sum(1 for r in recent if r.result == "L")
        points = sum(r.points for r in recent)
        gf = [r.goals_for for r in recent]
        ga = [r.goals_against for r in recent]
        clean_sheets = sum(1 for r in recent if r.goals_against == 0)
        btts = sum(1 for r in recent if r.goals_for > 0 and r.goals_against > 0)
        over25 = sum(1 for r in recent if r.goals_for + r.goals_against > 2)

        return {
            f"win_rate_{window}": wins / n,
            f"draw_rate_{window}": draws / n,
            f"loss_rate_{window}": losses / n,
            f"ppg_{window}": points / n,
            f"goals_scored_avg_{window}": np.mean(gf),
            f"goals_conceded_avg_{window}": np.mean(ga),
            f"goal_diff_avg_{window}": np.mean(np.array(gf) - np.array(ga)),
            f"clean_sheet_rate_{window}": clean_sheets / n,
            f"btts_rate_{window}": btts / n,
            f"over25_rate_{window}": over25 / n,
        }

    def _compute_weighted_form(
        self, records: list[MatchRecord], decay: float = 0.9,
    ) -> dict[str, float]:
        """Compute exponentially weighted form.

        Recent matches are weighted exponentially more than older matches.
        decay=0.9 means a match 10 games ago has weight 0.9^10 ≈ 0.35.

        Args:
            records: Match records (most recent last).
            decay: Exponential decay factor (0 < decay < 1).

        Returns:
            Weighted form features.
        """
        if not records:
            return {
                "weighted_ppg": np.nan,
                "weighted_gf": np.nan,
                "weighted_ga": np.nan,
            }

        n = len(records)
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weight_sum = weights.sum()

        points = np.array([r.points for r in records])
        gf = np.array([r.goals_for for r in records])
        ga = np.array([r.goals_against for r in records])

        return {
            "weighted_ppg": float(np.dot(points, weights) / weight_sum),
            "weighted_gf": float(np.dot(gf, weights) / weight_sum),
            "weighted_ga": float(np.dot(ga, weights) / weight_sum),
        }

    def _compute_streaks(self, records: list[MatchRecord]) -> dict[str, int]:
        """Compute current streak features.

        Counts consecutive matches from the most recent backward.

        Args:
            records: Match records (most recent last).

        Returns:
            Current streak values.
        """
        if not records:
            return {
                "streak_unbeaten": 0,
                "streak_wins": 0,
                "streak_losses": 0,
                "streak_scoring": 0,
                "streak_clean_sheets": 0,
            }

        # Count from most recent backward
        streaks: dict[str, int] = {
            "streak_unbeaten": 0,
            "streak_wins": 0,
            "streak_losses": 0,
            "streak_scoring": 0,
            "streak_clean_sheets": 0,
        }

        for r in reversed(records):
            if r.result != "L" and streaks["streak_unbeaten"] == len(records) - records.index(r) - 1 + streaks["streak_unbeaten"]:
                pass  # Will use simpler logic below
            break

        # Simpler streak computation
        for key, condition in [
            ("streak_unbeaten", lambda r: r.result != "L"),
            ("streak_wins", lambda r: r.result == "W"),
            ("streak_losses", lambda r: r.result == "L"),
            ("streak_scoring", lambda r: r.goals_for > 0),
            ("streak_clean_sheets", lambda r: r.goals_against == 0),
        ]:
            count = 0
            for r in reversed(records):
                if condition(r):
                    count += 1
                else:
                    break
            streaks[key] = count

        return streaks

    def _compute_h2h(self, team: str, opponent: str) -> dict[str, float]:
        """Compute head-to-head features between two teams.

        Args:
            team: Team name.
            opponent: Opponent name.

        Returns:
            Head-to-head statistics.
        """
        key = (team, opponent)
        records = self.h2h.get(key, [])

        if not records:
            return {
                "h2h_win_rate": np.nan,
                "h2h_goals_avg": np.nan,
                "h2h_conceded_avg": np.nan,
                "h2h_matches": 0,
            }

        wins = sum(1 for r in records if r.result == "W")
        n = len(records)
        gf = [r.goals_for for r in records]
        ga = [r.goals_against for r in records]

        return {
            "h2h_win_rate": wins / n,
            "h2h_goals_avg": np.mean(gf),
            "h2h_conceded_avg": np.mean(ga),
            "h2h_matches": n,
        }

    def _compute_venue_form(self, records: list[MatchRecord], is_home: bool) -> dict[str, float]:
        """Compute venue-specific form (home or away performance).

        Args:
            records: All match records for the team.
            is_home: Whether to compute home or away form.

        Returns:
            Venue-specific form features.
        """
        venue_records = [r for r in records if r.is_home == is_home]
        recent = venue_records[-10:] if len(venue_records) >= 10 else venue_records

        if not recent:
            return {
                "venue_win_rate": np.nan,
                "venue_goals_avg": np.nan,
            }

        wins = sum(1 for r in recent if r.result == "W")
        gf = [r.goals_for for r in recent]

        return {
            "venue_win_rate": wins / len(recent),
            "venue_goals_avg": np.mean(gf),
        }

    def _compute_competition_form(self, records: list[MatchRecord]) -> dict[str, float]:
        """Compute competitive match form (excludes friendlies).

        Args:
            records: All match records.

        Returns:
            Competition-filtered form features.
        """
        competitive = [r for r in records if get_competition_weight(r.competition) > 0.3]
        recent = competitive[-10:] if len(competitive) >= 10 else competitive

        if not recent:
            return {"competitive_win_rate": np.nan, "competitive_ppg": np.nan}

        wins = sum(1 for r in recent if r.result == "W")
        pts = sum(r.points for r in recent)

        return {
            "competitive_win_rate": wins / len(recent),
            "competitive_ppg": pts / len(recent),
        }

    def _compute_momentum(self, records: list[MatchRecord]) -> dict[str, float]:
        """Compute momentum indicators (short-term vs long-term form delta).

        Positive momentum = improving form.

        Args:
            records: Match records.

        Returns:
            Momentum features.
        """
        if len(records) < 10:
            return {"momentum_ppg": np.nan, "momentum_gf": np.nan}

        short = records[-5:]
        long = records[-10:]

        short_ppg = sum(r.points for r in short) / len(short)
        long_ppg = sum(r.points for r in long) / len(long)
        short_gf = sum(r.goals_for for r in short) / len(short)
        long_gf = sum(r.goals_for for r in long) / len(long)

        return {
            "momentum_ppg": short_ppg - long_ppg,
            "momentum_gf": short_gf - long_gf,
        }

    def compute_features_for_team(
        self, team: str, opponent: str, is_home: bool,
    ) -> dict[str, float]:
        """Compute all form features for a team before a match.

        Args:
            team: Team name.
            opponent: Opponent name.
            is_home: Whether the team is playing at home.

        Returns:
            Dictionary of all form features for this team.
        """
        records = list(self.team_history[team])
        prefix = "home" if is_home else "away"

        features: dict[str, float] = {}

        # Rolling window stats
        for window in self.windows:
            stats = self._compute_rolling_stats(records, window)
            for key, val in stats.items():
                features[f"{prefix}_{key}"] = val

        # Weighted form
        weighted = self._compute_weighted_form(records)
        for key, val in weighted.items():
            features[f"{prefix}_{key}"] = val

        # Streaks
        streaks = self._compute_streaks(records)
        for key, val in streaks.items():
            features[f"{prefix}_{key}"] = val

        # Head-to-head
        h2h = self._compute_h2h(team, opponent)
        for key, val in h2h.items():
            features[f"{prefix}_{key}"] = val

        # Venue form
        venue = self._compute_venue_form(records, is_home)
        for key, val in venue.items():
            features[f"{prefix}_{key}"] = val

        # Competitive form
        comp_form = self._compute_competition_form(records)
        for key, val in comp_form.items():
            features[f"{prefix}_{key}"] = val

        # Momentum
        momentum = self._compute_momentum(records)
        for key, val in momentum.items():
            features[f"{prefix}_{key}"] = val

        # Matches played
        features[f"{prefix}_matches_played"] = len(records)

        return features

    def _record_match(
        self,
        team: str,
        opponent: str,
        goals_for: int,
        goals_against: int,
        competition: str,
        match_date: str,
        is_home: bool,
        xg_for: float = 0.0,
        xg_against: float = 0.0,
    ) -> None:
        """Record a match result in the team's history.

        Args:
            team: Team name.
            opponent: Opponent name.
            goals_for: Goals scored.
            goals_against: Goals conceded.
            competition: Competition name.
            match_date: Match date.
            is_home: Whether team was home.
            xg_for: Expected goals for.
            xg_against: Expected goals against.
        """
        if goals_for > goals_against:
            result = "W"
            points = 3
        elif goals_for == goals_against:
            result = "D"
            points = 1
        else:
            result = "L"
            points = 0

        record = MatchRecord(
            date=match_date,
            opponent=opponent,
            goals_for=goals_for,
            goals_against=goals_against,
            result=result,
            competition=competition,
            is_home=is_home,
            points=points,
            xg_for=xg_for,
            xg_against=xg_against,
        )

        self.team_history[team].append(record)
        self.h2h[(team, opponent)].append(record)

    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Process all matches chronologically and compute form features.

        For each match, computes features BEFORE the match occurs,
        then records the result to update future features.

        Args:
            matches_df: DataFrame sorted by match_date with columns:
                        home_team, away_team, home_score, away_score, competition.

        Returns:
            DataFrame with all form features added.
        """
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
            home_features = self.compute_features_for_team(home, away, is_home=True)
            away_features = self.compute_features_for_team(away, home, is_home=False)

            # Combine
            match_features = {**home_features, **away_features}

            # Add differential features
            for window in self.windows:
                h_ppg = home_features.get(f"home_ppg_{window}", np.nan)
                a_ppg = away_features.get(f"away_ppg_{window}", np.nan)
                match_features[f"ppg_diff_{window}"] = h_ppg - a_ppg

                h_gf = home_features.get(f"home_goals_scored_avg_{window}", np.nan)
                a_gf = away_features.get(f"away_goals_scored_avg_{window}", np.nan)
                match_features[f"goals_avg_diff_{window}"] = h_gf - a_gf

            all_features.append(match_features)

            # Record this match result for future features
            self._record_match(
                home, away,
                int(row["home_score"]), int(row["away_score"]),
                row.get("competition", ""), date_str, is_home=True,
            )
            self._record_match(
                away, home,
                int(row["away_score"]), int(row["home_score"]),
                row.get("competition", ""), date_str, is_home=False,
            )

        features_df = pd.DataFrame(all_features)
        result = pd.concat([matches_df.reset_index(drop=True), features_df], axis=1)

        logger.info(
            "form_features_computed",
            matches=len(result),
            features=len(features_df.columns),
        )
        return result
