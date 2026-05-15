"""Team rating systems: Elo, Glicko-2, and decomposed attack/defense ratings.

This module implements multiple rating systems that track team strength
over time using only historical match results. These ratings are among
the most predictive individual features in football forecasting.

All rating systems are strictly temporal: a team's rating at time T
is computed using only matches that occurred before T. This prevents
any form of lookahead bias or data leakage.

Rating systems implemented:
1. Standard Elo with competition-weighted K-factor
2. Margin-of-victory adjusted Elo
3. Glicko-2 (Elo with uncertainty tracking)
4. Decomposed attack/defense Elo

References:
    - Elo: https://en.wikipedia.org/wiki/Elo_rating_system
    - Glicko-2: Glickman (2001) "Dynamic paired comparison models"
    - Football Elo: https://www.eloratings.net/about
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.constants import get_competition_weight
from src.utils.logger import get_logger

logger = get_logger(__name__)


# -- Elo Rating System -----------------------------------------------------


@dataclass
class EloState:
    """Current state of a team's Elo rating with history tracking."""

    rating: float = 1500.0
    matches_played: int = 0
    history: list[tuple[str, float]] = field(default_factory=list)  # (date, rating)


class EloRatingSystem:
    """Standard Elo rating system with football-specific adaptations.

    Adaptations over base Elo:
    1. Competition-weighted K-factor (World Cup matches matter more)
    2. Margin-of-victory adjustment (a 4-0 win shifts ratings more than 1-0)
    3. Home advantage adjustment (home team gets virtual +100 rating)
    4. Inter-tournament regression to mean (prevents stale ratings)

    The Elo system is the strongest single predictor in football forecasting,
    explaining ~30-40% of match outcome variance.

    Attributes:
        k_base: Base K-factor for rating updates.
        home_advantage: Virtual Elo bonus for the home team.
        initial_rating: Starting rating for new teams.
        ratings: Current rating state for each team.
    """

    def __init__(
        self,
        k_base: float = 40.0,
        home_advantage: float = 100.0,
        initial_rating: float = 1500.0,
        mov_weight: float = 1.0,
        mean_reversion_rate: float = 0.33,
        mean_reversion_gap_days: int = 365,
    ) -> None:
        self.k_base = k_base
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.mov_weight = mov_weight
        self.mean_reversion_rate = mean_reversion_rate
        self.mean_reversion_gap_days = mean_reversion_gap_days
        self.ratings: dict[str, EloState] = {}
        self._last_match_date: dict[str, str] = {}

    def _get_or_create(self, team: str) -> EloState:
        """Get existing rating or create a new one for a team."""
        if team not in self.ratings:
            self.ratings[team] = EloState(rating=self.initial_rating)
        return self.ratings[team]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Compute expected score for team A against team B.

        Uses the standard logistic curve:
            E(A) = 1 / (1 + 10^((R_B - R_A) / 400))

        Args:
            rating_a: Team A's Elo rating.
            rating_b: Team B's Elo rating.

        Returns:
            Expected score for team A (0.0 to 1.0).
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def margin_of_victory_multiplier(self, goal_diff: int) -> float:
        """Compute margin-of-victory multiplier.

        Larger victories produce larger rating changes, but with diminishing
        returns. Uses log scaling to prevent excessive swings from blowouts.

        Formula: M = ln(1 + |goal_diff|) × weight + (1 - weight)
        When weight=0, M=1 always. When weight=1, M = ln(1 + |GD|).

        Args:
            goal_diff: Absolute goal difference.

        Returns:
            Multiplier >= 1.0.
        """
        if self.mov_weight == 0:
            return 1.0
        return (
            math.log(1 + abs(goal_diff)) * self.mov_weight
            + (1 - self.mov_weight)
        )

    def _apply_mean_reversion(self, team: str, match_date: str) -> None:
        """Regress team rating toward the mean if there's been a long gap.

        International teams often have long gaps between matches (3-6 months).
        Without mean reversion, a team that had a hot streak 2 years ago
        retains an inflated rating. This corrects for that.

        Args:
            team: Team name.
            match_date: Current match date string (YYYY-MM-DD).
        """
        if team not in self._last_match_date:
            return

        last_date = pd.Timestamp(self._last_match_date[team])
        current_date = pd.Timestamp(match_date)
        gap_days = (current_date - last_date).days

        if gap_days > self.mean_reversion_gap_days:
            state = self.ratings[team]
            # Regress toward global mean proportional to gap length
            reversion_fraction = min(
                self.mean_reversion_rate * (gap_days / self.mean_reversion_gap_days),
                self.mean_reversion_rate,
            )
            global_mean = self._compute_global_mean()
            state.rating = state.rating + reversion_fraction * (global_mean - state.rating)

    def _compute_global_mean(self) -> float:
        """Compute the current mean Elo across all rated teams."""
        if not self.ratings:
            return self.initial_rating
        return sum(s.rating for s in self.ratings.values()) / len(self.ratings)

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        competition: str = "",
        match_date: str = "",
        is_neutral: bool = False,
    ) -> tuple[float, float, float, float]:
        """Update ratings after a match result.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            home_score: Goals scored by home team.
            away_score: Goals scored by away team.
            competition: Competition name (for K-factor weighting).
            match_date: Match date for history tracking.
            is_neutral: Whether the match is at a neutral venue.

        Returns:
            Tuple of (home_rating_before, away_rating_before,
                      home_rating_after, away_rating_after).
        """
        # Apply mean reversion for long gaps
        self._apply_mean_reversion(home_team, match_date)
        self._apply_mean_reversion(away_team, match_date)

        home_state = self._get_or_create(home_team)
        away_state = self._get_or_create(away_team)

        # Effective ratings (home team gets advantage unless neutral venue)
        home_elo_eff = home_state.rating + (0 if is_neutral else self.home_advantage)
        away_elo_eff = away_state.rating

        # Store pre-update ratings
        home_before = home_state.rating
        away_before = away_state.rating

        # Expected scores
        e_home = self.expected_score(home_elo_eff, away_elo_eff)
        e_away = 1.0 - e_home

        # Actual scores (1 = win, 0.5 = draw, 0 = loss)
        if home_score > away_score:
            s_home, s_away = 1.0, 0.0
        elif home_score == away_score:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0

        # Competition-weighted K-factor
        comp_weight = get_competition_weight(competition)
        k = self.k_base * comp_weight

        # Margin of victory multiplier
        mov_mult = self.margin_of_victory_multiplier(abs(home_score - away_score))

        # Rating updates
        home_delta = k * mov_mult * (s_home - e_home)
        away_delta = k * mov_mult * (s_away - e_away)

        home_state.rating += home_delta
        away_state.rating += away_delta
        home_state.matches_played += 1
        away_state.matches_played += 1

        # Track history
        if match_date:
            home_state.history.append((match_date, home_state.rating))
            away_state.history.append((match_date, away_state.rating))
            self._last_match_date[home_team] = match_date
            self._last_match_date[away_team] = match_date

        return home_before, away_before, home_state.rating, away_state.rating

    def get_rating(self, team: str) -> float:
        """Get the current Elo rating for a team.

        Args:
            team: Team name.

        Returns:
            Current Elo rating (default 1500 if unknown).
        """
        return self._get_or_create(team).rating

    def predict_match(
        self, home_team: str, away_team: str, is_neutral: bool = False,
    ) -> dict[str, float]:
        """Predict match outcome probabilities from Elo ratings.

        Converts Elo expected score into win/draw/loss probabilities
        using the 3-outcome extension of Elo.

        The draw probability is estimated from the win probability using
        the empirical relationship: P(draw) ≈ 1 - (P(home_win) - 0.5)² × 4
        bounded to historical international draw rate (~23%).

        Args:
            home_team: Home team name.
            away_team: Away team name.
            is_neutral: Whether the match is at a neutral venue.

        Returns:
            Dictionary with 'home_win', 'draw', 'away_win' probabilities.
        """
        home_state = self._get_or_create(home_team)
        away_state = self._get_or_create(away_team)

        home_elo = home_state.rating + (0 if is_neutral else self.home_advantage)
        away_elo = away_state.rating

        e_home = self.expected_score(home_elo, away_elo)

        # Convert expected score to 3-way probabilities
        # Draw probability peaks when teams are evenly matched
        draw_prob = max(0.15, 0.38 - 1.2 * abs(e_home - 0.5))
        draw_prob = min(draw_prob, 0.35)

        home_win = e_home * (1 - draw_prob / (1.0))
        away_win = (1 - e_home) * (1 - draw_prob / (1.0))

        # Normalize
        total = home_win + draw_prob + away_win
        home_win /= total
        draw_prob /= total
        away_win /= total

        return {
            "home_win": home_win,
            "draw": draw_prob,
            "away_win": away_win,
        }

    def process_match_history(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Process a chronological match DataFrame and compute Elo ratings.

        CRITICAL: Matches MUST be sorted by date. This function processes
        them in order, updating ratings after each match. The rating
        recorded for each match is the BEFORE-match rating (no leakage).

        Args:
            matches_df: DataFrame with columns: match_date, home_team,
                        away_team, home_score, away_score, competition.

        Returns:
            DataFrame with added columns: home_elo, away_elo, elo_diff,
                                          home_elo_after, away_elo_after.
        """
        matches_df = matches_df.sort_values("match_date").reset_index(drop=True)
        # Drop rows with missing scores (some datasets have incomplete records)
        matches_df = matches_df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        n = len(matches_df)

        home_elo_before = np.zeros(n)
        away_elo_before = np.zeros(n)
        home_elo_after = np.zeros(n)
        away_elo_after = np.zeros(n)

        for i, row in matches_df.iterrows():
            idx = int(i)  # type: ignore[arg-type]
            h_before, a_before, h_after, a_after = self.update(
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                competition=row.get("competition", ""),
                match_date=str(row["match_date"].date())
                if hasattr(row["match_date"], "date")
                else str(row["match_date"]),
            )
            home_elo_before[idx] = h_before
            away_elo_before[idx] = a_before
            home_elo_after[idx] = h_after
            away_elo_after[idx] = a_after

        matches_df["home_elo"] = home_elo_before
        matches_df["away_elo"] = away_elo_before
        matches_df["elo_diff"] = home_elo_before - away_elo_before
        matches_df["home_elo_after"] = home_elo_after
        matches_df["away_elo_after"] = away_elo_after

        logger.info(
            "elo_ratings_computed",
            matches=n,
            teams=len(self.ratings),
            mean_rating=round(self._compute_global_mean(), 1),
        )
        return matches_df


# -- Glicko-2 Rating System ------------------------------------------------


@dataclass
class GlickoState:
    """Glicko-2 rating state for a team.

    Attributes:
        mu: Rating on the Glicko-2 scale (default 0, maps to ~1500 Elo).
        phi: Rating deviation (uncertainty). Higher = more uncertain.
        sigma: Volatility (how much the team's strength tends to fluctuate).
    """

    mu: float = 0.0
    phi: float = 350.0 / 173.7178  # ~2.015 on Glicko-2 scale
    sigma: float = 0.06
    history: list[tuple[str, float, float]] = field(default_factory=list)

    @property
    def elo_scale_rating(self) -> float:
        """Convert to familiar Elo-scale rating (centered at 1500)."""
        return self.mu * 173.7178 + 1500

    @property
    def elo_scale_rd(self) -> float:
        """Convert rating deviation to Elo scale."""
        return self.phi * 173.7178


class Glicko2RatingSystem:
    """Glicko-2 rating system with uncertainty tracking.

    Key advantage over Elo: tracks rating deviation (uncertainty).
    Teams that haven't played recently have higher uncertainty,
    which naturally produces wider prediction intervals.

    This is especially valuable for international football where
    teams play infrequently (5-10 matches per year).

    The Glicko-2 algorithm:
    1. Convert ratings to Glicko-2 scale
    2. Compute variance and improvement from match results
    3. Update volatility using iterative algorithm
    4. Update rating deviation and rating

    Reference: Glickman, M.E. (2001). "Dynamic paired comparison models
    with stochastic variances." Journal of Applied Statistics.
    """

    TAU = 0.5  # System constant constraining volatility changes

    def __init__(self, initial_rd: float = 350.0, initial_vol: float = 0.06) -> None:
        self.initial_rd = initial_rd
        self.initial_vol = initial_vol
        self.ratings: dict[str, GlickoState] = {}

    def _get_or_create(self, team: str) -> GlickoState:
        if team not in self.ratings:
            self.ratings[team] = GlickoState(
                phi=self.initial_rd / 173.7178,
                sigma=self.initial_vol,
            )
        return self.ratings[team]

    @staticmethod
    def _g(phi: float) -> float:
        """Glicko-2 g-function: reduces impact based on opponent uncertainty."""
        return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / math.pi ** 2)

    @staticmethod
    def _e(mu: float, mu_j: float, phi_j: float) -> float:
        """Expected score against opponent j."""
        return 1.0 / (1.0 + math.exp(-Glicko2RatingSystem._g(phi_j) * (mu - mu_j)))

    def update(
        self,
        team: str,
        opponents: list[str],
        scores: list[float],
        match_date: str = "",
    ) -> None:
        """Update a team's rating after one or more matches.

        Args:
            team: Team name.
            opponents: List of opponent names.
            scores: List of scores (1.0 = win, 0.5 = draw, 0.0 = loss).
            match_date: Date string for history tracking.
        """
        state = self._get_or_create(team)
        opp_states = [self._get_or_create(opp) for opp in opponents]

        if not opponents:
            # No matches: increase rating deviation (uncertainty grows)
            state.phi = min(
                math.sqrt(state.phi ** 2 + state.sigma ** 2),
                self.initial_rd / 173.7178,
            )
            return

        # Step 3: Compute variance v
        v_inv = 0.0
        delta_sum = 0.0
        for opp_state, score in zip(opp_states, scores):
            g_val = self._g(opp_state.phi)
            e_val = self._e(state.mu, opp_state.mu, opp_state.phi)
            v_inv += g_val ** 2 * e_val * (1 - e_val)
            delta_sum += g_val * (score - e_val)

        v = 1.0 / v_inv if v_inv > 0 else 1e6
        delta = v * delta_sum

        # Step 4: Update volatility (iterative algorithm)
        a = math.log(state.sigma ** 2)
        eps = 1e-6

        def f(x: float) -> float:
            ex = math.exp(x)
            d2 = delta ** 2
            p2 = state.phi ** 2
            a1 = ex * (d2 - p2 - v - ex)
            a2 = 2 * (p2 + v + ex) ** 2
            return a1 / a2 - (x - a) / self.TAU ** 2

        # Find bounds
        big_a = a
        if delta ** 2 > state.phi ** 2 + v:
            big_b = math.log(delta ** 2 - state.phi ** 2 - v)
        else:
            k = 1
            while f(a - k * self.TAU) < 0:
                k += 1
            big_b = a - k * self.TAU

        # Bisection
        fa = f(big_a)
        fb = f(big_b)
        while abs(big_b - big_a) > eps:
            big_c = big_a + (big_a - big_b) * fa / (fb - fa)
            fc = f(big_c)
            if fc * fb <= 0:
                big_a = big_b
                fa = fb
            else:
                fa /= 2
            big_b = big_c
            fb = fc

        new_sigma = math.exp(big_a / 2)

        # Step 5: Update rating deviation
        phi_star = math.sqrt(state.phi ** 2 + new_sigma ** 2)

        # Step 6: Update rating and RD
        new_phi = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
        new_mu = state.mu + new_phi ** 2 * delta_sum

        state.mu = new_mu
        state.phi = new_phi
        state.sigma = new_sigma

        if match_date:
            state.history.append((match_date, state.elo_scale_rating, state.elo_scale_rd))

    def process_match_history(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Process matches chronologically and compute Glicko-2 ratings.

        Args:
            matches_df: Chronologically sorted match DataFrame.

        Returns:
            DataFrame with added Glicko-2 columns.
        """
        matches_df = matches_df.sort_values("match_date").reset_index(drop=True)
        matches_df = matches_df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        n = len(matches_df)

        glicko_home = np.zeros(n)
        glicko_away = np.zeros(n)
        glicko_rd_home = np.zeros(n)
        glicko_rd_away = np.zeros(n)

        for i, row in matches_df.iterrows():
            idx = int(i)  # type: ignore[arg-type]
            home = row["home_team"]
            away = row["away_team"]

            # Record BEFORE-match ratings
            h_state = self._get_or_create(home)
            a_state = self._get_or_create(away)
            glicko_home[idx] = h_state.elo_scale_rating
            glicko_away[idx] = a_state.elo_scale_rating
            glicko_rd_home[idx] = h_state.elo_scale_rd
            glicko_rd_away[idx] = a_state.elo_scale_rd

            # Determine scores
            if row["home_score"] > row["away_score"]:
                h_score, a_score = 1.0, 0.0
            elif row["home_score"] == row["away_score"]:
                h_score, a_score = 0.5, 0.5
            else:
                h_score, a_score = 0.0, 1.0

            date_str = (
                str(row["match_date"].date())
                if hasattr(row["match_date"], "date")
                else str(row["match_date"])
            )

            # Update both teams
            self.update(home, [away], [h_score], date_str)
            self.update(away, [home], [a_score], date_str)

        matches_df["home_glicko"] = glicko_home
        matches_df["away_glicko"] = glicko_away
        matches_df["home_glicko_rd"] = glicko_rd_home
        matches_df["away_glicko_rd"] = glicko_rd_away
        matches_df["glicko_diff"] = glicko_home - glicko_away

        logger.info(
            "glicko2_ratings_computed",
            matches=n,
            teams=len(self.ratings),
        )
        return matches_df


# -- Decomposed Attack/Defense Rating --------------------------------------


class AttackDefenseElo:
    """Elo system decomposed into separate attack and defense components.

    Instead of a single strength rating, each team has:
    - Attack rating: how many goals they score relative to expected
    - Defense rating: how many goals they concede relative to expected

    This captures the difference between "strong attack, weak defense"
    and "weak attack, strong defense" teams, which standard Elo misses.

    The attack/defense ratings are updated based on goal difference
    relative to the opponent's defensive/offensive strength.
    """

    def __init__(
        self,
        k_attack: float = 20.0,
        k_defense: float = 20.0,
        initial_rating: float = 0.0,
        avg_goals: float = 1.35,
    ) -> None:
        self.k_attack = k_attack
        self.k_defense = k_defense
        self.initial_rating = initial_rating
        self.avg_goals = avg_goals
        self.attack_ratings: dict[str, float] = {}
        self.defense_ratings: dict[str, float] = {}

    def _get_attack(self, team: str) -> float:
        return self.attack_ratings.get(team, self.initial_rating)

    def _get_defense(self, team: str) -> float:
        return self.defense_ratings.get(team, self.initial_rating)

    def expected_goals(self, team: str, opponent: str) -> float:
        """Compute expected goals for `team` against `opponent`.

        Formula: λ = avg_goals × exp(attack_team - defense_opponent)

        Args:
            team: Attacking team name.
            opponent: Defending team name.

        Returns:
            Expected goals (lambda parameter for Poisson distribution).
        """
        attack = self._get_attack(team)
        defense = self._get_defense(opponent)
        return self.avg_goals * math.exp(attack - defense)

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> None:
        """Update attack and defense ratings after a match.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            home_goals: Goals scored by home team.
            away_goals: Goals scored by away team.
        """
        # Expected goals for each team
        exp_home = self.expected_goals(home_team, away_team)
        exp_away = self.expected_goals(away_team, home_team)

        # Surprise factor (goals - expected, capped)
        home_attack_surprise = min(home_goals - exp_home, 3.0) / (exp_home + 1)
        away_attack_surprise = min(away_goals - exp_away, 3.0) / (exp_away + 1)

        # Update attack ratings (scoring more than expected -> increase)
        self.attack_ratings[home_team] = (
            self._get_attack(home_team) + self.k_attack * home_attack_surprise / 100
        )
        self.attack_ratings[away_team] = (
            self._get_attack(away_team) + self.k_attack * away_attack_surprise / 100
        )

        # Update defense ratings (conceding more than expected -> increase = worse)
        home_defense_surprise = min(away_goals - exp_away, 3.0) / (exp_away + 1)
        away_defense_surprise = min(home_goals - exp_home, 3.0) / (exp_home + 1)
        self.defense_ratings[home_team] = (
            self._get_defense(home_team) + self.k_defense * home_defense_surprise / 100
        )
        self.defense_ratings[away_team] = (
            self._get_defense(away_team) + self.k_defense * away_defense_surprise / 100
        )

    def process_match_history(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Process matches and compute attack/defense ratings.

        Args:
            matches_df: Chronologically sorted match DataFrame.

        Returns:
            DataFrame with added attack/defense rating columns.
        """
        matches_df = matches_df.sort_values("match_date").reset_index(drop=True)
        matches_df = matches_df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        n = len(matches_df)

        home_attack = np.zeros(n)
        away_attack = np.zeros(n)
        home_defense = np.zeros(n)
        away_defense = np.zeros(n)

        for i, row in matches_df.iterrows():
            idx = int(i)  # type: ignore[arg-type]

            # Record BEFORE-match ratings
            home_attack[idx] = self._get_attack(row["home_team"])
            away_attack[idx] = self._get_attack(row["away_team"])
            home_defense[idx] = self._get_defense(row["home_team"])
            away_defense[idx] = self._get_defense(row["away_team"])

            self.update(
                row["home_team"], row["away_team"],
                int(row["home_score"]), int(row["away_score"]),
            )

        matches_df["home_attack_rating"] = home_attack
        matches_df["away_attack_rating"] = away_attack
        matches_df["home_defense_rating"] = home_defense
        matches_df["away_defense_rating"] = away_defense
        matches_df["attack_rating_diff"] = home_attack - away_attack
        matches_df["defense_rating_diff"] = home_defense - away_defense

        logger.info(
            "attack_defense_ratings_computed",
            matches=n,
            teams=len(self.attack_ratings),
        )
        return matches_df
