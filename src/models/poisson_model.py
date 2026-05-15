"""Dixon-Coles Poisson goal model for score distribution prediction.

This model predicts the number of goals scored by each team using
bivariate Poisson distributions with a correlation correction for
low-scoring outcomes (Dixon & Coles, 1997).

Unlike classification models that predict W/D/L directly, the Poisson
model predicts the exact score distribution P(i, j) for all scorelines.
This is critical for:
1. Correct score probability markets
2. Monte Carlo tournament simulation (needs actual score samples)
3. Over/under and BTTS market predictions
4. Ensemble diversity (different modeling approach)

Reference: Dixon, M.J. & Coles, S.G. (1997). "Modelling Association
Football Scores and Inefficiencies in the Football Betting Market."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.models.base_model import BasePredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PoissonGoalModel(BasePredictor):
    """Dixon-Coles bivariate Poisson model for score prediction.

    Each team has:
    - attack parameter (α): offensive strength
    - defense parameter (δ): defensive weakness

    The expected goals for team i against team j are:
        λ_ij = exp(μ + home_adv + α_i - δ_j)

    Where:
    - μ is the average log-goals intercept
    - home_adv is a home advantage parameter
    - α_i is team i's attack strength (higher = scores more)
    - δ_j is team j's defense strength (higher = concedes more)

    The Dixon-Coles correction adjusts probabilities for low-scoring
    games (0-0, 1-0, 0-1, 1-1) which are empirically more/less
    common than independent Poisson would predict.

    Attributes:
        attack_params: Per-team attack strength parameters.
        defense_params: Per-team defense strength parameters.
        home_advantage: Home advantage parameter.
        rho: Dixon-Coles low-score correction parameter.
        mu: Intercept (average log-goals).
    """

    name = "poisson"

    def __init__(self, max_goals: int = 10, time_decay: float = 0.003) -> None:
        """Initialize the Poisson model.

        Args:
            max_goals: Maximum goals per team to compute in score matrix.
            time_decay: Exponential decay for weighting recent matches more.
                        Higher = more weight on recent data. 0.003 gives
                        half-life of ~231 days.
        """
        self.max_goals = max_goals
        self.time_decay = time_decay
        self.attack_params: dict[str, float] = {}
        self.defense_params: dict[str, float] = {}
        self.home_advantage: float = 0.25
        self.rho: float = -0.1  # Dixon-Coles correction
        self.mu: float = 0.2  # Intercept
        self.teams: list[str] = []
        self._fitted = False

    def _get_team_index(self) -> dict[str, int]:
        """Map team names to indices."""
        return {team: i for i, team in enumerate(self.teams)}

    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        home_teams: np.ndarray,
        away_teams: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute negative log-likelihood for optimization.

        Args:
            params: Flattened parameter vector:
                    [attack_0, ..., attack_n, defense_0, ..., defense_n,
                     home_advantage, rho, mu]
            home_teams: Array of home team indices.
            away_teams: Array of away team indices.
            home_goals: Array of home goals scored.
            away_goals: Array of away goals scored.
            weights: Time-decay weights for each match.

        Returns:
            Negative log-likelihood (to be minimized).
        """
        n_teams = len(self.teams)
        attack = params[:n_teams]
        defense = params[n_teams:2 * n_teams]
        home_adv = params[2 * n_teams]
        rho = params[2 * n_teams + 1]
        mu = params[2 * n_teams + 2]

        # Compute expected goals
        lambda_home = np.exp(mu + home_adv + attack[home_teams] - defense[away_teams])
        lambda_away = np.exp(mu + attack[away_teams] - defense[home_teams])

        # Clip to prevent numerical issues
        lambda_home = np.clip(lambda_home, 0.01, 10.0)
        lambda_away = np.clip(lambda_away, 0.01, 10.0)

        # Poisson log-likelihoods
        log_lik = (
            poisson.logpmf(home_goals, lambda_home)
            + poisson.logpmf(away_goals, lambda_away)
        )

        # Dixon-Coles correction for low scores
        dc_correction = np.ones_like(log_lik)
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_11 = (home_goals == 1) & (away_goals == 1)

        dc_correction[mask_00] = 1 + lambda_home[mask_00] * lambda_away[mask_00] * rho
        dc_correction[mask_10] = 1 - lambda_away[mask_10] * rho
        dc_correction[mask_01] = 1 - lambda_home[mask_01] * rho
        dc_correction[mask_11] = 1 + rho

        # Ensure positive corrections
        dc_correction = np.maximum(dc_correction, 1e-10)
        log_lik += np.log(dc_correction)

        # Weighted sum (recent matches matter more)
        total_nll = -np.sum(weights * log_lik)

        # L2 regularization on attack/defense params (keep them centered)
        reg = 0.001 * (np.sum(attack ** 2) + np.sum(defense ** 2))

        return total_nll + reg

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        min_team_matches: int = 5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fit the Poisson model to historical match data.

        Uses L-BFGS-B optimizer with bounded parameters for speed and
        numerical stability. Filters to teams with enough matches.

        Args:
            X_train: DataFrame with match data columns.
            y_train: Not used directly (scores are in X_train).
            min_team_matches: Minimum matches for a team to be included.
            **kwargs: Additional arguments.

        Returns:
            Training metadata.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("PoissonGoalModel requires DataFrame with team/score columns")

        df = X_train.copy()
        required_cols = {"home_team", "away_team", "home_score", "away_score"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

        # Drop NaN scores
        df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

        # Filter to teams with enough matches
        all_teams = pd.concat([df["home_team"], df["away_team"]])
        team_counts = all_teams.value_counts()
        valid_teams = set(team_counts[team_counts >= min_team_matches].index)
        mask = df["home_team"].isin(valid_teams) & df["away_team"].isin(valid_teams)
        df = df[mask].reset_index(drop=True)

        logger.info("poisson_data_filtered", valid_teams=len(valid_teams), matches=len(df))

        # Build team index
        self.teams = sorted(valid_teams)
        team_idx = self._get_team_index()
        n_teams = len(self.teams)

        home_teams = df["home_team"].map(team_idx).values
        away_teams = df["away_team"].map(team_idx).values
        home_goals = df["home_score"].values.astype(float)
        away_goals = df["away_score"].values.astype(float)

        # Compute time-decay weights
        if "match_date" in df.columns:
            dates = pd.to_datetime(df["match_date"])
            max_date = dates.max()
            days_ago = (max_date - dates).dt.days.values
            weights = np.exp(-self.time_decay * days_ago)
        else:
            weights = np.ones(len(df))

        # Initial parameters
        x0 = np.zeros(2 * n_teams + 3)
        x0[2 * n_teams] = 0.25  # home advantage
        x0[2 * n_teams + 1] = -0.1  # rho
        x0[2 * n_teams + 2] = 0.2  # mu

        # Bounds prevent exp overflow and keep params reasonable
        bounds = (
            [(-2.0, 2.0)] * n_teams        # attack
            + [(-2.0, 2.0)] * n_teams      # defense
            + [(0.0, 0.6)]                  # home advantage
            + [(-0.5, 0.5)]                 # rho
            + [(-0.5, 1.0)]                 # mu
        )

        # L-BFGS-B is much faster than SLSQP for this problem size
        result = minimize(
            self._negative_log_likelihood,
            x0,
            args=(home_teams, away_teams, home_goals, away_goals, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        # Post-hoc centering (identifiability)
        attack_raw = result.x[:n_teams]
        defense_raw = result.x[n_teams:2 * n_teams]
        attack_raw -= attack_raw.mean()
        defense_raw -= defense_raw.mean()

        # Extract parameters
        self.attack_params = dict(zip(self.teams, attack_raw.tolist()))
        self.defense_params = dict(zip(self.teams, defense_raw.tolist()))
        self.home_advantage = float(result.x[2 * n_teams])
        self.rho = float(result.x[2 * n_teams + 1])
        self.mu = float(result.x[2 * n_teams + 2])
        self._fitted = True

        metadata = {
            "converged": result.success,
            "n_teams": n_teams,
            "n_matches": len(df),
            "home_advantage": round(self.home_advantage, 4),
            "rho": round(self.rho, 4),
            "mu": round(self.mu, 4),
            "nll": round(float(result.fun), 2),
        }

        logger.info("poisson_model_fitted", **metadata)
        return metadata

    def predict_score_matrix(
        self, home_team: str, away_team: str, is_neutral: bool = False,
    ) -> np.ndarray:
        """Predict the full score probability matrix.

        Returns P(home_goals=i, away_goals=j) for all i,j ∈ [0, max_goals].

        Args:
            home_team: Home team name.
            away_team: Away team name.
            is_neutral: Whether venue is neutral (no home advantage).

        Returns:
            Array of shape (max_goals+1, max_goals+1).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        ha = 0 if is_neutral else self.home_advantage
        attack_h = self.attack_params.get(home_team, 0.0)
        defense_h = self.defense_params.get(home_team, 0.0)
        attack_a = self.attack_params.get(away_team, 0.0)
        defense_a = self.defense_params.get(away_team, 0.0)

        lambda_h = np.exp(self.mu + ha + attack_h - defense_a)
        lambda_a = np.exp(self.mu + attack_a - defense_h)

        lambda_h = np.clip(lambda_h, 0.01, 10.0)
        lambda_a = np.clip(lambda_a, 0.01, 10.0)

        # Independent Poisson probabilities
        goals = np.arange(self.max_goals + 1)
        prob_h = poisson.pmf(goals, lambda_h)
        prob_a = poisson.pmf(goals, lambda_a)

        # Outer product for joint distribution
        score_matrix = np.outer(prob_h, prob_a)

        # Dixon-Coles correction
        score_matrix[0, 0] *= max(1 + lambda_h * lambda_a * self.rho, 1e-10)
        score_matrix[1, 0] *= max(1 - lambda_a * self.rho, 1e-10)
        score_matrix[0, 1] *= max(1 - lambda_h * self.rho, 1e-10)
        score_matrix[1, 1] *= max(1 + self.rho, 1e-10)

        # Re-normalize
        score_matrix /= score_matrix.sum()

        return score_matrix

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict W/D/L probabilities from score distribution.

        Sums the score matrix across the appropriate regions:
        - Home win: sum of P(i,j) where i > j
        - Draw: sum of P(i,j) where i == j
        - Away win: sum of P(i,j) where i < j

        Args:
            X: DataFrame with home_team, away_team columns.

        Returns:
            Array of shape (n, 3): [P(home_win), P(draw), P(away_win)].
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PoissonGoalModel.predict_proba requires DataFrame")

        results = []
        for _, row in X.iterrows():
            home = row.get("home_team", "")
            away = row.get("away_team", "")

            if home not in self.attack_params or away not in self.attack_params:
                # Unknown team — return uniform
                results.append([1 / 3, 1 / 3, 1 / 3])
                continue

            matrix = self.predict_score_matrix(home, away)
            n = matrix.shape[0]

            home_win = sum(matrix[i, j] for i in range(n) for j in range(n) if i > j)
            draw = sum(matrix[i, i] for i in range(n))
            away_win = sum(matrix[i, j] for i in range(n) for j in range(n) if i < j)

            total = home_win + draw + away_win
            results.append([home_win / total, draw / total, away_win / total])

        return np.array(results)

    def sample_score(
        self, home_team: str, away_team: str, is_neutral: bool = False,
    ) -> tuple[int, int]:
        """Sample a random score from the predicted distribution.

        Used in Monte Carlo tournament simulation.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            is_neutral: Whether venue is neutral.

        Returns:
            Tuple of (home_goals, away_goals).
        """
        matrix = self.predict_score_matrix(home_team, away_team, is_neutral)
        flat_probs = matrix.flatten()
        idx = np.random.choice(len(flat_probs), p=flat_probs)
        home_goals = idx // matrix.shape[1]
        away_goals = idx % matrix.shape[1]
        return int(home_goals), int(away_goals)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model_data = {
            "attack_params": self.attack_params,
            "defense_params": self.defense_params,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
            "mu": self.mu,
            "teams": self.teams,
            "max_goals": self.max_goals,
            "time_decay": self.time_decay,
        }
        with open(path / "poisson_model.json", "w") as f:
            json.dump(model_data, f, indent=2)
        logger.info("poisson_saved", path=str(path))

    def load(self, path: Path) -> None:
        with open(path / "poisson_model.json") as f:
            data = json.load(f)
        self.attack_params = data["attack_params"]
        self.defense_params = data["defense_params"]
        self.home_advantage = data["home_advantage"]
        self.rho = data["rho"]
        self.mu = data["mu"]
        self.teams = data["teams"]
        self.max_goals = data.get("max_goals", 10)
        self.time_decay = data.get("time_decay", 0.003)
        self._fitted = True
        logger.info("poisson_loaded", path=str(path))

    def get_team_strengths(self) -> pd.DataFrame:
        """Get team attack and defense strength parameters.

        Returns:
            DataFrame with columns: team, attack, defense, expected_goals.
        """
        rows = []
        for team in self.teams:
            atk = self.attack_params.get(team, 0)
            dfn = self.defense_params.get(team, 0)
            # Expected goals against average opponent at neutral venue
            exp_goals = np.exp(self.mu + atk)
            rows.append({
                "team": team,
                "attack": round(atk, 4),
                "defense": round(dfn, 4),
                "expected_goals_per_match": round(exp_goals, 3),
            })
        return pd.DataFrame(rows).sort_values("attack", ascending=False)
