"""Global settings loaded from environment variables and .env file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


# Project root is two levels up from configs/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide configuration.

    Values are loaded from environment variables, with fallback to .env file.
    All paths are resolved relative to the project root.
    """

    # -- Paths ----------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="WC_DATA_DIR")
    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    features_dir: Path = Field(default=PROJECT_ROOT / "data" / "features")
    models_dir: Path = Field(default=PROJECT_ROOT / "data" / "models")
    logs_dir: Path = Field(default=PROJECT_ROOT / "logs")

    # -- Logging --------------------------------------------------------
    log_level: str = Field(default="INFO", alias="WC_LOG_LEVEL")

    # -- Reproducibility -----------------------------------------------
    random_seed: int = Field(default=42, alias="WC_RANDOM_SEED")

    # -- External APIs --------------------------------------------------
    football_data_api_key: Optional[str] = Field(default=None, alias="FOOTBALL_DATA_API_KEY")
    odds_api_key: Optional[str] = Field(default=None, alias="ODDS_API_KEY")

    # -- MLflow --------------------------------------------------------
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")

    # -- GPU ------------------------------------------------------------
    cuda_visible_devices: str = Field(default="0", alias="CUDA_VISIBLE_DEVICES")

    # -- StatsBomb ------------------------------------------------------
    statsbomb_base_url: str = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

    # -- Feature Engineering --------------------------------------------
    elo_k_factor_competitive: float = 40.0
    elo_k_factor_friendly: float = 20.0
    elo_k_factor_world_cup: float = 60.0
    elo_home_advantage: float = 100.0
    elo_initial_rating: float = 1500.0
    form_windows: list[int] = [3, 5, 10, 20]
    xg_rolling_windows: list[int] = [5, 10, 20]

    # -- Training ------------------------------------------------------
    n_optuna_trials: int = 100
    n_cv_folds: int = 5
    early_stopping_rounds: int = 50
    test_months: int = 6
    calibration_months: int = 6
    validation_months: int = 12

    model_config = {
        "env_file": PROJECT_ROOT / ".env", 
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True
    }

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for d in [
            self.data_dir, self.raw_dir, self.processed_dir,
            self.features_dir, self.models_dir, self.logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
