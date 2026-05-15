"""International football results dataset ingestor.

Downloads the comprehensive international match results dataset from
martj42/international_football_results on GitHub. This dataset contains
45,000+ international matches going back to 1872.

This is CRITICAL for training because StatsBomb only covers ~300 matches
(WC 2018/2022, Euro 2020/2024, Copa America 2024). We need thousands
of matches for robust feature engineering and model training.

Data source: https://github.com/martj42/international_football_results
License: CC0 / Public Domain

Columns:
- date, home_team, away_team, home_score, away_score
- tournament, city, country, neutral
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.utils.constants import normalize_team_name
from src.utils.logger import get_logger

logger = get_logger(__name__)

RESULTS_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)

SHOOTOUTS_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/shootouts.csv"
)

GOALSCORERS_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/goalscorers.csv"
)


class InternationalResultsIngestor:
    """Downloads and processes the international football results dataset.

    This dataset is the backbone of the training data - it provides
    45,000+ historical matches with results, tournament type, and
    venue information. Combined with StatsBomb xG data for recent
    matches, this gives us both volume and depth.
    """

    def __init__(self, cache_dir: Path = Path("data/raw/international_results")) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_csv(self, url: str, filename: str) -> Path:
        """Download a CSV file from URL, with local caching.

        Args:
            url: URL to download from.
            filename: Local filename for caching.

        Returns:
            Path to the downloaded/cached file.
        """
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            logger.info("using_cached_file", path=str(cache_path))
            return cache_path

        logger.info("downloading", url=url)
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        logger.info("downloaded", path=str(cache_path), size_kb=len(response.content) // 1024)
        return cache_path

    def fetch_results(self, min_year: int = 2000) -> pd.DataFrame:
        """Download and parse the main results dataset.

        Filters to matches from min_year onward. Older matches have
        limited relevance for predicting modern football.

        Args:
            min_year: Earliest year to include. Default 2000.

        Returns:
            DataFrame with columns: match_date, home_team, away_team,
            home_score, away_score, competition, neutral.
        """
        path = self._download_csv(RESULTS_CSV_URL, "results.csv")
        df = pd.read_csv(path, parse_dates=["date"])

        # Filter to recent-enough matches
        df = df[df["date"].dt.year >= min_year].copy()

        # Normalize column names
        df = df.rename(columns={
            "date": "match_date",
            "tournament": "competition",
        })

        # Normalize team names
        df["home_team"] = df["home_team"].apply(normalize_team_name)
        df["away_team"] = df["away_team"].apply(normalize_team_name)

        # Derive result
        df["result"] = "D"
        df.loc[df["home_score"] > df["away_score"], "result"] = "H"
        df.loc[df["home_score"] < df["away_score"], "result"] = "A"
        df["goal_diff"] = df["home_score"] - df["away_score"]

        # Create a match_id for compatibility (negative to avoid StatsBomb collision)
        df["match_id"] = range(-len(df), 0)

        # Sort chronologically
        df = df.sort_values("match_date").reset_index(drop=True)

        logger.info(
            "international_results_loaded",
            total_matches=len(df),
            date_range=f"{df['match_date'].min().date()} -> {df['match_date'].max().date()}",
            teams=df["home_team"].nunique(),
            competitions=df["competition"].nunique(),
        )

        return df

    def fetch_goalscorers(self, min_year: int = 2000) -> pd.DataFrame:
        """Download goalscorer data.

        Args:
            min_year: Earliest year to include.

        Returns:
            DataFrame with scorer information.
        """
        path = self._download_csv(GOALSCORERS_CSV_URL, "goalscorers.csv")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[df["date"].dt.year >= min_year].copy()
        df["team"] = df["team"].apply(normalize_team_name)
        logger.info("goalscorers_loaded", rows=len(df))
        return df

    def export_parquet(self, output_dir: Path = Path("data/processed")) -> dict[str, Path]:
        """Download, process, and export to Parquet.

        Args:
            output_dir: Output directory for Parquet files.

        Returns:
            Dictionary mapping dataset name to file path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.fetch_results()
        results_path = output_dir / "international_results.parquet"
        results.to_parquet(results_path, index=False, engine="pyarrow")

        logger.info("exported_international_results", path=str(results_path), rows=len(results))
        return {"results": results_path}
