"""StatsBomb Open Data ingestor.

Downloads and parses the StatsBomb open-data repository which provides
detailed match, event, and lineup data for international competitions
including FIFA World Cups, Euros, and Copa Americas.

This is the PRIMARY dataset for the forecasting engine. It provides:
- Match-level results and metadata
- Event-level data (shots with xG, passes, tackles, etc.)
- Lineup data (starters, substitutions, positions)

Data source: https://github.com/statsbomb/open-data
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils.constants import normalize_team_name
from src.utils.logger import get_logger

logger = get_logger(__name__)

# -- StatsBomb competition IDs for international football -------------------
# These are the VERIFIED competition IDs from StatsBomb open-data.
# Run check_comps.py to verify against the live catalog.
INTERNATIONAL_COMPETITION_IDS: dict[int, str] = {
    43: "FIFA World Cup",
    55: "UEFA Euro",
    223: "Copa America",
    1267: "African Cup of Nations",
    1470: "FIFA U20 World Cup",
    72: "Women's World Cup",
}

# Default: only men's senior international competitions (saves disk/time)
DEFAULT_COMPETITION_IDS: set[int] = {43, 55, 223}


@dataclass
class StatsBombMatch:
    """Parsed match record from StatsBomb data."""

    match_id: int
    match_date: str  # YYYY-MM-DD
    kick_off: str
    competition: str
    season: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    stadium: str | None = None
    referee: str | None = None
    home_managers: str | None = None
    away_managers: str | None = None
    competition_id: int = 0
    season_id: int = 0


@dataclass
class StatsBombEvent:
    """Parsed event record from StatsBomb data."""

    event_id: str
    match_id: int
    minute: int
    second: int
    period: int
    event_type: str
    team: str
    player: str | None = None
    x: float | None = None
    y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    xg: float | None = None
    outcome: str | None = None
    body_part: str | None = None
    technique: str | None = None
    pass_type: str | None = None


@dataclass
class StatsBombData:
    """Container for all ingested StatsBomb data."""

    matches: list[StatsBombMatch] = field(default_factory=list)
    events: list[StatsBombEvent] = field(default_factory=list)
    lineups: dict[int, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)


class StatsBombIngestor:
    """Fetches and parses data from the StatsBomb open-data GitHub repository.

    The ingestor:
    1. Fetches the competition catalog
    2. Filters to international competitions
    3. Downloads match data for each competition/season
    4. Downloads event data for each match
    5. Downloads lineup data for each match
    6. Parses everything into structured dataclasses
    7. Exports to Parquet for downstream use

    Attributes:
        base_url: StatsBomb open-data raw GitHub URL.
        raw_dir: Directory to cache raw JSON files.
        rate_limit_seconds: Minimum seconds between HTTP requests.
    """

    def __init__(
        self,
        base_url: str = "https://raw.githubusercontent.com/statsbomb/open-data/master/data",
        raw_dir: Path = Path("data/raw/statsbomb"),
        rate_limit_seconds: float = 0.5,
    ) -> None:
        self.base_url = base_url
        self.raw_dir = raw_dir
        self.rate_limit_seconds = rate_limit_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WorldCupAI/1.0 (research)"})
        self._last_request_time: float = 0.0

    def _rate_limited_get(self, url: str) -> requests.Response:
        """Make an HTTP GET request with rate limiting.

        Args:
            url: URL to fetch.

        Returns:
            HTTP response.

        Raises:
            requests.HTTPError: If the request fails.
        """
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)

        response = self.session.get(url, timeout=30)
        self._last_request_time = time.monotonic()
        response.raise_for_status()
        return response

    def _load_or_fetch_json(self, url: str, cache_path: Path) -> Any:
        """Load JSON from cache or fetch from URL.

        Args:
            url: Remote URL to fetch.
            cache_path: Local file path for caching.

        Returns:
            Parsed JSON data.
        """
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        response = self._rate_limited_get(url)
        data = response.json()

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        return data

    def _fetch_json_no_cache(self, url: str) -> Any:
        """Fetch JSON without caching to disk. Used for event data
        which is too large to cache with limited disk space.

        Args:
            url: Remote URL to fetch.

        Returns:
            Parsed JSON data.
        """
        response = self._rate_limited_get(url)
        return response.json()

    def fetch_competitions(self) -> list[dict[str, Any]]:
        """Fetch the list of available competitions from StatsBomb.

        Returns:
            List of competition/season records.
        """
        url = f"{self.base_url}/competitions.json"
        cache_path = self.raw_dir / "competitions.json"
        data = self._load_or_fetch_json(url, cache_path)
        logger.info("fetched_competitions", total=len(data))
        return data

    def get_international_competitions(self) -> list[dict[str, Any]]:
        """Filter competitions to international football only.

        Returns:
            List of competition/season records for international matches.
        """
        all_comps = self.fetch_competitions()
        international = [
            c for c in all_comps
            if c.get("competition_id") in INTERNATIONAL_COMPETITION_IDS
        ]
        logger.info(
            "filtered_international_competitions",
            total=len(all_comps),
            international=len(international),
        )
        return international

    def fetch_matches(self, competition_id: int, season_id: int) -> list[dict[str, Any]]:
        """Fetch all matches for a competition/season.

        Args:
            competition_id: StatsBomb competition ID.
            season_id: StatsBomb season ID.

        Returns:
            List of raw match records.
        """
        url = f"{self.base_url}/matches/{competition_id}/{season_id}.json"
        cache_path = self.raw_dir / f"matches/{competition_id}_{season_id}.json"

        try:
            data = self._load_or_fetch_json(url, cache_path)
            logger.info(
                "fetched_matches",
                competition_id=competition_id,
                season_id=season_id,
                count=len(data),
            )
            return data
        except requests.HTTPError as e:
            logger.warning(
                "failed_to_fetch_matches",
                competition_id=competition_id,
                season_id=season_id,
                error=str(e),
            )
            return []

    def fetch_events(self, match_id: int) -> list[dict[str, Any]]:
        """Fetch all events for a specific match.

        Uses no-cache mode to avoid filling disk with large event JSONs.
        Each match's events are ~500KB-2MB of JSON.

        Args:
            match_id: StatsBomb match ID.

        Returns:
            List of raw event records.
        """
        url = f"{self.base_url}/events/{match_id}.json"

        try:
            return self._fetch_json_no_cache(url)
        except requests.HTTPError as e:
            logger.warning("failed_to_fetch_events", match_id=match_id, error=str(e))
            return []
        except Exception as e:
            logger.warning("event_fetch_error", match_id=match_id, error=str(e))
            return []

    def fetch_lineups(self, match_id: int) -> list[dict[str, Any]]:
        """Fetch lineup data for a specific match.

        Args:
            match_id: StatsBomb match ID.

        Returns:
            List of team lineup records.
        """
        url = f"{self.base_url}/lineups/{match_id}.json"
        cache_path = self.raw_dir / f"lineups/{match_id}.json"

        try:
            return self._load_or_fetch_json(url, cache_path)
        except requests.HTTPError as e:
            logger.warning("failed_to_fetch_lineups", match_id=match_id, error=str(e))
            return []

    def parse_match(self, raw: dict[str, Any], competition_name: str) -> StatsBombMatch:
        """Parse a raw StatsBomb match record into a structured object.

        Args:
            raw: Raw match JSON from StatsBomb API.
            competition_name: Name of the competition.

        Returns:
            Parsed StatsBombMatch.
        """
        home_team = normalize_team_name(raw["home_team"]["home_team_name"])
        away_team = normalize_team_name(raw["away_team"]["away_team_name"])

        return StatsBombMatch(
            match_id=raw["match_id"],
            match_date=raw["match_date"],
            kick_off=raw.get("kick_off", "00:00:00.000"),
            competition=competition_name,
            season=raw.get("season", {}).get("season_name", ""),
            home_team=home_team,
            away_team=away_team,
            home_score=raw["home_score"],
            away_score=raw["away_score"],
            stadium=raw.get("stadium", {}).get("name") if raw.get("stadium") else None,
            referee=raw.get("referee", {}).get("name") if raw.get("referee") else None,
            home_managers=(
                raw.get("home_team", {}).get("managers", [{}])[0].get("name")
                if raw.get("home_team", {}).get("managers")
                else None
            ),
            away_managers=(
                raw.get("away_team", {}).get("managers", [{}])[0].get("name")
                if raw.get("away_team", {}).get("managers")
                else None
            ),
            competition_id=raw.get("competition", {}).get("competition_id", 0),
            season_id=raw.get("season", {}).get("season_id", 0),
        )

    def parse_events(self, raw_events: list[dict[str, Any]], match_id: int) -> list[StatsBombEvent]:
        """Parse raw StatsBomb events into structured objects.

        Extracts key event types: shots (with xG), passes, tackles, cards.
        Filters to actionable events only (skips camera/system events).

        Args:
            raw_events: List of raw event records.
            match_id: Associated match ID.

        Returns:
            List of parsed StatsBombEvent objects.
        """
        parsed: list[StatsBombEvent] = []
        # Only extract shots - the critical events for xG features.
        # Passes, tackles etc. are secondary and can be added later.
        # This reduces memory usage by ~90%.
        actionable_types = {"Shot"}

        for raw_evt in raw_events:
            event_type = raw_evt.get("type", {}).get("name", "")
            if event_type not in actionable_types:
                continue

            location = raw_evt.get("location", [None, None])
            end_location: list[float | None] = [None, None]

            # Extract end location from pass/shot
            if event_type == "Pass" and "pass" in raw_evt:
                end_location = raw_evt["pass"].get("end_location", [None, None])
            elif event_type == "Shot" and "shot" in raw_evt:
                end_location = raw_evt["shot"].get("end_location", [None, None, None])[:2]

            # Extract xG for shots
            xg: float | None = None
            if event_type == "Shot" and "shot" in raw_evt:
                xg = raw_evt["shot"].get("statsbomb_xg")

            # Extract outcome
            outcome: str | None = None
            if event_type == "Shot" and "shot" in raw_evt:
                outcome = raw_evt["shot"].get("outcome", {}).get("name")
            elif event_type == "Pass" and "pass" in raw_evt:
                outcome = raw_evt["pass"].get("outcome", {}).get("name")

            # Extract body part for shots
            body_part: str | None = None
            if event_type == "Shot" and "shot" in raw_evt:
                body_part = raw_evt["shot"].get("body_part", {}).get("name")

            # Extract technique for shots
            technique: str | None = None
            if event_type == "Shot" and "shot" in raw_evt:
                technique = raw_evt["shot"].get("technique", {}).get("name")

            team_name = raw_evt.get("team", {}).get("name", "")
            player_name = raw_evt.get("player", {}).get("name")

            parsed.append(StatsBombEvent(
                event_id=raw_evt["id"],
                match_id=match_id,
                minute=raw_evt.get("minute", 0),
                second=raw_evt.get("second", 0),
                period=raw_evt.get("period", 1),
                event_type=event_type,
                team=normalize_team_name(team_name),
                player=player_name,
                x=location[0] if len(location) > 0 else None,
                y=location[1] if len(location) > 1 else None,
                end_x=end_location[0] if len(end_location) > 0 else None,
                end_y=end_location[1] if len(end_location) > 1 else None,
                xg=xg,
                outcome=outcome,
                body_part=body_part,
                technique=technique,
            ))

        return parsed

    def ingest_all(
        self,
        include_events: bool = True,
        include_lineups: bool = False,
        competition_ids: set[int] | None = None,
    ) -> StatsBombData:
        """Run the full ingestion pipeline.

        Downloads international competitions, their matches, and shot events.
        Match metadata is cached locally; events are processed in-memory.

        Args:
            include_events: Whether to fetch event-level data (for xG).
            include_lineups: Whether to fetch lineup data.
            competition_ids: Filter to specific competition IDs.
                             Defaults to DEFAULT_COMPETITION_IDS.

        Returns:
            StatsBombData container with all parsed data.
        """
        logger.info("starting_statsbomb_ingestion")
        data = StatsBombData()

        competitions = self.get_international_competitions()
        # Use default competition set if not specified
        filter_ids = competition_ids or DEFAULT_COMPETITION_IDS
        competitions = [
            c for c in competitions
            if c["competition_id"] in filter_ids
        ]

        total_matches = 0

        for comp in competitions:
            comp_id = comp["competition_id"]
            season_id = comp["season_id"]
            comp_name = comp["competition_name"]
            season_name = comp.get("season_name", "")

            logger.info(
                "processing_competition",
                competition=comp_name,
                season=season_name,
            )

            raw_matches = self.fetch_matches(comp_id, season_id)

            for raw_match in raw_matches:
                match = self.parse_match(raw_match, comp_name)
                data.matches.append(match)
                total_matches += 1

                if include_events:
                    raw_events = self.fetch_events(match.match_id)
                    events = self.parse_events(raw_events, match.match_id)
                    data.events.extend(events)

                if include_lineups:
                    raw_lineups = self.fetch_lineups(match.match_id)
                    lineup_dict: dict[str, list[dict[str, Any]]] = {}
                    for team_lineup in raw_lineups:
                        team_name = normalize_team_name(team_lineup.get("team_name", ""))
                        lineup_dict[team_name] = team_lineup.get("lineup", [])
                    data.lineups[match.match_id] = lineup_dict

            logger.info(
                "completed_competition",
                competition=comp_name,
                season=season_name,
                matches_in_season=len(raw_matches),
            )

        logger.info(
            "statsbomb_ingestion_complete",
            total_matches=total_matches,
            total_events=len(data.events),
            total_lineups=len(data.lineups),
        )
        return data

    def to_dataframes(self, data: StatsBombData) -> dict[str, pd.DataFrame]:
        """Convert parsed StatsBomb data to Pandas DataFrames.

        Args:
            data: Parsed StatsBomb data container.

        Returns:
            Dictionary with 'matches', 'events' DataFrames.
        """
        matches_df = pd.DataFrame([
            {
                "match_id": m.match_id,
                "match_date": pd.to_datetime(m.match_date),
                "competition": m.competition,
                "season": m.season,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "home_score": m.home_score,
                "away_score": m.away_score,
                "stadium": m.stadium,
                "referee": m.referee,
                "home_manager": m.home_managers,
                "away_manager": m.away_managers,
                "competition_id": m.competition_id,
                "season_id": m.season_id,
            }
            for m in data.matches
        ])

        if len(matches_df) > 0:
            matches_df = matches_df.sort_values("match_date").reset_index(drop=True)
            # Derive result column
            matches_df["result"] = "D"
            matches_df.loc[
                matches_df["home_score"] > matches_df["away_score"], "result"
            ] = "H"
            matches_df.loc[
                matches_df["home_score"] < matches_df["away_score"], "result"
            ] = "A"
            # Goal difference
            matches_df["goal_diff"] = (
                matches_df["home_score"] - matches_df["away_score"]
            )

        events_df = pd.DataFrame([
            {
                "event_id": e.event_id,
                "match_id": e.match_id,
                "minute": e.minute,
                "second": e.second,
                "period": e.period,
                "event_type": e.event_type,
                "team": e.team,
                "player": e.player,
                "x": e.x,
                "y": e.y,
                "end_x": e.end_x,
                "end_y": e.end_y,
                "xg": e.xg,
                "outcome": e.outcome,
                "body_part": e.body_part,
                "technique": e.technique,
            }
            for e in data.events
        ])

        logger.info(
            "converted_to_dataframes",
            matches=len(matches_df),
            events=len(events_df),
        )
        return {"matches": matches_df, "events": events_df}

    def export_parquet(self, data: StatsBombData, output_dir: Path | None = None) -> dict[str, Path]:
        """Export parsed data to Parquet files.

        Args:
            data: Parsed StatsBomb data.
            output_dir: Output directory. Defaults to data/processed/.

        Returns:
            Dictionary mapping dataset name to file path.
        """
        output_dir = output_dir or Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        dfs = self.to_dataframes(data)
        paths: dict[str, Path] = {}

        for name, df in dfs.items():
            path = output_dir / f"statsbomb_{name}.parquet"
            df.to_parquet(path, index=False, engine="pyarrow")
            paths[name] = path
            logger.info("exported_parquet", name=name, path=str(path), rows=len(df))

        return paths
