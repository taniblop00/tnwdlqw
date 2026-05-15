"""Data merger - combines multiple data sources into a unified dataset.

Merges:
1. International results (45,000+ matches, no xG) - volume backbone
2. StatsBomb events (300+ matches, full xG) - quality depth

The merged dataset has:
- ALL international matches from the results dataset
- xG data attached where StatsBomb coverage exists
- Unified schema for downstream feature engineering
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def merge_datasets(
    international_df: pd.DataFrame,
    statsbomb_matches_df: pd.DataFrame | None = None,
    statsbomb_events_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge international results with StatsBomb data.

    Strategy:
    - Start with the full international results dataset
    - For matches that also exist in StatsBomb, attach xG data
    - Deduplicate by (date, home_team, away_team, score)

    Args:
        international_df: Full international results DataFrame.
        statsbomb_matches_df: Optional StatsBomb matches DataFrame.
        statsbomb_events_df: Optional StatsBomb events DataFrame.

    Returns:
        Tuple of (merged_matches, events_df).
    """
    logger.info(
        "starting_merge",
        international_matches=len(international_df),
        statsbomb_matches=len(statsbomb_matches_df) if statsbomb_matches_df is not None else 0,
    )

    # Ensure date columns are datetime
    international_df = international_df.copy()
    international_df["match_date"] = pd.to_datetime(international_df["match_date"])

    # If no StatsBomb data, just return international results
    if statsbomb_matches_df is None or statsbomb_matches_df.empty:
        logger.info("no_statsbomb_data_using_international_only")
        international_df["has_xg"] = False
        events_df = pd.DataFrame()
        return international_df, events_df

    statsbomb_matches_df = statsbomb_matches_df.copy()
    statsbomb_matches_df["match_date"] = pd.to_datetime(statsbomb_matches_df["match_date"])

    # Create merge key: date + teams + score (to handle name normalization edge cases)
    def make_key(df: pd.DataFrame) -> pd.Series:
        return (
            df["match_date"].dt.strftime("%Y-%m-%d")
            + "_" + df["home_team"]
            + "_" + df["away_team"]
        )

    international_df["merge_key"] = make_key(international_df)
    statsbomb_matches_df["merge_key"] = make_key(statsbomb_matches_df)

    # Find overlapping matches
    sb_keys = set(statsbomb_matches_df["merge_key"])
    international_df["has_xg"] = international_df["merge_key"].isin(sb_keys)

    overlap_count = international_df["has_xg"].sum()
    logger.info("matches_with_xg_overlap", count=overlap_count)

    # For overlapping matches, prefer StatsBomb data (has match_id for events)
    # Remove overlaps from international and add StatsBomb versions
    non_overlap = international_df[~international_df["has_xg"]].copy()
    non_overlap["match_id"] = range(-len(non_overlap), 0)

    # Ensure StatsBomb matches have all required columns
    sb_for_merge = statsbomb_matches_df.copy()
    sb_for_merge["has_xg"] = True

    # Add missing columns with defaults
    for col in ["neutral", "city", "country"]:
        if col not in sb_for_merge.columns:
            sb_for_merge[col] = np.nan

    # Standardize columns
    common_cols = [
        "match_id", "match_date", "home_team", "away_team",
        "home_score", "away_score", "competition", "result",
        "goal_diff", "has_xg",
    ]

    # Add missing columns to both
    for col in common_cols:
        if col not in non_overlap.columns:
            non_overlap[col] = np.nan
        if col not in sb_for_merge.columns:
            if col == "result":
                sb_for_merge["result"] = "D"
                sb_for_merge.loc[
                    sb_for_merge["home_score"] > sb_for_merge["away_score"], "result"
                ] = "H"
                sb_for_merge.loc[
                    sb_for_merge["home_score"] < sb_for_merge["away_score"], "result"
                ] = "A"
            elif col == "goal_diff":
                sb_for_merge["goal_diff"] = (
                    sb_for_merge["home_score"] - sb_for_merge["away_score"]
                )
            else:
                sb_for_merge[col] = np.nan

    # Concatenate
    merged = pd.concat(
        [non_overlap[common_cols], sb_for_merge[common_cols]],
        ignore_index=True,
    )
    merged = merged.sort_values("match_date").reset_index(drop=True)

    # Deduplicate any remaining duplicates (same date + teams)
    merged["dedup_key"] = make_key(merged)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset="dedup_key", keep="last")  # keep StatsBomb version
    merged = merged.drop(columns=["dedup_key"]).reset_index(drop=True)

    if before_dedup != len(merged):
        logger.info("deduplication", removed=before_dedup - len(merged))

    events_df = statsbomb_events_df if statsbomb_events_df is not None else pd.DataFrame()

    logger.info(
        "merge_complete",
        total_matches=len(merged),
        matches_with_xg=merged["has_xg"].sum(),
        date_range=f"{merged['match_date'].min().date()} -> {merged['match_date'].max().date()}",
        teams=merged["home_team"].nunique(),
    )

    return merged, events_df


def export_merged(
    merged_df: pd.DataFrame,
    events_df: pd.DataFrame,
    output_dir: Path = Path("data/processed"),
) -> dict[str, Path]:
    """Export merged datasets to Parquet.

    Args:
        merged_df: Merged match DataFrame.
        events_df: Events DataFrame.
        output_dir: Output directory.

    Returns:
        Paths to saved files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    matches_path = output_dir / "merged_matches.parquet"
    merged_df.to_parquet(matches_path, index=False, engine="pyarrow")

    paths = {"matches": matches_path}

    if not events_df.empty:
        events_path = output_dir / "merged_events.parquet"
        events_df.to_parquet(events_path, index=False, engine="pyarrow")
        paths["events"] = events_path

    logger.info(
        "merged_data_exported",
        matches=len(merged_df),
        events=len(events_df),
    )
    return paths
