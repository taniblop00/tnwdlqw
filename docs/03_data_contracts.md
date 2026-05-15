# World Cup AI — Data Contracts

## 1. StatsBomb Open Data

**Source**: `https://github.com/statsbomb/open-data`
**Format**: JSON files in Git repository
**Update Frequency**: Irregular (new competitions added periodically)
**License**: Free for non-commercial use

### 1.1 Matches Schema
| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| match_id | int | ✓ | > 0 | Unique match identifier |
| match_date | str | ✓ | ISO date format | YYYY-MM-DD |
| kick_off | str | | HH:MM:SS.SSS | Kickoff time |
| home_team.home_team_id | int | ✓ | > 0 | |
| home_team.home_team_name | str | ✓ | non-empty | |
| away_team.away_team_id | int | ✓ | > 0 | |
| away_team.away_team_name | str | ✓ | non-empty | |
| home_score | int | ✓ | >= 0 | |
| away_score | int | ✓ | >= 0 | |
| competition.competition_id | int | ✓ | | |
| competition.competition_name | str | ✓ | | |
| season.season_id | int | ✓ | | |
| stadium.name | str | | | |
| referee.name | str | | | |

**Missing Value Handling**:
- `stadium`, `referee`: Allow NULL — many historical matches lack this
- `kick_off`: Default to "00:00:00.000" if missing

**Deduplication**: Key = `(match_id)` — StatsBomb IDs are globally unique

**Normalization**:
- Team names → canonical mapping table (e.g., "USA" → "United States")
- Dates → UTC timezone

### 1.2 Events Schema
| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| id | str (UUID) | ✓ | UUID format | Event ID |
| index | int | ✓ | > 0 | Sequential within match |
| period | int | ✓ | 1-5 | Match period |
| timestamp | str | ✓ | MM:SS.SSS | |
| minute | int | ✓ | >= 0 | |
| second | int | ✓ | >= 0, < 60 | |
| type.id | int | ✓ | | Event type code |
| type.name | str | ✓ | | Shot/Pass/Tackle/etc |
| possession_team.id | int | ✓ | | |
| player.id | int | | | NULL for non-player events |
| player.name | str | | | |
| location | [float, float] | | x: 0-120, y: 0-80 | Pitch coordinates |
| shot.statsbomb_xg | float | | 0.0-1.0 | Only for shot events |
| shot.outcome.name | str | | | Goal/Saved/Off T/Blocked |
| pass.end_location | [float, float] | | | Only for passes |
| pass.outcome.name | str | | | Complete/Incomplete |

**Missing Value Handling**:
- `location`: NULL for events like half_start, half_end
- `player`: NULL for system events
- `shot.statsbomb_xg`: Only present for shots — skip for other events

**Deduplication**: Key = `(match_id, id)` — UUID is unique per match

### 1.3 Lineups Schema
| Field | Type | Required | Validation |
|-------|------|----------|------------|
| team_id | int | ✓ | > 0 |
| team_name | str | ✓ | non-empty |
| lineup[].player_id | int | ✓ | > 0 |
| lineup[].player_name | str | ✓ | |
| lineup[].player_nickname | str | | |
| lineup[].jersey_number | int | ✓ | 1-99 |
| lineup[].positions[].position | str | | |
| lineup[].positions[].from | str | | timestamp |
| lineup[].positions[].to | str | | timestamp |

---

## 2. FBref

**Source**: `https://fbref.com/`
**Format**: HTML tables (scraped via BeautifulSoup)
**Update Frequency**: Daily during active competitions
**Rate Limiting**: 3 seconds between requests (respect robots.txt)

### 2.1 Team Stats Schema
| Field | Type | Required | Validation | Source Table |
|-------|------|----------|------------|-------------|
| team_name | str | ✓ | non-empty | all |
| matches_played | int | ✓ | > 0 | Standard Stats |
| goals | int | ✓ | >= 0 | Standard Stats |
| assists | int | ✓ | >= 0 | Standard Stats |
| xg | float | ✓ | >= 0.0 | Standard Stats |
| xag | float | ✓ | >= 0.0 | Standard Stats |
| possession_pct | float | ✓ | 0-100 | Possession |
| passes_completed | int | ✓ | >= 0 | Passing |
| pass_accuracy_pct | float | ✓ | 0-100 | Passing |
| progressive_passes | int | ✓ | >= 0 | Passing |
| progressive_carries | int | ✓ | >= 0 | Possession |
| sca | int | ✓ | >= 0 | Shot Creation |
| gca | int | ✓ | >= 0 | Goal Creation |
| tackles | int | ✓ | >= 0 | Defensive |
| interceptions | int | ✓ | >= 0 | Defensive |
| blocks | int | ✓ | >= 0 | Defensive |
| clearances | int | ✓ | >= 0 | Defensive |
| pressures | int | ✓ | >= 0 | Defensive |
| ppda | float | | >= 0 | Derived |

**Missing Value Handling**:
- xG/xAG: Only available for leagues with tracking data; mark as NULL for others
- Per-90 stats: Compute from raw totals (avoid scraping pre-computed per-90)

**Normalization**:
- Team names → canonical mapping (FBref uses full names, map to FIFA codes)
- Stats → per-90 normalization after ingestion

**Deduplication**: Key = `(team_name, season, competition)`

### 2.2 Player Stats Schema
| Field | Type | Required | Validation |
|-------|------|----------|------------|
| player_name | str | ✓ | non-empty |
| player_url | str | ✓ | valid URL (contains fbref ID) |
| nation | str | ✓ | |
| position | str | ✓ | GK/DF/MF/FW |
| age | str | ✓ | YYY-DDD format |
| minutes_played | int | ✓ | >= 0 |
| goals | int | ✓ | >= 0 |
| assists | int | ✓ | >= 0 |
| xg | float | | >= 0.0 |
| xa | float | | >= 0.0 |
| progressive_passes | int | | >= 0 |
| progressive_carries | int | | >= 0 |
| shots_total | int | | >= 0 |
| shots_on_target | int | | >= 0 |
| key_passes | int | | >= 0 |
| tackles_won | int | | >= 0 |
| interceptions | int | | >= 0 |
| sca | int | | >= 0 |
| gca | int | | >= 0 |

---

## 3. Understat

**Source**: `https://understat.com`
**Format**: JSON embedded in JavaScript (parse from HTML)
**Update Frequency**: Daily during active leagues
**Rate Limiting**: 2 seconds between requests

### 3.1 Player xG Schema
| Field | Type | Required | Validation |
|-------|------|----------|------------|
| player_id | int | ✓ | > 0 |
| player_name | str | ✓ | non-empty |
| team | str | ✓ | |
| games | int | ✓ | > 0 |
| minutes | int | ✓ | >= 0 |
| goals | int | ✓ | >= 0 |
| xG | float | ✓ | >= 0.0 |
| assists | int | ✓ | >= 0 |
| xA | float | ✓ | >= 0.0 |
| shots | int | ✓ | >= 0 |
| key_passes | int | ✓ | >= 0 |
| npg | int | ✓ | >= 0 | (non-penalty goals)
| npxG | float | ✓ | >= 0.0 |
| xGChain | float | | >= 0.0 |
| xGBuildup | float | | >= 0.0 |

### 3.2 Shot-Level Schema
| Field | Type | Required | Validation |
|-------|------|----------|------------|
| id | int | ✓ | > 0 |
| minute | int | ✓ | >= 0 |
| X | float | ✓ | 0.0-1.0 (normalized) |
| Y | float | ✓ | 0.0-1.0 |
| xG | float | ✓ | 0.0-1.0 |
| result | str | ✓ | Goal/SavedShot/MissedShots/BlockedShot |
| situation | str | ✓ | OpenPlay/SetPiece/FromCorner/DirectFreekick/Penalty |
| shotType | str | ✓ | RightFoot/LeftFoot/Head |
| player | str | ✓ | |
| match_id | int | ✓ | > 0 |

**Normalization**: X, Y coordinates are 0-1 normalized; convert to 120x80 pitch coords

---

## 4. Football-Data.org API

**Source**: `https://api.football-data.org/v4/`
**Format**: JSON REST API
**Auth**: API key in `X-Auth-Token` header
**Rate Limit**: 10 req/min (free tier)
**Update Frequency**: Real-time during matches

### 4.1 Match Schema
| Field | Type | Required | Validation |
|-------|------|----------|------------|
| id | int | ✓ | > 0 |
| utcDate | str | ✓ | ISO 8601 |
| status | str | ✓ | SCHEDULED/LIVE/FINISHED/POSTPONED |
| matchday | int | | > 0 |
| stage | str | | GROUP_STAGE/ROUND_OF_16/etc |
| group | str | | Group A, Group B, etc |
| homeTeam.id | int | ✓ | |
| homeTeam.name | str | ✓ | |
| awayTeam.id | int | ✓ | |
| awayTeam.name | str | ✓ | |
| score.fullTime.home | int | | >= 0 |
| score.fullTime.away | int | | >= 0 |
| score.halfTime.home | int | | >= 0 |
| score.halfTime.away | int | | >= 0 |
| referees[].name | str | | |

### 4.2 Standings Schema
| Field | Type | Required | Validation |
|-------|------|----------|------------|
| stage | str | ✓ | |
| group | str | | |
| table[].position | int | ✓ | > 0 |
| table[].team.id | int | ✓ | |
| table[].team.name | str | ✓ | |
| table[].playedGames | int | ✓ | >= 0 |
| table[].won | int | ✓ | >= 0 |
| table[].draw | int | ✓ | >= 0 |
| table[].lost | int | ✓ | >= 0 |
| table[].goalsFor | int | ✓ | >= 0 |
| table[].goalsAgainst | int | ✓ | >= 0 |
| table[].goalDifference | int | ✓ | |
| table[].points | int | ✓ | >= 0 |

---

## 5. Transfermarkt

**Source**: `https://www.transfermarkt.com/`
**Format**: HTML (scraped via BeautifulSoup + Selenium)
**Update Frequency**: Weekly (market values updated quarterly)
**Rate Limiting**: 5 seconds between requests, rotate user agents

### 5.1 Player Value Schema
| Field | Type | Required | Validation |
|-------|------|----------|------------|
| player_name | str | ✓ | non-empty |
| player_url | str | ✓ | contains transfermarkt path |
| date_of_birth | str | ✓ | YYYY-MM-DD |
| nationality | str | ✓ | |
| club | str | ✓ | |
| position | str | ✓ | |
| market_value_eur | int | ✓ | > 0 |
| contract_until | str | | YYYY-MM-DD |
| agent | str | | |

### 5.2 Injury History Schema
| Field | Type | Required | Validation |
|-------|------|----------|------------|
| player_name | str | ✓ | |
| injury | str | ✓ | injury description |
| from_date | str | ✓ | YYYY-MM-DD |
| until_date | str | | YYYY-MM-DD or "?" |
| days_missed | int | | >= 0 |
| games_missed | int | | >= 0 |

**Missing Value Handling**:
- `until_date = "?"`: Injury is ongoing — set `is_active = TRUE`
- `market_value_eur`: Parse strings like "€50.00m" → 50000000

**Deduplication**: Key = `(player_name, club, date_of_birth)`

---

## 6. Odds APIs

**Sources**: OddsAPI (`https://the-odds-api.com/`), scraped Pinnacle/Bet365
**Format**: JSON REST API
**Update Frequency**: Every 15 minutes during active fixtures

### 6.1 Odds Schema
| Field | Type | Required | Validation |
|-------|------|----------|------------|
| match_id | str | ✓ | internal ID |
| home_team | str | ✓ | |
| away_team | str | ✓ | |
| commence_time | str | ✓ | ISO 8601 |
| bookmaker | str | ✓ | pinnacle/bet365/betfair |
| market | str | ✓ | h2h/totals/spreads |
| home_odds | float | ✓ | > 1.0 |
| draw_odds | float | ✓ | > 1.0 |
| away_odds | float | ✓ | > 1.0 |
| last_update | str | ✓ | ISO 8601 |

**Derived Fields** (computed on ingestion):
- `overround = 1/home + 1/draw + 1/away` (typically 1.02-1.10)
- `implied_home = (1/home_odds) / overround`
- `implied_draw = (1/draw_odds) / overround`
- `implied_away = (1/away_odds) / overround`

**Deduplication**: Key = `(match_id, bookmaker, last_update)` — keep time series of odds

---

## 7. Global Data Validation Rules

1. **Schema Enforcement**: Every ingested record passes through Pydantic model validation
2. **Type Coercion**: Strings → appropriate types with explicit error handling
3. **Range Checks**: All percentages 0-100, probabilities 0-1, coordinates within pitch
4. **Temporal Integrity**: No future dates in historical data; match_date ≤ current date
5. **Referential Integrity**: All team/player IDs must exist in master tables
6. **Completeness Score**: Each record gets a completeness score (0-1); records below 0.3 are flagged
7. **Cross-Source Reconciliation**: Match scores verified across ≥2 sources
