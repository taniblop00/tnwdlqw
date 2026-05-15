"""Constants, mappings, and canonical reference data for the football domain.

This module is the single source of truth for all team name normalization,
confederation mappings, competition importance weights, and other reference
data used throughout the pipeline.
"""

from __future__ import annotations

# -- FIFA Confederation Mapping ---------------------------------------------
CONFEDERATIONS: dict[str, str] = {
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL", "Ecuador": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    "Germany": "UEFA", "France": "UEFA", "Spain": "UEFA", "England": "UEFA",
    "Italy": "UEFA", "Netherlands": "UEFA", "Portugal": "UEFA",
    "Belgium": "UEFA", "Croatia": "UEFA", "Denmark": "UEFA",
    "Switzerland": "UEFA", "Austria": "UEFA", "Sweden": "UEFA",
    "Wales": "UEFA", "Poland": "UEFA", "Czech Republic": "UEFA",
    "Serbia": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA",
    "Hungary": "UEFA", "Ukraine": "UEFA", "Norway": "UEFA",
    "Romania": "UEFA", "Greece": "UEFA", "Slovakia": "UEFA",
    "Republic of Ireland": "UEFA", "Slovenia": "UEFA", "Albania": "UEFA",
    "North Macedonia": "UEFA", "Finland": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Iceland": "UEFA", "Montenegro": "UEFA", "Georgia": "UEFA",
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF", "Honduras": "CONCACAF",
    "Panama": "CONCACAF", "El Salvador": "CONCACAF",
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC",
    "Iran": "AFC", "Saudi Arabia": "AFC", "Qatar": "AFC",
    "China PR": "AFC", "Iraq": "AFC", "United Arab Emirates": "AFC",
    "Uzbekistan": "AFC", "Jordan": "AFC",
    "Senegal": "CAF", "Morocco": "CAF", "Nigeria": "CAF",
    "Cameroon": "CAF", "Ghana": "CAF", "Egypt": "CAF",
    "Algeria": "CAF", "Tunisia": "CAF", "Ivory Coast": "CAF",
    "South Africa": "CAF", "DR Congo": "CAF", "Mali": "CAF",
    "Burkina Faso": "CAF",
    "New Zealand": "OFC",
}

# -- Team Name Normalization ------------------------------------------------
# Maps common aliases and variant spellings to canonical names.
TEAM_NAME_MAP: dict[str, str] = {
    # StatsBomb variants
    "United States of America": "United States",
    "USA": "United States",
    "US": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "China": "China PR",
    "Eire": "Republic of Ireland",
    "Northern Ireland": "Northern Ireland",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    # Short codes
    "BRA": "Brazil", "ARG": "Argentina", "FRA": "France",
    "GER": "Germany", "ESP": "Spain", "ENG": "England",
    "ITA": "Italy", "NED": "Netherlands", "POR": "Portugal",
    "BEL": "Belgium", "CRO": "Croatia", "URU": "Uruguay",
    "COL": "Colombia", "MEX": "Mexico", "JPN": "Japan",
    "KOR": "South Korea", "USA_CODE": "United States",
    "SEN": "Senegal", "MAR": "Morocco", "NGA": "Nigeria",
    "GHA": "Ghana", "CMR": "Cameroon", "EGY": "Egypt",
    "AUS": "Australia", "CAN": "Canada", "QAT": "Qatar",
    "KSA": "Saudi Arabia", "IRN": "Iran", "SUI": "Switzerland",
    "DEN": "Denmark", "SWE": "Sweden", "POL": "Poland",
    "WAL": "Wales", "AUT": "Austria", "SRB": "Serbia",
    "CRC": "Costa Rica", "TUN": "Tunisia", "ALG": "Algeria",
    "ECU": "Ecuador", "PAR": "Paraguay", "PER": "Peru",
    "CHI": "Chile", "BOL": "Bolivia", "VEN": "Venezuela",
}

# -- Competition Importance Weights -----------------------------------------
# Used for Elo K-factor scaling. Higher = more weight on result.
COMPETITION_WEIGHTS: dict[str, float] = {
    "FIFA World Cup": 1.0,
    "UEFA Euro": 0.85,
    "Copa America": 0.85,
    "Africa Cup of Nations": 0.80,
    "AFC Asian Cup": 0.75,
    "CONCACAF Gold Cup": 0.70,
    "FIFA Confederations Cup": 0.70,
    "UEFA Nations League": 0.65,
    "FIFA World Cup qualification": 0.55,
    "UEFA Euro qualification": 0.50,
    "Friendly": 0.25,
    "International Friendly": 0.25,
}

# -- Match Result Encoding -------------------------------------------------
RESULT_MAP: dict[str, int] = {
    "home_win": 0,
    "draw": 1,
    "away_win": 2,
    "H": 0,
    "D": 1,
    "A": 2,
}

# -- Position Encoding -----------------------------------------------------
POSITION_MAP: dict[str, str] = {
    "Goalkeeper": "GK",
    "Right Back": "DEF", "Left Back": "DEF", "Center Back": "DEF",
    "Right Center Back": "DEF", "Left Center Back": "DEF",
    "Right Wing Back": "DEF", "Left Wing Back": "DEF",
    "Right Defensive Midfield": "MID", "Left Defensive Midfield": "MID",
    "Center Defensive Midfield": "MID",
    "Right Center Midfield": "MID", "Left Center Midfield": "MID",
    "Center Midfield": "MID",
    "Right Midfield": "MID", "Left Midfield": "MID",
    "Right Center Forward": "FWD", "Left Center Forward": "FWD",
    "Center Forward": "FWD", "Striker": "FWD",
    "Right Wing": "FWD", "Left Wing": "FWD",
    "Center Attacking Midfield": "MID",
    "Right Attacking Midfield": "MID", "Left Attacking Midfield": "MID",
}


def normalize_team_name(name: str) -> str:
    """Normalize a team name to its canonical form.

    Args:
        name: Raw team name from any data source.

    Returns:
        Canonical team name.
    """
    return TEAM_NAME_MAP.get(name, name)


def get_competition_weight(competition: str) -> float:
    """Get the importance weight for a competition.

    Falls back to 0.3 for unknown competitions (assumed minor).

    Args:
        competition: Competition name.

    Returns:
        Weight between 0.0 and 1.0.
    """
    # Try exact match first, then substring match
    if competition in COMPETITION_WEIGHTS:
        return COMPETITION_WEIGHTS[competition]
    for key, weight in COMPETITION_WEIGHTS.items():
        if key.lower() in competition.lower():
            return weight
    return 0.3
