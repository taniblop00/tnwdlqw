# World Cup AI — Feature Engineering Specification

## Feature Catalog: 300+ Engineered Features

All features use STRICT temporal filtering: only data available BEFORE match_date is used.

---

## Category 1: Team Form Features (40 features)

### 1.1 Rolling Match Results
| # | Feature | Formula | Window | Source | Why It Matters |
|---|---------|---------|--------|--------|---------------|
| 1 | `form_win_rate_5` | wins / matches in last 5 | 5 matches | matches | Short-term momentum |
| 2 | `form_win_rate_10` | wins / matches in last 10 | 10 matches | matches | Medium-term form |
| 3 | `form_win_rate_20` | wins / matches in last 20 | 20 matches | matches | Long-term consistency |
| 4 | `form_draw_rate_5` | draws / matches in last 5 | 5 | matches | Defensive tendency |
| 5 | `form_loss_rate_5` | losses / matches in last 5 | 5 | matches | Vulnerability |
| 6 | `form_points_per_game_5` | (3W + D) / 5 | 5 | matches | PPG form |
| 7 | `form_points_per_game_10` | (3W + D) / 10 | 10 | matches | PPG stability |
| 8 | `form_goals_scored_avg_5` | mean(GF) last 5 | 5 | matches | Attack output |
| 9 | `form_goals_scored_avg_10` | mean(GF) last 10 | 10 | matches | |
| 10 | `form_goals_conceded_avg_5` | mean(GA) last 5 | 5 | matches | Defensive solidity |
| 11 | `form_goals_conceded_avg_10` | mean(GA) last 10 | 10 | matches | |
| 12 | `form_goal_diff_avg_5` | mean(GF - GA) last 5 | 5 | matches | Net dominance |
| 13 | `form_clean_sheets_5` | count(GA=0) last 5 | 5 | matches | Defensive excellence |
| 14 | `form_btts_rate_5` | count(GF>0 & GA>0) / 5 | 5 | matches | Open match tendency |
| 15 | `form_over25_rate_5` | count(GF+GA > 2.5) / 5 | 5 | matches | High-scoring tendency |

### 1.2 Weighted Form (exponential decay)
| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 16 | `form_weighted_points_exp` | Σ(points_i × 0.9^(n-i)) / Σ(0.9^(n-i)) | Exponential decay: recent matches weigh more; λ=0.9 |
| 17 | `form_weighted_gf_exp` | Same formula applied to goals scored | |
| 18 | `form_weighted_ga_exp` | Same formula applied to goals conceded | |

### 1.3 Venue-Adjusted Form
| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 19 | `form_home_win_rate_10` | Home wins / home matches (last 10 home) | Home advantage signal |
| 20 | `form_away_win_rate_10` | Away wins / away matches (last 10 away) | Away resilience |
| 21 | `form_home_goals_avg` | Mean goals scored at home (last 10) | |
| 22 | `form_away_goals_avg` | Mean goals scored away (last 10) | |

### 1.4 Competition-Specific Form
| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 23 | `form_competitive_win_rate` | Win rate in competitive matches only (exclude friendlies) | Friendlies have different intensity |
| 24 | `form_tournament_win_rate` | Win rate in major tournaments (WC, Euros, Copa) | Tournament-specific pressure |
| 25 | `form_knockout_win_rate` | Win rate in knockout stages | Clutch performance |
| 26 | `form_vs_top20_win_rate` | Win rate vs FIFA top-20 teams | Quality of opposition |
| 27 | `form_vs_confederation_rate` | Win rate vs same confederation | Regional familiarity |

### 1.5 Streak Features
| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 28 | `streak_unbeaten` | Consecutive matches without loss | Confidence indicator |
| 29 | `streak_wins` | Consecutive wins | Peak momentum |
| 30 | `streak_losses` | Consecutive losses | Crisis indicator |
| 31 | `streak_clean_sheets` | Consecutive clean sheets | Defensive peak |
| 32 | `streak_scoring` | Consecutive matches with a goal | Attacking reliability |

### 1.6 Head-to-Head
| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 33 | `h2h_win_rate` | Historical win rate vs opponent | Psychological edge |
| 34 | `h2h_goals_avg` | Mean goals vs opponent | Matchup-specific attack |
| 35 | `h2h_conceded_avg` | Mean goals conceded vs opponent | |
| 36 | `h2h_matches_count` | Number of historical meetings | Sample reliability |
| 37 | `h2h_recent_result` | Encoded result of last meeting (W=3/D=1/L=0) | Recent memory |
| 38 | `h2h_win_rate_last5` | Win rate in last 5 meetings | Recent head-to-head trend |

### 1.7 Momentum Indicators
| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 39 | `momentum_form_delta` | form_5 - form_10 | Improving or declining |
| 40 | `momentum_goal_delta` | goals_avg_5 - goals_avg_10 | Attack trend |

---

## Category 2: Rating Features (20 features)

| # | Feature | Formula | Source | Description |
|---|---------|---------|--------|-------------|
| 41 | `elo_rating` | Standard Elo: R_new = R + K × (S - E) where E = 1/(1+10^((R_opp-R)/400)), K=40 for WC, 30 competitive, 20 friendly | matches | Overall strength |
| 42 | `elo_delta_last5` | Elo change over last 5 matches | derived | Trajectory |
| 43 | `elo_delta_last20` | Elo change over last 20 | derived | Long-term trajectory |
| 44 | `elo_diff_vs_opponent` | team_elo - opponent_elo | derived | Match strength gap |
| 45 | `elo_home_advantage` | Add +100 Elo for home team | derived | Standard adjustment |
| 46 | `fifa_ranking` | Official FIFA ranking position | FIFA | Institutional rating |
| 47 | `fifa_ranking_points` | FIFA ranking points | FIFA | Continuous rating |
| 48 | `fifa_rank_diff` | team_rank - opponent_rank | derived | Rank gap |
| 49 | `glicko_rating` | Glicko-2 rating | derived | Bayesian rating |
| 50 | `glicko_rd` | Glicko rating deviation | derived | Uncertainty |
| 51 | `glicko_volatility` | Glicko volatility σ | derived | Rating stability |
| 52 | `attack_rating` | Elo decomposed: offensive component based on goals scored relative to expected | derived | |
| 53 | `defense_rating` | Elo decomposed: defensive component based on goals conceded relative to expected | derived | |
| 54 | `elo_weighted_competitive` | Elo using only competitive matches (K=0 for friendlies) | derived | |
| 55 | `elo_surface` | Elo computed only on same surface type | derived | Surface-specific |
| 56 | `pi_rating_home` | Pi-rating home component | derived | Home/away decomposition |
| 57 | `pi_rating_away` | Pi-rating away component | derived | |
| 58 | `rating_consistency` | Std deviation of Elo over last 20 matches | derived | Consistency |
| 59 | `rating_peak_ratio` | Current Elo / max Elo in last 2 years | derived | Peak form |
| 60 | `rating_confederation_adjusted` | Elo adjusted for confederation strength | derived | Cross-confederation comparison |

---

## Category 3: Expected Goals Features (30 features)

| # | Feature | Formula | Window | Description |
|---|---------|---------|--------|-------------|
| 61 | `xg_per90_5` | mean(xG / (minutes/90)) last 5 | 5 | Attack quality |
| 62 | `xg_per90_10` | mean(xG / (minutes/90)) last 10 | 10 | |
| 63 | `xg_per90_20` | mean(xG / (minutes/90)) last 20 | 20 | |
| 64 | `xga_per90_5` | mean(xGA / (minutes/90)) last 5 | 5 | Defense quality |
| 65 | `xga_per90_10` | Same, last 10 | 10 | |
| 66 | `xga_per90_20` | Same, last 20 | 20 | |
| 67 | `xg_diff_5` | xg_per90_5 - xga_per90_5 | 5 | Net xG dominance |
| 68 | `xg_diff_10` | xg_per90_10 - xga_per90_10 | 10 | |
| 69 | `xg_overperformance_5` | mean(goals - xG) last 5 | 5 | Finishing above expected |
| 70 | `xga_overperformance_5` | mean(goals_conceded - xGA) last 5 | 5 | GK saving above expected |
| 71 | `xg_trend` | slope of linear regression on xG last 10 matches | 10 | xG trajectory |
| 72 | `xg_variance_10` | var(xG) last 10 | 10 | Attacking consistency |
| 73 | `npxg_per90_10` | mean(non-penalty xG per 90) last 10 | 10 | Open play quality |
| 74 | `shot_quality_avg_10` | mean(xG per shot) last 10 | 10 | Chance quality |
| 75 | `shots_per90_10` | mean(shots per 90) last 10 | 10 | Shot volume |
| 76 | `sot_per90_10` | mean(shots on target per 90) last 10 | 10 | Shot accuracy |
| 77 | `conversion_rate_10` | goals / shots last 10 | 10 | Finishing efficiency |
| 78 | `big_chances_created_10` | count(xG > 0.3) last 10 | 10 | High-quality chances |
| 79 | `xg_from_set_pieces_10` | mean(set piece xG) last 10 | 10 | Set piece threat |
| 80 | `xg_from_open_play_10` | mean(open play xG) last 10 | 10 | Open play threat |
| 81 | `xg_first_half_10` | mean(first-half xG) last 10 | 10 | Early game intensity |
| 82 | `xg_second_half_10` | mean(second-half xG) last 10 | 10 | Late game intensity |
| 83-90 | `xg_diff_match_*` | xg_per90 difference between the two teams for each window | | Match-specific differential |

---

## Category 4: Tactical & Possession Features (35 features)

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 91 | `possession_avg_10` | mean(possession %) last 10 | Ball dominance |
| 92 | `ppda_avg_10` | mean(passes per defensive action) last 10 | Pressing intensity (lower = more pressing) |
| 93 | `ppda_allowed_avg_10` | mean(opponent PPDA) last 10 | Ability to play through press |
| 94 | `progressive_passes_per90_10` | mean(progressive passes per 90) last 10 | Vertical passing |
| 95 | `progressive_carries_per90_10` | mean(progressive carries per 90) last 10 | Ball carrying |
| 96 | `pass_accuracy_avg_10` | mean(pass completion %) last 10 | Technical quality |
| 97 | `long_pass_accuracy_10` | mean(long pass completion %) last 10 | Direct play ability |
| 98 | `crosses_per90_10` | mean(crosses per 90) last 10 | Width in attack |
| 99 | `sca_per90_10` | mean(shot creating actions per 90) last 10 | Chance creation |
| 100 | `gca_per90_10` | mean(goal creating actions per 90) last 10 | Direct goal involvement |
| 101 | `tackles_per90_10` | mean(tackles per 90) last 10 | Defensive aggression |
| 102 | `interceptions_per90_10` | mean(interceptions per 90) last 10 | Reading the game |
| 103 | `blocks_per90_10` | mean(blocks per 90) last 10 | Shot blocking |
| 104 | `clearances_per90_10` | mean(clearances per 90) last 10 | Defensive workload |
| 105 | `aerials_won_pct_10` | mean(aerial duels won %) last 10 | Physical dominance |
| 106 | `fouls_committed_per90` | mean(fouls per 90) last 10 | Discipline |
| 107 | `fouls_won_per90` | mean(fouls drawn per 90) last 10 | Drawing fouls (skill) |
| 108 | `corners_per90_10` | mean(corners per 90) last 10 | Attacking pressure |
| 109 | `defensive_line_height_10` | mean(avg Y of back line) last 10 | High/low block |
| 110 | `counter_attack_goals_10` | mean(goals from counters) last 10 | Counter attacking ability |
| 111 | `set_piece_goals_10` | mean(goals from set pieces) last 10 | Set piece threat |
| 112 | `pressing_success_rate_10` | mean(successful pressures / pressures) last 10 | Press effectiveness |
| 113 | `ball_recovery_time_avg` | Mean seconds to recover ball | Transition speed |
| 114 | `build_up_speed` | Mean passes before shot | Direct vs patient |
| 115-125 | `tactical_diff_*` | Differential between team and opponent for each tactical feature | Matchup advantages |

---

## Category 5: Player Quality Features (45 features)

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 126 | `squad_total_market_value` | Sum of all squad market values (€) | Overall quality proxy |
| 127 | `squad_avg_market_value` | Mean squad market value | Average quality |
| 128 | `squad_median_market_value` | Median squad market value | Robust average |
| 129 | `squad_max_market_value` | Max single player value | Star power |
| 130 | `starting11_total_value` | Sum of expected starting 11 values | Starting quality |
| 131 | `bench_depth_value` | squad_total - starting11_total | Bench depth |
| 132 | `squad_avg_age` | Mean age of squad | Experience vs youth |
| 133 | `squad_avg_caps` | Mean international caps | International experience |
| 134 | `squad_total_goals` | Sum of goals in last season | Goalscoring depth |
| 135 | `squad_total_xg` | Sum of xG in last season | Expected output |
| 136 | `top_scorer_xg_per90` | Best player's xG/90 | Star attacker quality |
| 137 | `top_scorer_goals` | Best player's goal tally | Star output |
| 138 | `gk_save_pct` | Starting GK save percentage | Goalkeeper quality |
| 139 | `gk_psxg_minus_ga` | GK post-shot xG minus GA | GK shot stopping above expected |
| 140 | `defense_avg_tackle_rate` | Mean tackle success % of defenders | Defensive solidity |
| 141 | `midfield_progressive_passes` | Sum of midfield progressive passes per 90 | Midfield creativity |
| 142 | `attack_combined_xg` | Sum of forwards' xG per 90 | Forward line quality |
| 143 | `squad_injury_count` | Number of injured players | Availability |
| 144 | `squad_injury_impact` | Sum of injured players' market values | Quality of missing players |
| 145 | `key_player_available` | 1 if top-3 players by value are available, else fraction | Star availability |
| 146 | `lineup_stability` | Jaccard similarity of last 3 starting lineups | Team cohesion |
| 147 | `minutes_concentration` | Herfindahl index of minutes distribution | Squad rotation level |
| 148 | `youth_ratio` | Fraction of squad under 23 | Youth development |
| 149 | `veteran_ratio` | Fraction of squad over 30 | Experience |
| 150 | `champions_league_players` | Count of players in UCL knockout stage | Elite club experience |
| 151-160 | `position_depth_*` | Number of quality options per position (GK/DEF/MID/FWD) | Positional depth |
| 161-170 | `player_form_aggregate_*` | Aggregated recent club form of top-11 players | Current player form |

---

## Category 6: Fatigue & Rest Features (25 features)

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 171 | `days_since_last_match` | match_date - last_match_date | Rest |
| 172 | `rest_differential` | team_days_rest - opponent_days_rest | Rest advantage |
| 173 | `matches_last_30_days` | Count matches in 30-day window | Fixture congestion |
| 174 | `matches_last_60_days` | Count matches in 60-day window | Season load |
| 175 | `minutes_last_30_days_avg` | Mean minutes per player in 30 days | Player fatigue |
| 176 | `travel_distance_km` | Haversine distance from last venue | Travel fatigue |
| 177 | `timezone_shift_hours` | abs(tz_current - tz_last_match) | Jet lag |
| 178 | `altitude_difference_m` | abs(current_altitude - team_home_altitude) | Altitude adjustment |
| 179 | `cumulative_fatigue_index` | Σ(minutes_i × decay^(days_since_i)) for last 90 days | Accumulated fatigue with decay |
| 180 | `sprint_load_30d` | Sum of sprint distances in 30 days (if available) | Physical load |
| 181 | `injury_risk_score` | f(fatigue_index, age, injury_history) = 0.3×fatigue + 0.3×age_factor + 0.4×injury_history | |
| 182 | `season_stage` | 0=early, 0.5=mid, 1=end of domestic season | Season fatigue |
| 183 | `tournament_stage_fatigue` | Cumulative minutes in current tournament | Tournament fatigue |
| 184 | `extra_time_in_tournament` | Count of ET matches in current tournament | Extra load |
| 185-195 | `player_fatigue_*` | Individual fatigue scores for key positions | Position-specific fatigue |

---

## Category 7: Tournament & Pressure Features (25 features)

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 196 | `tournament_experience_caps` | Sum of tournament caps across squad | Tournament pedigree |
| 197 | `world_cup_appearances` | Number of WC appearances for team | Historical presence |
| 198 | `best_wc_finish` | Encoded: Winner=7, Final=6, SF=5, QF=4, R16=3, Group=2, Never=1 | Historical ceiling |
| 199 | `years_since_last_wc` | Years since last WC appearance | Recent tournament exposure |
| 200 | `manager_tournament_experience` | Manager's previous tournament campaigns | Tactical tournament experience |
| 201 | `manager_tenure_days` | Days since manager appointed | Squad familiarity |
| 202 | `group_stage_must_win` | Binary: 1 if must win to qualify | Pressure situation |
| 203 | `knockout_pressure` | 1 / (remaining_matches_to_final + 1) | Increasing pressure |
| 204 | `is_host` | Binary: playing in host country | Home crowd advantage |
| 205 | `is_neighbor` | Binary: playing in neighboring country | Fan proximity |
| 206 | `historical_rivalry` | Binary: known historical rivalry | Extra motivation |
| 207 | `group_position_entering` | Current points in group | Must-win context |
| 208 | `qualification_probability` | Pre-match probability of qualifying | Desperation level |
| 209 | `penalty_shootout_record` | Historical penalty win rate | Penalty confidence |
| 210 | `elimination_match` | Binary: loss means elimination | Pressure multiplier |
| 211-220 | `stage_encoded_*` | One-hot encoding of tournament stage | Stage-specific patterns |

---

## Category 8: Odds & Market Features (30 features)

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 221 | `odds_implied_home_prob` | 1/home_odds / overround (Pinnacle) | Market probability |
| 222 | `odds_implied_draw_prob` | 1/draw_odds / overround | |
| 223 | `odds_implied_away_prob` | 1/away_odds / overround | |
| 224 | `odds_home_opening` | Opening decimal odds for home | Opening line |
| 225 | `odds_home_closing` | Closing decimal odds for home | Closing line (incorporates late info) |
| 226 | `odds_movement_home` | closing - opening odds | Sharp money direction |
| 227 | `odds_movement_magnitude` | abs(odds_movement_home) | Movement strength |
| 228 | `odds_pinnacle_vs_avg` | Pinnacle implied prob - mean(all bookmaker probs) | Sharp vs soft line |
| 229 | `model_vs_market_home` | model_prob_home - market_prob_home | Value detection |
| 230 | `model_vs_market_draw` | model_prob_draw - market_prob_draw | |
| 231 | `model_vs_market_away` | model_prob_away - market_prob_away | |
| 232 | `overround` | Sum of 1/odds for all outcomes | Market efficiency |
| 233 | `odds_over25_implied` | 1/over25_odds | Total goals expectation |
| 234 | `odds_btts_implied` | 1/btts_odds | Both teams scoring expectation |
| 235 | `kelly_criterion_home` | (prob × odds - 1) / (odds - 1) | Optimal bet fraction |
| 236 | `expected_value_home` | prob × odds - 1 | Positive EV signal |
| 237 | `market_consensus_entropy` | -Σ(p × log(p)) for market probs | Market uncertainty |
| 238-250 | `odds_bookmaker_spread_*` | Variance across bookmakers for each outcome | Line disagreement |

---

## Category 9: Weather & Venue Features (15 features)

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 251 | `temperature_c` | Match temperature | Climate adaptation |
| 252 | `humidity_pct` | Match humidity | Physical impact |
| 253 | `altitude_m` | Stadium altitude | Fitness impact |
| 254 | `is_rain` | Binary: precipitation > 0 | Pitch conditions |
| 255 | `wind_speed_kmh` | Wind speed | Ball trajectory |
| 256 | `temp_diff_from_home` | abs(match_temp - team_avg_home_temp) | Climate familiarity |
| 257 | `altitude_diff_from_home` | abs(match_alt - team_home_alt) | Altitude adjustment |
| 258 | `kickoff_hour_local` | Local time of kickoff (0-23) | Circadian performance |
| 259 | `is_night_match` | kickoff_hour >= 18 | Night game flag |
| 260 | `pitch_surface` | Encoded: natural=0, artificial=1, hybrid=0.5 | Surface adaptation |
| 261-265 | `venue_familiarity_*` | How many times team played at this venue or city | Familiarity |

---

## Category 10: Referee Features (10 features)

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 266 | `referee_fouls_per_match` | Historical average | Strictness |
| 267 | `referee_yellows_per_match` | Historical average | Card tendency |
| 268 | `referee_reds_per_match` | Historical average | Sending off tendency |
| 269 | `referee_penalties_per_match` | Historical average | Penalty tendency |
| 270 | `referee_home_win_rate` | Historical home win rate under referee | Home bias |
| 271 | `referee_avg_goals_per_match` | Mean total goals | Flow preference |
| 272 | `referee_confederation` | Same confederation as either team? | Familiarity |
| 273-275 | `referee_interaction_*` | Cross features with team play style | Style-referee interactions |

---

## Category 11: Sentiment & NLP Features (15 features)

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 276 | `sentiment_twitter_avg` | Mean sentiment score from Twitter/X | Public mood |
| 277 | `sentiment_reddit_avg` | Mean sentiment from r/soccer | Fan confidence |
| 278 | `sentiment_news_avg` | Mean sentiment from news articles | Media narrative |
| 279 | `sentiment_combined` | Weighted average of all sources | Overall sentiment |
| 280 | `sentiment_trend_3d` | Sentiment change over 3 days | Momentum of opinion |
| 281 | `injury_rumor_count` | Count of injury-related mentions | Hidden injury signals |
| 282 | `controversy_score` | Count of controversy mentions | Team distraction |
| 283 | `confidence_score` | NLP-extracted confidence level | Team morale |
| 284 | `manager_pressure_score` | Negative mentions of manager | Management stability |
| 285-290 | `topic_*` | Topic-specific sentiment (tactics, lineup, fitness) | Granular sentiment |

---

## Category 12: Interaction & Matchup Features (30 features)

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 291 | `xg_diff_match` | team_xg_per90 - opponent_xga_per90 | Attack vs defense matchup |
| 292 | `possession_diff` | team_possession - opponent_possession | Ball control matchup |
| 293 | `ppda_ratio` | team_ppda / opponent_ppda | Pressing matchup |
| 294 | `value_ratio` | team_squad_value / opponent_squad_value | Quality ratio |
| 295 | `elo_ratio` | team_elo / opponent_elo | Rating ratio |
| 296 | `age_diff` | team_avg_age - opponent_avg_age | Experience vs energy |
| 297 | `style_matchup_score` | Cosine similarity of tactical feature vectors | Tactical similarity |
| 298 | `counter_vs_possession` | team_counter_goals × opponent_possession | Counter exploitation |
| 299 | `set_piece_vs_aerial` | team_set_piece_xg × (1 - opponent_aerials_pct) | Set piece vulnerability |
| 300 | `press_resistance_matchup` | team_pass_accuracy_under_press / opponent_pressing_success | Press resistance |
| 301-320 | `diff_*` | Pairwise differences for all key stats between team and opponent | All statistical differentials |

---

## Leakage Prevention Rules

1. **Temporal Cutoff**: All rolling features computed using `match_date < current_match_date` (strict less-than)
2. **No Target Encoding on Full Data**: Target encoding (if used) fitted only on training fold
3. **No Future Odds**: Only use opening/pre-match odds, never in-play or closing odds available after match start
4. **No In-Tournament Stats for Tournament Predictions**: When predicting a tournament match, features use only pre-tournament + completed tournament matches
5. **Walk-Forward CV**: Each fold's features computed only from data available at that point in time
6. **Feature Snapshot**: At prediction time, freeze all features to avoid temporal contamination

## Normalization Strategy

- **Ratings** (Elo, Glicko): No normalization (inherently calibrated)
- **Per-90 stats**: Already normalized by playing time
- **Raw counts**: StandardScaler fitted on training set only
- **Percentages**: MinMaxScaler to [0, 1]
- **Market values**: Log-transform then StandardScaler (heavy right skew)
- **Categorical**: Label encoding for ordinals, one-hot for nominals
