# World Cup AI — Monte Carlo Simulation Engine Design

## 1. Overview

The Monte Carlo engine simulates the entire FIFA World Cup tournament 100,000+ times to produce probabilistic outcomes for every team.

### Target Outputs
- Champion probability per team
- Final appearance probability
- Semifinal probability
- Quarterfinal probability
- Group exit probability
- Expected goals scored/conceded per team
- Golden Boot probability per player
- Most likely bracket paths

---

## 2. Tournament Structure (2026 World Cup)

```
48 teams → 12 groups of 4
    │
    ▼ (Top 2 + 8 best 3rd-place advance = 32 teams)
Round of 32
    │
    ▼
Round of 16
    │
    ▼
Quarter-finals
    │
    ▼
Semi-finals
    │
    ▼
Third-place match + Final
```

### Configuration (loaded from `configs/tournament_2026.yaml`)
```yaml
tournament:
  name: "FIFA World Cup 2026"
  num_groups: 12
  teams_per_group: 4
  group_matches: 3  # per team
  advance_top: 2  # per group
  advance_best_third: 8
  knockout_rounds: [R32, R16, QF, SF, F]
  extra_time: true
  penalties: true
  third_place_match: true

hosts: ["USA", "Canada", "Mexico"]
host_advantage_elo_bonus: 100
```

---

## 3. Match Simulation Model

### 3.1 Score Generation (Poisson-based)

For each simulated match between Team A and Team B:

```python
def simulate_match(team_a, team_b, neutral=False):
    # Get expected goals from Poisson model
    lambda_a = poisson_model.predict_lambda(team_a, team_b, is_home=not neutral)
    lambda_b = poisson_model.predict_lambda(team_b, team_a, is_home=False)
    
    # Apply fatigue adjustment
    lambda_a *= fatigue_multiplier(team_a)
    lambda_b *= fatigue_multiplier(team_b)
    
    # Apply injury adjustment
    lambda_a *= injury_multiplier(team_a)
    lambda_b *= injury_multiplier(team_b)
    
    # Sample goals from Poisson
    goals_a = np.random.poisson(lambda_a)
    goals_b = np.random.poisson(lambda_b)
    
    return goals_a, goals_b
```

### 3.2 Fatigue Accumulation Model

```python
def fatigue_multiplier(team):
    """
    Fatigue reduces expected goals by up to 15%.
    
    fatigue_score = Σ (minutes_played_in_match_i × recency_weight_i)
    recency_weight = exp(-0.15 × days_since_match)
    
    multiplier = max(0.85, 1.0 - 0.02 × normalized_fatigue)
    """
    fatigue = sum(
        match.minutes * math.exp(-0.15 * days_since(match))
        for match in team.tournament_matches
    )
    normalized = fatigue / 90.0  # normalize by one full match
    return max(0.85, 1.0 - 0.02 * normalized)
```

### 3.3 Injury During Tournament

```python
def check_injuries(team, match_num):
    """
    Each player has a per-match injury probability.
    Base rate: 2% per match, modified by:
      - Age factor: +1% per year over 30
      - Fatigue: +0.5% per tournament match played
      - Injury history: +2% if injury-prone
    """
    for player in team.squad:
        p_injury = 0.02
        p_injury += max(0, player.age - 30) * 0.01
        p_injury += match_num * 0.005
        p_injury += 0.02 if player.is_injury_prone else 0
        
        if random.random() < p_injury:
            player.injured = True
            player.return_match = match_num + random.randint(1, 3)
```

### 3.4 Extra Time & Penalties (Knockout Stages)

```python
def simulate_knockout_match(team_a, team_b):
    goals_a, goals_b = simulate_match(team_a, team_b, neutral=True)
    
    if goals_a == goals_b:
        # Extra time: reduced goal expectation (30 min vs 90 min)
        et_factor = 0.33  # 30/90 minutes
        fatigue_penalty = 0.85  # players are tired
        
        lambda_a_et = lambda_a * et_factor * fatigue_penalty
        lambda_b_et = lambda_b * et_factor * fatigue_penalty
        
        et_goals_a = np.random.poisson(lambda_a_et)
        et_goals_b = np.random.poisson(lambda_b_et)
        goals_a += et_goals_a
        goals_b += et_goals_b
        
        if goals_a == goals_b:
            # Penalty shootout
            winner = simulate_penalties(team_a, team_b)
            return goals_a, goals_b, winner
    
    winner = team_a if goals_a > goals_b else team_b
    return goals_a, goals_b, winner


def simulate_penalties(team_a, team_b):
    """
    5-round penalty simulation with sudden death.
    Base conversion rate: 76% (historical average).
    Adjusted by: GK save rate, team penalty history, pressure.
    """
    score_a, score_b = 0, 0
    
    for round_num in range(5):
        # Team A shoots
        p_score_a = 0.76 * team_a.penalty_ability / team_b.gk_save_factor
        if random.random() < p_score_a:
            score_a += 1
        
        # Check if already decided
        remaining = 4 - round_num
        if score_a - score_b > remaining:
            return team_a
        if score_b - score_a > remaining:
            return team_b
        
        # Team B shoots
        p_score_b = 0.76 * team_b.penalty_ability / team_a.gk_save_factor
        if random.random() < p_score_b:
            score_b += 1
        
        if score_a - score_b > remaining - 1:
            return team_a
        if score_b - score_a > remaining - 1:
            return team_b
    
    # Sudden death
    while score_a == score_b:
        if random.random() < 0.76:
            score_a += 1
        if random.random() < 0.76:
            score_b += 1
        if score_a != score_b:
            return team_a if score_a > score_b else team_b
        score_a, score_b = 0, 0  # Reset for next sudden death pair
```

---

## 4. Group Stage Simulation

```python
def simulate_group(group_teams, n_matches=3):
    """
    Round-robin within group. Each team plays n_matches.
    Standings determined by: Points > GD > GF > H2H > Fair play > Drawing of lots
    """
    standings = {team: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for team in group_teams}
    
    for team_a, team_b in itertools.combinations(group_teams, 2):
        goals_a, goals_b = simulate_match(team_a, team_b, neutral=True)
        
        standings[team_a]["gf"] += goals_a
        standings[team_a]["ga"] += goals_b
        standings[team_a]["gd"] += goals_a - goals_b
        standings[team_b]["gf"] += goals_b
        standings[team_b]["ga"] += goals_a
        standings[team_b]["gd"] += goals_b - goals_a
        
        if goals_a > goals_b:
            standings[team_a]["pts"] += 3
        elif goals_a == goals_b:
            standings[team_a]["pts"] += 1
            standings[team_b]["pts"] += 1
        else:
            standings[team_b]["pts"] += 3
    
    # Sort: points, then GD, then GF
    sorted_teams = sorted(
        group_teams,
        key=lambda t: (standings[t]["pts"], standings[t]["gd"], standings[t]["gf"]),
        reverse=True
    )
    
    return sorted_teams, standings
```

---

## 5. Vectorized Simulation Architecture

### Key Insight
Instead of simulating one tournament at a time (slow), simulate ALL 100K tournaments in parallel using NumPy vectorization.

```python
def vectorized_simulate_all(n_simulations=100_000):
    """
    Vectorized: simulate all matches across all simulations simultaneously.
    
    For a group of 4 teams, there are 6 matches.
    For 12 groups: 72 group-stage matches.
    
    Shape: (n_simulations, n_matches, 2)  → goals for each team in each match
    """
    n_group_matches = 72  # 12 groups × 6 matches per group
    
    # Pre-compute all lambdas: (n_group_matches, 2)
    lambdas = np.array([
        [poisson_lambda(match.team_a, match.team_b),
         poisson_lambda(match.team_b, match.team_a)]
        for match in all_group_matches
    ])
    
    # Broadcast to (n_simulations, n_group_matches, 2)
    lambdas_broadcast = np.broadcast_to(lambdas, (n_simulations, n_group_matches, 2))
    
    # Sample all goals at once: (n_simulations, n_group_matches, 2)
    all_goals = np.random.poisson(lambdas_broadcast)
    
    # Compute standings vectorized
    # ... (vectorized point/GD/GF accumulation)
    
    return all_goals
```

### Performance Estimates
| Approach | Time for 100K sims | Memory |
|----------|-------------------|--------|
| Sequential Python | ~45 minutes | 500 MB |
| Vectorized NumPy | ~30 seconds | 4 GB |
| Vectorized + Multiprocessing (8 cores) | ~5 seconds | 8 GB |

### Multiprocessing Strategy
```python
def run_simulations(n_total=100_000, n_workers=8):
    chunk_size = n_total // n_workers
    
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(vectorized_simulate_all, chunk_size)
            for _ in range(n_workers)
        ]
        
        results = [f.result() for f in futures]
    
    return aggregate_results(results)
```

---

## 6. Golden Boot Simulation

```python
def track_golden_boot(simulation_results):
    """
    Track individual player goals across the tournament.
    
    For each simulated match, distribute goals to players based on:
    - Player's share of team xG (pre-tournament)
    - Position weighting (FWD > MID > DEF)
    - Minutes played probability
    """
    player_goals = defaultdict(list)  # player → [goals across sims]
    
    for sim in simulations:
        sim_goals = defaultdict(int)
        for match in sim.matches:
            team_goals = match.goals  # total goals for team in this match
            
            for player in match.team.likely_squad:
                # Probability of scoring = player_xg_share × team_goals
                p_goal = player.xg_share * player.minutes_share
                player_match_goals = np.random.binomial(team_goals, p_goal)
                sim_goals[player.id] += player_match_goals
        
        # Find top scorer
        top_scorer = max(sim_goals, key=sim_goals.get)
        player_goals[top_scorer].append(sim_goals[top_scorer])
    
    # Compute Golden Boot probabilities
    golden_boot_probs = {
        player: count / n_simulations
        for player, count in Counter(top_scorers).items()
    }
    return golden_boot_probs
```

---

## 7. Output Schema

```python
@dataclass
class SimulationOutput:
    """Complete output from Monte Carlo simulation."""
    
    # Per-team probabilities
    champion_prob: Dict[str, float]       # {"Brazil": 0.142, "France": 0.128, ...}
    final_prob: Dict[str, float]
    semifinal_prob: Dict[str, float]
    quarterfinal_prob: Dict[str, float]
    r16_prob: Dict[str, float]
    group_exit_prob: Dict[str, float]
    
    # Per-team expected stats
    expected_goals: Dict[str, float]       # Expected total goals in tournament
    expected_conceded: Dict[str, float]
    expected_group_points: Dict[str, float]
    
    # Golden Boot
    golden_boot_prob: Dict[str, float]     # {"Mbappé": 0.08, "Haaland": 0.07, ...}
    expected_top_scorer_goals: float        # ~6.2 goals
    
    # Most likely outcomes
    most_likely_champion: str
    most_likely_final: Tuple[str, str]
    most_likely_group_winners: Dict[str, str]
    
    # Metadata
    n_simulations: int
    simulation_time_seconds: float
    model_version: str
    timestamp: datetime
```

---

## 8. Validation

### Historical Backtesting
- Run simulator on past World Cups (2010, 2014, 2018, 2022)
- Compare predicted champion probabilities vs actual outcomes
- Evaluate calibration: teams given X% chance should win X% of the time

### Convergence Check
- Run with increasing N: 1K, 10K, 50K, 100K, 500K
- Check that probabilities stabilize (std < 0.001 between 100K and 500K)
- Typical convergence: champion probs stable at ~50K sims

### Sensitivity Analysis
- Vary fatigue parameters ±20%: measure impact on top-4 probabilities
- Vary injury rates ±50%: measure impact
- Vary home advantage ±30%: measure impact
- If small parameter changes cause large outcome changes → flag instability
