# World Cup AI — File-by-File Implementation Plan & Execution Order

## Complete Project Tree

```
world_cup_ai/
│
├── configs/
│   ├── __init__.py
│   ├── settings.py                    # Global settings (Pydantic BaseSettings)
│   ├── train_config.yaml              # Training hyperparameters
│   ├── tournament_2026.yaml           # Tournament structure
│   ├── data_sources.yaml              # API keys, URLs, rate limits
│   └── logging_config.yaml            # Logging configuration
│
├── data/
│   ├── raw/                           # Raw ingested data (JSON, CSV)
│   ├── processed/                     # Cleaned data
│   ├── features/                      # Feature matrices (Parquet)
│   ├── models/                        # Saved model artifacts
│   ├── odds/                          # Historical odds data
│   ├── injuries/                      # Injury data snapshots
│   └── simulations/                   # Simulation outputs
│
├── src/
│   ├── __init__.py
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── engine.py                  # SQLAlchemy engine & session factory
│   │   ├── models.py                  # All SQLAlchemy ORM models
│   │   └── migrations/               # Alembic migrations
│   │       ├── env.py
│   │       ├── alembic.ini
│   │       └── versions/
│   │           └── 001_initial.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── base_ingestor.py           # Abstract base class for all ingestors
│   │   ├── statsbomb_ingestor.py      # StatsBomb open data loader
│   │   ├── fbref_scraper.py           # FBref HTML scraper
│   │   ├── understat_scraper.py       # Understat JSON-from-HTML parser
│   │   ├── transfermarkt_scraper.py   # Transfermarkt scraper (Selenium)
│   │   ├── football_data_api.py       # Football-data.org REST client
│   │   ├── odds_ingestor.py           # Odds API client
│   │   ├── news_ingestor.py           # News/social media scraper
│   │   └── orchestrator.py            # Run all ingestors in order
│   │
│   ├── cleaning/
│   │   ├── __init__.py
│   │   ├── base_cleaner.py            # Abstract cleaner interface
│   │   ├── match_cleaner.py           # Match data cleaning & dedup
│   │   ├── player_cleaner.py          # Player data cleaning & entity resolution
│   │   ├── team_cleaner.py            # Team name normalization
│   │   ├── event_cleaner.py           # Event data validation
│   │   ├── odds_cleaner.py            # Odds data normalization
│   │   └── pipeline.py                # Cleaning DAG orchestration
│   │
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── base_feature.py            # Feature computation interface
│   │   ├── registry.py                # Feature registry & dependency resolution
│   │   ├── team_form.py               # Rolling form features (40 features)
│   │   ├── ratings.py                 # Elo, Glicko, attack/defense ratings (20)
│   │   ├── expected_goals.py          # xG features (30)
│   │   ├── tactical.py                # Possession, PPDA, pressing features (35)
│   │   ├── player_quality.py          # Squad value, depth, star power (45)
│   │   ├── fatigue.py                 # Rest, travel, congestion features (25)
│   │   ├── tournament_pressure.py     # Stage, history, pressure features (25)
│   │   ├── odds_features.py           # Market probabilities, value features (30)
│   │   ├── weather_venue.py           # Weather, altitude, surface features (15)
│   │   ├── referee_features.py        # Referee tendency features (10)
│   │   ├── sentiment.py               # NLP sentiment features (15)
│   │   ├── interaction.py             # Matchup & differential features (30)
│   │   ├── builder.py                 # Build full feature matrix
│   │   └── store.py                   # Feature store (read/write Parquet)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py              # Abstract BasePredictor interface
│   │   ├── xgboost_model.py           # XGBoost classifier
│   │   ├── lightgbm_model.py          # LightGBM classifier
│   │   ├── catboost_model.py          # CatBoost classifier
│   │   ├── neural_net.py              # PyTorch neural network
│   │   ├── poisson_model.py           # Dixon-Coles Poisson model
│   │   ├── bayesian_model.py          # Bayesian hierarchical model
│   │   └── model_registry.py          # Model registration & loading
│   │
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── weighted_average.py        # Optimized weighted averaging
│   │   ├── stacking.py                # Stacking meta-learner
│   │   ├── calibration.py             # Platt, isotonic, temperature scaling
│   │   ├── uncertainty.py             # Uncertainty estimation
│   │   └── ensemble_model.py          # Full ensemble pipeline
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── match_simulator.py         # Single match Poisson simulation
│   │   ├── group_stage.py             # Group stage simulation logic
│   │   ├── knockout_stage.py          # Knockout + ET + penalties
│   │   ├── tournament_simulator.py    # Full tournament orchestrator
│   │   ├── fatigue_model.py           # Fatigue accumulation
│   │   ├── injury_model.py            # In-tournament injury model
│   │   ├── golden_boot.py             # Golden Boot tracking
│   │   └── vectorized_engine.py       # NumPy vectorized simulation
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Log loss, Brier, ECE, RPS
│   │   ├── calibration.py             # Reliability diagrams
│   │   ├── roi_analysis.py            # Betting ROI simulation
│   │   ├── shap_analysis.py           # SHAP explanations
│   │   ├── drift_detector.py          # Feature drift detection
│   │   ├── report_generator.py        # Generate evaluation reports
│   │   └── walk_forward.py            # Walk-forward CV implementation
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                     # FastAPI application factory
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── predictions.py         # /predict_match endpoint
│   │   │   ├── simulations.py         # /simulate_tournament endpoint
│   │   │   ├── golden_boot.py         # /golden_boot endpoint
│   │   │   ├── team_strength.py       # /team_strength endpoint
│   │   │   └── live.py                # /live_predictions endpoint
│   │   ├── schemas.py                 # Pydantic request/response models
│   │   ├── dependencies.py            # Dependency injection
│   │   └── middleware.py              # Auth, logging, CORS middleware
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py                     # Streamlit main app
│   │   ├── pages/
│   │   │   ├── match_predictions.py   # Match prediction page
│   │   │   ├── team_rankings.py       # Team power rankings page
│   │   │   ├── player_analytics.py    # Player stats page
│   │   │   ├── tournament_sim.py      # Tournament simulation page
│   │   │   ├── odds_comparison.py     # Odds analysis page
│   │   │   └── model_explainer.py     # SHAP explanations page
│   │   └── components/
│   │       ├── charts.py              # Reusable chart components
│   │       ├── tables.py              # Reusable table components
│   │       └── sidebar.py             # Navigation sidebar
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Main training orchestrator
│   │   ├── splitter.py                # Temporal data splitting
│   │   ├── optimizer.py               # Optuna HPO wrapper
│   │   ├── callbacks.py               # Early stopping, checkpointing
│   │   └── experiment.py              # MLflow experiment management
│   │
│   ├── live/
│   │   ├── __init__.py
│   │   ├── predictor.py               # Live prediction engine
│   │   ├── data_refresher.py          # Real-time data fetching
│   │   ├── cache_manager.py           # Redis cache management
│   │   └── celery_tasks.py            # Celery task definitions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                  # Structured logging setup
│       ├── config.py                  # Configuration loader
│       ├── validators.py              # Pydantic validators & schemas
│       ├── decorators.py              # Retry, timing, caching decorators
│       ├── constants.py               # Team codes, confederations, mappings
│       ├── geo.py                     # Haversine distance, timezone calculations
│       └── async_utils.py             # Async HTTP helpers, rate limiting
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_feature_analysis.ipynb      # Feature importance analysis
│   ├── 03_model_comparison.ipynb      # Model performance comparison
│   └── 04_simulation_analysis.ipynb   # Simulation results analysis
│
├── docker/
│   ├── Dockerfile.api                 # FastAPI container
│   ├── Dockerfile.dashboard           # Streamlit container
│   ├── Dockerfile.worker              # Celery worker container
│   ├── Dockerfile.training            # GPU training container
│   └── nginx.conf                     # Nginx reverse proxy config
│
├── scripts/
│   ├── setup_linux.sh                 # Linux server setup
│   ├── setup_gpu.sh                   # CUDA/GPU setup
│   ├── start_services.sh             # Start all services
│   ├── stop_services.sh              # Stop all services
│   ├── run_ingestion.sh              # Manual ingestion trigger
│   ├── run_training.sh               # Training wrapper
│   ├── backup_db.sh                  # Database backup
│   └── healthcheck.sh                # Service health checks
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_ingestion/
│   ├── test_cleaning/
│   ├── test_features/
│   ├── test_models/
│   ├── test_ensemble/
│   ├── test_simulation/
│   ├── test_evaluation/
│   └── test_api/
│
├── logs/                              # Application logs
├── mlruns/                            # MLflow tracking data
│
├── train.py                           # Main training entry point
├── predict.py                         # CLI prediction tool
├── simulate_world_cup.py             # Tournament simulation CLI
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project metadata & tool config
├── docker-compose.yml                 # Full stack compose
├── .env.example                       # Environment variables template
├── Makefile                           # Common commands
└── README.md                          # Project documentation
```

---

## File Details

### configs/settings.py
- **Responsibility**: Global configuration via Pydantic BaseSettings (env vars + .env file)
- **Classes**: `Settings`, `DatabaseSettings`, `RedisSettings`, `MLflowSettings`
- **Dependencies**: pydantic-settings
- **Execution**: Imported by all modules at startup

### src/db/engine.py
- **Responsibility**: SQLAlchemy async engine, session factory, connection pooling
- **Functions**: `get_engine()`, `get_session()`, `init_db()`, `dispose_engine()`
- **Dependencies**: sqlalchemy[asyncio], asyncpg

### src/db/models.py
- **Responsibility**: All 15 ORM models matching database schema
- **Classes**: `Team`, `Player`, `Tournament`, `Stadium`, `Referee`, `Manager`, `Match`, `MatchEvent`, `MatchLineup`, `MatchWeather`, `TeamStats`, `PlayerStats`, `TeamRating`, `MatchOdds`, `PlayerInjury`, `Prediction`, `Simulation`, `SimulationResult`, `NewsSentiment`
- **Dependencies**: sqlalchemy, src/db/engine.py

### src/ingestion/base_ingestor.py
- **Responsibility**: Abstract base class defining ingestor interface
- **Classes**: `BaseIngestor` (abstract) with methods: `fetch()`, `validate()`, `store()`, `run()`
- **Features**: Rate limiting, retry logic, logging, progress tracking

### src/ingestion/statsbomb_ingestor.py
- **Responsibility**: Download and parse StatsBomb open data (matches, events, lineups)
- **Classes**: `StatsBombIngestor`
- **Functions**: `fetch_competitions()`, `fetch_matches()`, `fetch_events()`, `fetch_lineups()`
- **Dependencies**: requests, src/db/models.py

### src/ingestion/fbref_scraper.py
- **Responsibility**: Scrape FBref for team/player stats tables
- **Classes**: `FBrefScraper`
- **Functions**: `scrape_team_stats()`, `scrape_player_stats()`, `parse_stats_table()`
- **Dependencies**: beautifulsoup4, requests, src/utils/async_utils.py

### src/ingestion/understat_scraper.py
- **Responsibility**: Parse Understat for xG/xA data
- **Classes**: `UnderstatScraper`
- **Functions**: `fetch_player_data()`, `fetch_shot_data()`, `parse_js_data()`
- **Dependencies**: beautifulsoup4, aiohttp

### src/ingestion/transfermarkt_scraper.py
- **Responsibility**: Scrape market values, injuries, transfers
- **Classes**: `TransfermarktScraper`
- **Functions**: `scrape_squad_values()`, `scrape_injuries()`, `scrape_transfers()`
- **Dependencies**: selenium/playwright, beautifulsoup4

### src/ingestion/odds_ingestor.py
- **Responsibility**: Fetch odds from The Odds API
- **Classes**: `OddsIngestor`
- **Functions**: `fetch_match_odds()`, `compute_implied_probs()`
- **Dependencies**: aiohttp

### src/cleaning/pipeline.py
- **Responsibility**: Orchestrate all cleaning stages in dependency order
- **Classes**: `CleaningPipeline`
- **Functions**: `run()`, `validate_output()`, `log_stats()`
- **Execution order**: teams → players → matches → events → stats → odds

### src/feature_engineering/registry.py
- **Responsibility**: Register feature functions, resolve dependencies, compute in order
- **Classes**: `FeatureRegistry`
- **Pattern**: Decorator-based registration: `@registry.register("team_form")`

### src/feature_engineering/builder.py
- **Responsibility**: Build complete feature matrix for all matches
- **Classes**: `FeatureBuilder`
- **Functions**: `build_features()`, `build_match_features()`, `get_feature_names()`
- **Key logic**: Iterate matches chronologically; for each match compute all 300+ features using only prior data

### src/models/base_model.py
- **Responsibility**: Define interface all models must implement
- **Abstract methods**: `train()`, `predict_proba()`, `save()`, `load()`, `get_params()`
- **Concrete methods**: `evaluate()`, `feature_importance()`

### src/ensemble/ensemble_model.py
- **Responsibility**: Full ensemble pipeline: base models → stacking → calibration
- **Classes**: `EnsembleModel`
- **Functions**: `fit()`, `predict()`, `predict_calibrated()`, `get_model_contributions()`

### src/simulation/tournament_simulator.py
- **Responsibility**: Orchestrate full tournament MC simulation
- **Classes**: `TournamentSimulator`
- **Functions**: `simulate()`, `run_group_stage()`, `run_knockout()`, `aggregate_results()`

### src/simulation/vectorized_engine.py
- **Responsibility**: NumPy-vectorized batch simulation for performance
- **Functions**: `vectorized_simulate_groups()`, `vectorized_simulate_knockout()`
- **Key**: All 100K simulations run in parallel via array operations

### src/api/app.py
- **Responsibility**: FastAPI application factory with all middleware
- **Functions**: `create_app()`, register routes, configure CORS/auth/logging

### src/dashboard/app.py
- **Responsibility**: Streamlit multi-page dashboard entry point
- **Pages**: Match Predictions, Team Rankings, Player Analytics, Tournament Sim, Odds, SHAP

### train.py
- **Responsibility**: Single CLI entry point for full training pipeline
- **Flow**: Config → Ingestion → Cleaning → Features → Train Models → Ensemble → Evaluate → Register

### predict.py
- **Responsibility**: CLI tool for ad-hoc match predictions
- **Usage**: `python predict.py --home Brazil --away Germany --date 2026-06-15`

### simulate_world_cup.py
- **Responsibility**: CLI tool for tournament simulation
- **Usage**: `python simulate_world_cup.py --n-sims 100000 --output data/simulations/`

---

## Execution Order

### Phase 1: Foundation (Day 1)
```
1. configs/settings.py
2. src/utils/logger.py
3. src/utils/config.py
4. src/utils/constants.py
5. src/utils/decorators.py
6. src/utils/validators.py
7. src/utils/geo.py
8. src/utils/async_utils.py
9. src/db/engine.py
10. src/db/models.py
11. src/db/migrations/
12. requirements.txt
13. pyproject.toml
14. .env.example
```

### Phase 2: Data Pipeline (Day 2-3)
```
15. src/ingestion/base_ingestor.py
16. src/ingestion/statsbomb_ingestor.py
17. src/ingestion/fbref_scraper.py
18. src/ingestion/understat_scraper.py
19. src/ingestion/transfermarkt_scraper.py
20. src/ingestion/football_data_api.py
21. src/ingestion/odds_ingestor.py
22. src/ingestion/news_ingestor.py
23. src/ingestion/orchestrator.py
24. src/cleaning/base_cleaner.py
25. src/cleaning/team_cleaner.py
26. src/cleaning/player_cleaner.py
27. src/cleaning/match_cleaner.py
28. src/cleaning/event_cleaner.py
29. src/cleaning/odds_cleaner.py
30. src/cleaning/pipeline.py
```

### Phase 3: Feature Engineering (Day 4-5)
```
31. src/feature_engineering/base_feature.py
32. src/feature_engineering/registry.py
33. src/feature_engineering/team_form.py
34. src/feature_engineering/ratings.py
35. src/feature_engineering/expected_goals.py
36. src/feature_engineering/tactical.py
37. src/feature_engineering/player_quality.py
38. src/feature_engineering/fatigue.py
39. src/feature_engineering/tournament_pressure.py
40. src/feature_engineering/odds_features.py
41. src/feature_engineering/weather_venue.py
42. src/feature_engineering/referee_features.py
43. src/feature_engineering/sentiment.py
44. src/feature_engineering/interaction.py
45. src/feature_engineering/builder.py
46. src/feature_engineering/store.py
```

### Phase 4: Models (Day 6-8)
```
47. src/models/base_model.py
48. src/models/xgboost_model.py
49. src/models/lightgbm_model.py
50. src/models/catboost_model.py
51. src/models/neural_net.py
52. src/models/poisson_model.py
53. src/models/bayesian_model.py
54. src/models/model_registry.py
55. src/ensemble/weighted_average.py
56. src/ensemble/stacking.py
57. src/ensemble/calibration.py
58. src/ensemble/uncertainty.py
59. src/ensemble/ensemble_model.py
```

### Phase 5: Training & Evaluation (Day 9-10)
```
60. src/training/splitter.py
61. src/training/callbacks.py
62. src/training/optimizer.py
63. src/training/experiment.py
64. src/training/trainer.py
65. src/evaluation/metrics.py
66. src/evaluation/calibration.py
67. src/evaluation/walk_forward.py
68. src/evaluation/roi_analysis.py
69. src/evaluation/shap_analysis.py
70. src/evaluation/drift_detector.py
71. src/evaluation/report_generator.py
72. train.py
```

### Phase 6: Simulation (Day 11)
```
73. src/simulation/match_simulator.py
74. src/simulation/fatigue_model.py
75. src/simulation/injury_model.py
76. src/simulation/group_stage.py
77. src/simulation/knockout_stage.py
78. src/simulation/golden_boot.py
79. src/simulation/vectorized_engine.py
80. src/simulation/tournament_simulator.py
81. simulate_world_cup.py
```

### Phase 7: Serving (Day 12-13)
```
82. src/api/schemas.py
83. src/api/dependencies.py
84. src/api/middleware.py
85. src/api/routes/predictions.py
86. src/api/routes/simulations.py
87. src/api/routes/golden_boot.py
88. src/api/routes/team_strength.py
89. src/api/routes/live.py
90. src/api/app.py
91. predict.py
92. src/live/cache_manager.py
93. src/live/data_refresher.py
94. src/live/predictor.py
95. src/live/celery_tasks.py
96. src/dashboard/components/charts.py
97. src/dashboard/components/tables.py
98. src/dashboard/components/sidebar.py
99. src/dashboard/pages/match_predictions.py
100. src/dashboard/pages/team_rankings.py
101. src/dashboard/pages/player_analytics.py
102. src/dashboard/pages/tournament_sim.py
103. src/dashboard/pages/odds_comparison.py
104. src/dashboard/pages/model_explainer.py
105. src/dashboard/app.py
```

### Phase 8: Infrastructure (Day 14)
```
106. docker/Dockerfile.api
107. docker/Dockerfile.dashboard
108. docker/Dockerfile.worker
109. docker/Dockerfile.training
110. docker/nginx.conf
111. docker-compose.yml
112. scripts/setup_linux.sh
113. scripts/setup_gpu.sh
114. scripts/start_services.sh
115. scripts/stop_services.sh
116. scripts/run_ingestion.sh
117. scripts/run_training.sh
118. scripts/backup_db.sh
119. scripts/healthcheck.sh
120. Makefile
121. README.md
122. configs/train_config.yaml
123. configs/tournament_2026.yaml
124. configs/data_sources.yaml
125. configs/logging_config.yaml
```

### Phase 9: Tests (Day 15)
```
126. tests/conftest.py
127-140. tests/test_*/test_*.py (one per module)
```

**Total: ~140 files**
