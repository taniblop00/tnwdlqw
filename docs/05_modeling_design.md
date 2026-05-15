# World Cup AI — Modeling Design

## 1. Model Overview

| Model | Task | Output | GPU | Framework |
|-------|------|--------|-----|-----------|
| XGBoost | Match outcome classification | P(H), P(D), P(A) | ✓ gpu_hist | xgboost |
| LightGBM | Match outcome classification | P(H), P(D), P(A) | ✓ gpu | lightgbm |
| CatBoost | Match outcome classification | P(H), P(D), P(A) | ✓ GPU | catboost |
| PyTorch NN | Match outcome classification | P(H), P(D), P(A) | ✓ CUDA AMP | torch |
| Poisson | Score prediction | P(score=i,j) for all i,j | ✗ | scipy/numpy |
| Bayesian | Probabilistic strength estimation | Posterior distributions | ✗ | pymc/numpy |

---

## 2. Model 1: XGBoost

### Architecture
- **Objective**: `multi:softprob` (3-class: home/draw/away)
- **Booster**: `gbtree`
- **Tree Method**: `gpu_hist` (GPU) or `hist` (CPU fallback)

### Input Features
- All 300+ features from feature catalog
- Categorical features label-encoded (XGBoost handles natively in recent versions)
- Missing values: XGBoost handles natively (learns optimal split direction)

### Hyperparameter Search Space (Optuna)
```python
{
    "n_estimators": IntUniform(200, 3000),
    "max_depth": IntUniform(3, 12),
    "learning_rate": LogUniform(0.005, 0.3),
    "min_child_weight": IntUniform(1, 20),
    "subsample": Uniform(0.5, 1.0),
    "colsample_bytree": Uniform(0.3, 1.0),
    "colsample_bylevel": Uniform(0.3, 1.0),
    "gamma": LogUniform(1e-8, 10.0),
    "reg_alpha": LogUniform(1e-8, 100.0),
    "reg_lambda": LogUniform(1e-8, 100.0),
    "scale_pos_weight": Uniform(0.5, 3.0),
}
```

### Loss Function
- Primary: Multi-class log loss (cross-entropy)
- Early stopping on validation log loss (patience=50 rounds)

### Validation
- Walk-forward temporal CV: 5 folds
- Each fold: train on all data before cutoff, validate on next 6-month window
- Final model trained on all data except held-out calibration set

### GPU Optimization
```python
xgb.XGBClassifier(
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    gpu_id=0,
    n_jobs=1,  # GPU handles parallelism
)
```

---

## 3. Model 2: LightGBM

### Architecture
- **Objective**: `multiclass` with `num_class=3`
- **Boosting**: `gbdt` (also try `dart` in HPO)
- **Device**: `gpu` with `gpu_use_dp=True`

### Hyperparameter Search Space
```python
{
    "n_estimators": IntUniform(200, 3000),
    "num_leaves": IntUniform(15, 255),
    "max_depth": IntUniform(-1, 15),  # -1 = no limit
    "learning_rate": LogUniform(0.005, 0.3),
    "min_child_samples": IntUniform(5, 100),
    "subsample": Uniform(0.5, 1.0),
    "colsample_bytree": Uniform(0.3, 1.0),
    "reg_alpha": LogUniform(1e-8, 100.0),
    "reg_lambda": LogUniform(1e-8, 100.0),
    "min_split_gain": LogUniform(1e-8, 1.0),
    "path_smooth": Uniform(0, 10),
}
```

### Key Differences from XGBoost
- Leaf-wise growth (vs level-wise) — often faster convergence
- Native categorical feature support via `categorical_feature` parameter
- GOSS (Gradient-based One-Side Sampling) for faster training on large datasets

### Loss Function
- Multi-class log loss with early stopping (patience=50)

---

## 4. Model 3: CatBoost

### Architecture
- **Objective**: `MultiClass`
- **Task Type**: `GPU`
- **Boosting**: Ordered boosting (CatBoost's default — reduces overfitting)

### Hyperparameter Search Space
```python
{
    "iterations": IntUniform(200, 3000),
    "depth": IntUniform(3, 10),
    "learning_rate": LogUniform(0.005, 0.3),
    "l2_leaf_reg": LogUniform(1e-3, 100.0),
    "bagging_temperature": Uniform(0, 10),
    "random_strength": LogUniform(1e-3, 10.0),
    "border_count": IntUniform(32, 255),
    "grow_policy": Categorical(["SymmetricTree", "Depthwise", "Lossguide"]),
    "min_data_in_leaf": IntUniform(1, 50),
}
```

### Key Advantages
- Best native categorical handling (target statistics with ordered boosting)
- Most robust to overfitting among GBDT models
- Built-in feature importance and SHAP

### GPU Config
```python
catboost.CatBoostClassifier(
    task_type="GPU",
    devices="0",
    bootstrap_type="Poisson",  # required for GPU
)
```

---

## 5. Model 4: PyTorch Neural Network

### Architecture
```
Input (N features)
    │
    ▼
BatchNorm1d(N)
    │
    ▼
Linear(N, 512) → GELU → Dropout(0.3)
    │
    ▼
ResidualBlock(512, 512) → GELU → Dropout(0.2)
    │
    ▼
ResidualBlock(512, 256) → GELU → Dropout(0.2)
    │
    ▼
ResidualBlock(256, 128) → GELU → Dropout(0.15)
    │
    ▼
Linear(128, 64) → GELU → Dropout(0.1)
    │
    ▼
Linear(64, 3) → Softmax
    │
    ▼
Output: P(H), P(D), P(A)
```

**ResidualBlock**:
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.act = nn.GELU()
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return self.act(out + residual)
```

### Training Configuration
```python
{
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingWarmRestarts",
    "T_0": 20,
    "T_mult": 2,
    "batch_size": 256,
    "max_epochs": 200,
    "early_stopping_patience": 20,
}
```

### GPU Optimization
- **Mixed Precision**: `torch.cuda.amp.autocast()` for FP16 forward pass
- **GradScaler**: Prevents FP16 underflow in gradients
- **Pin Memory**: `DataLoader(pin_memory=True)` for faster CPU→GPU transfer
- **Compiled Model**: `torch.compile(model)` for PyTorch 2.0+ graph optimization
- **Gradient Accumulation**: Effective batch size = batch_size × accumulation_steps

### Loss Function
- Cross-entropy loss with label smoothing (ε=0.05)
- Optional: Focal loss for handling class imbalance (draws are underrepresented)

### Hyperparameter Search (Optuna)
```python
{
    "hidden_dims": Categorical([[512,512,256,128], [256,256,128,64], [1024,512,256,128]]),
    "dropout": Uniform(0.05, 0.4),
    "lr": LogUniform(1e-5, 1e-2),
    "weight_decay": LogUniform(1e-6, 1e-2),
    "batch_size": Categorical([128, 256, 512]),
    "label_smoothing": Uniform(0.0, 0.1),
}
```

---

## 6. Model 5: Poisson Goal Model

### Architecture
This model predicts the number of goals each team scores using independent Poisson distributions.

**Mathematical Foundation**:
```
Goals_home ~ Poisson(λ_home)
Goals_away ~ Poisson(λ_away)

λ_home = exp(β_0 + β_home + α_home_attack + δ_away_defense + X_match @ γ)
λ_away = exp(β_0 + α_away_attack + δ_home_defense + X_match @ γ)

Where:
  β_0 = intercept (log of average goals ≈ log(1.3))
  β_home = home advantage parameter
  α_team_attack = team-specific attack strength
  δ_team_defense = team-specific defense weakness
  X_match = match-level features (rest, travel, weather, etc.)
  γ = feature coefficients
```

### Fitting Method
- Maximum Likelihood Estimation via `scipy.optimize.minimize`
- Regularization: L2 penalty on attack/defense parameters (prevents extreme values)
- Objective: Negative log-likelihood of observed scores under Poisson model

### Score Distribution
For each match, generate full score matrix:
```
P(i, j) = P(Goals_home = i) × P(Goals_away = j)   for i,j ∈ {0,...,10}

With Dixon-Coles correction for low scores (0-0, 1-0, 0-1, 1-1):
P_DC(0,0) = P(0,0) × (1 + λ_home × λ_away × ρ)
P_DC(1,0) = P(1,0) × (1 - λ_away × ρ)
P_DC(0,1) = P(0,1) × (1 - λ_home × ρ)
P_DC(1,1) = P(1,1) × (1 + ρ)

Where ρ is a correction parameter fitted from data (typically ρ ≈ -0.1)
```

### Deriving Match Outcome Probabilities
```
P(Home Win)  = Σ_{i>j} P(i, j)
P(Draw)      = Σ_{i=j} P(i, i)
P(Away Win)  = Σ_{j>i} P(i, j)
```

### Advantages
- Produces exact score probabilities (useful for correct score markets)
- Interpretable attack/defense parameters
- Feeds directly into Monte Carlo simulator

---

## 7. Model 6: Bayesian Probability Model

### Architecture
Bayesian hierarchical model with team-level latent strengths.

**Generative Model**:
```
# Priors
μ_attack ~ Normal(0, 1)
σ_attack ~ HalfNormal(0.5)
μ_defense ~ Normal(0, 1)
σ_defense ~ HalfNormal(0.5)
home_advantage ~ Normal(0.3, 0.2)

# Team-level parameters (for each team t)
attack_t ~ Normal(μ_attack, σ_attack)
defense_t ~ Normal(μ_defense, σ_defense)

# Match-level likelihood
log(λ_home) = home_advantage + attack_home - defense_away
log(λ_away) = attack_away - defense_home

goals_home ~ Poisson(λ_home)
goals_away ~ Poisson(λ_away)
```

### Inference
- **Method**: MCMC via NumPyro/PyMC (NUTS sampler)
- **Chains**: 4 parallel chains
- **Warmup**: 1000 samples
- **Samples**: 2000 post-warmup per chain (8000 total)
- **Convergence**: Check R-hat < 1.01, ESS > 400

### Time-Varying Strengths
- Exponential decay weighting: matches further in the past have less influence
- `weight_i = exp(-decay × days_since_match_i)` with `decay = 0.003` (half-life ≈ 230 days)

### Output
- Posterior distributions of team strengths (full uncertainty quantification)
- Predictive distributions: sample from posterior → compute λ → sample Poisson → get match outcome distribution
- 95% credible intervals for all probabilities

### Advantages Over Frequentist Models
- Full uncertainty quantification (not just point estimates)
- Natural handling of small sample sizes (shrinkage toward prior)
- Intuitive team strength interpretation
- Can incorporate external priors (e.g., FIFA rankings as informative priors)

---

## 8. Ensemble System

### 8.1 Level 1: Weighted Averaging
```
P_avg(outcome) = Σ_m (w_m × P_m(outcome))

Where weights w_m are optimized to minimize log loss on validation set:
  minimize -Σ_i Σ_c y_ic × log(Σ_m w_m × p_mic)
  subject to: Σ_m w_m = 1, w_m ≥ 0
```

Optimization via `scipy.optimize.minimize` with Nelder-Mead or SLSQP.

### 8.2 Level 2: Stacking Meta-Learner
```
Input: [P_xgb(H), P_xgb(D), P_xgb(A), P_lgb(H), ..., P_bay(A)]  → 18 features
       + original match features (Elo diff, form, etc.)              → ~20 features
       ──────────────────────────────────────────────────
       Total: ~38 input features

Meta-Learner: Logistic Regression with L2 regularization
  - Simple model to avoid over-fitting to base model outputs
  - Trained on out-of-fold predictions from walk-forward CV
  - C = 1.0 (tuned via cross-validation)
```

**Critical**: Base model predictions for training the meta-learner MUST be out-of-fold predictions — never train on the same predictions used to train base models.

### 8.3 Level 3: Probability Calibration
```
After stacking, apply calibration:

Option A: Platt Scaling (parametric)
  - Fit logistic regression: P_calibrated = sigmoid(a × P_raw + b)
  - Per-class calibration

Option B: Isotonic Regression (non-parametric)
  - Fit monotonic step function mapping raw → calibrated probabilities
  - Better for larger datasets

Option C: Temperature Scaling
  - P_calibrated = softmax(logits / T)
  - Single temperature parameter T fitted on calibration set

Use: Isotonic for large datasets (>5000 matches), Platt for smaller datasets
```

### 8.4 Uncertainty Estimation
```
For each prediction, compute:
1. Model disagreement: std(P_m(outcome)) across models
2. Bayesian posterior width: 95% CI from Bayesian model
3. Prediction entropy: H = -Σ P(c) × log(P(c))

Uncertainty score = weighted combination of all three
```

---

## 9. Ensemble Weighting Strategy

### Initial Weights (based on typical performance)
| Model | Initial Weight | Rationale |
|-------|---------------|-----------|
| XGBoost | 0.22 | Strong baseline, handles interactions |
| LightGBM | 0.22 | Fast, good with sparse features |
| CatBoost | 0.20 | Best with categoricals, robust |
| PyTorch NN | 0.15 | Captures non-linear patterns |
| Poisson | 0.12 | Provides score distributions |
| Bayesian | 0.09 | Uncertainty estimation, priors |

### Dynamic Weight Optimization
Weights re-optimized monthly on rolling 6-month validation window.
Log loss is the optimization target.

---

## 10. Calibration Analysis

### Reliability Diagram
- Bin predictions into 10 deciles
- Plot mean predicted probability vs observed frequency
- Perfect calibration = 45° line
- Metric: Expected Calibration Error (ECE) = Σ |bin_accuracy - bin_confidence| × bin_weight

### Calibration Requirements
- ECE < 0.03 for production deployment
- Per-class calibration error < 0.05
- Sharpe ratio of Kelly criterion bets > 0 on historical data
