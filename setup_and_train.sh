#!/bin/bash
#ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•
# WORLD CUP AI ג€” ONE-SHOT SETUP & TRAINING SCRIPT
#ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•
#
# This script does EVERYTHING on a fresh Linux server:
#   1. Installs system dependencies
#   2. Creates Python virtual environment
#   3. Installs all Python packages (with GPU support)
#   4. Creates required directories
#   5. Downloads ALL data (international results + StatsBomb)
#   6. Builds the feature matrix (200+ features)
#   7. Trains all models (XGBoost, LightGBM, CatBoost, Poisson)
#   8. Builds calibrated ensemble
#   9. Evaluates and reports
#
# Usage:
#   chmod +x setup_and_train.sh
#   ./setup_and_train.sh             # Full pipeline with GPU
#   ./setup_and_train.sh --no-gpu    # CPU only
#   ./setup_and_train.sh --fast      # Skip StatsBomb (faster, no xG)
#
#ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•

set -e  # Exit on any error

# ג”€ג”€ Configuration ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
PYTHON_VERSION="python3"
VENV_DIR=".venv"
USE_GPU=true
SKIP_STATSBOMB=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --no-gpu)  USE_GPU=false ;;
        --fast)    SKIP_STATSBOMB=true ;;
        *)         echo "Unknown argument: $arg" ;;
    esac
done

echo ""
echo "ג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆ"
echo "  WORLD CUP AI ג€” AUTOMATED SETUP & TRAINING"
echo "ג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆ"
echo ""
echo "  GPU:            $USE_GPU"
echo "  Skip StatsBomb: $SKIP_STATSBOMB"
echo "  Python:         $PYTHON_VERSION"
echo ""

# ג”€ג”€ Step 0: System Dependencies ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"
echo "  STEP 0: SYSTEM DEPENDENCIES"
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"

if command -v apt-get &> /dev/null; then
    echo "  ג†’ Installing system packages (apt)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3 python3-pip python3-venv python3-dev \
        build-essential cmake \
        libgomp1 \
        > /dev/null 2>&1
    echo "  ג“ System packages installed"
elif command -v yum &> /dev/null; then
    echo "  ג†’ Installing system packages (yum)..."
    sudo yum install -y -q \
        python3 python3-pip python3-devel \
        gcc gcc-c++ cmake \
        > /dev/null 2>&1
    echo "  ג“ System packages installed"
else
    echo "  ג  Unknown package manager, assuming dependencies exist"
fi

# ג”€ג”€ Step 1: Python Virtual Environment ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
echo ""
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"
echo "  STEP 1: PYTHON ENVIRONMENT"
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"

if [ ! -d "$VENV_DIR" ]; then
    echo "  ג†’ Creating virtual environment..."
    $PYTHON_VERSION -m venv $VENV_DIR
    echo "  ג“ Virtual environment created"
else
    echo "  ג†’ Virtual environment already exists"
fi

source $VENV_DIR/bin/activate
echo "  ג†’ Activated: $(which python)"
echo "  ג†’ Python version: $(python --version)"

# Upgrade pip
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "  ג“ pip upgraded"

# ג”€ג”€ Step 2: Install Python Packages ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
echo ""
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"
echo "  STEP 2: PYTHON PACKAGES"
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"

echo "  ג†’ Installing core packages..."
pip install \
    pandas numpy polars pyarrow \
    scikit-learn scipy \
    requests beautifulsoup4 lxml \
    pydantic pydantic-settings pyyaml python-dotenv \
    structlog rich \
    optuna mlflow shap \
    2>&1 | tail -1

echo "  ג†’ Installing ML frameworks..."
pip install xgboost lightgbm catboost 2>&1 | tail -1

if [ "$USE_GPU" = true ]; then
    echo "  ג†’ Installing PyTorch with CUDA..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1

    # Check GPU availability
    python -c "
import torch
if torch.cuda.is_available():
    print(f'  ג“ GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'    CUDA version: {torch.version.cuda}')
    print(f'    GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  ג  No GPU detected - falling back to CPU')
" 2>/dev/null || echo "  ג  PyTorch GPU check failed"
else
    echo "  ג†’ Installing PyTorch CPU..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
fi

echo "  ג“ All packages installed"

# Verify critical imports
echo "  ג†’ Verifying imports..."
python -c "
import pandas, numpy, sklearn, xgboost, lightgbm, catboost, scipy, structlog
print('  ג“ All imports verified')
"

# ג”€ג”€ Step 3: Create Directory Structure ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
echo ""
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"
echo "  STEP 3: DIRECTORY STRUCTURE"
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"

mkdir -p data/{raw/{statsbomb,international_results},processed,features,models}
mkdir -p logs
echo "  ג“ Directory structure created"

# ג”€ג”€ Step 4: Create .env if not exists ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
if [ ! -f ".env" ]; then
    cp .env.example .env 2>/dev/null || cat > .env << 'EOF'
WC_DATA_DIR=./data
WC_LOG_LEVEL=INFO
WC_RANDOM_SEED=42
MLFLOW_TRACKING_URI=./mlruns
CUDA_VISIBLE_DEVICES=0
EOF
    echo "  ג“ .env file created"
fi

# ג”€ג”€ Step 5: Run Training Pipeline ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
echo ""
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"
echo "  STEP 5: RUNNING FULL TRAINING PIPELINE"
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"
echo ""

TRAIN_ARGS=""

if [ "$USE_GPU" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --gpu"
fi

if [ "$SKIP_STATSBOMB" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --no-statsbomb"
fi

echo "  Command: python train.py $TRAIN_ARGS"
echo ""

python train.py $TRAIN_ARGS

# ג”€ג”€ Step 6: Post-training verification ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
echo ""
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"
echo "  STEP 6: POST-TRAINING VERIFICATION"
echo "ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•"

python -c "
import json
from pathlib import Path

# Check model files
models_dir = Path('data/models')
models = ['xgboost', 'lightgbm', 'catboost', 'poisson', 'ensemble']
for model in models:
    model_dir = models_dir / model
    if model_dir.exists():
        files = list(model_dir.iterdir())
        print(f'  ג“ {model}: {len(files)} files saved')
    else:
        print(f'  ג— {model}: NOT FOUND')

# Check evaluation report
report_path = models_dir / 'evaluation_report.json'
if report_path.exists():
    with open(report_path) as f:
        results = json.load(f)
    print(f'\n  Evaluation Summary:')
    for name, metrics in sorted(results.items(), key=lambda x: x[1].get('log_loss', 99)):
        ll = metrics.get('log_loss', 'N/A')
        print(f'    {name:<15s} Log Loss: {ll}')
else:
    print('  ג— Evaluation report not found')

# Check feature matrix
feat_path = Path('data/features/feature_matrix.parquet')
if feat_path.exists():
    import pandas as pd
    df = pd.read_parquet(feat_path)
    print(f'\n  Feature matrix: {len(df):,} matches ֳ— {len(df.columns)} columns')
    print(f'  Disk usage: {feat_path.stat().st_size / 1e6:.1f} MB')
"

echo ""
echo "ג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆ"
echo "  ג“ SETUP & TRAINING COMPLETE"
echo ""
echo "  To predict a match:"
echo "    source .venv/bin/activate"
echo "    python predict.py --home Brazil --away Germany --scores"
echo ""
echo "  To retrain with cached data:"
echo "    python train.py --skip-ingestion --gpu"
echo ""
echo "ג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆג–ˆ"
