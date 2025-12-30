#!/bin/bash
# Run 5-fold cross validation for LR model
# Output saved to data/kfold/

set -e

# Activate conda environment
source /home/akoehl/miniconda3/etc/profile.d/conda.sh
conda activate omnilib-stability

# Create output directory
mkdir -p data/kfold

# Run K-fold CV
python scripts/kfold_lr.py \
    --n-folds 5 \
    --epochs 10 \
    --output-dir data/kfold \
    2>&1 | tee data/kfold/kfold_lr.log

echo "Done. Output saved to data/kfold/"
