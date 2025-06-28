#!/bin/bash
echo "Cleaning old results..."
rm -rf ../results/single/*
mkdir -p ../results/single

models=(knn ncentroid dtree linsvc rbfsvc rforest ada bag binlr qda lda xgboost gradboost extratree)
days=(0 1 2 3 4 5)  # Adjust based on your dataset

for model in "${models[@]}"; do
  for day in "${days[@]}"; do
    echo "Running model $model on day $day"
    python ml.py -A $model -S Z -D $day --datadir data/CSV/ --resultdir results/
  done
done
python ../utils/extract_json_summary.py > latest_run_summary.json
