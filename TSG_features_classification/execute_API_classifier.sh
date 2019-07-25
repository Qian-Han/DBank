#!/usr/bin/env bash

# parameters for cross-validation
NFOLDS=10
NITER=1

FEATURES_PATH=./

RESULTS_PATH=.classification_results/

KFOLDS=('Stratified')
CLASSIFIERS=('RF' 'DT' 'NB' 'LR' 'KNN' 'SVM' 'ADA' 'GBDT' 'NN_MLP' 'KerasNN_MLP')


DATASETS=(
# "good_banker_ety"
"good_banker_API"
) 

# 0 means only using given feature, 1 means using given feature and SR feature
# input "_ety" feature and using 1 can produce single SR feature
SR_METHOD=('1')

echo "Creating directory $RESULTS_PATH..."
mkdir -p $RESULTS_PATH

echo "Removing files from $RESULTS_PATH..."

# prompt before cleaning results folder
rm -rI $RESULTS_PATH/*.csv

echo "Running the classifiers"

# Running the classifiers
python3 API_classifier.py --nfolds $NFOLDS --niterations $NITER --features "$FEATURES_PATH" --sr_method ${SR_METHOD[@]} --results $RESULTS_PATH --classifiers ${CLASSIFIERS[@]} --datasets ${DATASETS[@]} --kfolds ${KFOLDS[@]} --debug  --save-fi 

# echo "Removing incomplete results"
rm $RESULTS_PATH/*incomplete*.csv

echo "Script complete."
