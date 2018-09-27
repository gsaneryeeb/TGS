#!/usr/bin/env bash
set -e # abort if any command fails

source activate machinelearning

echo "PREDICT"
echo "---"

BATCH=2 # batch size
for FOLD in 0 1 2 3 4
do
    echo "=========="
    echo "FOLD $FOLD"
    echo "BATCH $BATCH"
    echo "=========="
    echo ""

    python predict_v1.py --fold $FOLD
done

# python merge.py

#echo "Generate final ensemble"
#python generate_sub_final_ensemble.py -j=4

source deactivate