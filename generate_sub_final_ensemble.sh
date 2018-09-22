#!/usr/bin/env bash
set -e # abort if any command fails

source activate machinelearning


echo "Generate final ensemble"
python generate_sub_final_ensemble.py -j=4

source deactivate