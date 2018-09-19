#!/usr/bin/env bash
set -e # abort if any command fails

source activate machinelearning

echo "Merge Only"


python merge.py


source deactivate