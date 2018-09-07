#!/usr/bin/env bash
set -e # abort if any command fails

source activate machinelearning

python prepare_folds.py

for i in 0 1 2 3 4
do
   python task_v1.py --size 101x101 --device-ids 0,1 --batch-size 4 --fold $i --workers 12 --lr 0.0001 --n-epochs 5
done

source deactivate
