#!/bin/bash

for i in $(seq 1 50);
do
    echo "Running with time window $i"
    python3 logistic_regression_enlarge_only_rets.py $i
done

