#!/bin/bash

for i in $(seq 1 50);
do
    echo "Running with time window $i"
    python3 MultiLayer_Perceptron.py $i
done

