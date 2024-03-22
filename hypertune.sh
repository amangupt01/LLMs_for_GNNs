#!/bin/bash

# take the model name as an argument
MODEL_NAME=$1

# for loop over "random", "fixed" string
for i in "random" "fixed"
do
    for j in 'MLP' 'GCN' 'GAT'
    do
        echo python3 baseline.py --data_format $MODEL_NAME --split $i --dataset cora --seed_num 5 --model_name $j --mode sweep
        python3 baseline.py --data_format $MODEL_NAME --split $i --dataset cora --seed_num 5 --model_name $j --mode sweep
        echo "Done with $i $j"
    done
done


