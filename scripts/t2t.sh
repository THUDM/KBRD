#!/bin/bash
let num_runs=24

for i in $(seq 0 $((num_runs-1)));
do
    CUDA_VISIBLE_DEVICES=$((2 + (i % 6))) python parlai/tasks/redial/train_transformer.py -mf saved/transformer_$i &
    if [ $(( (i+1) % 6 )) -eq 0 ]; then
        wait
    fi
done

