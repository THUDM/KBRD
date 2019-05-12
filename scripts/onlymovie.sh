#!/bin/bash
let num_runs=24

for i in $(seq 0 $((num_runs-1)));
do
    CUDA_VISIBLE_DEVICES=0 python parlai/tasks/redial/train_ripplenet.py -mf saved/transformer_$i
done

