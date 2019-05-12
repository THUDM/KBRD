#!/bin/bash
let num_runs=32

for i in $(seq 0 $((num_runs-1)));
do
    CUDA_VISIBLE_DEVICES=1 python parlai/tasks/redial/train_ripplenet.py -mf saved/both_$i
done

