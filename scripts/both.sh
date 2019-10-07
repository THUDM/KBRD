#!/bin/bash
let num_runs=32

for i in $(seq 0 $((num_runs-1)));
do
    CUDA_VISIBLE_DEVICES=2 python parlai/tasks/redial/train_kbrd.py -mf saved/both_rgcn_$i
done

