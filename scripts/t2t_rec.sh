#!/bin/bash
let num_runs=11

for i in $(seq 0 $((num_runs-1)));
do
    if [ $(( (i+1) % 8 )) -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$((i % 8)) python parlai/tasks/redial/train_transformer_rec.py -mf saved/transformer_onlymovie_2fc_$i
    else
        CUDA_VISIBLE_DEVICES=$((i % 8)) python parlai/tasks/redial/train_transformer_rec.py -mf saved/transformer_onlymovie_2fc_$i &
    fi
done

