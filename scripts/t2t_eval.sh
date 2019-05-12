#!/bin/bash
let num_runs=16

for i in $(seq 0 $((num_runs-1)));
do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python examples/eval_model.py -t redial -mf saved/transformer_$i -dt test --beam-size 10 | tee saved/transformer_$i.eval &
    if [ $(( (i+1) % 8 )) -eq 0 ]; then
        wait
    fi
done

