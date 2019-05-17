#!/bin/bash
let num_runs=24

for i in $(seq 0 $((num_runs-1)));
do
    CUDA_VISIBLE_DEVICES=$((2 + (i % 6))) python examples/eval_model.py -t redial -mf saved/transformer_$i -dt test --beam-size 5 --beam-block-ngram 3 | tee saved/transformer_$i.eval &
    if [ $(( (i+1) % 6 )) -eq 0 ]; then
        wait
    fi
done

