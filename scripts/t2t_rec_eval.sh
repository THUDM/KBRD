#!/bin/bash
let num_runs=24

for i in $(seq 0 $((num_runs-1)));
do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python examples/eval_model.py -t redial -mf saved-05-17/transformer_rec_both_$i -dt test --beam-size 5 --beam-block-ngram 3 | tee saved-05-17/transformer_rec_both_$i.eval &
    if [ $(( (i+1) % 8 )) -eq 0 ]; then
        wait
    fi
done

