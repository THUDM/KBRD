#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Train a model using parlai's standard training loop.

For documentation, see parlai.scripts.train_model.
"""

from parlai.scripts.train_model import TrainLoop, setup_args

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='redial',
        model='transformer/generator',
        model_file='saved/transformer',
        dict_tokenizer='nltk',
        dict_lower=True,
        batchsize=64,
        truncate=1024,
        dropout=0.1,
        relu_dropout=0.1,
        validation_metric='nll_loss',
        validation_metric_mode='min',
        validation_every_n_secs=300,
        validation_patience=5,
        tensorboard_log=True,
        tensorboard_tag="task,model,batchsize,ffn_size,embedding_size,n_layers,learningrate,model_file",
        tensorboard_metrics="ppl,nll_loss,token_acc,bleu",
    )
    opt = parser.parse_args()
    TrainLoop(opt).train()
