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
        dict_tokenizer='split',
        model='seq2seq',
        model_file='saved/rnn',
        batchsize=64,
        validation_metric='token_acc',
        validation_every_n_secs=300,
        validation_patience=5,
        tensorboard_log=True,
        tensorboard_tag='task,model,batchsize,hiddensize,embeddingsize,attention,numlayers,rnn_class,learningrate,dropout,gradient_clip',
        tensorboard_metrics="ppl,nll_loss,token_acc,bleu",
    )
    opt = parser.parse_args()
    TrainLoop(opt).train()
