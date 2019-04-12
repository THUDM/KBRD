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
        model='transformer/generator',
        model_file='saved/transformer',
        batchsize=64,
        truncate=1024,
        validation_metric='token_acc',
        validation_every_n_secs=600,
        tensorboard_log=True,
        tensorboard_tag="task,model,batchsize,ffn_size,embedding_size,n_layers,learningrate,dropout,gradient_clip",
        tensorboard_metrics="ppl,nll_loss,token_acc,bleu",
    )
    opt = parser.parse_args()
    TrainLoop(opt).train()
