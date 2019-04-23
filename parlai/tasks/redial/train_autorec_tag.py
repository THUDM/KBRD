#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Train a model using parlai's standard training loop.

For documentation, see parlai.scripts.train_model.
"""

from parlai.scripts.train_model import TrainLoop, setup_args

if __name__ == "__main__":
    parser = setup_args()
    parser.set_defaults(
        task="redial",
        dict_tokenizer="split",
        model="autorec_tag",
        model_file="saved/autorec_tag",
        batchsize=64,
        n_movies=6924 + 4765,
        validation_metric="recall@50",
        validation_every_n_secs=5,
        validation_patience=5,
        tensorboard_log=True,
        tensorboard_tag="task,model,batchsize,hiddensize,learningrate",
        tensorboard_metrics="loss",
    )
    opt = parser.parse_args()
    TrainLoop(opt).train()
