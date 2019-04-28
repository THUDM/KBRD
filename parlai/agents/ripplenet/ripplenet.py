import copy
import os
import pickle as pkl
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from parlai.core.torch_agent import Output, TorchAgent
from parlai.core.utils import round_sigfigs

from .modules import RippleNet


def _get_ripple_set(kg, original_set, n_hop, n_memory):
    """Return: [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]"""

    ripple_set = []

    for h in range(n_hop):
        memories_h = []
        memories_r = []
        memories_t = []

        if h == 0:
            tails_of_last_hop = original_set
        else:
            tails_of_last_hop = ripple_set[-1][2]

        for entity in tails_of_last_hop:
            for tail_and_relation in kg[entity]:
                memories_h.append(entity)
                memories_r.append(tail_and_relation[0])
                memories_t.append(tail_and_relation[1])

        # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
        # this won't happen for h = 0, because only the items that appear in the KG have been selected
        # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
        if len(memories_h) == 0:
            ripple_set.append(ripple_set[-1])
        else:
            # sample a fixed-size 1-hop memory for each user
            replace = len(memories_h) < n_memory
            indices = np.random.choice(len(memories_h), size=n_memory, replace=replace)
            memories_h = [memories_h[i] for i in indices]
            memories_r = [memories_r[i] for i in indices]
            memories_t = [memories_t[i] for i in indices]
            ripple_set.append((memories_h, memories_r, memories_t))

    return ripple_set


class RipplenetAgent(TorchAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        super(RipplenetAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group("Arguments")
        agent.add_argument("-ne", "--n-entity", type=int)
        agent.add_argument("-nr", "--n-relation", type=int)
        agent.add_argument("-dim", "--dim", type=int, default=16)
        agent.add_argument("-hop", "--n-hop", type=int, default=2)
        agent.add_argument("-kgew", "--kge-weight", type=float, default=1)
        agent.add_argument("-l2w", "--l2-weight", type=float, default=2.5e-6)
        agent.add_argument("-nmem", "--n-memory", type=int, default=32)
        agent.add_argument(
            "-ium", "--item-update-mode", type=str, default="plus_transform"
        )
        agent.add_argument("-uah", "--using-all-hops", type=bool, default=True)
        agent.add_argument(
            "-lr", "--learningrate", type=float, default=1e-2, help="learning rate"
        )
        RipplenetAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        init_model, is_finetune = self._get_init_model(opt, shared)

        self.id = "RipplenetAgent"
        self.n_entity = opt["n_entity"]
        self.n_hop = opt["n_hop"]
        self.n_memory = opt["n_memory"]

        if not shared:
            # set up model from scratch

            # encoder captures the input text
            self.model = RippleNet(
                n_entity=opt["n_entity"],
                n_relation=opt["n_relation"],
                dim=opt["dim"],
                n_hop=opt["n_hop"],
                kge_weight=opt["kge_weight"],
                l2_weight=opt["l2_weight"],
                n_memory=opt["n_memory"],
                item_update_mode=opt["item_update_mode"],
                using_all_hops=opt["using_all_hops"],
            )
            if init_model is not None:
                # load model parameters if available
                print("[ Loading existing model params from {} ]" "".format(init_model))
                states = self.load(init_model)
                if "number_training_updates" in states:
                    self._number_training_updates = states["number_training_updates"]

            if self.use_cuda:
                self.model.cuda()
            self.kg = pkl.load(
                open(os.path.join(opt["datapath"], "redial", "subkg.pkl"), "rb")
            )
            self.movie_ids = pkl.load(
                open(os.path.join(opt["datapath"], "redial", "movie_ids.pkl"), "rb")
            )

        elif "ripplenet" in shared:
            # copy initialized data from shared table
            self.model = shared["ripplenet"]
            self.kg = shared["kg"]
            self.movie_ids = shared["movie_ids"]

        # self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            opt["learningrate"],
        )
        self.metrics = {"loss": 0.0, "base_loss": 0.0, "kge_loss": 0.0, "l2_loss": 0.0}
        self.metrics["recall@1"] = []
        self.metrics["recall@10"] = []
        self.metrics["recall@50"] = []
        self.metrics["num_tokens"] = 0
        self.metrics["num_batches"] = 0
        self.metrics["acc"] = 0.0
        self.metrics["auc"] = 0.0

    def report(self):
        """
        Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        base = super().report()
        m = {}
        m["num_tokens"] = self.metrics["num_tokens"]
        m["num_batches"] = self.metrics["num_batches"]
        m["loss"] = self.metrics["loss"] / m["num_batches"]
        m["base_loss"] = self.metrics["base_loss"] / m["num_batches"]
        m["kge_loss"] = self.metrics["kge_loss"] / m["num_batches"]
        m["l2_loss"] = self.metrics["l2_loss"] / m["num_batches"]
        m["acc"] = self.metrics["acc"] / m["num_tokens"]
        m["auc"] = self.metrics["auc"] / m["num_tokens"]
        for x in ["1", "10", "50"]:
            if f"recall@{x}" in self.metrics and self.metrics[f"recall@{x}"] != []:
                m[f"recall@{x}"] = np.mean(self.metrics[f"recall@{x}"])
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def reset_metrics(self):
        self.metrics["loss"] = 0.0
        self.metrics["base_loss"] = 0.0
        self.metrics["kge_loss"] = 0.0
        self.metrics["l2_loss"] = 0.0
        self.metrics["recall@1"] = []
        self.metrics["recall@10"] = []
        self.metrics["recall@50"] = []
        self.metrics["num_tokens"] = 0
        self.metrics["num_batches"] = 0
        self.metrics["acc"] = 0.0
        self.metrics["auc"] = 0.0

    def share(self):
        """Share internal states."""
        shared = super().share()
        shared["ripplenet"] = self.model
        shared["kg"] = self.kg
        shared["movie_ids"] = self.movie_ids
        return shared

    def vectorize(self, obs, history, **kwargs):
        if "text" not in obs:
            return obs
        # TODO: do sentiment analysis when testing or ignore like/dislike
        pattern = re.compile(r"@\d+")
        input_match = re.findall(pattern, history.get_history_str())
        input_match = [int(x[1:]) for x in input_match]

        if "labels" in obs:
            label_type = "labels"
        elif "eval_labels" in obs:
            label_type = "eval_labels"
        else:
            label_type = None
        if label_type is None:
            return obs
        labels_match = re.findall(pattern, obs[label_type][0])
        labels_match = [int(x[1:]) for x in labels_match]
        if input_match == [] or labels_match == []:
            del obs["text"], obs[label_type]
            return obs

        input_vec = torch.zeros(self.n_entity)
        labels_vec = torch.zeros(self.n_entity, dtype=torch.long)
        input_vec[input_match] = 1
        labels_vec[labels_match] = 1

        obs["text_vec"] = input_vec
        obs[label_type + "_vec"] = labels_vec

        return obs

    def train_step(self, batch):
        self.model.train()
        bs = (batch.label_vec == 1).sum().item()

        items = torch.zeros(2 * bs, dtype=torch.long)
        labels = torch.zeros(2 * bs, dtype=torch.long)
        ripple_set = []
        for i, (b, movieIdx) in enumerate(batch.label_vec.nonzero().tolist()):
            seed = batch.text_vec[b].nonzero().view(-1).tolist()
            ripple_set.append(_get_ripple_set(self.kg, seed, self.n_hop, self.n_memory))
            items[i] = movieIdx
            labels[i] = 1
            # Negative samples
            items[bs + i] = int(np.random.choice(self.movie_ids))
            labels[bs + i] = 0
        memories_h, memories_r, memories_t = [], [], []
        for i in range(self.n_hop):
            memories_h.append(
                torch.LongTensor([ripple_set[idx % bs][i][0] for idx in range(2 * bs)])
            )
            memories_r.append(
                torch.LongTensor([ripple_set[idx % bs][i][1] for idx in range(2 * bs)])
            )
            memories_t.append(
                torch.LongTensor([ripple_set[idx % bs][i][2] for idx in range(2 * bs)])
            )

        if self.use_cuda:
            items = items.cuda()
            labels = labels.cuda()
            memories_h = list(map(lambda x: x.cuda(), memories_h))
            memories_r = list(map(lambda x: x.cuda(), memories_r))
            memories_t = list(map(lambda x: x.cuda(), memories_t))

        return_dict = self.model(items, labels, memories_h, memories_r, memories_t)
        loss = return_dict["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metrics["base_loss"] += return_dict["base_loss"]
        self.metrics["kge_loss"] += return_dict["kge_loss"]
        self.metrics["l2_loss"] += return_dict["l2_loss"]
        self.metrics["loss"] += loss
        self.metrics["num_tokens"] += bs
        self.metrics["num_batches"] += 1
        self._number_training_updates += 1

        self.model.eval()
        acc, auc = self.model.evaluate(
            items, labels, memories_h, memories_r, memories_t
        )
        self.metrics["acc"] += acc * bs
        self.metrics["auc"] += auc * bs

    def eval_step(self, batch):
        if batch.text_vec is None:
            return

        self.model.eval()
        bs = (batch.label_vec == 1).sum().item()
        items = torch.zeros(2 * bs, dtype=torch.long)
        labels = torch.zeros(2 * bs, dtype=torch.long)
        ripple_set = []
        for i, (b, movieIdx) in enumerate(batch.label_vec.nonzero().tolist()):
            seed = batch.text_vec[b].nonzero().view(-1).tolist()
            ripple_set.append(_get_ripple_set(self.kg, seed, self.n_hop, self.n_memory))
            items[i] = movieIdx
            labels[i] = 1
            # Negative samples
            items[bs + i] = int(np.random.choice(self.movie_ids))
            labels[bs + i] = 0
        memories_h, memories_r, memories_t = [], [], []
        for i in range(self.n_hop):
            memories_h.append(
                torch.LongTensor([ripple_set[idx % bs][i][0] for idx in range(2 * bs)])
            )
            memories_r.append(
                torch.LongTensor([ripple_set[idx % bs][i][1] for idx in range(2 * bs)])
            )
            memories_t.append(
                torch.LongTensor([ripple_set[idx % bs][i][2] for idx in range(2 * bs)])
            )

        if self.use_cuda:
            items = items.cuda()
            labels = labels.cuda()
            memories_h = list(map(lambda x: x.cuda(), memories_h))
            memories_r = list(map(lambda x: x.cuda(), memories_r))
            memories_t = list(map(lambda x: x.cuda(), memories_t))

        return_dict = self.model(items, labels, memories_h, memories_r, memories_t)
        loss = return_dict["loss"]

        self.metrics["base_loss"] += return_dict["base_loss"]
        self.metrics["kge_loss"] += return_dict["kge_loss"]
        self.metrics["l2_loss"] += return_dict["l2_loss"]
        self.metrics["loss"] += loss
        self.metrics["num_tokens"] += bs
        self.metrics["num_batches"] += 1

        acc, auc = self.model.evaluate(
            items, labels, memories_h, memories_r, memories_t
        )
        self.metrics["acc"] += acc * bs
        self.metrics["auc"] += auc * bs

        # _, pred_idx = torch.topk(outputs, k=100, dim=1)

        # for b in range(bs):
        #     self.metrics["recall@1"].append(
        #         int(ys[b].item() in pred_idx[b][:1].tolist())
        #     )
        #     self.metrics["recall@10"].append(
        #         int(ys[b].item() in pred_idx[b][:10].tolist())
        #     )
        #     self.metrics["recall@50"].append(
        #         int(ys[b].item() in pred_idx[b][:50].tolist())
        #     )
