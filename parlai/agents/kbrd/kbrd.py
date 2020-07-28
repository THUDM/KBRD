import copy
import os
import pickle as pkl
import re
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import nltk


from parlai.core.torch_agent import Output, TorchAgent
from parlai.core.utils import round_sigfigs

from .modules import KBRD



def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = line[0]
            if entity not in entity2entityId:
                continue
            entityId = entity2entityId[entity]
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings

def _load_text_embeddings(entity2entityId, dim, abstract_path):
    entities = []
    texts = []
    sent_tok = nltk.data.load('tokenizers/punkt/english.pickle')
    word_tok = nltk.tokenize.treebank.TreebankWordTokenizer()
    def nltk_tokenize(text):
        return [token for sent in sent_tok.tokenize(text)
                for token in word_tok.tokenize(sent)]

    with open(abstract_path, 'r') as f:
        for line in f.readlines():
            try:
                entity = line[:line.index('>')+1]
                if entity not in entity2entityId:
                    continue
                line = line[line.index('> "')+2:len(line)-line[::-1].index('@')-1]
                entities.append(entity)
                texts.append(line.replace('\\', ''))
            except Exception:
                pass
    vec_dim = 64
    try:
        model = Doc2Vec.load('doc2vec')
    except Exception:
        corpus = [nltk_tokenize(text) for text in texts]
        corpus = [
            TaggedDocument(words, ['d{}'.format(idx)])
            for idx, words in enumerate(corpus)
        ]
        model = Doc2Vec(corpus, vector_size=vec_dim, min_count=5, workers=28)
        model.save('doc2vec')

    full_text_embeddings = torch.zeros(len(entity2entityId), vec_dim)
    for i, entity in enumerate(entities):
        full_text_embeddings[entity2entityId[entity]] = torch.from_numpy(model.docvecs[i])

    return full_text_embeddings

class KbrdAgent(TorchAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        super(KbrdAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group("Arguments")
        agent.add_argument("-ne", "--n-entity", type=int)
        agent.add_argument("-nr", "--n-relation", type=int)
        agent.add_argument("-dim", "--dim", type=int, default=128)
        agent.add_argument(
            "-lr", "--learningrate", type=float, default=3e-3, help="learning rate"
        )
        agent.add_argument("-nb", "--num-bases", type=int, default=8)
        KbrdAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        init_model, is_finetune = self._get_init_model(opt, shared)

        self.id = "KbrdAgent"
        self.n_entity = opt["n_entity"]

        if not shared:
            # set up model from scratch

            self.kg = pkl.load(
                open(os.path.join(opt["datapath"], "redial", "subkg.pkl"), "rb")
            )
            self.movie_ids = pkl.load(
                open(os.path.join(opt["datapath"], "redial", "movie_ids.pkl"), "rb")
            )
            entity2entityId = pkl.load(
                open(os.path.join(opt["datapath"], "redial", "entity2entityId.pkl"), "rb")
            )
            entity_kg_emb = None
            abstract_path = 'dbpedia/short_abstracts_en.ttl'
            entity_text_emb = None

            # encoder captures the input text
            self.model = KBRD(
                n_entity=opt["n_entity"],
                n_relation=opt["n_relation"],
                dim=opt["dim"],
                kg=self.kg,
                entity_kg_emb=entity_kg_emb,
                entity_text_emb=entity_text_emb,
                num_bases=opt["num_bases"]
            )
            if init_model is not None:
                # load model parameters if available
                print("[ Loading existing model params from {} ]" "".format(init_model))
                states = self.load(init_model)
                if "number_training_updates" in states:
                    self._number_training_updates = states["number_training_updates"]

            if self.use_cuda:
                self.model.cuda()
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                opt["learningrate"],
            )

        elif "kbrd" in shared:
            # copy initialized data from shared table
            self.model = shared["kbrd"]
            self.kg = shared["kg"]
            self.movie_ids = shared["movie_ids"]
            self.optimizer = shared["optimizer"]

        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)

    def report(self):
        """
        Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        base = super().report()
        m = {}
        m["num_tokens"] = self.counts["num_tokens"]
        m["num_batches"] = self.counts["num_batches"]
        m["loss"] = self.metrics["loss"] / m["num_batches"]
        m["base_loss"] = self.metrics["base_loss"] / m["num_batches"]
        m["acc"] = self.metrics["acc"] / m["num_tokens"]
        m["auc"] = self.metrics["auc"] / m["num_tokens"]
        # Top-k recommendation Recall
        for x in sorted(self.metrics):
            if x.startswith("recall") and self.counts[x] > 200:
                m[x] = self.metrics[x] / self.counts[x]
                m["num_tokens_" + x] = self.counts[x]
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def reset_metrics(self):
        for key in self.metrics:
            self.metrics[key] = 0.0
        for key in self.counts:
            self.counts[key] = 0

    def share(self):
        """Share internal states."""
        shared = super().share()
        shared["kbrd"] = self.model
        shared["kg"] = self.kg
        shared["movie_ids"] = self.movie_ids
        shared["optimizer"] = self.optimizer
        return shared

    def vectorize(self, obs, history, **kwargs):
        if "text" not in obs:
            return obs

        if "labels" in obs:
            label_type = "labels"
        elif "eval_labels" in obs:
            label_type = "eval_labels"
        else:
            label_type = None
        if label_type is None:
            return obs

        # mentioned movies
        input_match = list(map(int, obs['label_candidates'][1].split()))
        labels_match = list(map(int, obs['label_candidates'][2].split()))
        entities_match = list(map(int, obs['label_candidates'][3].split()))

        if labels_match == []:
            del obs["text"], obs[label_type]
            return obs

        input_vec = torch.zeros(self.n_entity)
        labels_vec = torch.zeros(self.n_entity, dtype=torch.long)
        input_vec[input_match] = 1
        input_vec[entities_match] = 1
        labels_vec[labels_match] = 1

        obs["text_vec"] = input_vec
        obs[label_type + "_vec"] = labels_vec

        # turn no.
        obs["turn"] = len(input_match)

        return obs

    def train_step(self, batch):
        self.model.train()
        bs = (batch.label_vec == 1).sum().item()
        labels = torch.zeros(bs, dtype=torch.long)

        # create subgraph for propagation
        seed_sets = []
        for i, (b, movieIdx) in enumerate(batch.label_vec.nonzero().tolist()):
            # seed set (i.e. mentioned movies + entitites)
            seed_set = batch.text_vec[b].nonzero().view(-1).tolist()
            labels[i] = movieIdx
            seed_sets.append(seed_set)

        if self.use_cuda:
            labels = labels.cuda()

        return_dict = self.model(seed_sets, labels)

        loss = return_dict["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metrics["base_loss"] += return_dict["base_loss"].item()
        self.metrics["loss"] += loss.item()

        self.counts["num_tokens"] += bs
        self.counts["num_batches"] += 1
        self._number_training_updates += 1

    def eval_step(self, batch):
        if batch.text_vec is None:
            return

        self.model.eval()
        bs = (batch.label_vec == 1).sum().item()
        labels = torch.zeros(bs, dtype=torch.long)

        # create subgraph for propagation
        seed_sets = []
        turns = []
        for i, (b, movieIdx) in enumerate(batch.label_vec.nonzero().tolist()):
            # seed set (i.e. mentioned movies + entitites)
            seed_set = batch.text_vec[b].nonzero().view(-1).tolist()
            labels[i] = movieIdx
            seed_sets.append(seed_set)
            turns.append(batch.turn[b])

        if self.use_cuda:
            labels = labels.cuda()

        return_dict = self.model(seed_sets, labels)

        loss = return_dict["loss"]

        self.metrics["base_loss"] += return_dict["base_loss"].item()
        self.metrics["loss"] += loss.item()
        self.counts["num_tokens"] += bs
        self.counts["num_batches"] += 1

        outputs = return_dict["scores"].cpu()
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        _, pred_idx = torch.topk(outputs, k=100, dim=1)
        for b in range(bs):
            target_idx = self.movie_ids.index(labels[b].item())
            self.metrics["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            self.metrics[f"recall@1@turn{turns[b]}"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics[f"recall@10@turn{turns[b]}"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics[f"recall@50@turn{turns[b]}"] += int(target_idx in pred_idx[b][:50].tolist())
            self.counts[f"recall@1@turn{turns[b]}"] += 1
            self.counts[f"recall@10@turn{turns[b]}"] += 1
            self.counts[f"recall@50@turn{turns[b]}"] += 1
            self.counts[f"recall@1"] += 1
            self.counts[f"recall@10"] += 1
            self.counts[f"recall@50"] += 1
        return Output(list(map(lambda x: str(self.movie_ids[x]), outputs.argmax(dim=1).tolist())))
