import math
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.rgcn_conv import RGCNConv


def kaiming_reset_parameters(linear_module):
    nn.init.kaiming_uniform_(linear_module.weight, a=math.sqrt(5))
    if linear_module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_module.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(linear_module.bias, -bound, bound)

class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        N = h.shape[0]
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e)
        return torch.matmul(attention, h)

def _edge_list(kg, n_entity):
    edge_list = []
    self_loop_id = None
    for entity in range(n_entity):
        if entity not in kg:
            continue
        for tail_and_relation in kg[entity]:
            if entity != tail_and_relation[1]:
                edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))
            else:
                self_loop_id = tail_and_relation[0]
    assert self_loop_id
    for entity in range(n_entity):
        # add self loop
        edge_list.append((entity, entity, self_loop_id))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    # Discard infrequent relations
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx)

class KBRD(nn.Module):
    def __init__(
        self,
        n_entity,
        n_relation,
        dim,
        kg,
        entity_kg_emb,
        entity_text_emb,
        num_bases
    ):
        super(KBRD, self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        nn.init.kaiming_uniform_(self.entity_emb.weight.data)

        self.criterion = nn.CrossEntropyLoss()
        self.kge_criterion = nn.Softplus()

        self.self_attn = SelfAttentionLayer(self.dim, self.dim)
        self.output = nn.Linear(self.dim, self.n_entity)

        self.kg = kg

        edge_list, self.n_relation = _edge_list(self.kg, self.n_entity)
        self.rgcn = RGCNConv(self.n_entity, self.dim, self.n_relation, num_bases=num_bases)
        edge_list = list(set(edge_list))
        edge_list_tensor = torch.LongTensor(edge_list).cuda()
        self.edge_idx = edge_list_tensor[:, :2].t()
        self.edge_type = edge_list_tensor[:, 2]

    def _get_triples(self, kg):
        triples = []
        for entity in kg:
            for relation, tail in kg[entity]:
                if entity != tail:
                    triples.append([entity, relation, tail])
        return triples

    def forward(
        self,
        seed_sets: list,
        labels: torch.LongTensor,
    ):
        # [batch size, dim]
        u_emb, nodes_features = self.user_representation(seed_sets)
        scores = F.linear(u_emb, nodes_features, self.output.bias)

        base_loss = self.criterion(scores, labels)

        loss = base_loss

        return dict(scores=scores.detach(), base_loss=base_loss, loss=loss)

    def user_representation(self, seed_sets):
        nodes_features = self.rgcn(None, self.edge_idx, self.edge_type)

        user_representation_list = []
        for i, seed_set in enumerate(seed_sets):
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).cuda())
                continue
            user_representation = nodes_features[seed_set]
            user_representation = self.self_attn(user_representation)
            user_representation_list.append(user_representation)
        return torch.stack(user_representation_list), nodes_features
