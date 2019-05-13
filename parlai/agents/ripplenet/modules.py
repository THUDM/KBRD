import math

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nhid, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        """x: shape (|V|, |D|); adj: shape(|V|, |V|)"""
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
        # return F.log_softmax(x, dim=1)

def _add_neighbors(kg, g, seed_set, hop):
    tails_of_last_hop = seed_set
    for h in range(hop):
        next_tails_of_last_hop = []
        for entity in tails_of_last_hop:
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                g.add_edge(entity, tail_and_relation[1])
                if entity != tail_and_relation[1]:
                    next_tails_of_last_hop.append(tail_and_relation[1])
        tails_of_last_hop = next_tails_of_last_hop

class RippleNet(nn.Module):
    def __init__(
        self,
        n_entity,
        n_relation,
        dim,
        n_hop,
        kge_weight,
        l2_weight,
        n_memory,
        item_update_mode,
        using_all_hops,
        kg,
        entity_kg_emb
    ):
        super(RippleNet, self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.n_hop = n_hop
        self.kge_weight = kge_weight
        self.l2_weight = l2_weight
        self.n_memory = n_memory
        self.item_update_mode = item_update_mode
        self.using_all_hops = using_all_hops

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)
        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        self.gcn = GCN(self.dim)
        self.output = nn.Linear(self.dim, self.n_entity)
        self.kg = kg
        # KG emb as initialization
        self.entity_emb.weight.data[entity_kg_emb != 0] = entity_kg_emb[entity_kg_emb != 0]

    def forward(
        self,
        seed_sets: list,
        labels: torch.LongTensor,
    ):
        # [batch size, dim]
        u_emb = self.user_representation(seed_sets)
        scores = self.output(u_emb)

        base_loss = self.criterion(scores, labels)

        # l2_loss = 0
        # for hop in range(self.n_hop):
        #     l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
        # l2_loss = self.l2_weight * l2_loss
        # l2_loss = torch.Tensor([0])

        return dict(scores=scores.detach(), base_loss=base_loss, loss=base_loss)


    def user_representation(self, seed_sets):
        # find subgraph for this batch
        g = nx.Graph()
        for seed_set in seed_sets:
            for seed in seed_set:
                g.add_node(seed)
            # _add_neighbors(self.kg, g, seed_set, hop=2)
        # add self-loops
        for node in g.nodes:
            g.add_edge(node, node)

        # to cuda
        nodes = torch.LongTensor(list(g.nodes))
        adj = torch.FloatTensor(nx.to_numpy_matrix(g))
        nodes = nodes.cuda()
        adj = adj.cuda()

        nodes_features = self.entity_emb(nodes)
        # print(nodes_features.shape, adj.shape)
        # nodes_features = self.gcn(nodes_features, adj)

        node2id = dict([(n, i) for i, n in enumerate(list(g.nodes))])
        user_representation_list = []
        for seed_set in seed_sets:
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).cuda())
                continue
            seed_set_ids = list(map(lambda x: node2id[x], seed_set))
            user_representation = nodes_features[seed_set_ids]
            user_representation = user_representation.mean(dim=0)
            # user_representation = torch.relu(user_representation)
            user_representation_list.append(user_representation)
        return torch.stack(user_representation_list)

