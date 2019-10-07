import math
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.rgcn_conv import RGCNConv


def kaiming_reset_parameters(linear_module):
    nn.init.kaiming_uniform_(linear_module.weight, a=math.sqrt(5))
    if linear_module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_module.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(linear_module.bias, -bound, bound)

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
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

        kaiming_reset_parameters(self)

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
    def __init__(self, ninp, nhid, dropout=0.5):
        super(GCN, self).__init__()

        # self.gc1 = GraphConvolution(ninp, nhid)
        self.gc2 = GraphConvolution(ninp, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        """x: shape (|V|, |D|); adj: shape(|V|, |V|)"""
        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
        # return F.log_softmax(x, dim=1)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        # self.a = nn.Parameter(torch.zeros(size=(2*self.dim, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        # self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        N = h.shape[0]
        assert self.dim == h.shape[1]
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.dim)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # attention = F.softmax(e, dim=1)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        return torch.matmul(attention, h)

class SelfAttentionLayer2(nn.Module):
    def __init__(self, dim, da):
        super(SelfAttentionLayer2, self).__init__()
        self.dim = dim
        self.Wq = nn.Parameter(torch.zeros(self.dim, self.dim))
        self.Wk = nn.Parameter(torch.zeros(self.dim, self.dim))
        nn.init.xavier_uniform_(self.Wq.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wk.data, gain=1.414)
        # self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        N = h.shape[0]
        assert self.dim == h.shape[1]
        q = torch.matmul(h, self.Wq)
        k = torch.matmul(h, self.Wk)
        e = torch.matmul(q, k.t()) / math.sqrt(self.dim)
        attention = F.softmax(e, dim=1)
        attention = attention.mean(dim=0)
        x = torch.matmul(attention, h)
        return x

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

        def forward(self, input, memory, mask=None):
            bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

            input = self.dropout(input)
            memory = self.dropout(memory)

            input_dot = self.input_linear(input)
            memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
            cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
            att = input_dot + memory_dot + cross_dot
            if mask is not None:
                att = att - 1e30 * (1 - mask[:,None])

                weight_one = F.softmax(att, dim=-1)
                output_one = torch.bmm(weight_one, memory)
                weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
                output_two = torch.bmm(weight_two, input)
            return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        # self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        # self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        N = input.size()[0]
        # edge = adj.nonzero().t()
        edge = adj._indices()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        # edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_h = h[edge[1, :], :].t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1)).cuda())
        # e_rowsum: N x 1

        # edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        # self.attentions = [SpGraphAttentionLayer(nfeat,
        #                                          nhid,
        #                                          dropout=dropout,
        #                                          alpha=alpha,
        #                                          concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        # self.out_att = SpGraphAttentionLayer(nhid * nheads,
        #                                      nclass,
        #                                      dropout=dropout,
        #                                      alpha=alpha,
        #                                      concat=False)
        self.out_att = SpGraphAttentionLayer(nhid,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        x = self.out_att(x, adj)
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

# http://dbpedia.org/ontology/director
EDGE_TYPES = [58, 172]
def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            # edge_list.append((entity, entity))
            # self_loop id = 185
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185 :# and tail_and_relation[0] in EDGE_TYPES:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
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
        n_hop,
        kge_weight,
        l2_weight,
        n_memory,
        item_update_mode,
        using_all_hops,
        kg,
        entity_kg_emb,
        entity_text_emb,
        num_bases
    ):
        super(KBRD, self).__init__()

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
        # self.entity_kg_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        # nn.init.uniform_(self.entity_emb.weight.data)
        nn.init.kaiming_uniform_(self.entity_emb.weight.data)
        # nn.init.xavier_uniform_(self.entity_kg_emb.weight.data)
        # nn.init.xavier_uniform_(self.relation_emb.weight.data)

        # self.entity_text_emb = entity_text_emb.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.kge_criterion = nn.Softplus()

        # self.gcn = GCN(self.dim, self.dim)
        # self.gcn = GCN(self.entity_text_emb.shape[1], self.dim)
        # self.transform = nn.Sequential(
        #     nn.Linear(self.entity_text_emb.shape[1], 128),
        #     # nn.ReLU(),
        #     # nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(128, self.dim),
        # )
        # self.gat = SpGAT(self.dim, self.dim, self.dim, dropout=0., alpha=0.2, nheads=4)
        # self.gcn = GCNConv(self.dim, self.dim)
        # self.gat = GATConv(self.dim, self.dim, dropout=0.1)

        self.self_attn = SelfAttentionLayer(self.dim, self.dim)
        # self.self_attn = SelfAttentionLayer2(self.dim, self.dim)
        # self.bi_attn = BiAttention(self.dim, dropout=0)
        self.output = nn.Linear(self.dim, self.n_entity)
        # kaiming_reset_parameters(self.output)
        # stdv = 1. / math.sqrt(self.output.weight.size(1))
        # nn.init.xavier_normal_(self.output.weight.data, gain=1.414)
        # if self.output.bias is not None:
        #     self.output.bias.data.uniform_(-stdv, stdv)

        self.kg = kg
        # triples = self._get_triples(kg)
        # np.random.shuffle(triples)
        # self.train_triples = triples[:int(len(triples) * 0.95)]
        # self.valid_triples = triples[int(len(triples) * 0.95):]
        # self.train_idx = 0
        # self.valid_idx = 0
        # KG emb as initialization
        # self.entity_emb.weight.data[entity_kg_emb != 0] = entity_kg_emb[entity_kg_emb != 0]
        # self.entity_emb.weight.requires_grad_(False)
        # self.entity_kg_emb.weight.data[entity_kg_emb != 0] = entity_kg_emb[entity_kg_emb != 0]
        # self.entity_kg_emb.weight.requires_grad_(False)
        # self.transform = nn.Sequential(
        #     nn.Linear(self.dim, self.dim),
        #     nn.ReLU(),
        #     nn.Linear(self.dim, self.dim),
        # )

        edge_list, self.n_relation = _edge_list(self.kg, self.n_entity, hop=2)
        self.rgcn = RGCNConv(self.n_entity, self.dim, self.n_relation, num_bases=num_bases)
        edge_list = list(set(edge_list))
        print(len(edge_list), self.n_relation)
        edge_list_tensor = torch.LongTensor(edge_list).cuda()
        # self.adj = torch.sparse.FloatTensor(edge_list_tensor[:, :2].t(), torch.ones(len(edge_list))).cuda()
        # self.edge_idx = self.adj._indices()
        self.edge_idx = edge_list_tensor[:, :2].t()
        self.edge_type = edge_list_tensor[:, 2]

    def _get_triples(self, kg):
        triples = []
        for entity in kg:
            for relation, tail in kg[entity]:
                if entity != tail:
                    triples.append([entity, relation, tail])
                    # triples.append([tail, self.n_relation + relation, entity])
        return triples

    def forward(
        self,
        seed_sets: list,
        labels: torch.LongTensor,
    ):
        # [batch size, dim]
        u_emb, nodes_features = self.user_representation(seed_sets)
        # scores = self.output(u_emb)
        # scores = F.linear(u_emb, self.entity_emb.weight, self.output.bias)
        scores = F.linear(u_emb, nodes_features, self.output.bias)
        # scores = F.linear(u_emb, torch.cat([self.transform(self.entity_text_emb), self.entity_emb.weight], dim=1), self.output.bias)

        base_loss = self.criterion(scores, labels)
        # base_loss = torch.Tensor([0])

        # kge_loss, l2_loss = self.compute_kge_loss()
        kge_loss = torch.Tensor([0])
        l2_loss = torch.Tensor([0])

        loss = base_loss# + kge_loss + l2_loss

        return dict(scores=scores.detach(), base_loss=base_loss, kge_loss=kge_loss, loss=loss, l2_loss=l2_loss)

    def _calc(self, h_re, h_im, r_re, r_im, t_re, t_im):
        return -torch.sum(
                h_re * r_re * t_re + h_im * r_re * t_im
                + h_re * r_im * t_im - h_im * r_im * t_re,
                -1
                )

    def compute_kge_loss(self):
        bs = 4096
        if self.training:
            triples = self.train_triples[self.train_idx:self.train_idx+bs]
            self.train_idx += bs
            if self.train_idx >= len(self.train_triples):
                np.random.shuffle(self.train_triples)
                self.train_idx = 0
        else:
            triples = self.valid_triples[self.valid_idx:self.valid_idx+bs]
            self.valid_idx += bs
            if self.valid_idx >= len(self.valid_triples):
                np.random.shuffle(self.valid_triples)
                self.valid_idx = 0
        triples_tensor = torch.LongTensor(triples).cuda()
        batch_h = triples_tensor[:, 0]
        batch_r = triples_tensor[:, 1]
        batch_t = triples_tensor[:, 2]
        # negative samples
        neg_batch_h = batch_h.clone()
        neg_batch_r = batch_r.clone()
        neg_batch_t = batch_t.clone()
        for i in range(batch_h.shape[0]):
            if np.random.random() < 0.5:
                neg_batch_h[i] = np.random.choice(self.n_entity)
            else:
                neg_batch_t[i] = np.random.choice(self.n_entity)
        ys = torch.cat([torch.ones(batch_h.shape[0]), -torch.ones(batch_h.shape[0])]).cuda()
        batch_h = torch.cat([batch_h, neg_batch_h])
        batch_r = torch.cat([batch_r, neg_batch_r])
        batch_t = torch.cat([batch_t, neg_batch_t])
        kge_loss, l2_loss = self._kge_loss(batch_h, batch_r, batch_t, ys)
        return kge_loss, l2_loss

    def _kge_loss(self, batch_h, batch_r, batch_t, ys):
        h_re, h_im = torch.chunk(self.entity_emb(batch_h), 2, dim=1)
        r_re, r_im = torch.chunk(self.relation_emb(batch_r), 2, dim=1)
        t_re, t_im = torch.chunk(self.entity_emb(batch_t), 2, dim=1)
        score = self._calc(h_re, h_im, r_re, r_im, t_re, t_im)
        regul = (
                torch.mean(h_re ** 2)
                + torch.mean(h_im ** 2)
                + torch.mean(r_re ** 2)
                + torch.mean(r_im ** 2)
                + torch.mean(t_re ** 2)
                + torch.mean(t_im ** 2)
        )
        lmbda = 0.0
        kge_loss = torch.mean(self.kge_criterion(score * ys))
        l2_loss = lmbda * regul
        return kge_loss, l2_loss

    def user_representation(self, seed_sets):
        # find subgraph for this batch
        # g = nx.Graph()
        # for seed_set in seed_sets:
            # for seed in seed_set:
            #     g.add_node(seed)
            # _add_neighbors(self.kg, g, seed_set, hop=2)
        # add self-loops
        # for node in g.nodes:
        #     g.add_edge(node, node)

        # create temporary nodes for user representation readout
        # n_readout_nodes = 0
        # for i, seed_set in enumerate(seed_sets):
        #     if seed_set == []:
        #         continue
        #     n_readout_nodes += 1
        #     for seed in seed_set:
        #         g.add_edge(-i, seed)

        # nodes_list = list(set([seed for seed_set in seed_sets for seed in seed_set]))
        # nodes = torch.LongTensor(nodes_list)
        # to cuda
        # nodes = torch.LongTensor(list(g.nodes))
        # adj = torch.FloatTensor(nx.to_numpy_matrix(g))
        # nodes = nodes.cuda()
        # adj = adj.cuda()

        # nodes_embed = torch.cat([self.entity_emb(nodes[:-n_readout_nodes]), torch.zeros(n_readout_nodes, self.dim).cuda()])
        # nodes_features = torch.cat([self.entity_text_emb[nodes[:-n_readout_nodes]].cuda(), torch.zeros(n_readout_nodes, self.entity_text_emb.shape[1]).cuda()])
        # nodes_features = self.transform(nodes_features)
        # nodes_features = self.entity_emb(nodes[:-n_readout_nodes])
        # nodes_features = self.transform(nodes_embed + nodes_features)
        # nodes_features = self.transform(self.entity_text_emb)
        # nodes_features = self.gcn(self.entity_emb.weight, self.adj)
        # nodes_features = self.entity_emb.weight
        # nodes_features += self.transform(self.entity_text_emb)
        # nodes_features = self.gcn(nodes_features, self.edge_idx)
        # nodes_features = self.gat(nodes_features, self.edge_idx)
        nodes_features = self.rgcn(None, self.edge_idx, self.edge_type)

        # node2id = dict([(n, i) for i, n in enumerate(list(g.nodes))])
        user_representation_list = []
        for i, seed_set in enumerate(seed_sets):
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).cuda())
                continue
            # seed_set_ids = list(map(lambda x: node2id[x], seed_set))
            # user_representation = torch.cat([nodes_features[seed_set_ids], nodes_embed[seed_set_ids]], dim=1)
            user_representation = nodes_features[seed_set]
            user_representation = self.self_attn(user_representation)
            # text_features = self.entity_text_emb[seed_set]
            # print(user_representation.shape, text_features.shape)
            # user_representation = self.bi_attn(user_representation.unsqueeze(0), text_features.unsqueeze(0))
            # user_representation = user_representation.mean(dim=0)
            # user_representation = torch.relu(user_representation)
            # user_representation = nodes_features[node2id[-i]]
            user_representation_list.append(user_representation)
        return torch.stack(user_representation_list), nodes_features

