import numpy as np
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

from layers1 import GATLayer

# GAT
def drop_node(feats, drop_rate, training):
    n = feats.shape[0]
    drop_rates = th.FloatTensor(np.ones(n) * drop_rate)

    if training:

        masks = th.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats

    else:
        feats = feats * (1. - drop_rate)

    return feats


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.input_dropout = nn.Dropout(input_droprate)
        self.hidden_dropout = nn.Dropout(hidden_droprate)
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):

        if self.use_bn:
            x = self.bn1(x)
        x = self.input_dropout(x)
        x = F.relu(self.layer1(x))

        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_dropout(x)
        x = self.layer2(x)

        return x


def GRANDConv(graph, feats, order):
    with graph.local_scope():
        ''' Calculate Symmetric normalized adjacency matrix   \hat{A} '''
        degs = graph.in_degrees().float().clamp(min=1)
        norm = th.pow(degs, -0.5).to(feats.device).unsqueeze(1)

        graph.ndata['norm'] = norm
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))

        ''' Graph Conv '''
        x = feats
        y = 0 + feats

        for i in range(order):
            graph.ndata['h'] = x
            graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y.add_(x)

    return y / (order + 1)


class GRAND(nn.Module):
    def __init__(self, G, hid_dim, n_class, S, K, batchnorm, num_diseases, num_mirnas,
                 d_sim_dim, m_sim_dim, out_dim, dropout, slope, node_dropout=0.5, input_droprate=0.0,
                 hidden_droprate=0.0):
        super(GRAND, self).__init__()
        self.G = G
        self.hid_dim = hid_dim
        self.S = S
        self.K = K
        self.n_class = n_class
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], hid_dim, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], hid_dim, bias=False)
        self.m_fc1 = nn.Linear(128, out_dim)
        self.d_fc1 = nn.Linear(128, out_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.mlp = MLP(hid_dim, out_dim, n_class, input_droprate, hidden_droprate, batchnorm)
        self.gat = GATLayer(G, hid_dim, out_dim)

        self.dropout = node_dropout
        self.node_dropout = nn.Dropout(node_dropout)
        self.predict = nn.Linear(out_dim * 2, 1)
        self.InnerProductDecoder = InnerProductDecoder()

    def forward(self, graph,  diseases, mirnas,  training=True):

        self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
        feats = self.G.ndata.pop('z')

        X = feats
        S = self.S

        if training:  # Training Mode
            feat0 = []
            drop_feat = drop_node(X, self.dropout, True)  # Drop node
            feat0 = GRANDConv(graph, drop_feat, self.K)
            #feat0 = th.log_softmax(self.gat(feat0), dim=-1)
            feat0 = self.gat(feat0)

            for s in range(S-1):
                drop_feat = drop_node(X, self.dropout, True)  # Drop node
                feat = GRANDConv(graph, drop_feat, self.K)  # Graph Convolution
                feat = th.log_softmax(self.gat(feat), dim=-1)
                # output_list.append(th.log_softmax(self.mlp(feat), dim=-1))  # Prediction
                feat1 = th.cat(feat0, feat)
                feat0 = feat1

            h_d = th.cat((feat0[:self.num_diseases], feats[:self.num_diseases]), dim=1)
            h_m = th.cat((feat0[self.num_diseases:], feats[self.num_diseases:878]), dim=1)

            h_m = self.dropout1(F.elu(self.m_fc1(h_m)))  # （383,64）
            h_d = self.dropout1(F.elu(self.d_fc1(h_d)))
            h = th.cat((h_d, h_m), dim=0)

            h_diseases = h[diseases]  # disease中有重复的疾病名称;(17376,64)
            h_mirnas = h[mirnas]

            h_concat = th.cat((h_diseases, h_mirnas), 1)  # (17376,128)
            predict_score = th.sigmoid(self.predict(h_concat))

            # predict_score = self.InnerProductDecoder(h_diseases, h_mirnas)
            return predict_score
        else:  # Inference Mode
            drop_feat = drop_node(X, self.dropout, False)
            X = GRANDConv(graph, drop_feat, self.K)
            feat0 = th.log_softmax(self.gat(X), dim=-1)
            h_d = th.cat((feat0[:self.num_diseases], feats[:self.num_diseases]), dim=1)
            h_m = th.cat((feat0[self.num_diseases:], feats[self.num_diseases:878]), dim=1)

            h_m = self.dropout1(F.elu(self.m_fc1(h_m)))  # （383,64）
            h_d = self.dropout1(F.elu(self.d_fc1(h_d)))
            h = th.cat((h_d, h_m), dim=0)

            h_diseases = h[diseases]  # disease中有重复的疾病名称;(17376,64)
            h_mirnas = h[mirnas]

            h_concat = th.cat((h_diseases, h_mirnas), 1)  # (17376,128)
            predict_score = th.sigmoid(self.predict(h_concat))

            #predict_score = self.InnerProductDecoder(h_diseases, h_mirnas)
            return predict_score

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, h_diseases, h_mirnas):
        x = th.mul(h_diseases, h_mirnas).sum(1)
        x = th.reshape(x, [-1])
        outputs = F.sigmoid(x)

        return outputs
