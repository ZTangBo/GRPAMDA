import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, G, in_dim, out_dim):
        super(GATLayer, self).__init__()

        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.G = G
        self.slope = 0.2
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], in_dim, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], in_dim, bias=False)
        self.dropout = nn.Dropout(0.5)
        # self.attn_fc = nn.Linear(feature_attn_size * 2, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # print('SRC size:', edges.src['z'].size())
        # print('DST size: ', edges.dst['z'].size())
        # z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        # a = self.attn_fc(z2)
        # return {'e': a}
        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1)
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'h': F.elu(h)}

    def forward(self, h):
        z = self.fc(h)
        self.G.ndata['z'] = z

        self.G.apply_edges(self.edge_attention)
        self.G.update_all(self.message_func, self.reduce_func)

        return self.G.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, G, feature_attn_size, num_heads, dropout, slope, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()

        self.G = G
        self.dropout = dropout
        self.slope = slope
        self.merge = merge

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(G, feature_attn_size, dropout, slope))

    def forward(self, G):
        head_outs = [attn_head(G) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs), dim=0)

class HAN_metapath_specific(nn.Module):
    def __init__(self, G, feature_attn_size, out_dim, dropout, slope):
        super(HAN_metapath_specific, self).__init__()

        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)
        self.lncrna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 2)
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)

        self.G = G
        self.slope = slope

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], feature_attn_size, bias=False)
        self.l_fc = nn.Linear(G.ndata['l_sim'].shape[1], feature_attn_size, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], feature_attn_size, bias=False)

        self.m_fc1 = nn.Linear(feature_attn_size + 495, out_dim)   # 设置全连接层
        self.d_fc1 = nn.Linear(feature_attn_size + 383, out_dim)
        self.attn_fc = nn.Linear(feature_attn_size * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.l_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)

    def edge_attention(self, edges):
        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1)
        '''z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)'''
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'h': F.elu(h)}

    def forward(self, new_g, meta_path):

        if meta_path == 'ml':
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.l_fc(nodes.data['l_sim']))}, self.lncrna_nodes)
            new_g.apply_edges(self.edge_attention)
            new_g.update_all(self.message_func, self.reduce_func)

            h_ml = new_g.ndata.pop('h')

            return h_ml

        elif meta_path == 'dl':
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.l_fc(nodes.data['l_sim']))}, self.lncrna_nodes)
            new_g.apply_edges(self.edge_attention)
            new_g.update_all(self.message_func, self.reduce_func)

            h_dl = new_g.ndata.pop('h')

            return h_dl

