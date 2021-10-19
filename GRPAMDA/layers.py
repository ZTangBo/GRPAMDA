import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        # z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        # a = self.attn_fc(z2)
        # return {'e': F.leaky_relu(a)}

        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1)
        return {'e': F.leaky_relu(a, negative_slope=0.2)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


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