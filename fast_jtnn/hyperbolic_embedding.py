import torch
import torch.nn as nn
from layers.hyp_layers import LorentzLinear

class HyperbolicEmbedding(nn.Module):
    def __init__(self, in_dim, feat_dim, out_dim, manifold):
        super(HyperbolicEmbedding, self).__init__()

        self.manifold = manifold
        self.embedding = nn.Embedding(in_dim, feat_dim)
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            feat_dim = feat_dim + 1

        self.linear = LorentzLinear(self.manifold, feat_dim, out_dim, dropout=0.0)

    def forward(self, x):
        # print("embed in", x[0])
        x = self.embedding(x)
        # print("embed mid", x[0])
        # print(x.shape)
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)
        # print('bl', self.manifold.check_point_on_manifold(x))
        # print(x.shape)
        x = self.linear(x)
        # if not (self.manifold.check_point_on_manifold(x)):
        #     print(x)
        return x