import torch
from layers.hyp_layers import LorentzLinear
import manifolds

class Generator(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, depth = 2, bias = True, dropout = 0.0):
        super(Generator, self).__init__()
        manifold = manifolds.Lorentz()
        layers = []
        for i in range(depth):
            layers.append(
                LorentzLinear(
                in_features = hid_dim if i != 0 else in_dim, 
                out_features = hid_dim if i != depth - 1 else out_dim,
                manifold = manifold,
                bias = bias,
                dropout = dropout
            ))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)