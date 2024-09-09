import torch
from layers.hyp_layers import LorentzLinear, LorentzCentroidDistance
import manifolds

class Critic(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, depth = 2, bias = True, dropout = 0.0):
        super(Critic, self).__init__()
        manifold = manifolds.Lorentz()
        layers = []
        for i in range(depth - 1):
            layers.append(
                LorentzLinear(
                in_features = hid_dim if i != 0 else in_dim, 
                out_features = hid_dim,
                manifold = manifold,
                bias = bias,
                dropout = dropout
            ))
        layers.append(
            LorentzCentroidDistance(
                dim = hid_dim, 
                n_classes = 1,
                manifold = manifold,
                bias = bias
            )
        )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)