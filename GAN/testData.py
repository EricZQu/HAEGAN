import torch
from torch.utils.data import Dataset, DataLoader
import manifolds
from distributions.Lorentz_wrapped_normal import LorentzWrappedNormal

from GAN.Generator import Generator

# We here use the midpoint of some Wrapped Normal distributions (which is still a Gaussian)

class TestDataset(Dataset):
    def __init__(self, manifold, num, dim, N, std = 5):
        self.manifold = manifold
        self.dat = []
        self.mu = []
        self.dist = []
        # dim *= 2
        for _ in range(num):
            self.mu.append(self.manifold.random_normal(dim, std = std))
            # self.dist.append(LorentzWrappedNormal(self.mu[-1], std, self.manifold, dim))
            self.dist.append(LorentzWrappedNormal(self.manifold.origin(dim), std, self.manifold, dim))
            self.dat.append(self.dist[-1].sample((1, N))[0])
        self.dat = torch.stack(self.dat).permute((1, 0, 2))
        self.dat = self.manifold.mid_point(self.dat)

        # Net = Generator(dim, dim, dim // 2, 1)
        # with torch.no_grad():
        #     self.dat = Net(self.dat)

    def __len__(self):
            return len(self.dat)

    def __getitem__(self, idx):
            return self.dat[idx]