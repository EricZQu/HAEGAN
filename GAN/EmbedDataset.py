import torch
from torch.utils.data import Dataset, DataLoader

class TreeDataset(Dataset):
    def __init__(self, path = 'MoleculeGeneration/Embed/tree.pt'):
        self.dat = torch.load(path)
        if len(self.dat.size()) == 3:
            self.dat = self.dat.view((self.dat.size(0)*self.dat.size(1), self.dat.size(2)))

    def __len__(self):
            return len(self.dat)

    def __getitem__(self, idx):
            return self.dat[idx]

class GraphDataset(Dataset):
    def __init__(self, path = 'MoleculeGeneration/Embed/graph.pt'):
        self.dat = torch.load(path)
        if len(self.dat.size()) == 3:
            self.dat = self.dat.view((self.dat.size(0)*self.dat.size(1), self.dat.size(2)))

    def __len__(self):
            return len(self.dat)

    def __getitem__(self, idx):
            return self.dat[idx]

class GTDataset(Dataset):
    def __init__(self, gpath = 'MoleculeGeneration/Embed/graph.pt', tpath = 'MoleculeGeneration/Embed/tree.pt'):
        self.gdat = torch.load(gpath)
        self.tdat = torch.load(tpath)
        if len(self.gdat.size()) == 3:
            self.gdat = self.gdat.view((self.gdat.size(0)*self.gdat.size(1), self.gdat.size(2)))
        if len(self.tdat.size()) == 3:
            self.tdat = self.tdat.view((self.tdat.size(0)*self.tdat.size(1), self.tdat.size(2)))

    def __len__(self):
            return len(self.gdat)

    def __getitem__(self, idx):
            return torch.stack([self.tdat[idx], self.gdat[idx]], 0)