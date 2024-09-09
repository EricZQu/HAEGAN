import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from .nnutils import *
from .chemutils import get_mol
from utils.data_utils import DataProcess

from layers.hyp_layers import LorentzGraphConvolution, LorentzLinear
import manifolds

import networkx as nx
import numpy as np
import scipy.sparse as sp

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

class HyperbolicMPN(nn.Module):

    def __init__(self, args):
        super(HyperbolicMPN, self).__init__()
        self.manifold = getattr(manifolds, args.manifold)()
        self.in_feat = ATOM_FDIM
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            self.in_feat += 1
        self.hidden_size = args.dim
        self.depth = args.num_layers_graph

        hgc_layers = []
        if not args.act:
            act = lambda x: x
        else:
            act = getattr(F, args.act)
        for i in range(self.depth):
            hgc_layers.append(
                LorentzGraphConvolution(
                    manifold = self.manifold, 
                    # in_features = self.hidden_size,
                    in_features = self.hidden_size if i != 0 else self.in_feat,
                    out_features = self.hidden_size, 
                    use_bias = args.bias, 
                    dropout = args.dropout, 
                    use_att = args.use_att, 
                    local_agg = args.local_agg, 
                    nonlin = act if i != 0 else None
            ))
        self.layers = nn.Sequential(*hgc_layers)

        # self.Elinear = nn.Linear(ATOM_FDIM, self.hidden_size)
        # self.Hlinear = LorentzLinear(
        #     in_features = self.hidden_size + 1,
        #     out_features = self.hidden_size,
        #     manifold = self.manifold,
        #     bias = args.bias,
        #     dropout = args.dropout,
        # )

    def forward(self, adj, x, scope):
        x = create_var(x)
        adj = create_var(adj)

        # x = self.Elinear(x)

        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)

        # x = self.Hlinear(x)

        h, adj = self.layers.forward((x, adj))

        batch_vecs = []
        for st, le in scope:
            batch_vecs.append(self.manifold.mid_point(h[st: st + le]))

        mol_vecs = torch.stack(batch_vecs, dim=0)
        return mol_vecs

        # fatoms = create_var(fatoms)
        # fbonds = create_var(fbonds)
        # agraph = create_var(agraph)
        # bgraph = create_var(bgraph)

        # binput = self.W_i(fbonds)
        # message = F.relu(binput)

        # for i in range(self.depth - 1):
        #     nei_message = index_select_ND(message, 0, bgraph)
        #     nei_message = nei_message.sum(dim=1)
        #     nei_message = self.W_h(nei_message)
        #     message = F.relu(binput + nei_message)

        # nei_message = index_select_ND(message, 0, agraph)
        # nei_message = nei_message.sum(dim=1)
        # ainput = torch.cat([fatoms, nei_message], dim=1)
        # atom_hiddens = F.relu(self.W_o(ainput))

        # max_len = max([x for _,x in scope])
        # batch_vecs = []
        # for st,le in scope:
        #     cur_vecs = atom_hiddens[st : st + le].mean(dim=0)
        #     batch_vecs.append( cur_vecs )

        # mol_vecs = torch.stack(batch_vecs, dim=0)
        # return mol_vecs 

    @staticmethod
    def tensorize(mol_batch):
        padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
        fatoms,fbonds = [],[padding] #Ensure bond is 1-indexed
        in_bonds,all_bonds = [],[(-1,-1)] #Ensure bond is 1-indexed
        scope = []
        total_atoms = 0

        for smiles in mol_batch:
            mol = get_mol(smiles)
            #mol = Chem.MolFromSmiles(smiles)
            n_atoms = mol.GetNumAtoms()
            for atom in mol.GetAtoms():
                fatoms.append( atom_features(atom) )
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms

                b = len(all_bonds) 
                all_bonds.append((x,y))
                fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
                in_bonds[y].append(b)

                b = len(all_bonds)
                all_bonds.append((y,x))
                fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
                in_bonds[x].append(b)
            
            scope.append((total_atoms,n_atoms))
            total_atoms += n_atoms

        features = torch.stack(fatoms, 0)

        G = nx.Graph()

        for i in range(total_atoms):
            G.add_node(i)

        for i in range(1, len(all_bonds)):
            x, y = all_bonds[i]
            G.add_edge(x, y)

        adj = nx.adjacency_matrix(G)

        adj, features = DataProcess(adj, features)

        return (adj, features, scope)

        # total_bonds = len(all_bonds)
        # fatoms = torch.stack(fatoms, 0)
        # fbonds = torch.stack(fbonds, 0)
        # agraph = torch.zeros(total_atoms,MAX_NB).long()
        # bgraph = torch.zeros(total_bonds,MAX_NB).long()

        # for a in range(total_atoms):
        #     for i,b in enumerate(in_bonds[a]):
        #         agraph[a,i] = b

        # for b1 in range(1, total_bonds):
        #     x,y = all_bonds[b1]
        #     for i,b2 in enumerate(in_bonds[x]):
        #         if all_bonds[b2][0] != y:
        #             bgraph[b1,i] = b2

        # return (fatoms, fbonds, agraph, bgraph, scope)

