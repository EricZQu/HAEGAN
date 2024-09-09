import torch
import torch.nn as nn
import torch.nn.functional as F
from .nnutils import create_var, index_select_ND
from .chemutils import get_mol
import rdkit.Chem as Chem
import networkx as nx
from utils.data_utils import DataProcess
import manifolds

from layers.hyp_layers import LorentzGraphConvolution, LorentzLinear

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5 
MAX_NB = 15

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    return torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()])

class HyperbolicJTMPN(nn.Module):

    def __init__(self, args):
        super(HyperbolicJTMPN, self).__init__()
        self.manifold = getattr(manifolds, args.manifold)()
        # self.in_feat = args.dim
        self.in_feat = ATOM_FDIM
        # self.in_feat = ATOM_FDIM + BOND_FDIM
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            self.in_feat += 1
        self.hidden_size = args.dim
        self.depth = args.num_layers_graph

        # self.Elinear = nn.Linear(ATOM_FDIM, self.hidden_size)

        self.Hlinear = LorentzLinear(
            in_features = self.in_feat,
            out_features = self.hidden_size,
            manifold = self.manifold,
            bias = args.bias,
            dropout = args.dropout,
        )

        hgc_layers = []
        if not args.act:
            act = lambda x: x
        else:
            act = getattr(F, args.act)
        for i in range(self.depth):
            hgc_layers.append(
                LorentzGraphConvolution(
                    manifold = self.manifold, 
                    in_features = self.hidden_size, 
                    out_features = self.hidden_size, 
                    use_bias = args.bias, 
                    dropout = args.dropout, 
                    use_att = args.use_att, 
                    local_agg = args.local_agg, 
                    nonlin = act if i != 0 else None
            ))
        self.layers = nn.Sequential(*hgc_layers)

        # self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        # self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, adj, graph_features, tree_features, scope): 
        adj = create_var(adj)
        graph_features = create_var(graph_features)
        tree_features = create_var(tree_features)

        # graph_features = self.Elinear(graph_features)

        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(graph_features)
            graph_features = torch.cat([o[:, 0:1], graph_features], dim=1)
            if self.manifold.name == 'Lorentz':
                graph_features = self.manifold.expmap0(graph_features)

        graph_features = self.Hlinear(graph_features)

        x = torch.cat([tree_features, graph_features], dim = 0)

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
        # graph_message = F.relu(binput)

        # for i in range(self.depth - 1):
        #     message = torch.cat([tree_message,graph_message], dim=0) 
        #     nei_message = index_select_ND(message, 0, bgraph)
        #     nei_message = nei_message.sum(dim=1) #assuming tree_message[0] == vec(0)
        #     nei_message = self.W_h(nei_message)
        #     graph_message = F.relu(binput + nei_message)

        # message = torch.cat([tree_message,graph_message], dim=0)
        # nei_message = index_select_ND(message, 0, agraph)
        # nei_message = nei_message.sum(dim=1)
        # ainput = torch.cat([fatoms, nei_message], dim=1)
        # atom_hiddens = F.relu(self.W_o(ainput))
        
        # mol_vecs = []
        # for st,le in scope:
        #     mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
        #     mol_vecs.append(mol_vec)

        # mol_vecs = torch.stack(mol_vecs, dim=0)
        # return mol_vecs

    # This is only using ATOM information
    @staticmethod
    def tensorize(cand_batch, adj):
        features = []
        cnt = adj.shape[0]
        scope = []
        G = nx.Graph()
        for i in range(cnt):
            G.add_node(i)
        for i in range(cnt):
            for j in range(cnt):
                if adj[i][j] != 0:
                    G.add_edge(i, j)

        for smiles, all_nodes, ctr_node in cand_batch:
            mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(mol) 
            n_atoms = mol.GetNumAtoms()
            offset = cnt
            
            for atom in mol.GetAtoms():
                features.append( atom_features(atom) )
                G.add_node(cnt)
                cnt += 1

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + offset
                y = a2.GetIdx() + offset
                #Here x_nid,y_nid could be 0
                x_nid,y_nid = a1.GetAtomMapNum(),a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                G.add_edge(x, y)
                G.add_edge(y, x)

                # bfeature = bond_features(bond)

                # b = total_mess + len(all_bonds)  #bond idx offseted by total_mess
                # all_bonds.append((x,y))
                # fbonds.append( torch.cat([fatoms[x], bfeature], 0) )
                # in_bonds[y].append(b)

                # b = total_mess + len(all_bonds)
                # all_bonds.append((y,x))
                # fbonds.append( torch.cat([fatoms[y], bfeature], 0) )
                # in_bonds[x].append(b)

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if adj[x_bid][y_bid] != 0:
                        G.add_edge(x_bid, y)
                    if adj[y_bid][x_bid] != 0:
                        G.add_edge(y_bid, x)
                    # if (x_bid,y_bid) in mess_dict:
                    #     mess_idx = mess_dict[(x_bid,y_bid)]
                    #     in_bonds[y].append(mess_idx)
                    # if (y_bid,x_bid) in mess_dict:
                    #     mess_idx = mess_dict[(y_bid,x_bid)]
                    #     in_bonds[x].append(mess_idx)
            
            scope.append((offset, n_atoms))

        features = torch.stack(features, dim = 0)
        adj = nx.adjacency_matrix(G)

        adj, features = DataProcess(adj, features)

        return (adj, features, scope)

    # Use Bond and Atom information
    @staticmethod
    def tensorize_bond(cand_batch, adj):
        cnt = adj.shape[0]
        scope = []
        G = nx.Graph()
        fatoms,fbonds = [],[] 
        in_bonds,all_bonds = [],[] 
        total_atoms = 0
        total_mess = adj.shape[0]
        scope = []
        for i in range(cnt):
            for j in range(cnt):
                if adj[i][j] != 0:
                    G.add_edge(i, j)

        for smiles,all_nodes,ctr_node in cand_batch:
            mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(mol) #The original jtnn version kekulizes. Need to revisit why it is necessary
            n_atoms = mol.GetNumAtoms()
            ctr_bid = ctr_node.idx

            for atom in mol.GetAtoms():
                fatoms.append( atom_features(atom) )
                in_bonds.append([]) 
        
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms
                #Here x_nid,y_nid could be 0
                x_nid,y_nid = a1.GetAtomMapNum(),a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                bfeature = bond_features(bond)

                b = total_mess + len(all_bonds)  #bond idx offseted by total_mess
                all_bonds.append((x,y))
                fbonds.append( torch.cat([fatoms[x], bfeature], 0) )
                in_bonds[y].append(b)

                b = total_mess + len(all_bonds)
                all_bonds.append((y,x))
                fbonds.append( torch.cat([fatoms[y], bfeature], 0) )
                in_bonds[x].append(b)

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if adj[x_bid][y_bid] != 0:
                        G.add_edge(y, x_bid)
                        # in_bonds[y].append(x_bid)
                    if adj[y_bid][x_bid] != 0:
                        # in_bonds[x].append(y_bid)
                        G.add_edge(x, y_bid)
                    # if (x_bid,y_bid) in mess_dict:
                    #     mess_idx = mess_dict[(x_bid,y_bid)]
                    #     in_bonds[y].append(mess_idx)
                    # if (y_bid,x_bid) in mess_dict:
                    #     mess_idx = mess_dict[(y_bid,x_bid)]
                    #     in_bonds[x].append(mess_idx)
            
            scope.append((total_atoms,n_atoms))
            total_atoms += n_atoms
        
        total_bonds = len(all_bonds)
        # fatoms = torch.stack(fatoms, 0)
        features = torch.stack(fbonds, 0)
        # agraph = torch.zeros(total_atoms,MAX_NB).long()
        # bgraph = torch.zeros(total_bonds,MAX_NB).long()

        # for a in range(total_atoms):
        #     for i,b in enumerate(in_bonds[a]):
        #         agraph[a,i] = b
        G.add_nodes_from(range(total_mess + total_bonds))
        for b1 in range(total_bonds):
            x,y = all_bonds[b1]
            for i,b2 in enumerate(in_bonds[x]): #b2 is offseted by total_mess
                if b2 < total_mess or all_bonds[b2-total_mess][0] != y:
                    # bgraph[b1,i] = b2
                    G.add_edge(b1, b2)

        adj = nx.adjacency_matrix(G)

        adj, features = DataProcess(adj, features)

        return (adj, features, scope)
       
    # @staticmethod
    # def tensorize(cand_batch, mess_dict):
    #     fatoms,fbonds = [],[] 
    #     in_bonds,all_bonds = [],[] 
    #     total_atoms = 0
    #     total_mess = len(mess_dict) + 1 #must include vec(0) padding
    #     scope = []

    #     for smiles,all_nodes,ctr_node in cand_batch:
    #         mol = Chem.MolFromSmiles(smiles)
    #         Chem.Kekulize(mol) #The original jtnn version kekulizes. Need to revisit why it is necessary
    #         n_atoms = mol.GetNumAtoms()
    #         ctr_bid = ctr_node.idx

    #         for atom in mol.GetAtoms():
    #             fatoms.append( atom_features(atom) )
    #             in_bonds.append([]) 
        
    #         for bond in mol.GetBonds():
    #             a1 = bond.GetBeginAtom()
    #             a2 = bond.GetEndAtom()
    #             x = a1.GetIdx() + total_atoms
    #             y = a2.GetIdx() + total_atoms
    #             #Here x_nid,y_nid could be 0
    #             x_nid,y_nid = a1.GetAtomMapNum(),a2.GetAtomMapNum()
    #             x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
    #             y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

    #             bfeature = bond_features(bond)

    #             b = total_mess + len(all_bonds)  #bond idx offseted by total_mess
    #             all_bonds.append((x,y))
    #             fbonds.append( torch.cat([fatoms[x], bfeature], 0) )
    #             in_bonds[y].append(b)

    #             b = total_mess + len(all_bonds)
    #             all_bonds.append((y,x))
    #             fbonds.append( torch.cat([fatoms[y], bfeature], 0) )
    #             in_bonds[x].append(b)

    #             if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
    #                 if (x_bid,y_bid) in mess_dict:
    #                     mess_idx = mess_dict[(x_bid,y_bid)]
    #                     in_bonds[y].append(mess_idx)
    #                 if (y_bid,x_bid) in mess_dict:
    #                     mess_idx = mess_dict[(y_bid,x_bid)]
    #                     in_bonds[x].append(mess_idx)
            
    #         scope.append((total_atoms,n_atoms))
    #         total_atoms += n_atoms
        
    #     total_bonds = len(all_bonds)
    #     fatoms = torch.stack(fatoms, 0)
    #     fbonds = torch.stack(fbonds, 0)
    #     agraph = torch.zeros(total_atoms,MAX_NB).long()
    #     bgraph = torch.zeros(total_bonds,MAX_NB).long()

    #     for a in range(total_atoms):
    #         for i,b in enumerate(in_bonds[a]):
    #             agraph[a,i] = b

    #     for b1 in range(total_bonds):
    #         x,y = all_bonds[b1]
    #         for i,b2 in enumerate(in_bonds[x]): #b2 is offseted by total_mess
    #             if b2 < total_mess or all_bonds[b2-total_mess][0] != y:
    #                 bgraph[b1,i] = b2

    #     return (fatoms, fbonds, agraph, bgraph, scope)

