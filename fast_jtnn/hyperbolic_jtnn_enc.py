from networkx.readwrite.json_graph import tree
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from collections import deque
from .mol_tree import Vocab, MolTree
from .nnutils import create_var, index_select_ND
from utils.data_utils import DataProcess

from layers.hyp_layers import LorentzGraphConvolution
import manifolds

import networkx as nx

class HyperbolicJTNNEncoder(nn.Module):

    def __init__(self, args, embedding):
        super(HyperbolicJTNNEncoder, self).__init__()
        self.manifold = getattr(manifolds, args.manifold)()
        # if self.manifold.name in ['Lorentz', 'Hyperboloid']:
        #     args.feat_dim = args.feat_dim + 1
        self.hidden_size = args.dim
        self.depth = args.num_layers_tree

        self.embedding = embedding
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

    def forward(self, adj, x, scope):
        x = create_var(x)
        adj = create_var(adj)

        x = self.embedding(x)

        h, adj = self.layers.forward((x, adj))

        st = 0
        batch_vecs = []
        for i in range(len(scope)):
            _, ed = scope[i]
            batch_vecs.append(self.manifold.mid_point(h[st: st + ed]))
            st += ed

        tree_vecs = torch.stack(batch_vecs, dim=0)
        return tree_vecs, h
        # fnode = create_var(fnode)
        # fmess = create_var(fmess)
        # node_graph = create_var(node_graph)
        # mess_graph = create_var(mess_graph)
        # messages = create_var(torch.zeros(mess_graph.size(0), self.hidden_size))

        # fnode = self.embedding(fnode)
        # fmess = index_select_ND(fnode, 0, fmess)
        # messages = self.GRU(messages, fmess, mess_graph)

        # mess_nei = index_select_ND(messages, 0, node_graph)
        # node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=-1)
        # node_vecs = self.outputNN(node_vecs)

        # max_len = max([x for _,x in scope])
        # batch_vecs = []
        # for st,le in scope:
        #     cur_vecs = node_vecs[st] #Root is the first node
        #     batch_vecs.append( cur_vecs )

        # tree_vecs = torch.stack(batch_vecs, dim=0)
        # return tree_vecs, messages

    @staticmethod
    def tensorize(tree_batch):
        node_batch = [] 
        scope = []
        for tree in tree_batch:
            scope.append( (len(node_batch), len(tree.nodes)) )
            node_batch.extend(tree.nodes)

        return HyperbolicJTNNEncoder.tensorize_nodes(node_batch, scope)
        

    @staticmethod
    def tensorize_nodes(node_batch, scope):
        # Convert mol_tree to sparse tensor
        G = nx.Graph()
        for node in node_batch:
            G.add_node(node.idx)
        for node in node_batch:
            for nei in node.neighbors:
                G.add_edge(node.idx, nei.idx)
                G.add_edge(nei.idx, node.idx)
        
        adj = nx.adjacency_matrix(G)

        features = np.zeros(G.number_of_nodes())
        for node in node_batch:
            features[node.idx] = node.wid

        adj, features = DataProcess(adj, features, long_tensor=True)

        return (adj, features, scope), adj
    
    # @staticmethod
    # def tensorize_nodes(node_batch, scope):
    #     messages,mess_dict = [None],{}
    #     fnode = []
    #     for x in node_batch:
    #         fnode.append(x.wid)
    #         for y in x.neighbors:
    #             mess_dict[(x.idx,y.idx)] = len(messages)
    #             messages.append( (x,y) )

    #     node_graph = [[] for i in range(len(node_batch))]
    #     mess_graph = [[] for i in range(len(messages))]
    #     fmess = [0] * len(messages)

    #     for x,y in messages[1:]:
    #         mid1 = mess_dict[(x.idx,y.idx)]
    #         fmess[mid1] = x.idx 
    #         node_graph[y.idx].append(mid1)
    #         for z in y.neighbors:
    #             if z.idx == x.idx: continue
    #             mid2 = mess_dict[(y.idx,z.idx)]
    #             mess_graph[mid2].append(mid1)

    #     max_len = max([len(t) for t in node_graph] + [1])
    #     for t in node_graph:
    #         pad_len = max_len - len(t)
    #         t.extend([0] * pad_len)

    #     max_len = max([len(t) for t in mess_graph] + [1])
    #     for t in mess_graph:
    #         pad_len = max_len - len(t)
    #         t.extend([0] * pad_len)

    #     mess_graph = torch.LongTensor(mess_graph)
    #     node_graph = torch.LongTensor(node_graph)
    #     fmess = torch.LongTensor(fmess)
    #     fnode = torch.LongTensor(fnode)
    #     return (fnode, fmess, node_graph, mess_graph, scope), mess_dict

class GraphGRU(nn.Module):

    def __init__(self, input_size, hidden_size, depth):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h, x, mess_graph):
        mask = torch.ones(h.size(0), 1)
        mask[0] = 0 #first vector is padding
        mask = create_var(mask)
        for it in range(self.depth):
            h_nei = index_select_ND(h, 0, mess_graph)
            sum_h = h_nei.sum(dim=1)
            z_input = torch.cat([x, sum_h], dim=1)
            z = F.sigmoid(self.W_z(z_input))

            r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
            r_2 = self.U_r(h_nei)
            r = F.sigmoid(r_1 + r_2)
            
            gated_h = r * h_nei
            sum_gated_h = gated_h.sum(dim=1)
            h_input = torch.cat([x, sum_gated_h], dim=1)
            pre_h = F.tanh(self.W_h(h_input))
            h = (1.0 - z) * sum_h + z * pre_h
            h = h * mask

        return h
