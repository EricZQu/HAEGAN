import torch
import torch.nn as nn
import torch.nn.functional as F
from .mol_tree import Vocab, MolTree, MolTreeNode
from .nnutils import create_var, GRU
from .chemutils import enum_assemble, set_atommap
from layers.hyp_layers import LorentzMultiHeadedAttention, LorentzLinear, LorentzCentroidDistance
import manifolds
import copy

MAX_NB = 15
MAX_DECODE_LEN = 100

class HyperbolicJTNNDecoder(nn.Module):

    def __init__(self, args, vocab, embedding):
        super(HyperbolicJTNNDecoder, self).__init__()
        self.manifold = getattr(manifolds, args.manifold)()
        self.hidden_size = args.dim
        self.latent_size = args.latent_dim
        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = embedding

        #Attention
        # self.attnet = LorentzMultiHeadedAttention(args.head_count, self.hidden_size, self.manifold)

        # #GRU Weights
        # self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        # self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.W_r = nn.Linear(hidden_size, hidden_size)
        # self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        self.A0 = LorentzLinear(
                in_features = self.hidden_size, 
                out_features = self.hidden_size,
                manifold = self.manifold,
                bias = args.bias,
                dropout = args.dropout
            )
        self.A1 = LorentzLinear(
                in_features = self.hidden_size * 2 - 1, 
                out_features = self.hidden_size,
                manifold = self.manifold,
                bias = args.bias,
                dropout = args.dropout
            )

        #Word Prediction Weights 
        # self.W = LorentzLinear(
        #         in_features = self.hidden_size + self.latent_size - 1, 
        #         out_features = self.hidden_size,
        #         manifold = self.manifold,
        #         bias = args.bias,
        #         dropout = args.dropout
        #     )
        self.W_o = LorentzCentroidDistance(
                dim = self.hidden_size + self.latent_size - 1, 
                n_classes = self.vocab.size(),
                manifold = self.manifold,
                bias = args.bias
            )
        
        # self.W_h = nn.Sequential(
        #     LorentzLinear(
        #         in_features = self.hidden_size, 
        #         out_features = self.hidden_size,
        #         manifold = self.manifold,
        #         bias = args.bias,
        #         dropout = args.dropout
        #     ),
        #     LorentzCentroidDistance(
        #         dim = self.hidden_size, 
        #         n_classes = self.hidden_size, 
        #         manifold = self.manifold, 
        #         bias = args.bias
        #     )
        # )
        # self.W_t = nn.Sequential(
        #     LorentzLinear(
        #         in_features = self.latent_size, 
        #         out_features = self.latent_size,
        #         manifold = self.manifold,
        #         bias = args.bias,
        #         dropout = args.dropout
        #     ),
        #     LorentzCentroidDistance(
        #         dim = self.latent_size, 
        #         n_classes = self.latent_size, 
        #         manifold = self.manifold, 
        #         bias = args.bias
        #     )
        # )

        #Stop Prediction Weights
        # self.U_i = LorentzLinear(
        #         in_features = self.hidden_size * 2 - 1, 
        #         out_features = self.hidden_size,
        #         manifold = self.manifold,
        #         bias = args.bias,
        #         dropout = args.dropout
        #     )
        # self.U = LorentzLinear(
        #         in_features = self.hidden_size * 2 + self.latent_size - 2, 
        #         out_features = self.hidden_size,
        #         manifold = self.manifold,
        #         bias = args.bias,
        #         dropout = args.dropout
        #     )
        self.U_o = LorentzCentroidDistance(
                dim = self.hidden_size * 2 + self.latent_size - 2, 
                n_classes = 2,
                manifold = self.manifold,
                bias = args.bias
            )
        # self.U_x = nn.Sequential(
        #     LorentzLinear(
        #         in_features = self.hidden_size, 
        #         out_features = self.hidden_size,
        #         manifold = self.manifold,
        #         bias = args.bias,
        #         dropout = args.dropout
        #     ),
        #     LorentzCentroidDistance(
        #         dim = self.hidden_size, 
        #         n_classes = self.hidden_size, 
        #         manifold = self.manifold, 
        #         bias = args.bias
        #     )
        # )
        # self.U_h = nn.Sequential(
        #     LorentzLinear(
        #         in_features = self.hidden_size, 
        #         out_features = self.hidden_size,
        #         manifold = self.manifold,
        #         bias = args.bias,
        #         dropout = args.dropout
        #     ),
        #     LorentzCentroidDistance(
        #         dim = self.hidden_size, 
        #         n_classes = self.hidden_size, 
        #         manifold = self.manifold, 
        #         bias = args.bias
        #     )
        # )
        # self.U_t = nn.Sequential(
        #     LorentzLinear(
        #         in_features = self.latent_size, 
        #         out_features = self.latent_size,
        #         manifold = self.manifold,
        #         bias = args.bias,
        #         dropout = args.dropout
        #     ),
        #     LorentzCentroidDistance(
        #         dim = self.latent_size, 
        #         n_classes = self.latent_size, 
        #         manifold = self.manifold, 
        #         bias = args.bias
        #     )
        # )
        # self.U_i = LorentzLinear(
        #     in_features = 2 * self.hidden_size,
        #     out_features = self.hidden_size,
        #     manifold = self.manifold,
        #     bias = args.bias,
        #     dropout = args.droupout)

        #Output Weights
        # self.W_o = nn.Linear(self.hidden_size + self.latent_size, self.vocab.size())
        # self.U_o = nn.Linear(2 * self.hidden_size + self.latent_size, 1)

        #Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stop_loss = nn.CrossEntropyLoss(reduction='sum')
        # self.stop_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def aggregate(self, hiddens, contexts, x_tree_vecs, mode):
        if mode == 'word':
            tree_contexts = x_tree_vecs.index_select(0, contexts)
            # print('word', hiddens.size(), tree_contexts.size())
            h = self.manifold.Concat(hiddens, tree_contexts)
            # print('word', h.size())
            # h1 = self.W_h(hiddens)
            # h2 = self.W_t(tree_contexts)
            # input_vec = torch.relu(torch.cat([h1, h2], dim=-1))
            # h = self.W(h)
            # print('word', h[0])
            return self.W_o(h)
        elif mode == 'stop':
            tree_contexts = x_tree_vecs.index_select(0, contexts)
            x, h = hiddens
            # h1 = self.U_x(x)
            # h2 = self.U_h(h)
            # h3 = self.U_t(tree_contexts)
            # input_vec = torch.relu(torch.cat([h1, h2, h3], dim=-1))
            # print(x.size(), h.size())
            # print('stop', x.size(), h.size(), tree_contexts.size())
            # print('stop in', x[0], h[0])
            h = self.manifold.Concat(x, h)
            # h = self.U_i(h)
            h = self.manifold.Concat(h, tree_contexts)
            # h = self.U(h)
            # print('stop', h.size())
            # print('stop out', h[0])
            return self.U_o(h)
        else:
            raise ValueError('aggregate mode is wrong')

    def forward(self, mol_batch, x_tree_vecs):
        pred_hiddens,pred_contexts,pred_targets = [],[],[]
        stop_hiddens_x, stop_hiddens_o, stop_contexts,stop_targets = [],[],[],[]
        traces = []
        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], -1)
            traces.append(s)
            for node in mol_tree.nodes:
                node.neighbors = []

        #Predict Root
        batch_size = len(mol_batch)
        pred_hiddens.append(create_var(self.manifold.origin(batch_size,self.hidden_size)))
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
        pred_contexts.append( create_var( torch.LongTensor(list(range(batch_size))) ) )

        max_iter = max([len(tr) for tr in traces])
        padding = create_var(self.manifold.origin(self.hidden_size), False)
        h = {}

        for t in range(max_iter):
            prop_list = []
            batch_list = []
            for i,plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x = []
            cur_h_nei,cur_o_nei = [],[]
            scope_o, scope_h = [0], [0]

            for node_x, real_y, _ in prop_list:
                #Neighbors for message passing (target not included)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
                if len(cur_nei) == 0:
                    cur_h_nei.append(padding)
                else:
                    cur_h_nei.append(torch.stack(cur_nei, dim=0))

                #Neighbors for stop prediction (all neighbors)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
                if len(cur_nei) == 0:
                    cur_o_nei.append(padding)
                else:
                    cur_o_nei.append(torch.stack(cur_nei, dim=0))

                #Current clique embedding
                cur_x.append(node_x.wid)

            #Clique embedding
            cur_x = create_var(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x) 

            #Message passing
            new_h = []
            for i, nei in enumerate(cur_h_nei): #TODO: Use mask to parallel this
                if len(nei) == 1 or len(nei) == self.hidden_size:
                    h1 = nei.view(1, self.hidden_size)
                else:
                    h1 = nei.view(len(nei), self.hidden_size)
                h2 = cur_x[i].view(1, self.hidden_size)
                # h0 = torch.cat([h2, h1], dim = -2)
                h1 = self.A0(h1)
                h1 = self.manifold.mid_point(h1).view(1, self.hidden_size)
                h0 = self.manifold.Concat(h2, h1)
                h0 = self.A1(h0)
                new_h.append(h0.view(self.hidden_size))
                # new_h.append(self.attnet(
                #     key = h1, 
                #     value = h1, 
                #     query = h2
                #     ).view(self.hidden_size))
            new_h = torch.stack(new_h, dim=0)

            #Node Aggregate
            cur_o = []
            for nei in cur_o_nei:
                if len(nei.size()) == 1:
                    nei = nei.view(1, nei.size()[0])
                cur_o.append(self.manifold.mid_point(nei))
            cur_o = torch.stack(cur_o, dim=0).view(-1,self.hidden_size)

            #Gather targets
            pred_target,pred_list = [],[]
            stop_target = []
            for i,m in enumerate(prop_list):
                node_x,node_y,direction = m
                x,y = node_x.idx,node_y.idx
                h[(x,y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i) 
                stop_target.append(direction)

            # print('test')
            #Hidden states for stop prediction
            cur_batch = create_var(torch.LongTensor(batch_list))
            # stop_hidden = torch.cat([cur_x,cur_o], dim=1)
            stop_hiddens_x.append( cur_x )
            stop_hiddens_o.append( cur_o )
            stop_contexts.append( cur_batch )
            stop_targets.extend( stop_target )
            
            #Hidden states for clique prediction
            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = create_var(torch.LongTensor(batch_list))
                pred_contexts.append( cur_batch )

                cur_pred = create_var(torch.LongTensor(pred_list))
                pred_hiddens.append( new_h.index_select(0, cur_pred) )
                pred_targets.extend( pred_target )

        #Last stop at root
        cur_x,cur_o_nei = [],[]
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
            if len(cur_nei) == 0:
                cur_o_nei.append(padding)
            else:
                cur_o_nei.append(torch.stack(cur_nei, dim=0))

        cur_x = create_var(torch.LongTensor(cur_x))
        cur_x = self.embedding(cur_x)
        # print(self.manifold.check_point_on_manifold(cur_x))
        cur_o = []
        for nei in cur_o_nei:
            if len(nei.size()) == 1:
                nei = nei.view(1, nei.size()[0])
            cur_o.append(self.manifold.mid_point(nei))
        cur_o = torch.stack(cur_o, dim=0).view(-1,self.hidden_size)
        # cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
        # cur_o = cur_o_nei.sum(dim=1)

        # stop_hidden = torch.cat([cur_x,cur_o], dim=1)
        # stop_hiddens.append( stop_hidden )
        stop_hiddens_x.append( cur_x )
        stop_hiddens_o.append( cur_o )
        stop_contexts.append( create_var( torch.LongTensor(list(range(batch_size))) ) )
        stop_targets.extend( [0] * len(mol_batch) )

        #Predict next clique
        pred_contexts = torch.cat(pred_contexts, dim=0)
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_scores = self.aggregate(pred_hiddens, pred_contexts, x_tree_vecs, 'word')
        pred_targets = create_var(torch.LongTensor(pred_targets))
        # print(pred_scores.shape, pred_targets.shape)
        # print(pred_targets.min(), pred_targets.max())

        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)
        _,preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        #Predict stop
        stop_contexts = torch.cat(stop_contexts, dim=0)
        stop_hiddens_x = torch.cat(stop_hiddens_x, dim=0)
        stop_hiddens_o = torch.cat(stop_hiddens_o, dim=0)
        stop_targets = create_var(torch.LongTensor(stop_targets))
        # stop_hiddens = F.relu( self.U_i(stop_hiddens) )
        # print(stop_hiddens_o.shape, stop_hiddens_x.shape, stop_contexts.shape)
        # print(self.manifold.check_point_on_manifold(stop_hiddens_x))
        # print(self.manifold.check_point_on_manifold(stop_hiddens_o))
        # print(stop_targets)
        stop_scores = self.aggregate((stop_hiddens_x, stop_hiddens_o), stop_contexts, x_tree_vecs, 'stop')
        stop_scores = stop_scores.squeeze(-1)
        # print(stop_scores.shape, stop_targets.shape)
        # print(stop_targets)
        
        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(mol_batch)
        _,stops = torch.max(stop_scores, dim=1)
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()
        # stops = torch.ge(stop_scores, 0).float()
        # stop_acc = torch.eq(stops, stop_targets).float()
        # stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()
    
    def decode(self, x_tree_vecs, prob_decode):
        assert x_tree_vecs.size(0) == 1

        stack = []
        init_hiddens = create_var( self.manifold.origin(1, self.hidden_size) )
        zero_pad = create_var(self.manifold.origin(1,1,self.hidden_size))
        contexts = create_var( torch.LongTensor(1).zero_() )

        #Root Prediction
        root_score = self.aggregate(init_hiddens, contexts, x_tree_vecs, 'word')
        _,root_wid = torch.max(root_score, dim=1)
        root_wid = root_wid.item()

        root = MolTreeNode(self.vocab.get_smiles(root_wid))
        root.wid = root_wid
        root.idx = 0
        stack.append( (root, self.vocab.get_slots(root.wid)) )

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x,fa_slot = stack[-1]
            cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
            else:
                cur_h_nei = zero_pad

            cur_x = create_var(torch.LongTensor([node_x.wid]))
            cur_x = self.embedding(cur_x)

            #Predict stop
            # cur_h = cur_h_nei.sum(dim=1)
            # stop_hiddens = torch.cat([cur_x,cur_h], dim=1)
            # stop_hiddens = F.relu( self.U_i(stop_hiddens) )
            cur_h = self.manifold.mid_point(cur_h_nei[0]).view(1, self.hidden_size)
            stop_score = self.aggregate((cur_x, cur_h), contexts, x_tree_vecs, 'stop')
            
            if prob_decode:
                backtrack = (torch.bernoulli( torch.sigmoid(stop_score) ).item() == 0)
            else:
                backtrack = (stop_score.item() < 0) 

            if not backtrack: #Forward: Predict next clique
                # new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                new_h = self.attnet(
                    key = cur_h_nei, 
                    value = cur_h_nei, 
                    query = cur_x.view(cur_x.size()[0], 1, cur_x.size()[1])
                    ).view(cur_x.size()[0], cur_x.size()[1])
                pred_score = self.aggregate(new_h, contexts, x_tree_vecs, 'word')

                if prob_decode:
                    sort_wid = torch.multinomial(F.softmax(pred_score, dim=1).squeeze(), 5)
                else:
                    _,sort_wid = torch.sort(pred_score, dim=1, descending=True)
                    sort_wid = sort_wid.data.squeeze()

                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = MolTreeNode(self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True #No more children can be added
                else:
                    node_y = MolTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.wid = next_wid
                    node_y.idx = len(all_nodes)
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx,node_y.idx)] = new_h[0]
                    stack.append( (node_y,next_slots) )
                    all_nodes.append(node_y)

            if backtrack:
                if len(stack) == 1: 
                    break #At root, terminate

                node_fa,_ = stack[-2]
                cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                # new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                self.attnet(
                    key = cur_h_nei, 
                    value = cur_h_nei, 
                    query = cur_x.view(cur_x.size()[0], 1, cur_x.size()[1])
                    ).view(cur_x.size()[0], cur_x.size()[1])
                h[(node_x.idx,node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes

"""
Helper Functions:
"""
def dfs(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx: continue
        stack.append( (x,y,1) )
        dfs(stack, y, x.idx)
        stack.append( (y,x,0) )

def have_slots(fa_slots, ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i,s1 in enumerate(fa_slots):
        a1,c1,h1 = s1
        for j,s2 in enumerate(ch_slots):
            a2,c2,h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append( (i,j) )

    if len(matches) == 0: return False

    fa_match,ch_match = list(zip(*matches))
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: #never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: #never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True

def can_assemble(node_x, node_y):
    node_x.nid = 1
    node_x.is_leaf = False
    set_atommap(node_x.mol, node_x.nid)

    neis = node_x.neighbors + [node_y]
    for i,nei in enumerate(neis):
        nei.nid = i + 2
        nei.is_leaf = (len(nei.neighbors) <= 1)
        if nei.is_leaf:
            set_atommap(nei.mol, 0)
        else:
            set_atommap(nei.mol, nei.nid)

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands,aroma_scores = enum_assemble(node_x, neighbors)
    return len(cands) > 0# and sum(aroma_scores) >= 0

if __name__ == "__main__":
    smiles = [
        "O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1",
        "O=C([O-])CC[C@@]12CCCC[C@]1(O)OC(=O)CC2", 
        "ON=C1C[C@H]2CC3(C[C@@H](C1)c1ccccc12)OCCO3", 
        "C[C@H]1CC(=O)[C@H]2[C@@]3(O)C(=O)c4cccc(O)c4[C@@H]4O[C@@]43[C@@H](O)C[C@]2(O)C1", 
        'Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br', 
        'CC(C)(C)c1ccc(C(=O)N[C@H]2CCN3CCCc4cccc2c43)cc1', 
        "O=c1c2ccc3c(=O)n(-c4nccs4)c(=O)c4ccc(c(=O)n1-c1nccs1)c2c34", 
        "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1"]
    for s in smiles:
        print(s)
        tree = MolTree(s)
        for i,node in enumerate(tree.nodes):
            node.idx = i

        stack = []
        dfs(stack, tree.nodes[0], -1)
        for x,y,d in stack:
            print((x.smiles, y.smiles, d))
        print('------------------------------')
