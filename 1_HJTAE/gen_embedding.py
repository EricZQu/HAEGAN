import torch
import torch.nn as nn

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit
from config import parser

def load_model(args, vocab, model_path):

    model = HyperbolicJTNNAE(args, vocab)
    dict_buffer = torch.load(model_path)
    model.load_state_dict(dict_buffer)
    model = model.cuda()

    # torch.manual_seed(0)
    return model

# model_path = 'logs/2022_1_9/0/model.iter-48000'
# config_path = 'logs/2022_1_9/0/config.json'
# output = '../2_WGAN/Embed/'

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)
args = parser.parse_args()

model_path = args.model_path
config_path = args.config_path
output = args.embed_path

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = load_model(args, vocab, model_path)
model.eval()

tree_embed = []
graph_embed = []

loader = HyperbolicMolTreeFolder(args.data_path, vocab, args.batch_size)
for batch in loader:
    with torch.no_grad():
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = batch
        tree_vecs, tree_mess, mol_vecs = model.encode(x_jtenc_holder, x_mpn_holder)

        tree_embed.append(tree_vecs.cpu())
        graph_embed.append(mol_vecs.cpu())

    # print(tree_vecs.cpu().shape)
    # break

tree_embed = torch.stack(tree_embed, 0)
graph_embed = torch.stack(graph_embed, 0)

print(tree_embed.shape)
print(graph_embed.shape)

torch.save(tree_embed, output + 'tree.pt')
torch.save(graph_embed, output + 'graph.pt')