from re import M
import torch
from GAN.EmbedDataset import TreeDataset, GraphDataset
from GAN.Generator import Generator
# import torch.optim as optim
from optim import RiemannianAdam, RiemannianSGD
from GAN.Critic import Critic
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import manifolds
from fast_jtnn import *
import rdkit
import time
import io
from config import parser

args = parser.parse_args()

LEARNING_RATE = args.lr
DIM = args.dim
BATCH_SIZE = args.batch_size
Z_DIM = args.z_dim
NUM_EPOCHS = args.epochs
CRITIC_HID = args.critic_dim
CRITIC_DEP = args.critic_depth
GEN_HID = args.generator_dim
GEN_DEP = args.generator_depth
CRITIC_ITERATIONS = args.critic_iterations
LAMBDA_GP = args.lambda_gp
NUM = args.num_samples
OUTPUT = './samples'


tree_model_path = args.model_path_tree
graph_model_path = args.model_path_graph

def load_model(model_path):

    model = Generator(Z_DIM, GEN_HID, DIM, GEN_DEP)
    dict_buffer = torch.load(model_path)
    model.load_state_dict(dict_buffer)
    model = model.cuda()

    # torch.manual_seed(0)
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
manifold = manifolds.Lorentz()

gen_tree = load_model(tree_model_path)
gen_graph = load_model(graph_model_path)

noise = manifold.random_normal((NUM, Z_DIM)).to(device)

with torch.no_grad():
    tree_embed = gen_tree(noise)
    graph_embed = gen_graph(noise)


def load_models(args, vocab, model_path):

    model = HyperbolicJTNNAE(args, vocab)
    dict_buffer = torch.load(model_path)
    model.load_state_dict(dict_buffer)
    model = model.cuda()

    # torch.manual_seed(0)
    return model

model_path = args.model_path
config_path = args.config_path

tree_embed = tree_embed.view((tree_embed.size(0)*tree_embed.size(1), tree_embed.size(2)))
graph_embed = graph_embed.view((graph_embed.size(0)*graph_embed.size(1), graph_embed.size(2)))

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = load_model(args, vocab, model_path)
model.eval()

res = []

for i in range(NUM):
    # tmp = torch.randint(0, tree_embed.size(0), (1,))

    t = tree_embed[i].view(1, tree_embed.size(1))
    g = graph_embed[i].view(1, graph_embed.size(1))

    res.append(model.decode(t, g, False))

with open(args.sample_path, 'w') as f:
    for item in res:
        f.write(item + '\n')