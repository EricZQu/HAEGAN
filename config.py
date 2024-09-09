import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.0005, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (20, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('radam', 'which optimizer to use, can be any of [rsgd, radam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (1, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/date/run/)'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs'),
        'print_iter': (25, 'number of iteration to print info'),
        'save_iter' : (2000, 'number of iteration to save model'),
        'anneal_rate' : (0.5, 'anneal rate to decrease lr'),
        'anneal_iter' : (20000, 'number of iteration to decrease lr')
    },
    'model_config': {
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc]'),
        'model': ('GCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN, HyboNet]'),
        'dim': (256, 'embedding dimension'),###
        'feat-dim': (256, 'embedding mapping dimension'),###
        'head-count': (16, 'head count of self attention'),
        'latent-dim': (256, 'latent dimension'),###
        'manifold': ('Lorentz', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall, Lorentz]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'margin': (2., 'margin of MarginLoss'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers-tree': (4, 'number of hidden layers in tree encoder'),
        'num-layers-graph': (4, 'number of hidden layers in graph encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('None', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'vocab': ('./vocab.txt', 'path to vocab file'),
        'data-path': ('../data/moses-processed', 'path to processed data files'),
        'batch-size': (32, 'batch size'),
        # 'dataset': ('cora', 'which dataset to use'),
        # 'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        # 'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
    },
    'embed_config': {
        'model-path': (None, 'path to save model'),
        'config-path': (None, 'path to config file'),
        'embed-path': ('./2_HWGAN/Embed/', 'path to saved embedding')
    },
    'GAN_config': {
        'z-dim': (128, 'zdim of GAN'),
        'critic-dim': (256, 'hiddem dim for critic'),
        'critic-depth': (3, 'depth for critic'),
        'generator-dim': (256, 'hidden dim for generator'),
        'generator-depth': (3, 'depth for generator'),
        'critic-iterations': (5, 'the number of iterations for training of critic per generator'),
        'lambda-gp': (10, 'labmda value for gradient penalty'),
        'sample-path': ('samples.out', 'path for output samples'),
        'num-smaples': (1000, 'number of samples'),
        'model-path-tree': (None, 'model path for tree generator'),
        'model-path-graph': (None, 'model path for graph generator')
    
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
