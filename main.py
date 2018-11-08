import argparse

from agents.max_sparse_net import MaxSparseNetAgent

import torch
import numpy as np
import random

torch.manual_seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(10)
random.seed(10)


def train(cfg):
    agent = MaxSparseNetAgent(cfg)
    agent.train()


def test(cfg):
    pass


if __name__ == '__main__':
    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument('MODE', type=str, choices=['train', 'test'])
    parser.add_argument('LR', type=float)

    cfg = parser.parse_args()

    cfg.NUM_X = 100
    cfg.NUM_Y = 20
    cfg.SPARSITY = 3

    print(cfg)

    # -------
    # Run
    # -------
    if cfg.MODE == 'train':
        train(cfg)
    else:
        test(cfg)
