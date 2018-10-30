import argparse

from agents.max_sparse_net import MaxSparseNetAgent


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
