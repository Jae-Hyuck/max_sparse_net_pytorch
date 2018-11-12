import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.uniform_sparse import UniformSparseDataset
from nets.max_sparse_net import MaxSparseNet
from utils.metrics import AverageMeter, eval_s_acc, eval_l_acc


def make_dic(M, N):
    A = 0
    for i in range(M):
        u = np.random.randn(M)
        v = np.random.randn(N)
        outer = np.outer(u, v)
        A += outer / (i+1) / (i+1)

    # Normalize columns
    A = A / np.sqrt(np.sum(np.abs(A)**2, axis=0))

    return A


class MaxSparseNetAgent():
    def __init__(self, cfg):
        # device cfg
        self.device = torch.device('cuda')

        # Hyper-parameters
        self.cfg = cfg
        train_batch_size = 250
        self.lr = cfg.LR

        # Network
        self.net = MaxSparseNet(cfg).to(self.device)

        # Optimizer
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.LR)
        # self.optim = torch.optim.RMSprop(self.net.parameters(), lr=cfg.LR)
        '''
        self.optim = torch.optim.SGD(self.net.parameters(), lr=cfg.LR,
                                     momentum=0.9, weight_decay=0.0001)
        '''
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=50, gamma=0.5)

        # Prepare dataset
        # np.random.seed(910103)
        self.A = make_dic(cfg.NUM_Y, cfg.NUM_X)

        train_dataset = UniformSparseDataset(self.A, cfg.SPARSITY, 600000, mode='train')
        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=train_batch_size,
            shuffle=False, num_workers=8,
            worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2**32)
        )
        valid_dataset = UniformSparseDataset(self.A, cfg.SPARSITY, 100000, mode='valid')
        self.valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=train_batch_size,
            shuffle=False, num_workers=1,
            worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2**32)
        )

    def train(self):
        num_epochs = 150
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.scheduler.step()
            self.train_one_epoch()
            self.validate()
        print(self.lr)

    def train_one_epoch(self):
        # Training mode
        self.net.train()

        # Init Average Meters
        epoch_loss = AverageMeter()
        epoch_s_acc = AverageMeter()
        epoch_l_acc = AverageMeter()

        tqdm_batch = tqdm(self.train_loader, f'Epoch-{self.current_epoch}-')
        for data in tqdm_batch:
            # Prepare data
            x = torch.tensor(data['x'], dtype=torch.float32, device=self.device)
            y = torch.tensor(data['y'], dtype=torch.float32, device=self.device)

            s_gt = torch.zeros_like(x)
            s_gt[x != 0] = 1

            # Forward pass
            s_pred = self.net(y)

            # Compute loss
            # cur_loss = F.binary_cross_entropy_with_logits(shat, s_gt)
            cur_loss = F.log_softmax(s_pred, dim=1)
            cur_loss = cur_loss * s_gt
            cur_loss = - cur_loss.sum(dim=1)
            cur_loss = cur_loss.mean()

            # Backprop and optimize
            self.optim.zero_grad()
            cur_loss.backward()
            self.optim.step()

            # Metrics
            s_acc = eval_s_acc(s_pred, s_gt)
            l_acc = eval_l_acc(s_pred, s_gt, self.cfg.NUM_Y)

            batch_size = x.shape[0]
            epoch_loss.update(cur_loss.item(), batch_size)
            epoch_s_acc.update(s_acc.item(), batch_size)
            epoch_l_acc.update(l_acc.item(), batch_size)

        tqdm_batch.close()

        print(f'Train at epoch- {self.current_epoch} |'
              f'loss: {epoch_loss.val} - s_acc: {epoch_s_acc.val} - l_acc: {epoch_l_acc.val}')

    def validate(self):
        # Eval mode
        self.net.eval()

        # Init Average Meters
        epoch_s_acc = AverageMeter()
        epoch_l_acc = AverageMeter()

        tqdm_batch = tqdm(self.valid_loader, f'Epoch-{self.current_epoch}-')
        with torch.no_grad():
            for data in tqdm_batch:
                # Prepare data
                x = torch.tensor(data['x'], dtype=torch.float32, device=self.device)
                y = torch.tensor(data['y'], dtype=torch.float32, device=self.device)

                s_gt = torch.zeros_like(x)
                s_gt[x != 0] = 1

                # Forward pass
                s_pred = self.net(y)

                # Metrics
                s_acc = eval_s_acc(s_pred, s_gt)
                l_acc = eval_l_acc(s_pred, s_gt, self.cfg.NUM_Y)

                batch_size = x.shape[0]
                epoch_s_acc.update(s_acc.item(), batch_size)
                epoch_l_acc.update(l_acc.item(), batch_size)

        tqdm_batch.close()

        print(f'Validate at epoch- {self.current_epoch} |'
              f's_acc: {epoch_s_acc.val} - l_acc: {epoch_l_acc.val}')
