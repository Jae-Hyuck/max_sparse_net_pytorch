import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(n_in, n_in),
            nn.BatchNorm1d(n_in),
            nn.ReLU(inplace=True),
            nn.Linear(n_in, n_out),
            # nn.BatchNorm1d(n_out),
        )

        if n_in == n_out:
            self.shortcut = (lambda x: x)  # skip connection
        else:
            self.shortcut = nn.Linear(n_in, n_out)  # upsample
            '''
            self.shortcut = nn.Sequential(
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out),
            )
            '''

    def forward(self, x):
        x = self.block(x) + self.shortcut(x)
        x = F.relu(x, inplace=True)

        return x


class MaxSparseNet(nn.Module):
    def __init__(self, cfg):
        super(MaxSparseNet, self).__init__()

        n_in = cfg.NUM_Y
        n_out = cfg.NUM_X

        self.net = nn.Sequential(
            nn.Linear(n_in, n_in),
            nn.BatchNorm1d(n_in, n_in),
            ResBlock(n_in, n_in),
            ResBlock(n_in, n_in),
            ResBlock(n_in, n_in),
            ResBlock(n_in, n_in),

            ResBlock(n_in, n_out),

            ResBlock(n_out, n_out),
            ResBlock(n_out, n_out),
            ResBlock(n_out, n_out),
            ResBlock(n_out, n_out),
            nn.Linear(n_out, n_out),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.net(x)

        return x
