from torch.utils.data import Dataset
import numpy as np
import math


def make_sparse_uniform_sig(N, sparsity):
    '''
    non-sparse element has value of +-[0.1, 0.4].
    '''

    x = np.random.uniform(-0.4, 0.4, size=sparsity)
    x += np.sign(x) * 0.1

    x = np.append(x, np.zeros(N-sparsity))
    x = np.random.permutation(x)

    return x


def add_awgn(sig, snr):
    snr_lin = 10 ** (snr/10)  # dB scale to linear scale
    sig_power = np.sum(np.abs(sig)**2) / sig.size
    noise_power = sig_power / snr_lin
    noise = math.sqrt(noise_power) * np.random.randn(sig.size)
    return sig + noise


class UniformSparseDataset(Dataset):
    def __init__(self, A, sparsity, epoch_size, mode):
        super(UniformSparseDataset, self).__init__()

        self.A = A
        self.NUM_X = A.shape[1]
        self.K = sparsity
        self.epoch_size = epoch_size
        self.mode = mode

    def __getitem__(self, idx):
        K = self.K
        # K = idx % 7 + 3
        x = make_sparse_uniform_sig(self.NUM_X, K)

        # Generate measurements
        y = self.A @ x

        '''
        # Add noise
        if self.mode == 'train':
            snr = np.random.randint(60, 80)
            y = add_awgn(y, snr)
        elif self.mode == 'valid':
            pass
        else:
            raise ValueError(f'Unknown Mode: {self.mode}')
        '''

        # sample return
        sample = {'x': x, 'y': y}

        return sample

    def __len__(self):
        return self.epoch_size
