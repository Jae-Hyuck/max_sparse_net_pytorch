import torch


class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


def eval_s_acc(x_pred, x_gt):
    '''
    Strict Accuracy
    x_pred: Batch x N, float, high value means high probabilities.
    x_gt: Batch x N, float
    '''
    x_pred_masked = x_pred.clone()
    x_pred_masked[x_gt == 0] = float('inf')

    min_vals = x_pred_masked.min(dim=1, keepdim=True)[0]

    correct = ((x_pred >= min_vals) == (x_gt != 0)).all(dim=1).float()

    return correct.mean()


def eval_l_acc(x_pred, x_gt, n_measure):
    '''
    Loose Accuracy
    x_pred: Batch x N, float, high value means high probabilities.
    x_gt: Batch x N, float
    '''
    topk_vals = x_pred.topk(n_measure, dim=1)[0]
    kth_vals = topk_vals[:, -1:]

    x_pred_loose = torch.zeros_like(x_pred)
    x_pred_loose[x_pred >= kth_vals] = 1

    match = ((x_pred_loose != 0) & (x_gt != 0)).sum(dim=1).float()
    match /= (x_gt != 0).sum(dim=1).float()

    return match.mean()
