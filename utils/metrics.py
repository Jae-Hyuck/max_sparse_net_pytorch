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


def eval_s_acc(x_pred, x_gt, sparsity):
    '''
    Strict Accuracy
    Batch x N 형태로 들어온다고 가정
    '''
    topk_vals = x_pred.topk(sparsity, dim=1)[0]
    kth_vals = topk_vals[:, -1:]

    x_pred_strict = torch.zeros_like(x_pred)
    x_pred_strict[x_pred >= kth_vals] = 1

    correct = (x_pred_strict == x_gt).all(dim=1).float()

    return correct.mean()


def eval_l_acc(x_pred, x_gt, n_measure):
    '''
    Loose Accuracy
    Batch x N 형태로 들어온다고 가정
    '''
    topk_vals = x_pred.topk(n_measure, dim=1)[0]
    kth_vals = topk_vals[:, -1:]

    x_pred_loose = torch.zeros_like(x_pred)
    x_pred_loose[x_pred >= kth_vals] = 1

    match = ((x_pred_loose != 0) & (x_gt != 0)).sum(dim=1).float()
    match /= (x_gt != 0).sum(dim=1).float()

    return match.mean()
