
import numpy as np
import torch


def dice_coefficient(pred, target, smooth=1.):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)

    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    pred = torch.sigmoid(pred)

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)

    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    score = (2. * intersection + smooth) / (A_sum + B_sum + smooth)

    return score

