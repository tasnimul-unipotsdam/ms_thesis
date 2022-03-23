from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


class DiceCoef(nn.Module):
    print("getting dice")

    def __init__(self):

        super(DiceCoef, self).__init__()

        """
        Dice coef over the batch.
        Dividing by num_samples gives dice for each image
        """

    def forward(self, pred, target, smooth=1.):

        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)

        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        pred = torch.sigmoid(pred)

        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)

        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        # this computes mean IOU over batch
        # it means that is is sum(IOU_img)/num samples in each batch
        score = (2. * intersection + smooth) / (A_sum + B_sum + smooth)

        # print(f'mean IOU over batch in DiceCoef:{score:0.4f}')
        return score
