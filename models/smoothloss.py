import torch.nn.functional as F
import torch.nn as nn


class SmoothL1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, disp1, disp2, disp3, target):
        loss1 = F.smooth_l1_loss(disp1, target)
        loss2 = F.smooth_l1_loss(disp2, target)
        loss3 = F.smooth_l1_loss(disp3, target)

        return loss1, loss2, loss3
