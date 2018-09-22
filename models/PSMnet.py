import math
import torch
import torch.nn as nn
from models.costnet import CostNet
from models.stackedhourglass import StackedHourglass


class PSMNet(nn.Module):

    def __init__(self, max_disp):
        super().__init__()

        self.cost_net = CostNet()
        self.stackedhourglass = StackedHourglass(max_disp)
        self.D = max_disp

        self.__init_params()

    def forward(self, left_img, right_img):
        original_size = [self.D, left_img.size(2), left_img.size(3)]

        left_cost = self.cost_net(left_img)  # [B, 32, 1/4H, 1/4W]
        right_cost = self.cost_net(right_img)  # [B, 32, 1/4H, 1/4W]
        # cost = torch.cat([left_cost, right_cost], dim=1)  # [B, 64, 1/4H, 1/4W]
        # B, C, H, W = cost.size()

        # print('left_cost')
        # print(left_cost[0, 0, :3, :3])

        B, C, H, W = left_cost.size()

        cost_volume = torch.zeros(B, C * 2, self.D // 4, H, W).type_as(left_cost)  # [B, 64, D, 1/4H, 1/4W]

        # for i in range(self.D // 4):
        #     cost_volume[:, :, i, :, i:] = cost[:, :, :, i:]

        for i in range(self.D // 4):
            if i > 0:
                cost_volume[:, :C, i, :, i:] = left_cost[:, :, :, i:]
                cost_volume[:, C:, i, :, i:] = right_cost[:, :, :, :-i]
            else:
                cost_volume[:, :C, i, :, :] = left_cost
                cost_volume[:, C:, i, :, :] = right_cost

        disp1, disp2, disp3 = self.stackedhourglass(cost_volume, out_size=original_size)

        return disp1, disp2, disp3

    def __init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
