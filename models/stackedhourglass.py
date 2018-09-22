import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedHourglass(nn.Module):
    '''
    inputs --- [B, 64, 1/4D, 1/4H, 1/4W]
    '''

    def __init__(self, max_disp):
        super().__init__()

        self.conv0 = nn.Sequential(
            Conv3dBn(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True)
        )
        self.conv1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=False)
        )
        self.hourglass1 = Hourglass()
        self.hourglass2 = Hourglass()
        self.hourglass3 = Hourglass()

        self.out1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )
        self.out2 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )
        self.out3 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )

        self.regression = DisparityRegression(max_disp)

    def forward(self, inputs, out_size):

        conv0_out = self.conv0(inputs)     # [B, 32, 1/4D, 1/4H, 1/4W]
        conv1_out = self.conv1(conv0_out)
        conv1_out = conv0_out + conv1_out  # [B, 32, 1/4D, 1/4H, 1/4W]

        hourglass1_out1, hourglass1_out3, hourglass1_out4 = self.hourglass1(conv1_out, scale1=None, scale2=None, scale3=conv1_out)
        hourglass2_out1, hourglass2_out3, hourglass2_out4 = self.hourglass2(hourglass1_out4, scale1=hourglass1_out3, scale2=hourglass1_out1, scale3=conv1_out)
        hourglass3_out1, hourglass3_out3, hourglass3_out4 = self.hourglass3(hourglass2_out4, scale1=hourglass2_out3, scale2=hourglass1_out1, scale3=conv1_out)

        out1 = self.out1(hourglass1_out4)  # [B, 1, 1/4D, 1/4H, 1/4W]
        out2 = self.out2(hourglass2_out4) + out1
        out3 = self.out3(hourglass3_out4) + out2

        cost1 = F.upsample(out1, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]
        cost2 = F.upsample(out2, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]
        cost3 = F.upsample(out3, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]

        prob1 = F.softmax(-cost1, dim=1)  # [B, D, H, W]
        prob2 = F.softmax(-cost2, dim=1)
        prob3 = F.softmax(-cost3, dim=1)

        disp1 = self.regression(prob1)
        disp2 = self.regression(prob2)
        disp3 = self.regression(prob3)

        return disp1, disp2, disp3


class DisparityRegression(nn.Module):

    def __init__(self, max_disp):
        super().__init__()

        self.disp_score = torch.range(0, max_disp - 1)  # [D]
        self.disp_score = self.disp_score.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, D, 1, 1]

    def forward(self, prob):
        disp_score = self.disp_score.expand_as(prob).type_as(prob)  # [B, D, H, W]
        out = torch.sum(disp_score * prob, dim=1)  # [B, H, W]
        return out


class Hourglass(nn.Module):

    def __init__(self):
        super().__init__()

        self.net1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=False)
        )
        self.net2 = nn.Sequential(
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True)
        )
        self.net3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(num_features=64)
            # nn.ReLU(inplace=True)
        )
        self.net4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(num_features=32)
        )

    def forward(self, inputs, scale1=None, scale2=None, scale3=None):
        net1_out = self.net1(inputs)  # [B, 64, 1/8D, 1/8H, 1/8W]

        if scale1 is not None:
            net1_out = F.relu(net1_out + scale1, inplace=True)
        else:
            net1_out = F.relu(net1_out, inplace=True)

        net2_out = self.net2(net1_out)  # [B, 64, 1/16D, 1/16H, 1/16W]
        net3_out = self.net3(net2_out)  # [B, 64, 1/8D, 1/8H, 1/8W]

        if scale2 is not None:
            net3_out = F.relu(net3_out + scale2, inplace=True)
        else:
            net3_out = F.relu(net3_out + net1_out, inplace=True)

        net4_out = self.net4(net3_out)

        if scale3 is not None:
            net4_out = net4_out + scale3

        return net1_out, net3_out, net4_out


class Conv3dBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_relu=True):
        super().__init__()

        net = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
               nn.BatchNorm3d(out_channels)]
        if use_relu:
            net.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out
