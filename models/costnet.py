import torch
import torch.nn as nn
import torch.nn.functional as F


class CostNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = CNN()
        self.spp = SPP()
        self.fusion = nn.Sequential(
                Conv2dBn(in_channels=320, out_channels=128, kernel_size=3, stride=1, padding=1, use_relu=True),
                nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
            )

    def forward(self, inputs):
        conv2_out, conv4_out = self.cnn(inputs)           # [B, 64, 1/4H, 1/4W], [B, 128, 1/4H, 1/4W]

        spp_out = self.spp(conv4_out)                    # [B, 128, 1/4H, 1/4W]
        out = torch.cat([conv2_out, conv4_out, spp_out], dim=1)  # [B, 320, 1/4H, 1/4W]
        out = self.fusion(out)                            # [B, 32, 1/4H, 1/4W]

        return out


class SPP(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch1 = self.__make_branch(kernel_size=64, stride=64)
        self.branch2 = self.__make_branch(kernel_size=32, stride=32)
        self.branch3 = self.__make_branch(kernel_size=16, stride=16)
        self.branch4 = self.__make_branch(kernel_size=8, stride=8)

    def forward(self, inputs):

        out_size = inputs.size(2), inputs.size(3)
        branch1_out = F.upsample(self.branch1(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        # print('branch1_out')
        # print(branch1_out[0, 0, :3, :3])
        branch2_out = F.upsample(self.branch2(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        branch3_out = F.upsample(self.branch3(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        branch4_out = F.upsample(self.branch4(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        out = torch.cat([branch4_out, branch3_out, branch2_out, branch1_out], dim=1)  # [B, 128, 1/4H, 1/4W]

        return out

    @staticmethod
    def __make_branch(kernel_size, stride):
        branch = nn.Sequential(
                nn.AvgPool2d(kernel_size, stride),
                Conv2dBn(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True)  # kernel size maybe 1
            )
        return branch


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv0 = nn.Sequential(
                Conv2dBn(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, use_relu=True),  # downsample
                Conv2dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True),
                Conv2dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True)
            )

        self.conv1 = StackedBlocks(n_blocks=3, in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = StackedBlocks(n_blocks=16, in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1)  # downsample
        self.conv3 = StackedBlocks(n_blocks=3, in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2)  # dilated
        self.conv4 = StackedBlocks(n_blocks=3, in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4)  # dilated

    def forward(self, inputs):
        conv0_out = self.conv0(inputs)
        conv1_out = self.conv1(conv0_out)  # [B, 32, 1/2H, 1/2W]
        conv2_out = self.conv2(conv1_out)  # [B, 64, 1/4H, 1/4W]
        conv3_out = self.conv3(conv2_out)  # [B, 128, 1/4H, 1/4W]
        conv4_out = self.conv4(conv3_out)  # [B, 128, 1/4H, 1/4W]

        return conv2_out, conv4_out


class StackedBlocks(nn.Module):

    def __init__(self, n_blocks, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()

        if stride == 1 and in_channels == out_channels:
            downsample = False
        else:
            downsample = True
        net = [ResidualBlock(in_channels, out_channels, kernel_size, stride, padding, dilation, downsample)]

        for i in range(n_blocks - 1):
            net.append(ResidualBlock(out_channels, out_channels, kernel_size, 1, padding, dilation, downsample=False))
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, downsample=False):
        super().__init__()

        self.net = nn.Sequential(
                Conv2dBn(in_channels, out_channels, kernel_size, stride, padding, dilation, use_relu=True),
                Conv2dBn(out_channels, out_channels, kernel_size, 1, padding, dilation, use_relu=False)
            )

        self.downsample = None
        if downsample:
            self.downsample = Conv2dBn(in_channels, out_channels, 1, stride, use_relu=False)

    def forward(self, inputs):
        out = self.net(inputs)
        if self.downsample:
            inputs = self.downsample(inputs)
        out = out + inputs

        return out


class Conv2dBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_relu=True):
        super().__init__()

        net = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
               nn.BatchNorm2d(out_channels)]
        if use_relu:
            net.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out