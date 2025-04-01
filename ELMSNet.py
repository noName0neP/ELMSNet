import torch
from thop import profile
from torch import nn
from torchsummary import summary


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                                 stride=1, padding=padding, groups=in_channels, bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       nn.LeakyReLU(inplace=True)
                                       )

        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                 stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.LeakyReLU(inplace=True)
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MR_Block(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(MR_Block, self).__init__()
        out_channel = in_channel // ratio
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 8, kernel_size=1, bias=False),
            DepthwiseSeparableConv(out_channel * 8, out_channel, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 4, kernel_size=1, bias=False),
            DepthwiseSeparableConv(out_channel * 4, out_channel, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            DepthwiseSeparableConv(out_channel, out_channel, kernel_size=7, padding=3)
        )

    def forward(self, x):
        shortcut = x
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        output = torch.cat([b1, b2, b3, b4], dim=1)
        output = output + shortcut
        return output


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.conv3x3 = DepthwiseSeparableConv(channels // self.groups, channels // self.groups, kernel_size=3,
                                              padding=1)  # Improvement

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h,
                                                                            w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class ELMSNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, num_classes=10):
        super(ELMSNet, self).__init__()
        self.Embedding_Block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, 4, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )
        self.feature = []
        for i in range(4):
            self.feature.append(MR_Block(out_channel))
            if i < 3:
                self.feature.append(nn.Sequential(
                    EMA(out_channel),
                    nn.Conv2d(out_channel, out_channel * 2, 2, 2, bias=False),
                    nn.BatchNorm2d(out_channel * 2),
                    nn.ReLU6(inplace=True)
                ))  # AM Module
                out_channel = out_channel * 2
        self.feature = nn.Sequential(*self.feature)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU6(inplace=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.Embedding_Block(x)
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.relu(x)
        x = self.fc(x.flatten(1))
        return x


model = ELMSNet()
print(model)
model.to('cuda')
summary(model, input_size=(3, 224, 224))
model = ELMSNet()
input = torch.randn(1, 3, 224, 224)
FLOPs, params = profile(model, inputs=(input,))
print('Flops: % .4fG' % (FLOPs / 1000000000))
print('params: % .4fM' % (params / 1000000))
