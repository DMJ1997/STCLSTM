import torch
import torch.nn.functional as F
import torch.nn as nn

class _UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()
        self.conv1 = nn.Conv2d(num_input_features, num_input_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

        def forward(self, x, size):
            #True，输入的角像素将与输出张量对齐
            x = F.upsample(x, size=size, mode='bilinear', align_corners=True)
            x_conv1 = self.relu(self.bn1(self.conv1(x)))
            bran1 = self.bn1_2(self.conv1_2(x_conv1))
            bran2 = self.bn2(self.conv2(x))

            out = self.relu(bran1 + bran2)

            return out




class D(nn.Module):
    def __init__(self, num_features = 2048):
        super(D, self).__init__()
        # // 整数除法
        self.conv = nn.Conv2d(num_features, num_features//2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)

        self.up1 = _UpProjection(num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up2 = _UpProjection(num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up3 = _UpProjection(num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up4 = _UpProjection(num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(2)])
        x_d2 = self.up1(x_d1, [x_block2.size(2), x_block2.size(2)])
        x_d3 = self.up1(x_d2, [x_block1.size(2), x_block1.size(2)])
        x_d4 = self.up1(x_d3, [x_block1.size(2)*2, x_block1.size(2)*2])

        return x_d4

class MFF(nn.Module):
    def __init__(self, block_channel, num_features=64):
        super(MFF, self).__init__()
        self.up1 = _UpProjection(num_features=block_channel[0],num_output_features=16)
        self.up2 = _UpProjection(num_features=block_channel[1],num_output_features=16)
        self.up3 = _UpProjection(num_features=block_channel[2],num_output_features=16)
        self.up4 = _UpProjection(num_features=block_channel[3],num_output_features=16)

        self.conv = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)
        #cat 拼接channel
        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))

        x = F.relu(x)

        return x
