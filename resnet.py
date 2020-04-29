import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

#in_channel:　输入数据的通道数，
#out_channel: 输出数据的通道数
#stride：步长，默认为1，与kennel_size类似，stride=2,意味着步长上下左右扫描皆为2， stride=（2,3），左右扫描步长为2，上下为3；
# padding：　填充
#(B, C , H , W)
#后有bn，bias=false
def conv3X3(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self,inplanes,planes,srtide=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3X3(inplanes, planes, srtide)
       #channel
        self.bn1 = nn.BatchNorm2d(planes)
        #inplace 覆盖
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3X3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = srtide

    def forward(self, x):
        residual = x

        out =self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 =




        def _make_layer(self, block, planes, blocks, strides=1):




