import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np


#in_channel:　输入数据的通道数，
#out_channel: 输出数据的通道数
#stride：步长，默认为1，与kennel_size类似，stride=2,意味着步长上下左右扫描皆为2， stride=（2,3），左右扫描步长为2，上下为3；
# padding：　填充
#(B, C , H , W)
#后有bn，bias=false
def conv3X3(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    #expansion是残差结构中输出维度是输入维度的多少倍
    expansion =1
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

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 =self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #[batch_size, in_features]的张量变换成了[batch_size, out_features]
        self.fc = nn.Linear(512*block.expansion, num_classes)
        #一个对象是否是一个已知的类型
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                #正态分布的标准差设置为math.sqrt(2. / n)
                m.weight.data.norm(0,math.sqrt(2./n))
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            #Sequential容器，模块将按照构造函数中传递的顺序添加到模块中
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        #输入也可以是list, 然后输入的时候用 * 来引用
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        s = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #-1就代表这个位置由其他位置的数字来推断
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
#**来进行函数调用，首先需要一个字典，就像使用*进行函数调用时需要列表或者元组一样
def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


