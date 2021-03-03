'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SmallResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, d_h=32):
        super(SmallResNet, self).__init__()
        self.in_planes = 64
        if d_h == 32:
            pass
        elif d_h == 16:
            num_blocks = num_blocks[1:]
        elif d_h == 8:
            num_blocks = num_blocks[2:]
        elif d_h == 4:
            num_blocks = num_blocks[3:]
        else:
            raise ValueError("d_h must be 32, 16, 8 or 4")

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        layers = [self._make_layer(block, 64, num_blocks[0], stride=1)]
        planes = 64
        for blocks in num_blocks[1:]:
            planes *= 2
            layers.append(
                self._make_layer(block, planes, blocks, stride=2))
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(planes*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SmallResNet18(d_h=32):
    return SmallResNet(BasicBlock, [2, 2, 2, 2], d_h=d_h)


def SmallResNet34(d_h=32):
    return SmallResNet(BasicBlock, [3, 4, 6, 3], d_h=d_h)


def SmallResNet50(d_h=32):
    return SmallResNet(Bottleneck, [3, 4, 6, 3], d_h=d_h)


def SmallResNet101(d_h=32):
    return SmallResNet(Bottleneck, [3, 4, 23, 3], d_h=d_h)


def SmallResNet152(d_h=32):
    return SmallResNet(Bottleneck, [3, 8, 36, 3], d_h=d_h)


def test():
    d_h = 4
    net = SmallResNet18(d_h)
    y = net(torch.randn(1, 3, d_h, d_h))
    print(y.size())

# test()
