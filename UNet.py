import torch
import torch.nn as nn
from torch.nn.functional import interpolate

class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize=3, act=nn.ReLU):
        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels, outChannels, filterSize, stride=2, padding=1)
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding='same')
        self.act = act()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class up(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize=3, act=nn.ReLU):
        super(up, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, filterSize, 1, padding='same')
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, filterSize, 1, padding='same')
        self.act = act()

    def forward(self, x, skpCn):
        h, w = skpCn.shape[-2:]
        x = interpolate(x, size=(h, w), mode='bilinear')
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(torch.cat((x, skpCn), 1)))
        return x


class SmallUNet(nn.Module):
    def __init__(self, inChannels, outChannels, dim=32, act=nn.ReLU):
        super(SmallUNet, self).__init__()
        # Initialize neural network blocks.
        self.lvl0 = nn.Sequential(
            nn.Conv2d(inChannels, dim, 7, stride=1, padding=3),
            act(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            act(),
        )
        self.lvl1 = down(dim, dim*2, 3, act=act)
        self.lvl2 = down(dim*2, dim*4, 3, act=act)
        self.up2 = up(dim*4, dim*2, act=act)
        self.up1 = up(dim*2, dim, act=act)
        self.up0 = nn.Conv2d(dim, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        res = x
        s1 = self.lvl0(x)
        s2 = self.lvl1(s1)
        s3 = self.lvl2(s2)
        x = self.up2(s3, s2)
        x = self.up1(x, s1)
        x = self.up0(x)
        return x + res

