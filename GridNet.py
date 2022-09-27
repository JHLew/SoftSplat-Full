import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class GridNet(nn.Module):
    def __init__(self, dim=32, act=nn.ReLU):
        super(GridNet, self).__init__()
        self.lateral = nn.ModuleList([
            nn.ModuleList([
                LateralBlock(dim, act=act),
                LateralBlock(dim, act=act),
                LateralBlock(dim, act=act),
                LateralBlock(dim, act=act),
                LateralBlock(dim, act=act),
                LateralBlock(dim, act=act),
            ]),
            nn.ModuleList([
                LateralBlock(dim*2, act=act),
                LateralBlock(dim*2, act=act),
                LateralBlock(dim*2, act=act),
                LateralBlock(dim*2, act=act),
                LateralBlock(dim*2, act=act),
            ]),
            nn.ModuleList([
                LateralBlock(dim*3, act=act),
                LateralBlock(dim*3, act=act),
                LateralBlock(dim*3, act=act),
                LateralBlock(dim*3, act=act),
                LateralBlock(dim*3, act=act),
            ])
        ])
        self.down = nn.ModuleList([
            nn.ModuleList([
                DownBlock(dim, dim*2, act=act),
                DownBlock(dim, dim*2, act=act),
                DownBlock(dim, dim*2, act=act),
            ]),
            nn.ModuleList([
                DownBlock(dim*2, dim*3, act=act),
                DownBlock(dim*2, dim*3, act=act),
                DownBlock(dim*2, dim*3, act=act),
            ])
        ])
        self.up = nn.ModuleList([
            nn.ModuleList([
                UpBlock(dim*2, dim, act=act),
                UpBlock(dim*2, dim, act=act),
                UpBlock(dim*2, dim, act=act),
            ]),
            nn.ModuleList([
                UpBlock(dim*3, dim*2, act=act),
                UpBlock(dim*3, dim*2, act=act),
                UpBlock(dim*3, dim*2, act=act),
            ])
        ])
        self.compress = nn.ModuleList([
            nn.Conv2d((dim + 3) * 2, dim, 3, 1, 1),
            nn.Conv2d(dim*4, dim*2, 3, 1, 1),
            nn.Conv2d(dim*6, dim*3, 3, 1, 1)
        ])
        self.to_RGB = nn.Sequential(
            act(),
            nn.Conv2d(dim, 3, 3, 1, 1)
        )

    def forward(self, pyramid):
        x_orig, x_0_0, x_1_0, x_2_0 = pyramid

        '''
        compress dim first. (32 x 2 -> 32)
        this channel reduction / compression process is not explicitly mentioned in the Softsplat paper,
        but only says it adopted the GridNet of CtxSyn, which has a dimension of 32, 64, 96 at the three levels.
        Thus this dimension compression part may be different from the original implementation.
        '''
        x_0_0 = torch.cat([x_orig, x_0_0], dim=1)
        x_0_0 = self.compress[0](x_0_0)
        x_1_0 = self.compress[1](x_1_0)
        x_2_0 = self.compress[2](x_2_0)

        # first half: down & lateral
        x_0_1 = self.lateral[0][0](x_0_0)
        x_0_2 = self.lateral[0][1](x_0_1)
        x_0_3 = self.lateral[0][2](x_0_2)

        x_1_0 = x_1_0 + self.down[0][0](x_0_0)
        x_2_0 = x_2_0 + self.down[1][0](x_1_0)

        x_1_1 = self.lateral[1][0](x_1_0)
        x_2_1 = self.lateral[2][0](x_2_0)

        x_1_1 = x_1_1 + self.down[0][1](x_0_1)
        x_2_1 = x_2_1 + self.down[1][1](x_1_1)

        x_1_2 = self.lateral[1][1](x_1_1)
        x_2_2 = self.lateral[2][1](x_2_1)

        x_1_2 = x_1_2 + self.down[0][2](x_0_2)
        x_2_2 = x_2_2 + self.down[1][2](x_1_2)

        x_1_3 = self.lateral[1][2](x_1_2)
        x_2_3 = self.lateral[2][2](x_2_2)

        # second half: up & lateral
        x_2_4 = self.lateral[2][3](x_2_3)
        x_2_5 = self.lateral[2][4](x_2_4)

        x_1_3 = x_1_3 + interpolate(self.up[1][0](x_2_3), size=x_1_3.shape[-2:], mode='bilinear', align_corners=False)
        x_0_3 = x_0_3 + interpolate(self.up[0][0](x_1_3), size=x_0_3.shape[-2:], mode='bilinear', align_corners=False)

        x_1_4 = self.lateral[1][3](x_1_3)
        x_0_4 = self.lateral[0][3](x_0_3)

        x_1_4 = x_1_4 + interpolate(self.up[1][1](x_2_4), size=x_1_4.shape[-2:], mode='bilinear', align_corners=False)
        x_0_4 = x_0_4 + interpolate(self.up[0][1](x_1_4), size=x_0_4.shape[-2:], mode='bilinear', align_corners=False)

        x_1_5 = self.lateral[1][4](x_1_4)
        x_0_5 = self.lateral[0][4](x_0_4)

        x_1_5 = x_1_5 + interpolate(self.up[1][2](x_2_5), size=x_1_5.shape[-2:], mode='bilinear', align_corners=False)
        x_0_5 = x_0_5 + interpolate(self.up[0][2](x_1_5), size=x_0_5.shape[-2:], mode='bilinear', align_corners=False)

        # final synthesis
        output = self.lateral[0][5](x_0_5)
        output = self.to_RGB(output)
        return output


class LateralBlock(nn.Module):
    def __init__(self, dim, act=nn.ReLU):
        super(LateralBlock, self).__init__()
        self.layers = nn.Sequential(
            act(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            act(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU):
        super(DownBlock, self).__init__()
        self.layers = nn.Sequential(
            act(),
            nn.Conv2d(in_dim, out_dim, 3, 2, 1),
            act(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU):
        super(UpBlock, self).__init__()
        self.layers = nn.Sequential(
            act(),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            act(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        x = interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.layers(x)